# This file is part of ts_wep.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Stamp cutting and donut quality rejection."""

__all__ = ["CutDonutStampsConfig", "CutDonutStampsTask"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from astropy.table import QTable

import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Detector
from lsst.afw.image import Exposure

from .dataStructures import Donut
from .utils import _INSTRUMENT


@dataclass(frozen=True)
class _ExposureContext:
    """Per-exposure values every stamp in one `run` call shares.

    Extracted from the exposure once, so `CutDonutStampsTask._cut_stamp` can be
    a method taking explicit inputs rather than a closure over `run`'s locals.
    Frozen because nothing downstream of the extraction may rebind any of it;
    the two arrays are views into the exposure's pixels and are only read.
    """

    detector: Detector
    band: str
    visit_id: int
    det_id: int
    n_quarter: int
    # Image and mask pixels, and the bit `rejected_sat` tests for.
    image: npt.NDArray[np.float64]
    mask: npt.NDArray[np.integer]
    sat_bit: int

    @classmethod
    def from_exposure(cls, exposure: Exposure) -> "_ExposureContext":
        """Read the per-exposure values off a post-ISR exposure."""
        detector = exposure.getDetector()
        return cls(
            detector=detector,
            band=exposure.filter.bandLabel,
            visit_id=exposure.getInfo().getVisitInfo().id,
            det_id=detector.getId(),
            n_quarter=detector.getOrientation().getNQuarter(),
            image=exposure.image.array,
            mask=exposure.mask.array,
            sat_bit=exposure.mask.getPlaneBitMask("SAT"),
        )


@dataclass(frozen=True)
class _RefcatArrays:
    """The refcat columns the nearby-source box query needs, as plain arrays.

    Pulled out of the `QTable` once per `run` call rather than per donut: the
    query is a vectorized comparison against all of them, and column access on
    a `QTable` is not free.

    `None` on the blitz-detection path, where there is no refcat at all -- so a
    caller holding `None` skips the query entirely and every donut gets empty
    neighbor lists.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    donut_id: npt.NDArray
    photo_mag: npt.NDArray[np.float64]
    astrom_mag: npt.NDArray[np.float64]

    @classmethod
    def from_table(cls, refcat: QTable | None) -> "_RefcatArrays | None":
        """Extract the arrays, or `None` if there is no refcat."""
        if refcat is None:
            return None
        return cls(
            x=np.asarray(refcat["centroid_x"], dtype=float),
            y=np.asarray(refcat["centroid_y"], dtype=float),
            donut_id=np.asarray(refcat["donut_id"]),
            photo_mag=np.asarray(refcat["photo_mag"], dtype=float),
            astrom_mag=np.asarray(refcat["astrom_mag"], dtype=float),
        )

    def nearby(
        self,
        donut_id,
        x_det: float,
        y_det: float,
        half: int,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        """Both nearby-source lists for one donut's stamp box.

        Parameters
        ----------
        donut_id
            The donut's own refcat id, excluded from its own neighbor lists.
        x_det, y_det : float
            The donut's un-rounded centroid, in detector pixels.
        half : int
            Half the stamp side, ``stampSize // 2``.

        Returns
        -------
        nearby_photo, nearby_astrom : list of (float, float, float)
            ``(dx, dy, mag)`` per neighbor, the two lists differing only in
            which magnitude they carry. The box query is shared.
        """
        # Membership is against the *rounded* centroid, because that is what
        # the stamp bounds were cut on -- these are the sources actually
        # inside the stamp.
        cx, cy = round(x_det), round(y_det)
        box_mask = (np.abs(self.x - cx) <= half) & (np.abs(self.y - cy) <= half)
        # Drop this donut itself: it is a refcat source too, so the box always
        # contains it at zero offset. Matched on refcat id rather than on a
        # distance threshold -- this object exists only when the selections
        # were drawn from the refcat, so the id comparison is exact.
        box_mask &= self.donut_id != donut_id
        # Offsets are from x_det/y_det, not the rounded cx/cy, so that
        # ``x_det + nearby_*_dx_det`` is the neighbor's detector x with no
        # correction term. Anything wanting stamp-display coordinates has to
        # add the rounding residual ``x_det - round(x_det)``; see `_xform` in
        # donutBlitzPlot.
        dx_box = (self.x[box_mask] - x_det).tolist()
        dy_box = (self.y[box_mask] - y_det).tolist()
        return (
            list(zip(dx_box, dy_box, self.photo_mag[box_mask].tolist())),
            list(zip(dx_box, dy_box, self.astrom_mag[box_mask].tolist())),
        )


class CutDonutStampsConfig(pexConfig.Config):
    """Config for cutting donut stamps and evaluating rejection criteria."""

    stampSize: pexConfig.Field[int] = pexConfig.Field[int](
        doc=(
            "Side length in pixels of the square stamp cut around each donut "
            "centroid. The binned size (stampSize // binning) must be odd for "
            "the Danish fitting stage. The default 167 bins down to odd sizes "
            "for binnings of 1 through 7 (167/83/55/41/33/27/23). The binning "
            "stage forces the result to be odd if needed. Must also be large "
            "enough to contain the main photometric annulus: "
            "stampSize/2 >= donut_radius * (1 + apertureMarginFrac)."
        ),
        default=167,
    )
    innerFracThreshold: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Reject a donut if |inner_frac| exceeds this.",
        default=0.1,
    )
    outerFracThreshold: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Reject a donut if |outer_frac| exceeds this.",
        default=0.1,
    )
    minStampSnr: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Reject a donut if its per-stamp SNR falls below this.",
        default=100.0,
    )
    maxDonuts: pexConfig.Field[int] = pexConfig.Field[int](
        doc="Maximum number of accepted donuts to keep per detector, brightest-first.",
        default=8,
    )
    maxRejectDonuts: pexConfig.Field[int] = pexConfig.Field[int](
        doc=(
            "Maximum number of quality-rejected donuts to keep per detector "
            "(brightest-first) for downstream diagnostics."
        ),
        default=8,
    )


class CutDonutStampsTask(pipeBase.Task):
    """Cut donut stamps and evaluate rejection criteria.

    For each measured candidate, cuts a stamp centered on the centroid,
    computes per-stamp geometry (field angle, nearby refcat sources) and the
    SAT flag, then applies the quality cuts (SAT, inner/outer flux fraction,
    SNR) to split accepted from rejected.

    Candidates are sorted flux-descending internally, so both output lists are
    filled brightest-first and the cut loop early-exits once both the accepted
    (``maxDonuts``) and rejected (``maxRejectDonuts``) buckets are full --
    avoiding stamp cuts on the faint tail that would be discarded anyway.

    Donut radius comes from the module-level instrument (`_INSTRUMENT`), not
    config.
    """

    ConfigClass = CutDonutStampsConfig
    _DefaultName = "cutDonutStamps"
    config: CutDonutStampsConfig

    def run(
        self,
        exposure: Exposure,
        measurements: QTable,
        refcat: QTable | None,
        donut_radius: float | None = None,
    ) -> pipeBase.Struct:
        """Cut stamps and split accepted vs. quality-rejected donuts.

        Parameters
        ----------
        exposure : Exposure
            Background-subtracted post-ISR science exposure.
        measurements : QTable
            Measured candidates from the measurement task, with columns
            ``donut_id``, ``centroid_x``, ``centroid_y``, ``flux``,
            ``inner_frac``, ``outer_frac``, ``outer_sector_minmax_frac``,
            ``snr``, ``bkg_std``, ``bkg``, plus every column in
            `lsst.ts.wep.blitz.utils._REFCAT_COLUMNS` -- present whichever
            selection path ran, NaN-valued on the blitz-detection one.
            Photometric metrics are carried onto the Donut objects as-is; only
            geometry and the SAT flag are computed here. Row order is not
            assumed -- the table is sorted flux-descending internally before
            stamps are cut.
        refcat : QTable or None
            Full refcat with ``donut_id``, ``centroid_x``, ``centroid_y``,
            ``photo_mag``, ``astrom_mag``, or ``None`` in the blitz-detection
            fallback. Supplies the nearby-source lists; ``donut_id`` is what
            excludes each donut from its own neighbor list.
        donut_radius : float or None, optional
            Measured donut radius in un-binned pixels, or None/NaN if
            unmeasured. If None, the nominal `_INSTRUMENT.donutRadius` is used.

        Returns
        -------
        pipeBase.Struct
            ``donuts`` : list of Donut
                Accepted donuts, brightest-first, at most ``maxDonuts``.
            ``rejected_donuts`` : list of Donut
                Quality-rejected donuts, brightest-first, at most
                ``maxRejectDonuts``.
        """
        if donut_radius is None:
            donut_radius = _INSTRUMENT.donutRadius

        context = _ExposureContext.from_exposure(exposure)
        refcat_arrays = _RefcatArrays.from_table(refcat)

        # Sort candidates flux-descending up front so the fill loop below keeps
        # brightest-first and can early-exit once both buckets are full,
        # without depending on the upstream measurement task's row order.
        if len(measurements) > 1:
            measurements = measurements[np.argsort(measurements["flux"])[::-1]]

        max_donuts = self.config.maxDonuts
        max_reject = self.config.maxRejectDonuts

        # --- Single pass over candidates: accept or quality-reject ---
        # Input is flux-descending, so both lists fill brightest-first. Once
        # both buckets are full, every remaining candidate is fainter than
        # everything kept, so we stop cutting stamps entirely.
        donuts: list[Donut] = []
        rejected_donuts: list[Donut] = []
        for row in measurements:
            if len(donuts) >= max_donuts and len(rejected_donuts) >= max_reject:
                break
            d = self._cut_stamp(row, context, refcat_arrays, donut_radius)
            if d is None:
                continue
            if d.rejected:
                if len(rejected_donuts) < max_reject:
                    rejected_donuts.append(d)
                continue
            if len(donuts) < max_donuts:
                donuts.append(d)

        return pipeBase.Struct(donuts=donuts, rejected_donuts=rejected_donuts)

    def _cut_stamp(
        self,
        row,
        context: _ExposureContext,
        refcat_arrays: _RefcatArrays | None,
        donut_radius: float,
    ) -> Donut | None:
        """Cut one stamp and compute its metrics; None if it runs off the edge.

        Parameters
        ----------
        row
            One row of the measurements table; see `run`.
        context : _ExposureContext
            The per-exposure values shared by every stamp in this call.
        refcat_arrays : _RefcatArrays or None
            The refcat columns for the nearby-source query, or None on the
            blitz-detection path, where the neighbor lists come back empty.
        donut_radius : float
            Measured donut radius in un-binned pixels, carried onto the Donut.

        Returns
        -------
        Donut or None
            None if the stamp box falls outside the image, in which case the
            candidate is dropped.
        """
        # Cut a stamp of configured size, centered on the rounded centroid.
        # Odd-size preference is enforced during binning in
        # _prep_donut_for_danish.
        x_det = row["centroid_x"]
        y_det = row["centroid_y"]
        cx, cy = round(x_det), round(y_det)
        half_before = self.config.stampSize // 2
        half_after = self.config.stampSize - half_before - 1
        rmin, rmax = cy - half_before, cy + half_after + 1
        cmin, cmax = cx - half_before, cx + half_after + 1
        image = context.image
        if rmin < 0 or rmax > image.shape[0] or cmin < 0 or cmax > image.shape[1]:
            return None
        stamp = np.array(image[rmin:rmax, cmin:cmax])
        stamp_ccs = np.rot90(stamp, k=-context.n_quarter).T

        if refcat_arrays is None:
            nearby_photo, nearby_astrom = [], []
        else:
            nearby_photo, nearby_astrom = refcat_arrays.nearby(row["donut_id"], x_det, y_det, half_before)

        field_angle = context.detector.transform([lsst.geom.Point2D(x_det, y_det)], PIXELS, FIELD_ANGLE)[0]

        rejected_sat = bool(np.any(context.mask[rmin:rmax, cmin:cmax] & context.sat_bit))
        rejected_inner_frac = bool(
            np.isfinite(row["inner_frac"]) and abs(row["inner_frac"]) > self.config.innerFracThreshold
        )
        rejected_outer_frac = bool(
            np.isfinite(row["outer_frac"]) and abs(row["outer_frac"]) > self.config.outerFracThreshold
        )
        rejected_snr = bool(np.isfinite(row["snr"]) and row["snr"] < self.config.minStampSnr)
        rejected = rejected_sat or rejected_inner_frac or rejected_outer_frac or rejected_snr

        return Donut(
            det_name=context.detector.getName(),
            stamp=stamp_ccs,
            thx_ccs=field_angle[1],  # DVCS -> CCS (sitcomtn-003)
            thy_ccs=field_angle[0],
            flux=row["flux"],
            band=context.band,
            det_id=context.det_id,
            visit_id=context.visit_id,
            x_det=x_det,
            y_det=y_det,
            donut_id=row["donut_id"],
            inner_frac=row["inner_frac"],
            outer_frac=row["outer_frac"],
            outer_sector_minmax_frac=row["outer_sector_minmax_frac"],
            donut_radius=donut_radius,
            snr=row["snr"],
            bkg=row["bkg"],
            bkg_std=row["bkg_std"],
            n_quarter=context.n_quarter,
            # The donut's own refcat values ride the selections table, a row
            # subset of the refcat on that path. `_REFCAT_COLUMNS` guarantees
            # they are present on the blitz path too, NaN-filled, so there is
            # nothing to test for here.
            photo_mag=float(row["photo_mag"]),
            astrom_mag=float(row["astrom_mag"]),
            coord_ra=float(row["coord_ra"]),
            coord_dec=float(row["coord_dec"]),
            nearby_photo=nearby_photo,
            nearby_astrom=nearby_astrom,
            rejected_sat=rejected_sat,
            rejected_inner_frac=rejected_inner_frac,
            rejected_outer_frac=rejected_outer_frac,
            rejected_snr=rejected_snr,
            rejected=rejected,
        )
