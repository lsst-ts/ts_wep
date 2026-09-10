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

import numpy as np
from astropy.table import QTable

import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.afw.image import Exposure

from .dataStructures import Donut
from .utils import _INSTRUMENT


class CutDonutStampsConfig(pexConfig.Config):
    """Config for cutting donut stamps and evaluating rejection criteria."""

    stampSize: pexConfig.Field = pexConfig.Field(
        doc=(
            "Side length in pixels of the square stamp cut around each donut "
            "centroid. The binned size (stampSize // binning) must be odd for "
            "the Danish fitting stage. The default 167 bins down to odd sizes "
            "for binnings of 1 through 7 (167/83/55/41/33/27/23). The binning "
            "stage forces the result to be odd if needed. Must also be large "
            "enough to contain the main photometric annulus: "
            "stampSize/2 >= donutRadius * (1 + apertureMarginFrac)."
        ),
        dtype=int,
        default=167,
    )
    innerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc="Reject a donut if |inner_frac| exceeds this.",
        dtype=float,
        default=0.1,
    )
    outerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc="Reject a donut if |outer_frac| exceeds this.",
        dtype=float,
        default=0.1,
    )
    minStampSnr: pexConfig.Field = pexConfig.Field(
        doc="Reject a donut if its per-stamp SNR falls below this.",
        dtype=float,
        default=100.0,
    )
    maxDonuts: pexConfig.Field = pexConfig.Field(
        doc="Maximum number of accepted donuts to keep per detector, brightest-first.",
        dtype=int,
        default=8,
    )
    maxRejectDonuts: pexConfig.Field = pexConfig.Field(
        doc=(
            "Maximum number of quality-rejected donuts to keep per detector "
            "(brightest-first) for downstream diagnostics."
        ),
        dtype=int,
        default=8,
    )


class CutDonutStampsTask(pipeBase.Task):
    """Cut donut stamps and evaluate rejection criteria.

    For each measured candidate, cuts a stamp centered on the centroid,
    computes per-stamp geometry (field angle, nearby refcat sources) and the SAT
    flag, then applies the quality cuts (SAT, inner/outer flux fraction, SNR) to
    split accepted from rejected.

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
        donutRadius: float | None = None,
    ) -> pipeBase.Struct:
        """Cut stamps and split accepted vs. quality-rejected donuts.

        Parameters
        ----------
        exposure : Exposure
            Background-subtracted post-ISR science exposure.
        measurements : QTable
            Measured candidates from the measurement task, with columns
            ``donut_id``, ``centroid_x``, ``centroid_y``, ``flux``, ``inner_frac``,
            ``outer_frac``, ``outer_sector_minmax_frac``, ``snr``, ``bkg_std``,
            ``bkg``, plus every column in
            `lsst.ts.wep.blitz.utils._REFCAT_COLUMNS` -- present whichever
            selection path ran, NaN-valued on the blind-detection one.
            Photometric metrics are carried onto the Donut objects as-is; only
            geometry and the SAT flag are computed here. Row order is not
            assumed -- the table is sorted flux-descending internally before
            stamps are cut.
        refcat : QTable or None
            Full refcat with ``donut_id``, ``centroid_x``, ``centroid_y``,
            ``photo_mag``, ``astrom_mag``, or ``None`` in the blind-detection
            fallback. Supplies the nearby-source lists; ``donut_id`` is what
            excludes each donut from its own neighbour list.
        donutRadius : float or None, optional
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
        if donutRadius is None:
            donutRadius = _INSTRUMENT.donutRadius

        detector = exposure.getDetector()
        band = exposure.filter.bandLabel
        visit_id = exposure.getInfo().getVisitInfo().id
        det_id = detector.getId()
        n_quarter = detector.getOrientation().getNQuarter()
        half = self.config.stampSize // 2

        arr = exposure.image.array
        mask_arr = exposure.mask.array
        sat_bit = exposure.mask.getPlaneBitMask("SAT")

        # Sort candidates flux-descending up front so the fill loop below keeps
        # brightest-first and can early-exit once both buckets are full, without
        # depending on the upstream measurement task's row order.
        if len(measurements) > 1:
            measurements = measurements[np.argsort(measurements["flux"])[::-1]]

        if refcat is not None:
            _rc_x = np.asarray(refcat["centroid_x"], dtype=float)
            _rc_y = np.asarray(refcat["centroid_y"], dtype=float)
            _rc_id = np.asarray(refcat["donut_id"])
            _rc_mag = {
                "photo_mag": np.asarray(refcat["photo_mag"], dtype=float),
                "astrom_mag": np.asarray(refcat["astrom_mag"], dtype=float),
            }
        else:
            _rc_x = _rc_y = _rc_id = None
            _rc_mag = {}

        def _cut_stamp(row) -> Donut | None:
            """Cut one stamp and compute metrics. Returns Donut or None on failure."""
            # Cut a stamp of configured size, centered on the rounded centroid.
            # Odd-size preference is enforced during binning in _prep_donut_for_danish.
            cx_f = row["centroid_x"]
            cy_f = row["centroid_y"]
            cx, cy = round(cx_f), round(cy_f)
            half_before = half
            half_after = self.config.stampSize - half_before - 1
            rmin, rmax = cy - half_before, cy + half_after + 1
            cmin, cmax = cx - half_before, cx + half_after + 1
            if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
                return None
            stamp = np.array(arr[rmin:rmax, cmin:cmax])
            stamp_ccs = np.rot90(stamp, k=-n_quarter).T

            # Vectorized box query over the precomputed refcat arrays.
            # Offsets are relative to the *rounded* centroid (cx, cy).
            if _rc_x is None:
                box_mask = None
                dx_box = dy_box = None
            else:
                # Membership is against the *rounded* centroid, because that is
                # what the stamp bounds were cut on -- these are the sources
                # actually inside the stamp.
                box_mask = (np.abs(_rc_x - cx) <= half_before) & (np.abs(_rc_y - cy) <= half_before)
                # Drop this donut itself: it is a refcat source too, so the box
                # always contains it at zero offset. Matched on refcat id rather
                # than on a distance threshold -- `refcat` is non-None only when
                # the selections were drawn from it, so the id comparison is
                # exact.
                box_mask &= _rc_id != row["donut_id"]
                # Offsets are from cx_f/cy_f, not the rounded cx/cy, so that
                # ``x_det + nearby_*_dx_det`` is the neighbour's detector x with
                # no correction term. Anything wanting stamp-display coordinates
                # has to add the rounding residual ``x_det - round(x_det)``; see
                # `_xform` in donutBlitzPlotTask.
                dx_box = _rc_x[box_mask] - cx_f
                dy_box = _rc_y[box_mask] - cy_f

            def _nearby(mag_col):
                if box_mask is None:
                    return []
                mag_box = _rc_mag[mag_col][box_mask]
                return list(zip(dx_box.tolist(), dy_box.tolist(), mag_box.tolist()))

            _fa = detector.transform(
                [lsst.geom.Point2D(cx_f, cy_f)], PIXELS, FIELD_ANGLE
            )[0]

            rejected_sat = bool(np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit))
            rejected_inner_frac = bool(np.isfinite(row["inner_frac"]) and abs(row["inner_frac"]) > self.config.innerFracThreshold)
            rejected_outer_frac = bool(np.isfinite(row["outer_frac"]) and abs(row["outer_frac"]) > self.config.outerFracThreshold)
            rejected_snr = bool(np.isfinite(row["snr"]) and row["snr"] < self.config.minStampSnr)
            rejected = rejected_sat or rejected_inner_frac or rejected_outer_frac or rejected_snr

            return Donut(
                det_name=detector.getName(),
                stamp=stamp_ccs,
                thx_ccs=_fa[1],
                thy_ccs=_fa[0],
                flux=row["flux"],
                band=band,
                det_id=det_id,
                visit_id=visit_id,
                x_det=cx_f,
                y_det=cy_f,
                donut_id=row["donut_id"],
                inner_frac=row["inner_frac"],
                outer_frac=row["outer_frac"],
                outer_sector_minmax_frac=row["outer_sector_minmax_frac"],
                donut_radius=donutRadius,
                snr=row["snr"],
                bkg=row["bkg"],
                bkg_std=row["bkg_std"],
                n_quarter=n_quarter,
                # The donut's own refcat values ride the selections table, a row
                # subset of the refcat on that path. `_REFCAT_COLUMNS` guarantees
                # they are present on the blind path too, NaN-filled, so there is
                # nothing to test for here.
                photo_mag=float(row["photo_mag"]),
                astrom_mag=float(row["astrom_mag"]),
                coord_ra=float(row["coord_ra"]),
                coord_dec=float(row["coord_dec"]),
                nearby_photo=_nearby("photo_mag"),
                nearby_astrom=_nearby("astrom_mag"),
                rejected_sat=rejected_sat,
                rejected_inner_frac=rejected_inner_frac,
                rejected_outer_frac=rejected_outer_frac,
                rejected_snr=rejected_snr,
                rejected=rejected
            )

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
            d = _cut_stamp(row)
            if d is None:
                continue
            if d.rejected:
                if len(rejected_donuts) < max_reject:
                    rejected_donuts.append(d)
                continue
            if len(donuts) < max_donuts:
                donuts.append(d)

        return pipeBase.Struct(donuts=donuts, rejected_donuts=rejected_donuts)
