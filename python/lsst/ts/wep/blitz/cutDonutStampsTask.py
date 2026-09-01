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
            "stampSize/2 >= donutRadius * apertureOuterMarginFrac."
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
    computes per-stamp geometry (field angle, nearby refcat sources, catalog
    offset) and the SAT flag, then applies the quality cuts (SAT, field
    distance, inner/outer flux fraction, SNR) to split accepted from rejected.

    Candidates are sorted flux-descending internally, so both output lists are
    filled brightest-first and the cut loop early-exits once both the accepted
    (``maxDonuts``) and rejected (``maxRejectDonuts``) buckets are full --
    avoiding stamp cuts on the faint tail that would be discarded anyway.

    Donut radius and obscuration come from the module-level instrument
    (`_INSTRUMENT`), not config.
    """

    ConfigClass = CutDonutStampsConfig
    _DefaultName = "cutDonutStamps"
    config: CutDonutStampsConfig

    def run(
        self,
        exposure: Exposure,
        measurements: QTable,
        refcat: QTable | None,
        blindDetections: QTable,
        donutRadius: float | None = None,
    ) -> pipeBase.Struct:
        """Cut stamps and split accepted vs. quality-rejected donuts.

        Parameters
        ----------
        exposure : Exposure
            Background-subtracted post-ISR science exposure.
        measurements : QTable
            Measured candidates from the measurement task, with columns
            ``id``, ``centroid_x``, ``centroid_y``, ``flux``, ``inner_frac``,
            ``outer_frac``, ``outer_sector_minmax_frac``, ``snr``, ``std``,
            ``bkg``. Photometric metrics are carried onto the Donut objects
            as-is; only geometry and the SAT flag are computed here. Row order
            is not assumed -- the table is sorted flux-descending internally
            before stamps are cut.
        refcat : QTable or None
            Full refcat with ``centroid_x``, ``centroid_y``, ``photo_mag``,
            ``astrom_mag``, or ``None`` in the blind-detection fallback.
        blindDetections : QTable
            Centroids from ``BlindDetect``, used for
            ``catalog_centroid_offset_px``.
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
        obscuration = _INSTRUMENT.obscuration

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
            _rc_mag = {
                "photo_mag": np.asarray(refcat["photo_mag"], dtype=float),
                "astrom_mag": np.asarray(refcat["astrom_mag"], dtype=float),
            }
        else:
            _rc_x = _rc_y = None
            _rc_mag = {}

        def _cut_stamp(row, blind_cx=None, blind_cy=None) -> Donut | None:
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
                box_mask = (np.abs(_rc_x - cx) <= half_before) & (np.abs(_rc_y - cy) <= half_before)
                dx_box = _rc_x[box_mask] - cx
                dy_box = _rc_y[box_mask] - cy

            def _nearby(mag_col):
                if box_mask is None:
                    return []
                mag_box = _rc_mag[mag_col][box_mask]
                return list(zip(dx_box.tolist(), dy_box.tolist(), mag_box.tolist()))

            _fa = detector.transform(
                [lsst.geom.Point2D(cx_f, cy_f)], PIXELS, FIELD_ANGLE
            )[0]
            _field_dist_deg = np.degrees(np.hypot(_fa[0], _fa[1]))

            _nearby_photo_list = _nearby("photo_mag")
            _neighbor_dists = [
                np.hypot(dx, dy)
                for dx, dy, _ in _nearby_photo_list
                if np.hypot(dx, dy) >= 1.0
            ]

            _catalog_centroid_offset_px = (
                np.hypot(cx_f - blind_cx, cy_f - blind_cy)
                if blind_cx is not None and blind_cy is not None
                else float("nan")
            )

            rejected_sat = bool(np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit))
            rejected_inner_frac = bool(np.isfinite(row["inner_frac"]) and abs(row["inner_frac"]) > self.config.innerFracThreshold)
            rejected_outer_frac = bool(np.isfinite(row["outer_frac"]) and abs(row["outer_frac"]) > self.config.outerFracThreshold)
            rejected_snr = bool(np.isfinite(row["snr"]) and row["snr"] < self.config.minStampSnr)
            rejected = rejected_sat or rejected_inner_frac or rejected_outer_frac or rejected_snr

            return Donut(
                det_name=detector.getName(),
                stamp=stamp_ccs,
                fa_x_ccs=_fa[1],
                fa_y_ccs=_fa[0],
                flux=row["flux"],
                band=band,
                det_id=det_id,
                visit_id=visit_id,
                centroid_x_raw=cx_f,
                centroid_y_raw=cy_f,
                id=row["id"],
                inner_frac=row["inner_frac"],
                outer_frac=row["outer_frac"],
                outer_sector_minmax_frac=row["outer_sector_minmax_frac"],
                field_dist_deg=_field_dist_deg,
                donut_radius=donutRadius,
                obscuration=obscuration,
                snr=row["snr"],
                bkg=row["bkg"],
                bkg_std=row["std"],
                nearest_neighbor_dist_px=(
                    min(_neighbor_dists) if _neighbor_dists else float("nan")
                ),
                n_neighbors_in_stamp=len(_neighbor_dists),
                catalog_centroid_offset_px=_catalog_centroid_offset_px,
                n_quarter=n_quarter,
                nearby_photo=_nearby_photo_list,
                nearby_astrom=_nearby("astrom_mag"),
                rejected_sat=rejected_sat,
                rejected_inner_frac=rejected_inner_frac,
                rejected_outer_frac=rejected_outer_frac,
                rejected_snr=rejected_snr,
                rejected=rejected
            )

        # Match each centroid to the nearest blind detection for
        # catalog_centroid_offset_px.
        _blind_cx = (
            np.array(blindDetections["centroid_x"]) if len(blindDetections) > 0 else np.empty(0)
        )
        _blind_cy = (
            np.array(blindDetections["centroid_y"]) if len(blindDetections) > 0 else np.empty(0)
        )
        _match_tol = donutRadius * 0.5

        def _nearest_blind(cx_f, cy_f):
            if len(_blind_cx) == 0:
                return None, None
            dists = np.hypot(_blind_cx - cx_f, _blind_cy - cy_f)
            idx = np.argmin(dists)
            return (
                (_blind_cx[idx], _blind_cy[idx])
                if dists[idx] <= _match_tol
                else (None, None)
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
            _b_cx, _b_cy = _nearest_blind(row["centroid_x"], row["centroid_y"])
            d = _cut_stamp(row, blind_cx=_b_cx, blind_cy=_b_cy)
            if d is None:
                continue
            if d.rejected:
                if len(rejected_donuts) < max_reject:
                    rejected_donuts.append(d)
                continue
            if len(donuts) < max_donuts:
                donuts.append(d)

        return pipeBase.Struct(donuts=donuts, rejected_donuts=rejected_donuts)
