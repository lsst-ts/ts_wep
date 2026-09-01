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

"""Aperture photometry and quality metrics for candidate donuts."""

__all__ = ["MeasureDonutCandidatesConfig", "MeasureDonutCandidatesTask"]

import numpy as np
from astropy.table import QTable

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure

from .utils import _INSTRUMENT


class MeasureDonutCandidatesConfig(pexConfig.Config):
    """Config for donut candidate flux measurement and quality selection."""

    apertureOuterMarginFrac: pexConfig.Field = pexConfig.Field(
        doc=(
            "Outer edge of the main photometric aperture, as a multiple of "
            "the nominal donut radius. Adds margin beyond the nominal edge to "
            "tolerate PSF blur and centroiding error."
        ),
        dtype=float,
        default=1.05,
    )
    apertureInnerBufferFrac: pexConfig.Field = pexConfig.Field(
        doc=(
            "Inner edge of the background/blend-check region inside the "
            "obscuration, as a multiple of ``radius * obscuration``."
        ),
        dtype=float,
        default=0.67,
    )
    bkgAnnulusInnerFrac: pexConfig.Field = pexConfig.Field(
        doc=(
            "Inner edge of the outer background/blend-check annulus, as a "
            "multiple of the nominal donut radius."
        ),
        dtype=float,
        default=1.25,
    )
    bkgAnnulusOuterFrac: pexConfig.Field = pexConfig.Field(
        doc=(
            "Outer edge of the outer background/blend-check annulus, as a "
            "multiple of the nominal donut radius. Also sets the half-width "
            "of the photometry/quality-metric cutout window."
        ),
        dtype=float,
        default=1.4,
    )


class MeasureDonutCandidatesTask(pipeBase.Task):
    """Measure aperture flux and apply quality cuts to candidate donuts.

    For each candidate centroid, measures aperture flux and per-pixel noise
    over precomputed annular masks, then keeps only donuts passing the
    inner-fraction, outer-fraction, and SNR cuts. Survivors are returned
    brightest-first, truncated to ``maxDonuts``.

    The donut radius and obscuration come from the module-level instrument
    (`_INSTRUMENT`), not config -- they are fixed geometry, not tunable.
    """

    ConfigClass = MeasureDonutCandidatesConfig
    _DefaultName = "measureDonutCandidates"
    config: MeasureDonutCandidatesConfig

    def run(
        self,
        exposure: Exposure,
        selections: QTable,
        donutRadius: float | None = None,
    ) -> pipeBase.Struct:
        """Measure aperture flux and quality metrics for candidate donuts.

        Parameters
        ----------
        exposure : Exposure
            Background-subtracted post-ISR science exposure, in un-binned
            pixel coordinates.
        selections : QTable
            Catalog-selected (or blind-detection) centroids with columns
            ``centroid_x``, ``centroid_y``, ``id``.
        donutRadius : float or None, optional
            Measured donut radius in un-binned pixels, or None/NaN if
            unmeasured. If None, the nominal `_INSTRUMENT.donutRadius` is used.

        Returns
        -------
        pipeBase.Struct
            ``measurements`` : QTable
                A copy of ``selections`` with measurement columns added
                (``flux``, ``inner_frac``, ``outer_frac``,
                ``outer_sector_minmax_frac``, ``snr``, plus raw ``*_flux``,
                ``std``, ``bkg``). No rows are dropped and no ordering is
                imposed -- selection and culling happen downstream. An empty
                input is returned unmodified, without measurement columns.
        """
        if len(selections) == 0:
            return pipeBase.Struct(measurements=selections)
        return pipeBase.Struct(measurements=self._measureFlux(selections, exposure, donutRadius=donutRadius))

    def _measureFlux(self, selections: QTable, exposure: Exposure, donutRadius: float | None = None) -> QTable:
        """Measure aperture flux and per-pixel noise for each detected donut.

        For each peak a local background is estimated from an annular region
        (inner pupil + outer sky), subtracted, then flux is summed over the
        main annular aperture. Per-pixel noise is estimated from the IQR of
        first-differences in the background region.

        Returns a copy of ``selections`` with the measurement columns added;
        peaks too close to the image border get ``nan`` values. The input
        table is left unmodified.
        """
        if donutRadius is None:
            donutRadius = _INSTRUMENT.donutRadius
        radius = donutRadius
        obscuration = _INSTRUMENT.obscuration
        cfg = self.config

        arr = exposure.image.array
        half = round(radius * cfg.bkgAnnulusOuterFrac)

        gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
        r = np.hypot(gx, gy)
        sector_angle = np.arctan2(gy, gx)

        main_mask = (r < radius * cfg.apertureOuterMarginFrac) & (r > radius * obscuration)
        inner_mask = r < radius * obscuration * cfg.apertureInnerBufferFrac
        outer_mask = (r > radius * cfg.bkgAnnulusInnerFrac) & (r < radius * cfg.bkgAnnulusOuterFrac)
        bkg_mask = inner_mask | outer_mask
        n_main = np.sum(main_mask)
        outer_sector_masks = [
            outer_mask
            & (sector_angle >= -np.pi + k * np.pi / 4)
            & (sector_angle < -np.pi + (k + 1) * np.pi / 4)
            for k in range(8)
        ]

        flux_list, inner_flux_list, outer_flux_list, std_list = [], [], [], []
        outer_sector_minmax_list = []
        bkg_list = []

        for row, col in zip(selections["centroid_y"], selections["centroid_x"]):
            rmin, rmax = round(row) - half, round(row) + half + 1
            cmin, cmax = round(col) - half, round(col) + half + 1

            if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
                flux_list.append(np.nan)
                inner_flux_list.append(np.nan)
                outer_flux_list.append(np.nan)
                outer_sector_minmax_list.append(np.nan)
                std_list.append(np.nan)
                bkg_list.append(np.nan)
                continue

            stamp = arr[rmin:rmax, cmin:cmax]
            bkg = np.nanmedian(stamp[bkg_mask])
            bkg_list.append(bkg)
            stamp_sub = stamp - bkg

            flux_list.append(np.sum(stamp_sub[main_mask]))
            inner_flux_list.append(np.sum(stamp_sub[inner_mask]))
            outer_flux_list.append(np.sum(stamp_sub[outer_mask]))
            sector_fluxes = [np.sum(stamp_sub[m]) for m in outer_sector_masks]
            outer_sector_minmax_list.append(max(sector_fluxes) - min(sector_fluxes))

            diff = (stamp_sub - np.roll(stamp_sub, 1, axis=0))[bkg_mask]
            q75, q25 = np.nanpercentile(diff, [75, 25])
            std_list.append((q75 - q25) / 1.349 / np.sqrt(2))

        selections = selections.copy()  # avoid modifying the input in place
        selections["flux"] = flux_list
        selections["inner_flux"] = inner_flux_list
        selections["outer_flux"] = outer_flux_list
        selections["outer_sector_minmax_flux"] = outer_sector_minmax_list
        selections["std"] = std_list
        selections["bkg"] = bkg_list
        with np.errstate(invalid="ignore", divide="ignore"):
            selections["snr"] = (selections["flux"] / selections["std"]) / np.sqrt(n_main)
            selections["inner_frac"] = selections["inner_flux"] / selections["flux"]
            selections["outer_frac"] = selections["outer_flux"] / selections["flux"]
            selections["outer_sector_minmax_frac"] = (
                selections["outer_sector_minmax_flux"] / selections["flux"]
            )
        return selections
