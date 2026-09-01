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

__all__ = [
    "DonutBlitzMonolithTaskConnections",
    "DonutBlitzMonolithTaskConfig",
    "DonutBlitzMonolithTask",
    "DonutBlitzPlotTaskConfig",
    "DonutBlitzPlotTask",
]



import ast
import contextlib
import logging
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import batoid
import danish
import galsim
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import numpy as np
import numpy.typing as npt
from astropy.table import Table, QTable
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.afw.geom import SkyWcs
from lsst.afw.image import Exposure
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import (
    MagnitudeLimit,
    ReferenceObjectLoader,
    SubtractBackgroundTask,
)
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
)
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.donutDetectDiameterTask import DonutDetectDiameterTask
from lsst.ts.wep.utils import binArray
from lsst.utils.timer import timeMethod
from scipy.optimize import least_squares
from scipy.signal import correlate
from scipy.stats import median_abs_deviation
from skimage.feature import peak_local_max

_CALIB_STORE: dict = {}  # populated in parent before fork; workers inherit via COW

# Hard coding global wavefront sensor geometry for now
_INSTRUMENT: Instrument = Instrument(configFile="policy:instruments/LsstCam.yaml")

_EXTRA_FOCAL_DET_IDS = frozenset({191, 195, 199, 203})
_INTRA_FOCAL_DET_IDS = frozenset({192, 196, 200, 204})

# DZMultiDonutModel's field_radius has no effect on our fit (we don't model any
# field-dependent optics term), so any value works; set to roughly the Rubin
# field of view for a physically sensible default.
_DANISH_FIELD_RADIUS_RAD = np.deg2rad(1.85)

# SW0 = extra-focal, SW1 = intra-focal
CORNER_PAIRS = {
    "R00": ("R00_SW0", "R00_SW1"),
    "R04": ("R04_SW0", "R04_SW1"),
    "R40": ("R40_SW0", "R40_SW1"),
    "R44": ("R44_SW0", "R44_SW1"),
}
CORNER_DET_NAMES = frozenset(s for sw0, sw1 in CORNER_PAIRS.values() for s in (sw0, sw1))
# Detector name -> corner, derived from CORNER_PAIRS rather than re-encoded.
CORNER_BY_DET_NAME = {s: corner for corner, pair in CORNER_PAIRS.items() for s in pair}

# Display names for DonutSourceSelectorTask rejection reasons. Reasons absent
# here are shown verbatim. "faint_neighbor" folds into "blend" (both mean a
# neighbor spoiled this donut).
_SELECTOR_REASON_DISP = {
    "faint_neighbor": "blend",
    "too_many_blends": "many",
    "source_limit": "limit",
}

# Colors below are drawn from the colorblind-safe Okabe-Ito palette.
_COLOR_APERTURE = "#56B4E9"
_COLOR_BKG_ANNULUS = "#E69F00"
_COLOR_REJECTED = "#D55E00"
_COLOR_PHOTO_REFCAT = "#56B4E9"
_COLOR_ASTROM_REFCAT = "#E69F00"
_COLOR_CMAP_NEG = "#0072B2"
_COLOR_CMAP_MID = "#56B4E9"
_COLOR_CMAP_POS = "#D55E00"
_COLOR_COMA = "#F0E442"
_COLOR_ASTIGMATISM = "#E69F00"
_COLOR_TREFOIL = "#009E73"
_COLOR_QUADRAFOIL = "#0072B2"
_COLOR_PENTAFOIL = "#CC79A7"
_COLOR_HEXAFOIL = "#D55E00"

# ANSI escape codes for colorizing log messages (see colorLog config field).
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BLUE = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN = "\033[36m"

# Maximum nearby sources to store in the output table for each donut.
_MAX_NEARBY = 5


def _resolveColorLogEnabled(colorLog: bool | None) -> bool:
    """Resolve the colorLog config value to a concrete enabled/disabled bool.

    If ``colorLog`` is None, color is enabled only when stdout is attached
    to an interactive terminal.
    """
    if colorLog is None:
        return sys.stdout.isatty()
    return colorLog


def _colorize(text: str, *codes: str, enabled: bool = True) -> str:
    """Wrap ``text`` in the given ANSI escape code(s) if ``enabled``.

    Parameters
    ----------
    text : str
        The text to colorize.
    *codes : str
        One or more ANSI escape codes (e.g. ``_ANSI_RED``, ``_ANSI_BOLD``).
    enabled : bool, optional
        If False, ``text`` is returned unchanged. (the default is True)

    Returns
    -------
    str
        The colorized (or original) text.
    """
    if not enabled or not codes:
        return text
    return "".join(codes) + text + _ANSI_RESET


def _resolveDonutRadius(donutRadius: float | None) -> float:
    """Return a usable donut radius in pixels, falling back to the nominal one.

    The per-exposure radius measured by `DonutDetectDiameterTask` is preferred,
    but it is NaN whenever the sizing curve could not be formed (no surviving
    peaks, monotonic curve). Rather than propagate NaN into every mask radius
    downstream, fall back to the nominal `_INSTRUMENT.donutRadius`.

    Parameters
    ----------
    donutRadius : float or None
        Measured donut radius in un-binned pixels, or None/NaN if unmeasured.

    Returns
    -------
    float
        ``donutRadius`` if finite and positive, else ``_INSTRUMENT.donutRadius``.
    """
    if donutRadius is None:
        return _INSTRUMENT.donutRadius
    if not np.isfinite(donutRadius) or donutRadius <= 0:
        return _INSTRUMENT.donutRadius
    return donutRadius


def _buildAnnularTemplate(radius: float, innerFrac: float) -> np.ndarray:
    """Return a binary annular stamp for cross-correlation donut detection."""
    r_int = round(radius)
    cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
    r = np.hypot(cx, cy)
    return np.where((r < radius) & (r >= radius * innerFrac), 1.0, 0.0)


def _buildAfwSourceCat(blindDetections: QTable, wcs: SkyWcs) -> afwTable.SourceCatalog:
    """Convert blind-detect QTable into a minimal afwTable.SourceCatalog
    suitable for AstrometryTask.run().
    """
    sourceSchema = afwTable.SourceTable.makeMinimalSchema()
    measBase.SingleFrameMeasurementTask(schema=sourceSchema)

    sourceCat = afwTable.SourceCatalog(sourceSchema)
    sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
    sourceIdKey = sourceSchema["id"].asKey()
    sourceRAKey = sourceSchema["coord_ra"].asKey()
    sourceDecKey = sourceSchema["coord_dec"].asKey()

    # reserve() allocates the records as one block, which is what makes the
    # finished catalog contiguous -- AstrometryTask requires that. Without it,
    # addNew() spills into further blocks past ~100 records and the catalog
    # would need an explicit copy(deep=True) to compact it.
    sourceCat.reserve(len(blindDetections))
    for i, row in enumerate(blindDetections):
        x, y = row["centroid_x"], row["centroid_y"]
        sky = wcs.pixelToSky(x, y)
        src = sourceCat.addNew()
        src.set(sourceIdKey, i)
        src.set(sourceRAKey, sky.getRa())
        src.set(sourceDecKey, sky.getDec())
        src.set(sourceCentroidKey, lsst.geom.Point2D(x, y))

    return sourceCat


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


@dataclass
class Donut:
    """One cut donut stamp with its selection/quality metrics."""

    det_name: str
    stamp: np.ndarray  #  CCS
    fa_x_ccs: float
    fa_y_ccs: float
    flux: float
    band: str
    det_id: int
    visit_id: int
    centroid_x_raw: float
    centroid_y_raw: float
    id: int
    inner_frac: float
    outer_frac: float
    outer_sector_minmax_frac: float
    field_dist_deg: float
    donut_radius: float
    obscuration: float
    snr: float
    bkg: float
    bkg_std: float
    nearest_neighbor_dist_px: float
    n_neighbors_in_stamp: int
    catalog_centroid_offset_px: float
    n_quarter: int
    nearby_photo: list[tuple[float, float, float]]
    nearby_astrom: list[tuple[float, float, float]]
    intrinsic_zk: npt.NDArray[np.float64] | None = None
    # --- reject flags (default False = not rejected) ---
    rejected_sat: bool = False
    rejected_inner_frac: bool = False
    rejected_outer_frac: bool = False
    rejected_snr: bool = False
    rejected: bool = False


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


def _cutoutPipeline(det_name: str, t_dispatch: float) -> dict:
    """Run ISR, background subtraction, blind detection, WCS refit, catalog
    selection, and stamp cutting.

    Orchestrates the full per-detector cutout pipeline in a worker process.
    All inputs are read from the module-level ``_CALIB_STORE`` dict, which is
    populated by the parent process before forking.

    Parameters
    ----------
    det_name : str
        Detector name; used to look up per-detector calibrations in
        ``_CALIB_STORE``.
    t_dispatch : float
        ``time.time()`` timestamp at which the task was dispatched from the
        parent, used to measure dispatch-to-arrival latency.

    Returns
    -------
    dict
        Keys: ``det_name``, ``catalog`` (accepted donut dicts),
        ``rejected_catalog``, ``scatter_arcsec``, ``wcs_refit_error``,
        ``cat_select_error``, and timing floats ``dispatch_to_arrival``,
        ``isr_run``, ``bkg_run``, ``diam_run``, ``blind_detect_run``,
        ``wcs_refit_run``, ``catalog_select_run``, ``stamp_cut_run``.
    """
    t_arrival = time.time()
    entry = _CALIB_STORE[det_name]
    cutout_cfg = _CALIB_STORE["cutout_cfg"]

    # --- ISR ---
    t0 = time.perf_counter()
    isr_task = _CALIB_STORE["isr_task"]
    postIsr = isr_task.run(
        entry["raw"],
        ptc=entry["ptc"],
        flat=entry["flat"],
        linearizer=entry["linearizer"],
        crosstalk=entry["crosstalk"],
    ).exposure

    # --- background subtraction ---
    t1 = time.perf_counter()
    bkg_task = _CALIB_STORE["bkg_task"]
    bkg_task.run(exposure=postIsr)

    # --- detect diameter ---
    t2 = time.perf_counter()
    detect_diameter_task = _CALIB_STORE["detect_diameter_task"]
    donutDiameter = detect_diameter_task.run(postIsr).diameter
    donutRadius = _resolveDonutRadius(
        donutDiameter/ 2 if donutDiameter is not None else None
    )

    # --- blind detection ---
    t3 = time.perf_counter()
    blind_detect_task = _CALIB_STORE["blind_detect_task"]
    blindDetections = blind_detect_task.run(postIsr, donutRadius=donutRadius).detections

    if len(blindDetections) == 0:
        return {
            "det_name": det_name,
            "catalog": [],
            "dispatch_to_arrival": time.time() - t_dispatch,
            "isr_run": t1 - t0,
            "bkg_run": t2 - t1,
            "diam_run": t3 - t2,
            "blind_detect_run": time.perf_counter() - t2,
            "wcs_refit_run": 0.0,
            "catalog_select_run": 0.0,
            "stamp_cut_run": 0.0,
            "rejected_catalog": [],
            "scatter_arcsec": None,
            "wcs_refit_error": "No blind detections",
            "cat_select_error": "",
        }

    # --- astrometry ---
    t4 = time.perf_counter()
    astrom_task = _CALIB_STORE["astrom_task"]
    detector = postIsr.getDetector()
    refcat_handle = _CALIB_STORE["det_refcats"].get(detector.getName())
    scatter_arcsec = None
    wcs = None
    wcs_err = ""
    try:
        astrom_result = astrom_task.solve(
            exposure=postIsr,
            sourceCat=_buildAfwSourceCat(blindDetections, postIsr.getWcs()),
            load_result=refcat_handle,
        )
        scatter_arcsec = astrom_result.scatterOnSky.asArcseconds()
        if scatter_arcsec < cutout_cfg["maxFitScatter"]:
            wcs = postIsr.getWcs()
        else:
            wcs_err = f'scatter {scatter_arcsec:.2f}" >= {cutout_cfg["maxFitScatter"]}"'
    except Exception as exc:
        wcs_err = f"astrometry solve failed: {type(exc).__name__}: {exc}"
        logging.getLogger(__name__).warning(
            _colorize(
                "Astrometry solve failed for %s; falling back to blind detections: %s",
                _ANSI_BOLD,
                _ANSI_YELLOW
            ),
            det_name,
            wcs_err,
        )

    # --- catalog selection ---
    t5 = time.perf_counter()
    selections = blindDetections
    refcat = None
    cat_err = ""
    selection_source = None
    donut_selector = _CALIB_STORE["donut_selector_task"]

    if wcs is not None:
        try:
            photo_filter = cutout_cfg["photoRefFilter"]
            astrom_filter = cutout_cfg["astromRefFilter"]
            refcat = refcat_handle.refCat.copy(deep=True)
            afwTable.updateRefCentroids(wcs, refcat)
            # Much quicker to just copy the keys we need than convert the whole table to
            # astropy
            keys = [
                "id",
                "coord_ra", "coord_dec",
                "centroid_x", "centroid_y",
                f"{photo_filter}_flux", f"{astrom_filter}_flux"
            ]
            refcat = QTable({k: np.array(refcat[k]) for k in keys})
            refcat["photo_flux"] = refcat[f"{photo_filter}_flux"]
            refcat["astrom_flux"] = refcat[f"{astrom_filter}_flux"]
            with np.errstate(invalid="ignore", divide="ignore"):
                refcat["photo_mag"] = -2.5 * np.log10(refcat["photo_flux"]) + 31.4
                refcat["astrom_mag"] = -2.5 * np.log10(refcat["astrom_flux"]) + 31.4
            result = donut_selector.run(refcat, detector, photo_filter)
            selections = result.sourceCat
            selection_source = "refcat"
        except Exception as exc:
            cat_err = str(exc)
            refcat = None  # don't leave a partially-built refcat around

    # If the refcat path didn't produce a selection, run the blind detections
    # through the same selector.  If that also fails, then exit gracefully.
    if selection_source != "refcat":
        try:
            result = donut_selector.run(blindDetections, detector, "")
            selections = result.sourceCat
            selection_source = "blind_selected"
        except Exception as exc:
            cat_err = cat_err or str(exc)
            _log.warning(
                "Donut selector failed on blind detections for %s; "
                "dropping detector's donuts: %s",
                det_name,
                exc,
            )
            selections = blindDetections[:0]  # empty; flows through to empty catalog
            selection_source = "blind_failed"
    logging.getLogger(__name__).info(
        "Donut selection path: %s (%d sources)", selection_source, len(selections)
    )

    # --- stamp cutting ---
    t6 = time.perf_counter()
    measure_task = _CALIB_STORE["measure_candidates_task"]
    candidates = measure_task.run(
        postIsr,
        selections,
        donutRadius=donutRadius,
    ).measurements
    cut_stamps_task = _CALIB_STORE["cut_stamps_task"]
    cut_result = cut_stamps_task.run(
        postIsr,
        candidates,
        refcat,
        blindDetections,
        donutRadius=donutRadius
    )

    t7 = time.perf_counter()

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Annulus

    # fig, ax = plt.subplots(figsize=(10, 10))
    # vmin, vmax = np.nanquantile(postIsr.image.array, [0.01, 0.99])
    # ax.imshow(postIsr.image.array, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # ax.set_xlim(0, postIsr.image.array.shape[1])
    # ax.set_ylim(0, postIsr.image.array.shape[0])
    # ax.scatter(refcat["centroid_x"], refcat["centroid_y"], s=20, edgecolor="cyan", facecolor="none")
    # ax.scatter(blindDetections["centroid_x"], blindDetections["centroid_y"], s=50, edgecolor="blue", facecolor="none")
    # ax.scatter(selections["centroid_x"], selections["centroid_y"], s=80, edgecolor="red", facecolor="none")
    # ax.scatter(
    #     [d.centroid_x_raw for d in cut_result.donuts],
    #     [d.centroid_y_raw for d in cut_result.donuts],
    #     s=110, edgecolor="yellow", facecolor="none"
    # )
    # for d in cut_result.donuts:
    #     ax.annotate(
    #         f"{d.snr:.1f}",
    #         (d.centroid_x_raw, d.centroid_y_raw),
    #         xytext=(10, 10),
    #         textcoords="offset points", color="yellow", fontsize=10,
    #         annotation_clip=True,
    #     )
    # for d in cut_result.rejected_donuts:
    #     ax.annotate(
    #         f"{d.snr:.1f}",
    #         (d.centroid_x_raw, d.centroid_y_raw),
    #         xytext=(10, 10),
    #         textcoords="offset points", color="red", fontsize=10,
    #         annotation_clip=True,
    #     )
    # xform = detector.getTransform(FIELD_ANGLE, PIXELS)
    # mapping = xform.getMapping()
    # center = mapping.applyForward(np.array([[0.0], [0.0]]))
    # cx = float(center[0, 0])
    # cy = float(center[1, 0])
    # # Find points on the circle inside the detector bounds
    # th = np.linspace(0, 2 * np.pi, 1000)
    # x = np.deg2rad(donut_selector.config.maxFieldDist) * np.cos(th)
    # y = np.deg2rad(donut_selector.config.maxFieldDist) * np.sin(th)
    # xyPix = mapping.applyForward(np.vstack([x, y]))
    # keep = xyPix[0] >= 0
    # keep &= xyPix[0] < postIsr.image.array.shape[1]
    # keep &= xyPix[1] >= 0
    # keep &= xyPix[1] < postIsr.image.array.shape[0]
    # xyPix = xyPix[:, keep]
    # radius = float(np.mean(np.hypot(xyPix[0] - cx, xyPix[1] - cy)))
    # big_radius = radius * 2
    # ann = Annulus(
    #     (cx, cy), big_radius, big_radius - radius,
    #     facecolor="purple", alpha=0.2, edgecolor="none"
    # )
    # ax.add_patch(ann)
    # plt.show()

    return {
        "det_name": det_name,
        "catalog": cut_result.donuts,
        "dispatch_to_arrival": t_arrival - t_dispatch,
        "isr_run": t1 - t0,
        "bkg_run": t2 - t1,
        "diam_run": t3 - t2,
        "blind_detect_run": t4 - t3,
        "wcs_refit_run": t5 - t4,
        "catalog_select_run": t6 - t5,
        "stamp_cut_run": t7 - t6,
        "rejected_catalog": cut_result.rejected_donuts,
        "scatter_arcsec": scatter_arcsec,
        "wcs_refit_error": wcs_err,
        "cat_select_error": cat_err,
    }


def _run_cutout_worker(args: tuple) -> dict:
    det_name, t_dispatch = args
    return _cutoutPipeline(det_name, t_dispatch)


def _bin_stamp_odd(stamp: np.ndarray, binning: int) -> np.ndarray:
    """Bin a stamp and trim it to an odd pixel size.

    Danish wants an odd-sized image so the donut centre lands on a pixel
    centre. Shared by `_prep_donut_for_danish` and `_buildCatalog` so that
    donuts which never reached a fit (paired-mode surplus) still get a WF
    image on the same pixel grid as the fitted ones.
    """
    img = stamp.astype(float)
    if binning > 1:
        img = binArray(img, binning)
    if img.shape[0] % 2 == 0:
        img = img[:-1, :-1]
    return img


# Maximum Noll index fit/reported. Dense Noll-indexed arrays are length
# _ZK_JMAX + 1: index j holds Zernike j, and indices 0-3 are always 0.
_ZK_JMAX = 66


@dataclass
class WfResult:
    """One donut's wavefront-fit outputs, produced by `_wf_worker`.

    Consumed by `_buildCatalog`, keyed by ``(donut_id, det_name)``. A fit that
    timed out or raised still produces a WfResult with ``fit_success=False``
    and all-NaN Zernikes. `_NULL_WF` is the sentinel for "no fit consumed this
    donut" (paired-mode surplus with no partner).
    """

    donut_id: int
    det_name: str
    defocal: str
    zk_dev: npt.NDArray[np.float64]        # dense Noll 0.._ZK_JMAX, metres, NaN where unfit
    zk_intrinsic: npt.NDArray[np.float64]  # dense Noll 0.._ZK_JMAX, metres
    img: np.ndarray | None
    model_img: np.ndarray | None
    fit_success: bool
    fit_elapsed: float
    setup_elapsed: float
    fit_nfev: int
    fit_cost: float
    fit_dx: float
    fit_dy: float
    fit_flux: float
    fit_fwhm: float
    blend_frac: float
    group_id: str
    group_size: int
    fit_mode: str


# Sentinel for "no fit consumed this donut". All-NaN Zernikes, empty strings,
# fit_success=False -- so _buildCatalog's `used` derivation is naturally False.
_NULL_WF = WfResult(
    donut_id=-1,
    det_name="",
    defocal="",
    zk_dev=np.full(_ZK_JMAX + 1, np.nan),
    zk_intrinsic=np.full(_ZK_JMAX + 1, np.nan),
    img=None,
    model_img=None,
    fit_success=False,
    fit_elapsed=float("nan"),
    setup_elapsed=float("nan"),
    fit_nfev=0,
    fit_cost=float("nan"),
    fit_dx=float("nan"),
    fit_dy=float("nan"),
    fit_flux=float("nan"),
    fit_fwhm=float("nan"),
    blend_frac=float("nan"),
    group_id="",
    group_size=0,
    fit_mode="",
)


class _WfFitTimeoutError(Exception):
    pass


@contextlib.contextmanager
def _fit_timeout(seconds):
    """SIGALRM-based timeout context manager.

    Works on any POSIX platform, macOS included; it is Windows that has no
    ``SIGALRM``.  The real constraint is threads: ``signal.signal`` raises
    ``ValueError: signal only works in main thread of the main interpreter``,
    so this is usable from the WF pool (each forked worker runs the fit on its
    own main thread) but would break if fitting moved to a thread pool.
    """

    def _handler(_signum, _frame):
        raise _WfFitTimeoutError(f"WF fit exceeded {seconds:.0f}s timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _bkg_free_model(
    model_img: np.ndarray, danish_model, fit_params, donut_idx: int, bkg_order: int
) -> np.ndarray:
    """Return model_img with the fitted background subtracted.

    fit_params is the dict returned by model.unpack_params(), or None on fit
    failure.  For bkgOrder <= 0 (constant or no background) this is a cheap
    scalar subtraction.  For bkgOrder > 0 we re-evaluate the model with zeroed
    backgrounds, relying on Danish's cached optics component.
    """
    if model_img is None:
        return None
    if fit_params is None or bkg_order < 0:
        return model_img
    if bkg_order == 0:
        bkgs = fit_params.get("bkgs") or fit_params.get("bkg")
        if bkgs is None:
            return model_img
        if np.ndim(bkgs[0]) > 0:
            bkg_val = np.ravel(bkgs[donut_idx])[0]
        else:
            bkg_val = bkgs[0]
        return model_img - bkg_val
    # bkgOrder > 0: re-evaluate with zeroed backgrounds
    try:
        is_multi = "fluxes" in fit_params
        nbkg = danish_model.nbkg
        zero_bkgs = [[0.0] * nbkg] * len(fit_params["fluxes"]) if is_multi else [0.0] * nbkg
        bkg_key = "bkgs" if is_multi else "bkg"
        kw = {k: v for k, v in fit_params.items() if k not in ("bkgs", "bkg")}
        kw[bkg_key] = zero_bkgs
        result = danish_model.model(**kw)
        if isinstance(result, list):
            return result[donut_idx]
        return result
    except Exception:
        return model_img


def _blend_frac(
    resid: np.ndarray,
    model_img_bkg_free: np.ndarray,
    bkg_std: float,
    faint_frac: float = 0.05,
    sig_thresh: float = 2.0,
) -> float:
    """Fraction of significant residual flux in model-faint pixels, normalised by total model flux."""
    if resid is None or model_img_bkg_free is None or not np.isfinite(bkg_std) or bkg_std <= 0:
        return float("nan")
    model_peak = np.nanmax(model_img_bkg_free)
    total_model_flux = np.sum(model_img_bkg_free[model_img_bkg_free > 0])
    if model_peak <= 0 or total_model_flux <= 0:
        return float("nan")
    faint_mask = model_img_bkg_free < faint_frac * model_peak
    sig_mask = np.abs(resid) > sig_thresh * bkg_std
    return np.sum(np.abs(resid[faint_mask & sig_mask])) / total_model_flux


def _dense_intrinsic(donut: dict) -> np.ndarray:
    """Return intrinsic Zernikes in metres, dense over Noll 0..``_ZK_JMAX``.

    Indices with no supplied value are 0.0.
    """
    out = np.zeros(_ZK_JMAX + 1)
    raw = donut.intrinsic_zk  # µm, Noll 4.._ZK_JMAX
    if raw is not None:
        n_slots = _ZK_JMAX + 1 - 4  # Noll 4.._ZK_JMAX inclusive
        for idx in range(min(len(raw), n_slots)):
            out[idx + 4] = raw[idx] * 1e-6
    return out


def _dense_dev(zk_dev: np.ndarray, nollIndices) -> np.ndarray:
    """Return deviations in metres, dense over Noll 0..``_ZK_JMAX``.

    Indices that were not fitted are ``np.nan``.
    """
    out = np.full(_ZK_JMAX + 1, np.nan)
    out[0:4] = 0.0
    for k, j in enumerate(nollIndices):
        if k < len(zk_dev):
            out[j] = zk_dev[k]
    return out


def _zk_cols(suffix: str, zk: np.ndarray) -> dict:
    """Return ``{f"Z{j}_{suffix}": um}`` columns for Noll 4..``_ZK_JMAX``.

    ``zk`` is a Noll-indexed array in metres of length ``_ZK_JMAX + 1``; values
    are converted to µm, with NaN passed through as NaN.
    """
    return {
        f"Z{j}_{suffix}": zk[j] * 1e6 if not np.isnan(zk[j]) else float("nan")
        for j in range(4, _ZK_JMAX + 1)
    }


@dataclass
class _WfGroup:
    donuts: list  # donut dicts, ordered; each carries its own "det_name" key
    group_id: str
    band: str
    rtp: float | None  # Boresight rotation (spider angle), degrees or None
    alt: float | None  # Boresight altitude, radians or None


def _build_wf_groups(mode, results_by_det, band: str, rtp_deg: float | None, boresight_alt_rad: float | None):
    """Build _WfGroup list from per-detector catalogs, matching the mode dispatch logic.

    Groups for ``"paired"`` hold one extra/intra pair; all other modes group
    without regard to defocal type -- the invariant reported by
    `_mode_groups_are_pairs`.

    Returns (groups, unmatched_donuts).
    """
    groups = []
    unmatched_donuts = []
    if mode == "paired":
        for _corner, (sw0, sw1) in CORNER_PAIRS.items():
            extra_donuts = sorted(results_by_det.get(sw0, []), key=lambda d: d.snr, reverse=True)
            intra_donuts = sorted(results_by_det.get(sw1, []), key=lambda d: d.snr, reverse=True)
            for extra, intra in zip(extra_donuts, intra_donuts):
                gid = f"{extra.id}_{intra.id}"
                groups.append(_WfGroup(donuts=[extra, intra], group_id=gid, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
            n_pairs = min(len(extra_donuts), len(intra_donuts))
            unmatched_donuts.extend(extra_donuts[n_pairs:])
            unmatched_donuts.extend(intra_donuts[n_pairs:])
    elif mode == "unpaired":
        for det_donuts in results_by_det.values():
            for d in det_donuts:
                groups.append(_WfGroup(donuts=[d], group_id=str(d.id), band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    elif mode == "full_detector":
        for det_name, det_donuts in results_by_det.items():
            groups.append(_WfGroup(donuts=det_donuts, group_id=det_name, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    elif mode == "full_corner":
        for corner, (sw0, sw1) in CORNER_PAIRS.items():
            all_donuts = results_by_det.get(sw0, []) + results_by_det.get(sw1, [])
            groups.append(_WfGroup(donuts=all_donuts, group_id=corner, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    else:
        raise ValueError(f"Unknown WF mode {mode!r}")
    return groups, unmatched_donuts


# Module-level logger for the worker functions below. They are module-level
# (not methods) so the fork-based pools can pickle them by name, which means
# there is no `self` and so no `Task.log`. Consequence: worker output goes to
# the `lsst.ts.wep.task.donutBlitzMonolith` logger rather than the task's own
# `donutBlitzMonolithTask` hierarchy, so it is not affected by that task's log
# level. Parent-process code should keep using `self.log`.
_log = logging.getLogger(__name__)

_DZ_MODEL_KEYS = ("fluxes", "dxs", "dys", "fwhm", "wavefront_params", "bkgs")


def _wf_fitting_worker(group: "_WfGroup") -> dict:
    """Wavefront fitting worker for multiprocessing pool.

    Retrieves the WavefrontFittingTask from _CALIB_STORE and calls it
    on the group. Designed to be used with multiprocessing.Pool.map().
    """
    task = _CALIB_STORE["wf_fitting_task"]
    result = task.run(group)
    # Set fit_mode on each donut result from main config
    fit_mode = _CALIB_STORE["wfEstimationMode"]
    for wd in result.get("donuts", []):
        wd.fit_mode = fit_mode
    return result


class BlindDetectConfig(pexConfig.Config):
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Width of detector edge region to exclude from detection, in pixels.",
        dtype=int,
        default=80,
    )
    detectionBinning: pexConfig.Field = pexConfig.Field(
        doc=("Integer factor by which to bin the image before running the cross-correlation detection step."),
        dtype=int,
        default=8,
    )
    peakMinDistanceFactor: pexConfig.Field = pexConfig.Field(
        doc="Multiplier applied to the binned donut radius to set min_distance in peak_local_max.",
        dtype=float,
        default=1.6,
    )
    peakExcludeBorderFactor: pexConfig.Field = pexConfig.Field(
        doc="Multiplier applied to the binned donut radius to set exclude_border in peak_local_max.",
        dtype=float,
        default=1.15,
    )


class BlindDetect(pipeBase.Task):
    ConfigClass = BlindDetectConfig
    _DefaultName = "blindDetectTask"
    config: BlindDetectConfig

    """Detect donuts via annular template cross-correlation.

    Erodes the post-ISR exposure border by ``edgeMargin`` pixels, then calls
    `_detectPeaks`.

    Parameters
    ----------
    exposure : Exposure
        Science exposure; background is subtracted in-place.
    donutRadius : float or None
        Donut radius in pixels.  If None, uses the instrument's configured
        donut radius.

    Returns
    -------
    QTable
        Columns ``id``, ``centroid_x``, ``centroid_y`` in full-exposure pixel
        coordinates.  Empty table if no peaks are found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(
        self,
        exposure: Exposure,
        donutRadius: float | None = None,
    ) -> pipeBase.Struct:
        config = self.config
        if donutRadius is None:
            donutRadius = _INSTRUMENT.donutRadius

        trimmedBBox = exposure.getBBox().erodedBy(config.edgeMargin)
        binning = config.detectionBinning
        binned_donut_radius = donutRadius / binning
        template = _buildAnnularTemplate(
            binned_donut_radius,
            innerFrac=_INSTRUMENT.obscuration
        )

        if binning > 1:
            binnedImg = afwMath.binImage(exposure[trimmedBBox].image, binning)
            arr = binnedImg.array
        else:
            arr = exposure[trimmedBBox].image.array

        # Detect on the histogram equalized image
        heq = np.digitize(arr, np.nanquantile(arr, np.linspace(0, 1, 256)))
        det = correlate(heq.astype(float), template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(config.peakMinDistanceFactor * binned_donut_radius),
            exclude_border=int(config.peakExcludeBorderFactor * binned_donut_radius),
        )
        peaks = peaks * float(binning)
        return pipeBase.Struct(
            detections=QTable(
                {
                    "id": np.arange(1, len(peaks) + 1, dtype=np.int64),
                    "centroid_x": peaks[:, 1] + trimmedBBox.getMinX(),
                    "centroid_y": peaks[:, 0] + trimmedBBox.getMinY(),
                }
            )
        )


class WavefrontFittingTaskConfig(pexConfig.Config):
    """Configuration for wavefront fitting via Danish algorithm."""

    nollIndices: pexConfig.ListField = pexConfig.ListField(
        dtype=int,
        doc="Noll indices to fit with Danish.",
        default=list(range(4, 20)) + list(range(22, 27)),
    )
    lstsqKwargs: pexConfig.DictField = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        doc=(
            "Keyword arguments for scipy.optimize.least_squares passed to the Danish "
            "WF workers. Values are strings that will be eval()'d, e.g. "
            "{'method': 'trf', 'max_nfev': '200'}."
        ),
        default={
            "xtol": "1e-3",
            "ftol": "1e-3",
            "gtol": "1e-3",
            "x_scale": "'jac'",
        },
    )
    binning: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Binning factor applied to donut stamps before Danish fitting.",
    )
    modelSpiderShadows: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Include spider shadow modeling in Danish forward model.",
    )
    bkgOrder: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Background polynomial order for Danish (-1=none, 0=constant).",
    )
    doAoiThroughput: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply angle-of-incidence throughput correction in Danish forward model.",
    )
    systematicLossAlpha: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Fractional systematic uncertainty for Danish loss function (0=chi2).",
    )
    triangleMode: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Use DonutTriangleFactory instead of DonutFactory.",
    )
    wfFitTimeoutPerDonut: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=10.0,
        doc=(
            "Timeout in seconds per donut for a single WF fit. "
            "The total timeout for a work unit is this value times the number of donuts "
            "(1 for unpaired, 2 for paired, N for full_corner). "
            "Fits exceeding the limit are killed and return NaN Zernikes."
        ),
    )
    wfInitialGuessOnly: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=(
            "Skip the least-squares fit and return the initial-guess (x0) model only. "
            "Useful for diagnosing stamp orientation and model setup without "
            "waiting for convergence."
        ),
    )


@dataclass
class _LstsqFitResult:
    # All outputs from _run_lstsq_fit as first-class typed fields.
    zk_dev: npt.NDArray[np.float64]
    model_imgs: list
    blend_fracs: list
    success: bool
    elapsed: float
    fluxes: list
    dxs: list
    dys: list
    fwhm: float
    nfev: int = 0
    cost: float = float("nan")
    optimality: float = float("nan")
    njev: int = 0
    status: int = 0
    message: str = ""
    error: str = ""


class WavefrontFittingTask(pipeBase.Task):
    """Task to fit wavefront aberrations from grouped donut stamps using Danish algorithm.

    This task takes a pre-grouped collection of donuts (a _WfGroup) and performs
    joint wavefront fitting across all donuts in the group, returning Zernike
    coefficients for the wavefront error.
    """

    ConfigClass = WavefrontFittingTaskConfig
    _DefaultName = "wavefrontFittingTask"
    config: WavefrontFittingTaskConfig

    def run(self, group: "_WfGroup") -> dict:
        """Fit wavefront aberrations for a group of donuts.

        Parameters
        ----------
        group : _WfGroup
            Pre-grouped collection of donuts to fit jointly, including
            band, rtp, and alt for exposure-dependent parameters.

        Returns
        -------
        result : dict
            Dictionary containing:
            - group_id: str
            - group_size: int
            - zk_dev: ndarray
            - success: bool
            - fit_info: dict
            - donuts: list of WfResult
            - model_imgs: list or None
            - imgs: list
            - det_names: list of str
        """
        nollIndices = self.config.nollIndices
        all_donuts = group.donuts
        n = len(all_donuts)

        if not all_donuts:
            return {
                "group_id": group.group_id,
                "group_size": 0,
                "zk_dev": np.full(len(nollIndices), np.nan),
                "success": False,
                "fit_info": {},
                "donuts": [],
                "model_imgs": None,
                "imgs": [],
                "det_names": [],
            }

        t_setup0 = time.perf_counter()
        factory = self._build_wf_factory(group)
        preps = [self._prep_donut_for_danish(d) for d in all_donuts]
        imgs = [p[0] for p in preps]
        thxs = [p[1][0] for p in preps]
        thys = [p[1][1] for p in preps]
        zk_refs = [p[2] for p in preps]
        sky_lvl = [p[3] for p in preps]
        bkg_stds = [p[4] for p in preps]
        dz_terms = [(1, j) for j in nollIndices]

        npix = min(img.shape[0] for img in imgs)
        imgs = [img[:npix, :npix] for img in imgs]

        model = danish.DZMultiDonutModel(
            factory,
            z_refs=zk_refs,
            dz_terms=dz_terms,
            field_radius=_DANISH_FIELD_RADIUS_RAD,
            thxs=thxs,
            thys=thys,
            npix=npix,
            bkg_order=self.config.bkgOrder,
            loss_fn=self._build_loss_fn(),
        )
        fluxes_init = [np.clip(np.sum(img), 1e3, 1e9) for img in imgs]
        x0 = model.pack_params(
            fluxes=fluxes_init,
            dxs=[0.0] * n,
            dys=[0.0] * n,
            fwhm=1.0,
            bkgs=[[0.0] * model.nbkg] * n,
            wavefront_params=[0.0] * len(dz_terms),
        )
        bounds = model.pack_params(
            fluxes=[[0.0, np.inf]] * n,
            dxs=[[-np.inf, np.inf]] * n,
            dys=[[-np.inf, np.inf]] * n,
            fwhm=[0.1, 5.0],
            bkgs=[[[-np.inf, np.inf]] * model.nbkg] * n,
            wavefront_params=[[-np.inf, np.inf]] * len(dz_terms),
        )
        bounds = [list(b) for b in zip(*bounds)]
        x0 = np.clip(x0, bounds[0], bounds[1])
        timeout = self.config.wfFitTimeoutPerDonut * n
        _setup_elapsed = time.perf_counter() - t_setup0
        label = f"group={group.group_id} n={n}"
        self.log.info("WF %s npix=%d setup=%.2fs", label, npix, _setup_elapsed)

        fit_result = self._run_lstsq_fit(
            model, x0, bounds, imgs, sky_lvl, timeout, label
        )
        zk_dev_dense = _dense_dev(fit_result.zk_dev, nollIndices)

        donuts_out = []
        for i, d in enumerate(all_donuts):
            defocal = "intra" if int(d.det_id) in _INTRA_FOCAL_DET_IDS else "extra"
            _img = imgs[i] if i < len(imgs) else None
            donuts_out.append(
                WfResult(
                    donut_id=int(d.id),
                    det_name=d.det_name,
                    defocal=defocal,
                    zk_dev=zk_dev_dense,
                    zk_intrinsic=_dense_intrinsic(d),
                    img=_img,
                    model_img=fit_result.model_imgs[i],
                    fit_success=fit_result.success,
                    fit_elapsed=fit_result.elapsed,
                    setup_elapsed=_setup_elapsed,
                    fit_nfev=fit_result.nfev,
                    fit_cost=fit_result.cost,
                    fit_dx=float(fit_result.dxs[i]),
                    fit_dy=float(fit_result.dys[i]),
                    fit_flux=float(fit_result.fluxes[i]),
                    fit_fwhm=fit_result.fwhm,
                    blend_frac=fit_result.blend_fracs[i],
                    group_id=group.group_id,
                    group_size=n,
                    fit_mode="",  # Will be set by caller with wfEstimationMode
                )
            )
        return {
            "group_id": group.group_id,
            "group_size": n,
            "zk_dev": fit_result.zk_dev,
            "success": fit_result.success,
            "fit_info": {
                "elapsed": fit_result.elapsed,
                "nfev": fit_result.nfev,
                "cost": fit_result.cost,
                "optimality": fit_result.optimality,
                "njev": fit_result.njev,
                "status": fit_result.status,
                "message": fit_result.message,
                "error": fit_result.error,
            },
            "donuts": donuts_out,
            "model_imgs": fit_result.model_imgs,
            "imgs": imgs,
            "det_names": [d.det_name for d in all_donuts],
        }

    def _build_wf_factory(self, group: "_WfGroup") -> "danish.DonutFactory":
        """Build a Danish donut factory from config and group."""
        factory_class = danish.DonutTriangleFactory if self.config.triangleMode else danish.DonutFactory
        factory_kwargs = {}
        if self.config.doAoiThroughput and group.band:
            wavelength = _INSTRUMENT.wavelength.get(group.band)
            if wavelength:
                factory_kwargs["bandpass_filter"] = wavelength
            if group.alt is not None and np.isfinite(group.alt) and group.alt > 0:
                airmass = np.clip((1.0 / np.sin(group.alt)), 1.0, 2.5)
                factory_kwargs["airmass"] = airmass
            else:
                factory_kwargs["airmass"] = 1.2
        return factory_class(
            R_outer=_INSTRUMENT.radius,
            R_inner=_INSTRUMENT.radius * _INSTRUMENT.obscuration,
            mask_params=_INSTRUMENT.maskParams,
            focal_length=_INSTRUMENT.focalLength,
            pixel_scale=_INSTRUMENT.pixelSize * self.config.binning,
            spider_angle=group.rtp,
            **factory_kwargs,
        )

    def _build_loss_fn(self) -> Any:
        """Return a danish loss function from config, or None for standard chi-squared."""
        alpha = self.config.systematicLossAlpha
        if alpha <= 0:
            return None
        return danish.systematic_loss(alpha)

    def _prep_donut_for_danish(self, donut: "Donut") -> tuple:
        """Prepare a Donut for Danish fitting.

        Bins the stamp and forces it to an odd pixel size, estimates background
        noise, computes the reference Zernike array ``zk_ref`` from
        ``batoid.zernikeTA`` (with optional measured-intrinsics correction), and
        extracts the field angle.

        Parameters
        ----------
        donut : Donut
            Donut record. Uses ``stamp`` (2-D array), ``det_id``, ``band``,
            ``fa_x_ccs``, ``fa_y_ccs`` (field angles in radians), and
            ``intrinsic_zk`` (µm, Noll 4..``_ZK_JMAX``; ``None`` if uncalibrated).

        Returns
        -------
        img : np.ndarray
            ``(npix, npix)`` float stamp, binned and forced to odd size.
        angle_rad : np.ndarray
            ``[fa_x_ccs, fa_y_ccs]`` field angle in radians.
        zk_ref : np.ndarray
            Reference Zernike array in metres, Noll-indexed, shape
            ``(_ZK_JMAX + 1,)``.
            Equals ``W_TA_defoc`` at uncalibrated indices and
            ``W_TA_defoc + (W_meas - zk_opd_foc)`` at calibrated indices.
        bkg_var : float
            Background variance estimate (``bkg_std ** 2``) from pixel-difference
            MAD of the stamp.
        bkg_std : float
            Background standard deviation estimate.
        """
        binning = self.config.binning
        det_id = donut.det_id
        defocalSign = +1 if det_id in _EXTRA_FOCAL_DET_IDS else -1

        img = _bin_stamp_odd(donut.stamp, binning)
        diff = (img[1:] - img[:-1]).ravel()
        bkg_std = median_abs_deviation(diff, scale="normal") / np.sqrt(2.0)

        band = donut.band
        wavelength_by_band = {bl.value: wl for bl, wl in _INSTRUMENT.wavelength.items()}
        if band not in wavelength_by_band:
            raise RuntimeError(
                f"No wavelength configured for band {band!r}; the instrument supplies "
                f"{sorted(wavelength_by_band)}. Fitting with a wrong wavelength would "
                "bias the whole wavefront, so this is fatal rather than defaulted."
            )
        wavelength = wavelength_by_band[band]
        telescope = _CALIB_STORE["telescope"]
        telescope_dz = (
            _CALIB_STORE["telescope_extra"] if defocalSign > 0 else _CALIB_STORE["telescope_intra"]
        )
        eps = telescope.pupilObscuration
        nrad = 10
        zernikeTA_kwargs = dict(
            jmax=_ZK_JMAX,
            eps=eps,
            focal_length=_INSTRUMENT.focalLength,
            nrad=nrad,
            naz=int(2 * np.pi * nrad / (1 - eps)),
        )
        # W_TA_defoc: off-axis + nominal intrinsics + defocus in one call
        zk_ref = (
            batoid.zernikeTA(
                telescope_dz,
                donut.fa_x_ccs,
                donut.fa_y_ccs,
                wavelength,
                **zernikeTA_kwargs,
            )
            * wavelength
        )  # meters, shape (_ZK_JMAX + 1,)

        # Replace nominal on-axis model (zk_opd_foc) with measured intrinsics (W_meas)
        # for calibrated indices.
        intrinsic_zk = donut.intrinsic_zk
        if intrinsic_zk is not None:
            zk_opd_foc = (
                batoid.zernikeTA(
                    telescope,
                    donut.fa_x_ccs,
                    donut.fa_y_ccs,
                    wavelength,
                    **zernikeTA_kwargs,
                )
                * wavelength
            )  # meters
            calib_noll = np.arange(4, _ZK_JMAX + 1)
            for i, j in enumerate(calib_noll):
                if i < len(intrinsic_zk) and j <= _ZK_JMAX:
                    zk_ref[j] += intrinsic_zk[i] * 1e-6 - zk_opd_foc[j]

        angle_rad = np.array([donut.fa_x_ccs, donut.fa_y_ccs])
        return img, angle_rad, zk_ref, bkg_std**2, bkg_std

    def _run_lstsq_fit(self, model, x0, bounds, imgs, variances, timeout, label):
        """Run a DZMultiDonutModel least-squares fit and return results uniformly.

        Handles the ``wfInitialGuessOnly`` path, SIGALRM timeout, and all exception
        cases so each worker only needs to build the model and call this helper.

        Parameters
        ----------
        model : danish.DZMultiDonutModel
            Fully-constructed model ready to call ``.chi``, ``.jac``, ``.model``.
        x0 : np.ndarray
            Initial parameter vector from ``model.pack_params``.
        bounds : list
            Two-element ``[lower, upper]`` bound lists from ``model.pack_params``.
        imgs : list of np.ndarray
            Donut image stamps, one per model donut.
        variances : list of np.ndarray
            Per-pixel variance arrays aligned with ``imgs``.
        timeout : float
            SIGALRM timeout in seconds.
        label : str
            Short description used in log messages (e.g. ``"group=123 n=2"``).

        Returns
        -------
        _LstsqFitResult
            All fit outputs as first-class typed fields; see `_LstsqFitResult`.
        """
        nollIndices = list(self.config.nollIndices)
        n = len(imgs)
        _nan_blends = [float("nan")] * n
        t0 = time.perf_counter()
        if self.config.wfInitialGuessOnly:
            try:
                params = model.unpack_params(x0)
                zk_dev = np.zeros(len(nollIndices))
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in variances]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                elapsed = time.perf_counter() - t0
                self.log.info("WF %s (x0 only)", label)
                return _LstsqFitResult(
                    zk_dev=zk_dev,
                    model_imgs=model_imgs,
                    blend_fracs=blend_fracs,
                    success=True,
                    elapsed=elapsed,
                    fluxes=params["fluxes"],
                    dxs=params["dxs"],
                    dys=params["dys"],
                    fwhm=params["fwhm"],
                    nfev=0,
                    cost=float("nan"),
                    optimality=float("nan"),
                    njev=0,
                    status=0,
                    message="x0 only",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=str(exc),
                )
        else:
            galsim.errors.raise_fft_size_error = True
            try:
                with _fit_timeout(timeout):
                    lstsq_kwargs = {
                        k: ast.literal_eval(v)
                        for k, v in self.config.lstsqKwargs.items()
                    }
                    result = least_squares(
                        model.chi,
                        jac=model.jac,
                        x0=x0,
                        args=(imgs, variances),
                        bounds=bounds,
                        **lstsq_kwargs,
                    )
                elapsed = time.perf_counter() - t0
                params = model.unpack_params(result.x)
                zk_dev = np.array(params["wavefront_params"])
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in variances]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                self.log.info(
                    "WF %s success=%s nfev=%d elapsed=%.1fs",
                    label,
                    bool(result.success),
                    result.nfev,
                    elapsed,
                )
                return _LstsqFitResult(
                    zk_dev=zk_dev,
                    model_imgs=model_imgs,
                    blend_fracs=blend_fracs,
                    success=bool(result.success),
                    elapsed=elapsed,
                    fluxes=params["fluxes"],
                    dxs=params["dxs"],
                    dys=params["dys"],
                    fwhm=params["fwhm"],
                    nfev=result.nfev,
                    cost=result.cost,
                    optimality=result.optimality,
                    njev=result.njev,
                    status=result.status,
                    message=result.message,
                )
            except _WfFitTimeoutError:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s TIMED OUT after %.1fs", label, elapsed)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=f"timeout after {timeout:.0f}s",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=str(exc),
                )


class DonutBlitzMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzMonolithTask."""

    raws = connectionTypes.Input(
        doc="Raw corner wavefront sensor exposures (all 8 detectors).",
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    ptc = connectionTypes.PrerequisiteInput(
        name="ptc",
        storageClass="PhotonTransferCurveDataset",
        doc="Photon transfer curve calibration, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
    )
    flat = connectionTypes.PrerequisiteInput(
        name="flat",
        storageClass="ExposureF",
        doc="Flat field calibration, one per detector.",
        dimensions=["instrument", "detector", "physical_filter"],
        isCalibration=True,
        multiple=True,
    )
    linearizer = connectionTypes.PrerequisiteInput(
        name="linearizer",
        storageClass="Linearizer",
        doc="Linearity correction, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
    )
    crosstalk = connectionTypes.PrerequisiteInput(
        name="crosstalk",
        storageClass="CrosstalkCalib",
        doc="Crosstalk coefficients, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
    )
    refCat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog for both WCS fitting and donut selection.",
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
    )
    intrinsicZernikes = connectionTypes.PrerequisiteInput(
        doc="Intrinsic Zernike calibration, one per corner detector.",
        dimensions=("detector", "instrument", "physical_filter"),
        storageClass="IsrCalib",
        name="intrinsicZernikes",
        multiple=True,
        isCalibration=True,
        minimum=0,
    )
    blitzResults = connectionTypes.Output(
        doc=(
            "Per-donut catalog containing selection metrics, fit results, Zernikes, "
            "stamp/model images, and all metadata needed to regenerate diagnostic plots."
        ),
        name="donutBlitzResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class DonutBlitzMonolithTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzMonolithTaskConnections,  # type: ignore
):
    """Configuration for DonutBlitzMonolithTask."""

    isrTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="ISR subtask run on each corner wavefront sensor exposure.",
    )
    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Background subtraction subtask run before donut detection.",
    )
    detectDiameter: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutDetectDiameterTask,
        doc="Donut diameter detection subtask.",
    )
    blindDetect: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=BlindDetect,
        doc=(
            "Blind donut detection subtask run on each corner wavefront sensor "
            "exposure."
        ),
    )
    astromTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=AstrometryTask,
        doc="Astrometry subtask for WCS fitting.",
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask,
        doc="Donut source selector subtask.",
    )
    measureCandidatesTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=MeasureDonutCandidatesTask,
        doc="Donut candidate measurement subtask.",
    )
    cutStampsTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=CutDonutStampsTask,
        doc="Donut stamp cutting subtask.",
    )
    instConfigFile: pexConfig.Field = pexConfig.Field(
        doc=(
            "Path to an instrument configuration file to override the default. "
            "If begins with 'policy:' the path is relative to the ts_wep policy "
            "directory. If not provided, the default instrument for the camera "
            "will be loaded."
        ),
        dtype=str,
        optional=True,
    )
    maxFitScatter: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed on-sky scatter (arcsec) for WCS refit to be accepted.",
        dtype=float,
        default=1.0,
    )
    astromRefFilter: pexConfig.Field = pexConfig.Field(
        doc=(
            "Filter name to read from the reference catalog when fitting the "
            "WCS. Aliased over every filter via anyFilterMapsToThis, so it is "
            "what AstrometryTask resolves as its reference flux field."
        ),
        dtype=str,
        default="phot_g_mean",
    )
    photoRefFilter: pexConfig.Field = pexConfig.Field(
        doc=(
            "Explicit filter name to read from the reference catalog for donut "
            "selection (e.g. 'phot_g_mean'). Overrides photoRefFilterPrefix "
            "when set."
        ),
        dtype=str,
        optional=True,
    )
    photoRefFilterPrefix: pexConfig.Field = pexConfig.Field(
        doc=(
            "Filter prefix used for donut selection, combined with the exposure "
            "band label as '{prefix}_{band}'. Used when photoRefFilter is not "
            "set. The default matches the only per-band LSST-like fluxes "
            "present in the_monster_20250219 ('monster_ComCam_g' etc.); there "
            "are no monster_LSSTCam_* columns in any released refcat version, "
            "so override this once a matching set exists."
        ),
        dtype=str,
        default="monster_ComCam",
    )
    savePlots: pexConfig.Field = pexConfig.Field(
        doc=(
            "Generate diagnostic PNGs for each visit. "
            "Set False in production to skip plot generation and deliver "
            "Zernikes faster; plots can be generated later by calling "
            "plotTask.run() with the in-memory results."
        ),
        dtype=bool,
        default=False,
    )
    colorLog: pexConfig.Field = pexConfig.Field(
        doc=(
            "Colorize select log messages with ANSI escape codes. If None "
            "(the default), color is enabled only when stdout is an "
            "interactive terminal."
        ),
        dtype=bool,
        default=None,
        optional=True,
    )
    wfEstimationMode: pexConfig.ChoiceField = pexConfig.ChoiceField(
        doc="Wavefront estimation dispatch mode.",
        dtype=str,
        allowed={
            "paired": "Pair donuts from SW0/SW1 by SNR rank and dispatch as intra/extra pairs.",
            "unpaired": "Dispatch individual donuts independently.",
            "full_corner": "Dispatch all donuts from a corner (SW0+SW1) as one work unit.",
            "full_detector": "Dispatch all donuts on each detector as one work unit (8 fits per visit).",
        },
        default="paired",
    )
    wfFittingTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=WavefrontFittingTask,
        doc="Wavefront fitting subtask using Danish algorithm.",
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.isrTask.doAmpOffset = False
        self.isrTask.ampOffset.doApplyAmpOffset = False
        self.isrTask.doBrighterFatter = False
        self.isrTask.doSaturation = True
        self.isrTask.doStandardStatistics = False
        self.isrTask.doInterpolate = False
        self.isrTask.doVariance = False
        self.isrTask.doDeferredCharge = False
        self.isrTask.doDefect = False
        self.isrTask.doApplyGains = True
        self.isrTask.doBias = False
        self.isrTask.doFlat = True
        self.isrTask.doDark = False
        self.isrTask.doLinearize = True
        self.isrTask.doSuspect = False
        self.isrTask.doSetBadRegions = False
        self.isrTask.doBootstrap = False
        self.isrTask.doCrosstalk = True
        self.isrTask.crosstalk.doQuadraticCrosstalkCorrection = False
        self.isrTask.doITLEdgeBleedMask = False
        self.isrTask.qa.saveStats = False

        self.astromTask.wcsFitter.retarget(FitAffineWcsTask)
        self.astromTask.doMagnitudeOutlierRejection = False
        self.astromTask.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 18
        self.astromTask.referenceSelector.magLimit = magLimit
        self.astromTask.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astromTask.sourceSelector["science"].doRequirePrimary = False
        self.astromTask.sourceSelector["science"].doIsolated = False
        self.astromTask.sourceSelector["science"].doSignalToNoise = False
        self.astromTask.sourceSelector["science"].doCentroidErrorLimit = False
        self.astromTask.maxIter = 5
        self.astromTask.matcher.maxOffsetPix = 1000

        # Monster refcat uses full filter names (e.g. phot_g_mean), not band
        # labels, so the default mag-limit policy lookup by band would fail.
        # Use custom mag limits instead.
        self.donutSelector.useCustomMagLimit = True
        self.donutSelector.maxFieldDist = 1.725
        self.donutSelector.sourceLimit = 40
        self.donutSelector.allowFluxless = True


class DonutBlitzMonolithTask(pipeBase.PipelineTask):
    """Monolithic WEP task for corner wavefront sensors.

    Runs ISR, blind donut detection, WCS refit, catalog-based donut
    selection, and stamp cutting on all 8 corner detector raws in parallel
    using a multiprocessing pool.  Reference catalogs are loaded in the parent
    process before forking and inherited by workers via copy-on-write.
    """

    ConfigClass = DonutBlitzMonolithTaskConfig
    _DefaultName = "donutBlitzMonolithTask"
    config: DonutBlitzMonolithTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("isrTask")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("detectDiameter")
        self.makeSubtask("blindDetect")
        self.makeSubtask("astromTask")
        self.makeSubtask("donutSelector")
        self.makeSubtask("measureCandidatesTask")
        self.makeSubtask("cutStampsTask")
        self.makeSubtask("wfFittingTask")
        self.makeSubtask("plotTask")
        self._colorLogEnabled = _resolveColorLogEnabled(self.config.colorLog)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        self.log.info(
            _colorize(
                "DonutBlitzMonolithTask.runQuantum() on exposure %d",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            inputRefs.raws[0].dataId["exposure"],
        )
        raw_det_ids = {ref.dataId["detector"] for ref in inputRefs.raws}
        for attr in ("ptc", "flat", "linearizer", "crosstalk", "intrinsicZernikes"):
            refs = getattr(inputRefs, attr)
            setattr(inputRefs, attr, [r for r in refs if r.dataId["detector"] in raw_det_ids])

        # Time each input type separately to find I/O bottleneck.
        t0 = time.perf_counter()
        raws = butlerQC.get(inputRefs.raws)
        t1 = time.perf_counter()
        ptc = butlerQC.get(inputRefs.ptc)
        t2 = time.perf_counter()
        flat = butlerQC.get(inputRefs.flat)
        t3 = time.perf_counter()
        linearizer = butlerQC.get(inputRefs.linearizer)
        t4 = time.perf_counter()
        crosstalk = butlerQC.get(inputRefs.crosstalk)
        t5 = time.perf_counter()
        refCat = butlerQC.get(inputRefs.refCat)
        t6 = time.perf_counter()
        intrinsicZernikes = butlerQC.get(inputRefs.intrinsicZernikes)
        t7 = time.perf_counter()
        self.log.info(
            _colorize(
                "butlerQC.get timing: raws=%.3fs ptc=%.3fs flat=%.3fs"
                " linearizer=%.3fs crosstalk=%.3fs refCat=%.3fs"
                " intrinsicZernikes=%.3fs total=%.3fs",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
            t6 - t5,
            t7 - t6,
            t7 - t0,
        )
        butler_elapsed = t7 - t0
        butler_times = dict(
            raws=t1 - t0,
            ptc=t2 - t1,
            flat=t3 - t2,
            linearizer=t4 - t3,
            crosstalk=t5 - t4,
            refCat=t6 - t5,
            intrinsicZernikes=t7 - t6,
        )
        outputs = self.run(
            raws=raws,
            ptc=ptc,
            flat=flat,
            linearizer=linearizer,
            crosstalk=crosstalk,
            refCat=refCat,
            intrinsicZernikes=intrinsicZernikes,
            butler_elapsed=butler_elapsed,
            butler_times=butler_times,
            numCores=butlerQC.resources.num_cores,
        )
        t8 = time.perf_counter()
        self.log.info("run() execution: %.3fs", t8 - t7)
        butlerQC.put(outputs.blitzResults, outputRefs.blitzResults)

    @timeMethod
    def run(
        self,
        raws: list,
        ptc: list,
        flat: list,
        linearizer: list,
        crosstalk: list,
        refCat: list,
        intrinsicZernikes: list | None = None,
        butler_elapsed: float = 0.0,
        butler_times: dict | None = None,
        numCores: int = 1,
    ) -> pipeBase.Struct:
        """Run ISR, WCS refit, catalog selection, and stamp cutting on all 8
        corner raws in parallel.

        Parameters
        ----------
        raws : list of lsst.afw.image.Exposure
        ptc : list of lsst.ip.isr.PhotonTransferCurveDataset
        flat : list of lsst.afw.image.ExposureF
        linearizer : list of lsst.ip.isr.Linearizer
        crosstalk : list of lsst.ip.isr.CrosstalkCalib
        refCat : list of DeferredDatasetHandle or SimpleCatalog
            Shards used for both WCS fitting and donut selection, loaded once
            per detector.  The WCS fit reads ``astromRefFilter`` (resolved as the
            load's ``fluxField``) and donut selection reads the per-band
            ``photoRefFilter``/``photoRefFilterPrefix`` column off the same
            catalog.
        intrinsicZernikes : list of IntrinsicZernikes, optional
            One calibration per corner detector.  None or empty when absent.
        butler_elapsed : float, optional
            Total butlerQC.get() wall time in seconds, for logging and plot.
        butler_times : dict, optional
            Per-dataset butlerQC.get() times keyed by dataset type name.
        numCores : int
        """
        self.log.info(
            _colorize(
                "DonutBlitzMonolithTask.run() with %d cores, butler elapsed=%.3fs",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            numCores,
            butler_elapsed,
        )
        t_run0 = time.perf_counter()

        detNameById = {}
        rawByName = {}
        for exp in raws:
            det = exp.getDetector()
            detNameById[det.getId()] = det.getName()
            rawByName[det.getName()] = exp
        ptcByName = {p._detectorName: p for p in ptc}
        flatByName = {f.getDetector().getName(): f for f in flat}
        linearizerByName = {lin._detectorName: lin for lin in linearizer}
        crosstalkByName = {ct._detectorName: ct for ct in crosstalk}

        missing = CORNER_DET_NAMES - rawByName.keys()
        if missing:
            raise RuntimeError(f"Missing corner detector raws: {sorted(missing)}")

        if intrinsicZernikes:
            self.log.info("Loaded %d intrinsic Zernike calibration(s).", len(intrinsicZernikes))
        else:
            self.log.warning("No intrinsic Zernike calibrations provided.")
        self.intrinsicZernikes = list(intrinsicZernikes) if intrinsicZernikes else []
        intrinsicZernikesByName = {
            detNameById[iz.getMetadata()["LSST BUTLER DATAID DETECTOR"]]: iz
            for iz in self.intrinsicZernikes
        }

        band = next(iter(rawByName.values())).filter.bandLabel
        if self.config.photoRefFilter is not None:
            photo_filter_name = self.config.photoRefFilter
        else:
            photo_filter_name = f"{self.config.photoRefFilterPrefix}_{band}"

        loader = None
        if not refCat:
            self.log.warning("No reference catalog shards provided; skipping WCS refit and donut selection.")
        else:
            self.log.info("Loading reference catalog shards for WCS refit and donut selection.")
            loader = ReferenceObjectLoader(
                dataIds=[h.dataId for h in refCat],
                refCats=refCat,
            )
            loader.config.pixelMargin = 300  # extra tolerance for uncertain WCS

        t_refcat0 = time.perf_counter()
        det_refcats: dict = {}
        for name, raw in rawByName.items():
            raw_wcs = raw.getWcs()
            raw_bbox = raw.getBBox()
            raw_epoch = raw.getInfo().getVisitInfo().date.toAstropy()
            load_result = None
            if loader is not None:
                try:
                    load_result = loader.loadPixelBox(
                        bbox=raw_bbox,
                        wcs=raw_wcs,
                        filterName=self.config.astromRefFilter,
                        epoch=raw_epoch,
                    )
                except Exception as exc:
                    self.log.warning("Failed to load refcat for %s: %s", name, exc)
            det_refcats[name] = load_result
        t_refcat_elapsed = time.perf_counter() - t_refcat0

        # Stub loader: AstrometryTask.solve() calls refObjLoader.getMetadataBox()
        # unconditionally even when load_result is pre-supplied. That method is
        # pure geometry -- it never accesses catalog data, dataId.region, or the
        # flux aliases.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.pixelMargin = 0
        self.astromTask.setRefObjLoader(astrom_stub_loader)

        cutout_cfg = dict(
            maxFitScatter=self.config.maxFitScatter,
            astromRefFilter=self.config.astromRefFilter,
            photoRefFilter=photo_filter_name,
        )

        _CALIB_STORE.clear()
        _CALIB_STORE["isr_task"] = self.isrTask
        _CALIB_STORE["bkg_task"] = self.subtractBackground
        _CALIB_STORE["detect_diameter_task"] = self.detectDiameter
        _CALIB_STORE["blind_detect_task"] = self.blindDetect
        _CALIB_STORE["astrom_task"] = self.astromTask
        _CALIB_STORE["donut_selector_task"] = self.donutSelector
        _CALIB_STORE["measure_candidates_task"] = self.measureCandidatesTask
        _CALIB_STORE["cut_stamps_task"] = self.cutStampsTask
        _CALIB_STORE["cutout_cfg"] = cutout_cfg
        _CALIB_STORE["det_refcats"] = det_refcats
        for name in CORNER_DET_NAMES:
            missing_calib = [
                k for k, d in [
                    ("ptc", ptcByName),
                    ("flat", flatByName),
                    ("linearizer", linearizerByName),
                    ("crosstalk", crosstalkByName),
                ]
                if name not in d
            ]
            if missing_calib:
                raise RuntimeError(
                    f"Missing calibration(s) for detector {name}: {missing_calib}"
                )
            _CALIB_STORE[name] = dict(
                raw=rawByName[name],
                ptc=ptcByName[name],
                flat=flatByName[name],
                linearizer=linearizerByName[name],
                crosstalk=crosstalkByName[name],
            )

        # WF estimation config — populate CALIB_STORE for shared resources.
        visitInfo = next(iter(rawByName.values())).getInfo().getVisitInfo()
        boresight_rot_rad = visitInfo.boresightRotAngle.asRadians()
        boresight_par_rad = visitInfo.boresightParAngle.asRadians()
        boresight_alt_rad = visitInfo.boresightAzAlt.getLatitude().asRadians()
        rtp_deg = (
            (np.degrees(boresight_par_rad - boresight_rot_rad - np.pi / 2) + 180) % 360 - 180
            if self.wfFittingTask.config.modelSpiderShadows
            else None
        )

        # Store task and mode for WF worker access
        _CALIB_STORE["wf_fitting_task"] = self.wfFittingTask
        _CALIB_STORE["wfEstimationMode"] = self.config.wfEstimationMode

        # Keep old wf_cfg for backward compatibility with _wf_worker (deprecated)
        wavelength_by_band = {bl.value: wl for bl, wl in _INSTRUMENT.wavelength.items()}
        lstsq_kwargs_parsed = {k: ast.literal_eval(v) for k, v in self.wfFittingTask.config.lstsqKwargs.items()}
        _CALIB_STORE["wf_cfg"] = dict(
            nollIndices=np.array(list(self.wfFittingTask.config.nollIndices)),
            lstsqKwargs=lstsq_kwargs_parsed,
            binning=self.wfFittingTask.config.binning,
            bkgOrder=self.wfFittingTask.config.bkgOrder,
            modelSpiderShadows=self.wfFittingTask.config.modelSpiderShadows,
            doAoiThroughput=self.wfFittingTask.config.doAoiThroughput,
            systematicLossAlpha=self.wfFittingTask.config.systematicLossAlpha,
            triangleMode=self.wfFittingTask.config.triangleMode,
            rtp_deg=rtp_deg,
            boresight_alt_rad=boresight_alt_rad,
            band=band,
            calib_noll_indices=np.arange(4, _ZK_JMAX + 1),
            wavelength_by_band=wavelength_by_band,
            wfFitTimeoutPerDonut=self.wfFittingTask.config.wfFitTimeoutPerDonut,
            wfInitialGuessOnly=self.wfFittingTask.config.wfInitialGuessOnly,
            wfEstimationMode=self.config.wfEstimationMode,
        )
        # Telescope is band- and quantum-fixed; build once here and share via
        # COW instead of reloading "LSST_{band}.yaml" per donut in workers.
        _telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
        _CALIB_STORE["telescope"] = _telescope
        _CALIB_STORE["telescope_extra"] = _telescope.withLocallyShiftedOptic(
            "Detector", [0, 0, _INSTRUMENT.defocalOffset]
        )
        _CALIB_STORE["telescope_intra"] = _telescope.withLocallyShiftedOptic(
            "Detector", [0, 0, -_INSTRUMENT.defocalOffset]
        )

        cutout_args = sorted(CORNER_DET_NAMES)

        self.log.info(
            "Running cutout workers on %d corner detectors with %d core(s)",
            len(cutout_args),
            numCores,
        )
        t_cutout0 = time.perf_counter()
        if numCores == 1:
            t_dispatch = time.time()
            results = [_run_cutout_worker((arg, t_dispatch)) for arg in cutout_args]
        else:
            t_pool0 = time.perf_counter()
            # Never more workers than detectors to process, matching the WF pool
            # below. cutout_args is CORNER_DET_NAMES, so never empty.
            n_cutout_workers = min(numCores, len(cutout_args))
            with mp.get_context("fork").Pool(processes=n_cutout_workers) as pool:
                t_pool1 = time.perf_counter()
                t_dispatch = time.time()
                results = pool.map(_run_cutout_worker, [(arg, t_dispatch) for arg in cutout_args])
            t_pool2 = time.perf_counter()
            self.log.info(
                _colorize(
                    "Cutout pipeline: pool create: %.3fs, pool.map: %.3fs",
                    _ANSI_BOLD,
                    _ANSI_CYAN,
                    enabled=self._colorLogEnabled,
                ),
                t_pool1 - t_pool0,
                t_pool2 - t_pool1,
            )
        t_cutout1 = time.perf_counter()

        donuts = []
        for r in results:
            scatter_str = f'{r["scatter_arcsec"]:.3f}"' if r["scatter_arcsec"] is not None else "N/A"
            self.log.info(
                "  %s: dispatch=%.3fs  isr=%.3fs  bkg=%.3fs"
                "  diam=%.3fs  detect=%.3fs  wcs=%.3fs (scatter=%s)"
                "  select=%.3fs  cut=%.3fs  donuts=%d",
                r["det_name"],
                r["dispatch_to_arrival"],
                r["isr_run"],
                r["bkg_run"],
                r["diam_run"],
                r["blind_detect_run"],
                r["wcs_refit_run"],
                scatter_str,
                r["catalog_select_run"],
                r.get("stamp_cut_run", 0.0),
                len(r["catalog"]),
            )
            if r["wcs_refit_error"]:
                self.log.warning("  %s: WCS refit failed: %s", r["det_name"], r["wcs_refit_error"])
            if r["cat_select_error"]:
                self.log.warning(
                    "  %s: catalog selection failed: %s",
                    r["det_name"],
                    r["cat_select_error"],
                )
            donuts.extend(r["catalog"])

        # Annotate each accepted donut with realized intrinsic Zernikes.
        for r in results:
            calib = intrinsicZernikesByName.get(r["det_name"])
            for d in r["catalog"]:
                if calib is not None:
                    d.intrinsic_zk = np.squeeze(
                        calib.getIntrinsicZernikes(
                            np.degrees(d.fa_x_ccs),
                            np.degrees(d.fa_y_ccs),
                        )
                    )
                else:
                    d.intrinsic_zk = None

        # WF dispatch
        mode = self.config.wfEstimationMode
        results_by_det = {r["det_name"]: r["catalog"] for r in results}
        groups, unmatched_donuts = _build_wf_groups(mode, results_by_det, band, rtp_deg, boresight_alt_rad)

        self.log.info("WF dispatch (%s): %d work unit(s)", mode, len(groups))
        t_wf0 = time.perf_counter()
        if not groups:
            wf_results = []
        elif numCores == 1 or len(groups) == 1:
            wf_results = [_wf_fitting_worker(g) for g in groups]
        else:
            n_workers = min(numCores, len(groups))
            with mp.get_context("fork").Pool(processes=n_workers) as wf_pool:
                wf_results = wf_pool.map(_wf_fitting_worker, groups)
        t_wf1 = time.perf_counter()
        n_ok = sum(r.get("success") for r in wf_results)
        elapsed_fits = [r["fit_info"].get("elapsed", float("nan")) for r in wf_results]
        self.log.info(
            "WF results (%s): %d/%d succeeded  wall=%.1fs  fit_total=%.1fs  fit_mean=%.1fs",
            mode,
            n_ok,
            len(wf_results),
            t_wf1 - t_wf0,
            sum(e for e in elapsed_fits if not np.isnan(e)),
            np.nanmean(elapsed_fits) if elapsed_fits else float("nan"),
        )

        t_plot0 = time.perf_counter()
        self.log.info(
            _colorize(
                "Timing summary: butler=%.1fs  refcat=%.1fs  cutout=%.1fs  danish=%.1fs  total=%.1fs",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            butler_elapsed,
            t_refcat_elapsed,
            t_cutout1 - t_cutout0,
            t_wf1 - t_wf0,
            t_plot0 - t_run0,
        )
        visit_id = next(iter(raws)).getInfo().getVisitInfo().id

        catalog = self._buildCatalog(
            results=results,
            wf_results=wf_results,
            donuts=donuts,
            unmatched_donuts=unmatched_donuts,
            visit_id=visit_id,
            run_elapsed=t_plot0 - t_run0,
            refcat_elapsed=t_refcat_elapsed,
            butler_elapsed=butler_elapsed,
            butler_times=butler_times or {},
            cutout_elapsed=t_cutout1 - t_cutout0,
            danish_elapsed=t_wf1 - t_wf0,
            photo_filter_name=photo_filter_name,
            astrom_filter_name=self.config.astromRefFilter,
        )

        if self.config.savePlots:
            self.plotTask.run(catalog)
            self.log.info("Diagnostic plot: %.3fs", time.perf_counter() - t_plot0)

        return pipeBase.Struct(
            donuts=donuts,
            wf_results=wf_results,
            blitzResults=Table(catalog)
        )

    def _buildCatalog(
        self,
        results: list,
        wf_results: list,
        donuts: list,
        unmatched_donuts: list,
        visit_id: int,
        run_elapsed: float = 0.0,
        refcat_elapsed: float = 0.0,
        butler_elapsed: float = 0.0,
        butler_times: dict | None = None,
        cutout_elapsed: float = 0.0,
        danish_elapsed: float = 0.0,
        photo_filter_name: str = "",
        astrom_filter_name: str = "",
    ) -> QTable:
        """Build a per-donut QTable covering every donut cut from this visit.

        Parameters
        ----------
        results : list
            Per-detector cutout dicts from ``_cutoutPipeline`` (supplies rejected
            donuts and per-detector metadata).
        wf_results : list
            Per-fit WF result dicts from the WF worker pool.
        donuts : list
            Donut dicts that passed selection.
        unmatched_donuts : list
            Donut dicts with no intra/extra partner (paired mode only).  These
            also appear in ``donuts``; the table carries one row per donut.
        visit_id : int
            Visit identifier.

        Returns
        -------
        QTable
            Exactly one row per donut, keyed by ``(det_name, id)``, with two
            independent flags:

            ``candidate``
                The donut passed every selection and quality cut.  See
                ``reject_reasons`` and the ``rejected_*`` booleans for why not.
            ``used``
                A wavefront fit consumed the donut and returned a result.  A
                candidate can be unused either because no group claimed it
                (paired mode, surplus donut with no partner: ``fit_mode`` empty,
                ``group`` -1) or because its fit timed out or raised
                (``fit_success`` False).  Unused rows have all-NaN Zernikes.

            Array columns (``stamp``, ``model_img``, ``wf_img``) are zero-padded
            to a common shape.  ``wf_img`` is filled for every donut with a
            stamp -- from the fitter when a fit ran, otherwise by binning the
            stamp the same way -- so only ``model_img`` is all-NaN for unused
            donuts.  Visit-level and per-detector scalars are stored in
            ``table.meta``.
        """
        # Build lookup: (id, det_name) -> (wf donut entry, group index).
        # Key on both fields to handle the same refcat star on SW0 and SW1.
        wf_by_id: dict = {}
        for group_idx, r in enumerate(wf_results):
            for wd in r.get("donuts", []):
                wf_by_id[(wd.donut_id, wd.det_name)] = (wd, group_idx)

        # Build lookup: det_name -> per-detector metadata from cutout results.
        det_meta: dict = {}
        rejected_by_det: dict = {}
        for r in results:
            dname = str(r["det_name"])
            det_meta[dname] = {
                "scatter_arcsec": (
                    r["scatter_arcsec"] if r["scatter_arcsec"] is not None else float("nan")
                ),
                "wcs_refit_error": r["wcs_refit_error"],
                "cat_select_error": r["cat_select_error"],
                "isr_run": r.get("isr_run", float("nan")),
                "bkg_run": r.get("bkg_run", float("nan")),
                "diam_run": r.get("diam_run", float("nan")),
                "blind_detect_run": r.get("blind_detect_run", float("nan")),
                "wcs_refit_run": r.get("wcs_refit_run", float("nan")),
                "catalog_select_run": r.get("catalog_select_run", float("nan")),
            }
            rejected_by_det[dname] = r.get("rejected_catalog", [])

        def _encode_nearby(entries):
            """Return (x, y, mag) arrays of length ``_MAX_NEARBY`` for one donut.

            Neighbors are sorted brightest-first (ascending magnitude); entries with
            a NaN magnitude sort last. The brightest ``_MAX_NEARBY`` are kept and
            shorter lists are NaN-padded. True pre-truncation counts are captured
            separately (see n_nearby_*).
            """
            x = np.full(_MAX_NEARBY, np.nan, dtype=float)
            y = np.full(_MAX_NEARBY, np.nan, dtype=float)
            mag = np.full(_MAX_NEARBY, np.nan, dtype=float)
            brightest = sorted(entries, key=lambda e: (np.isnan(e[2]), e[2]))[:_MAX_NEARBY]
            n_nearby = len(brightest)
            x[:n_nearby] = [e[0] for e in brightest]
            y[:n_nearby] = [e[1] for e in brightest]
            mag[:n_nearby] = [e[2] for e in brightest]
            return x, y, mag

        # Collect every donut exactly once, tagged with whether it passed
        # selection ("candidate"). Whether a fit actually consumed it ("used")
        # is derived per row below from the wavefront result.
        #
        # `donuts` and `unmatched_donuts` overlap: in paired mode the surplus
        # donuts on whichever detector detected more pass selection but have no
        # partner, so they appear in both lists. Keyed dedupe keeps one row per
        # donut -- they stay candidates, they just never got fitted.
        def _key(d):
            return (d.det_name, d.id)

        all_donuts = []
        _seen = set()
        for d, candidate in (
            [(d, True) for d in donuts]
            + [
                (d, False)
                for r in results
                for d in rejected_by_det.get(str(r["det_name"]), [])
            ]
            + [(d, True) for d in unmatched_donuts]
        ):
            k = _key(d)
            if k in _seen:
                continue
            _seen.add(k)
            all_donuts.append((d, candidate))

        if not all_donuts:
            return QTable()

        # Pre-compute common sizes from config (all stamps and WF images are uniform).
        stamp_size = self.cutStampsTask.config.stampSize
        # WF images: binned and forced to odd size (see _prep_donut_for_danish).
        _binned = self.cutStampsTask.config.stampSize // self.wfFittingTask.config.binning
        wf_img_size = _binned if _binned % 2 == 1 else _binned - 1

        rows = []
        for d, candidate in all_donuts:
            sid = d.id
            wd, grp = wf_by_id.get((sid, d.det_name), (_NULL_WF, -1))

            zk_dev = wd.zk_dev
            zk_int = wd.zk_intrinsic

            stamp = (
                d.stamp.astype(float)
                if d.stamp is not None
                else np.full((stamp_size, stamp_size), np.nan, dtype=float)
            )

            # Donuts no fit consumed (paired-mode surplus) have no WF image from
            # the fitter, so bin their stamp here with the same prep the fitter
            # would have applied. Keeps them plottable as data-only rows.
            if wd.img is not None:
                wf_img = wd.img.astype(float)
            elif d.stamp is not None:
                wf_img = _bin_stamp_odd(d.stamp, self.wfFittingTask.config.binning)
            else:
                wf_img = np.full((wf_img_size, wf_img_size), np.nan, dtype=float)

            model_img = (
                wd.model_img.astype(float)
                if wd.model_img is not None
                else np.full((wf_img_size, wf_img_size), np.nan, dtype=float)
            )

            photo_x, photo_y, photo_mag = _encode_nearby(d.nearby_photo)
            astrom_x, astrom_y, astrom_mag = _encode_nearby(d.nearby_astrom)
            row = {
                # --- identity ---
                "visit_id": d.visit_id,
                "det_id": d.det_id,
                "det_name": d.det_name,
                "id": sid,
                # From the detector, not from the fit result: donuts no fit
                # consumed still belong to a defocal side.
                "defocal": "intra" if d.det_id in _INTRA_FOCAL_DET_IDS else "extra",
                "band": d.band,
                # candidate: passed every selection/quality cut.
                # used: a fit consumed it and returned a wavefront.
                # A candidate that is not used has no Zernikes (all NaN) -- see
                # fit_mode/group/fit_success for which of the two reasons.
                "candidate": bool(candidate),
                "used": bool(wd.fit_success),
                # --- geometry ---
                "centroid_x_raw": d.centroid_x_raw,
                "centroid_y_raw": d.centroid_y_raw,
                "fa_x_ccs": d.fa_x_ccs,
                "fa_y_ccs": d.fa_y_ccs,
                "field_dist_deg": np.degrees(np.hypot(d.fa_x_ccs, d.fa_y_ccs)),
                "n_quarter": d.n_quarter,
                # --- nearby refcat sources (brightest-first, padded to _MAX_NEARBY) ---
                "nearby_photo_x": photo_x * u.pix,
                "nearby_photo_y": photo_y * u.pix,
                "nearby_photo_mag": photo_mag * u.mag,
                "nearby_astrom_x": astrom_x * u.pix,
                "nearby_astrom_y": astrom_y * u.pix,
                "nearby_astrom_mag": astrom_mag * u.mag,
                "n_nearby_photo": len(d.nearby_photo),
                "n_nearby_astrom": len(d.nearby_astrom),
                # --- selection metrics ---
                "flux": d.flux,
                "snr": d.snr,
                "inner_frac": d.inner_frac,
                "outer_frac": d.outer_frac,
                "outer_sector_minmax_frac": d.outer_sector_minmax_frac,
                "bkg": d.bkg,
                "bkg_std": d.bkg_std,
                "donut_radius": d.donut_radius,
                "obscuration": d.obscuration,
                "nearest_neighbor_dist_px": d.nearest_neighbor_dist_px,
                "n_neighbors_in_stamp": d.n_neighbors_in_stamp,
                "catalog_centroid_offset_px": d.catalog_centroid_offset_px,
                "rejected_sat": d.rejected_sat,
                "rejected_inner_frac": d.rejected_inner_frac,
                "rejected_outer_frac": d.rejected_outer_frac,
                "rejected_snr": d.rejected_snr,
                "rejected": d.rejected,
                # --- fit results ---
                "fit_mode": wd.fit_mode,
                "group": grp,
                "group_size": wd.group_size,
                "fit_success": wd.fit_success,
                "fit_elapsed": wd.fit_elapsed,
                "setup_elapsed": wd.setup_elapsed,
                "fit_nfev": wd.fit_nfev,
                "fit_cost": wd.fit_cost,
                "fit_dx": wd.fit_dx,
                "fit_dy": wd.fit_dy,
                "fit_flux": wd.fit_flux,
                "fit_fwhm": wd.fit_fwhm,
                "blend_frac": wd.blend_frac,
                # --- per-Noll Zernikes (µm), Noll 4..``_ZK_JMAX`` ---
                **_zk_cols("dev", zk_dev),
                **_zk_cols("intrinsic", zk_int),
                # --- embedded images ---
                "stamp": stamp,
                "wf_img": wf_img,
                "model_img": model_img,
            }
            rows.append(row)

        table = QTable(rows)
        table.meta["visit_id"] = visit_id
        table.meta["run_elapsed"] = run_elapsed
        table.meta["refcat_elapsed"] = refcat_elapsed
        table.meta["butler_elapsed"] = butler_elapsed
        table.meta["butler_times"] = dict(butler_times or {})
        table.meta["cutout_elapsed"] = cutout_elapsed
        table.meta["danish_elapsed"] = danish_elapsed
        table.meta["photo_filter_name"] = photo_filter_name
        table.meta["astrom_filter_name"] = astrom_filter_name
        table.meta["noll_indices"] = list(self.wfFittingTask.config.nollIndices)
        table.meta["det_meta"] = det_meta
        table.meta["aperture_outer_margin_frac"] = self.measureCandidatesTask.config.apertureOuterMarginFrac
        table.meta["aperture_inner_buffer_frac"] = self.measureCandidatesTask.config.apertureInnerBufferFrac
        table.meta["bkg_annulus_inner_frac"] = self.measureCandidatesTask.config.bkgAnnulusInnerFrac
        table.meta["bkg_annulus_outer_frac"] = self.measureCandidatesTask.config.bkgAnnulusOuterFrac
        table.meta["max_donuts"] = self.cutStampsTask.config.maxDonuts
        table.meta["wf_mode"] = self.config.wfEstimationMode
        return table


class DonutBlitzPlotTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzPlotTask."""

    blitzResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzMonolithTask containing all data "
            "needed to regenerate diagnostic plots."
        ),
        name="donutBlitzResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
    )


class DonutBlitzPlotTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzPlotTaskConnections,  # type: ignore
):
    """Configuration for DonutBlitzPlotTask."""

    colorLog: pexConfig.Field = pexConfig.Field(
        doc=(
            "Colorize select log messages with ANSI escape codes. If None "
            "(the default), color is enabled only when stdout is an "
            "interactive terminal."
        ),
        dtype=bool,
        default=None,
        optional=True,
    )


class DonutBlitzPlotTask(pipeBase.PipelineTask):
    """PipelineTask that regenerates diagnostic plots from ``donutBlitzResults``.

    Can run standalone (reading from the butler) or be called as a subtask of
    ``DonutBlitzMonolithTask`` when ``savePlots=True``.
    """

    ConfigClass = DonutBlitzPlotTaskConfig
    _DefaultName = "donutBlitzPlotTask"
    config: DonutBlitzPlotTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._colorLogEnabled = _resolveColorLogEnabled(self.config.colorLog)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        catalog = inputs["blitzResults"].get(parameters={"strip_astropy_meta_yaml": False})
        self.run(catalog)

    def run(self, catalog: Table) -> None:
        """Generate donut and WF diagnostic plots from the blitzResults catalog.

        Parameters
        ----------
        catalog : QTable
            Per-donut table as produced by
            ``DonutBlitzMonolithTask._buildCatalog``.  Visit-level and
            per-detector metadata are in ``catalog.meta``.
        """
        catalog = QTable(catalog)
        self._saveDonutDiagnosticPlot(catalog)
        self._saveWfDiagnosticPlot(catalog)

    def _saveDonutDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a single diagnostic PNG with one section per detector.

        Layout per detector:
          - Left column: stats text (timing, WCS scatter, donut count, errors)
          - Remaining columns: donut stamps (up to maxDonuts), each annotated
            with flux and field angle

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.  Per-detector metadata is in
            ``catalog.meta["det_meta"]``; visit-level scalars are in
            ``catalog.meta``.
        """
        import matplotlib.patches as mpatches
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        run_elapsed = meta["run_elapsed"]
        refcat_elapsed = meta["refcat_elapsed"]
        butler_elapsed = meta["butler_elapsed"]
        photo_filter_label = meta["photo_filter_name"]
        astrom_filter_label = meta["astrom_filter_name"]
        visit_id = meta["visit_id"]
        det_meta = meta["det_meta"]

        # Group rows by detector; split on selection outcome. This plot is about
        # donut *selection*, so it splits on "candidate" -- a candidate that no
        # fit consumed still shows in the accepted panel, since it passed every
        # cut this plot reports on.
        det_name_col = np.asarray(catalog["det_name"], dtype=str)
        dets_with_data = []
        for det_name in sorted(set(det_name_col.tolist())):
            det_rows = catalog[det_name_col == det_name]
            acc = det_rows[det_rows["candidate"]]
            rej = det_rows[~det_rows["candidate"]]
            if len(acc) > 0 or len(rej) > 0:
                dets_with_data.append((det_name, acc, rej))

        n_dets = len(dets_with_data)
        if n_dets == 0:
            return

        STAMPS_PER_ROW = 8
        REJECTED_PER_ROW = 2
        STAMP_COL_W = 1.8 * 1.05
        STATS_COL_W = 2.8
        ROW_H = 1.7 * 1.05
        LEGEND_H = 0.35
        SPACER_W = 0.15
        SUPTITLE_H = 0.55

        N_COLS = 1 + STAMPS_PER_ROW + 1 + REJECTED_PER_ROW
        fig_w = STATS_COL_W + (STAMPS_PER_ROW + REJECTED_PER_ROW) * STAMP_COL_W + SPACER_W
        fig_h = n_dets * ROW_H + LEGEND_H + SUPTITLE_H

        fig = Figure(figsize=(fig_w, fig_h), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
        butler_str = f"  butler={butler_elapsed:.1f}s" if butler_elapsed > 0 else ""
        fig.suptitle(
            f"DonutBlitz diagnostics  visit={visit_id}"
            f"  refcat={refcat_elapsed:.1f}s{butler_str}  run={run_elapsed:.1f}s",
            fontsize=9,
        )

        w_stats = STATS_COL_W / STAMP_COL_W
        w_spacer = SPACER_W / STAMP_COL_W
        gs = GridSpec(
            n_dets + 1,
            N_COLS,
            figure=fig,
            height_ratios=[ROW_H] * n_dets + [LEGEND_H],
            width_ratios=[w_stats] + [1] * STAMPS_PER_ROW + [w_spacer] + [1] * REJECTED_PER_ROW,
        )
        COL_ACCEPTED_START = 1
        COL_SPACER = 1 + STAMPS_PER_ROW
        COL_REJECTED_START = COL_SPACER + 1

        # Per-detector radius/obscuration ride each donut row (from
        # Donut.donut_radius), so a detector's aperture/annulus circles match its
        # own detected donut size. The visit-level meta scalars below are the
        # fallback for rows that lack the columns -- e.g. an older blitzResults
        # catalog written before these columns existed, read back by a
        # standalone DonutBlitzPlotTask.
        _stamp_outer_margin_frac = catalog.meta["aperture_outer_margin_frac"]
        _stamp_inner_buffer_frac = catalog.meta["aperture_inner_buffer_frac"]
        _stamp_bkg_inner_frac = catalog.meta["bkg_annulus_inner_frac"]
        _stamp_bkg_outer_frac = catalog.meta["bkg_annulus_outer_frac"]

        # Every stamp's view is pinned to its own pixel extent, so a stamp fills
        # its axes exactly and consumes the same figure area no matter how many
        # pixels it contains. Setting the limits explicitly (rather than leaving
        # them to autoscale) is also required because ax.plot of the refcat
        # overlays triggers autoscale where add_patch alone does not, which would
        # otherwise pull in the annulus circles and shrink the stamp only on rows
        # that happen to have overlays.
        #
        # A config-derived view (e.g. donutRadius * bkgAnnulusOuterFrac) would
        # instead couple the drawn size to stampSize: at stampSize=215 the image
        # overflows its axes by ~15%.
        _STAMP_TEXT_FONTSIZE = 3.5

        def _draw_stamp(ax, row, rejected=False):
            stamp = np.array(row["stamp"])
            h_px = stamp.shape[0] // 2
            vmin, vmax = np.nanpercentile(stamp, [1, 99])
            _edge = h_px + 0.5
            ax.imshow(
                stamp,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="gray",
                aspect="equal",
                extent=[-_edge, _edge, -_edge, _edge],
            )

            # Per-detector radius/obscuration off the row, falling back to the
            # visit-level meta scalar if a row lacks a finite value (stale
            # catalog). row is an astropy Row, so membership is via colnames.
            dr = row["donut_radius"]
            ob = row["obscuration"]
            if dr is not None and ob is not None:
                _circ_specs = [
                    (dr * ob * _stamp_inner_buffer_frac, _COLOR_BKG_ANNULUS, "--"),
                    (dr * ob, _COLOR_APERTURE, "-"),
                    (dr * _stamp_outer_margin_frac, _COLOR_APERTURE, "-"),
                    (dr * _stamp_bkg_inner_frac, _COLOR_BKG_ANNULUS, "--"),
                    (dr * _stamp_bkg_outer_frac, _COLOR_BKG_ANNULUS, "--"),
                ]
                for _rad, _col, _ls in _circ_specs:
                    ax.add_patch(
                        mpatches.Circle(
                            (0, 0),
                            _rad,
                            fill=False,
                            edgecolor=_col,
                            linewidth=1.0,
                            linestyle=_ls,
                            alpha=0.45,
                            zorder=4,
                        )
                    )

            if rejected:
                ax.plot([-_edge, _edge], [-_edge, _edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)
                ax.plot([-_edge, _edge], [_edge, -_edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)

            nq = row["n_quarter"] % 4

            def _xform(dx, dy):
                """Map a raw-pixel offset to stamp display coords.

                Must mirror the stamp transform in `_cut_and_evaluate_stamps`,
                ``np.rot90(stamp, k=-n_quarter).T`` -- including the transpose.
                The loop applies the rot90 in (row, col) space; returning
                ``(r, c)`` rather than ``(c, r)`` is what applies the ``.T``.
                Under ``origin="lower"`` the displayed x axis is the column
                index and y is the row index, so the returned pair is
                ``(x, y)`` after transposition.
                """
                r, c = dy, dx
                for _ in range(nq):
                    r, c = c, -r
                return r, c

            n_photo = min(row["n_nearby_photo"], _MAX_NEARBY)
            px = row["nearby_photo_x"][:n_photo].to_value(u.pix)
            py = row["nearby_photo_y"][:n_photo].to_value(u.pix)
            pm = row["nearby_photo_mag"][:n_photo].to_value(u.mag)
            for dx, dy, mag in zip(px, py, pm):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "o", ms=6, mfc="none", mec=_COLOR_PHOTO_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty + 3, f"{mag:.2f}", color=_COLOR_PHOTO_REFCAT, fontsize=3.5, zorder=4)

            n_astrom = min(row["n_nearby_astrom"], _MAX_NEARBY)
            ax_ = row["nearby_astrom_x"][:n_astrom].to_value(u.pix)
            ay_ = row["nearby_astrom_y"][:n_astrom].to_value(u.pix)
            am_ = row["nearby_astrom_mag"][:n_astrom].to_value(u.mag)
            for dx, dy, mag in zip(ax_, ay_, am_):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "+", ms=6, mec=_COLOR_ASTROM_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty - 5, f"{mag:.2f}", color=_COLOR_ASTROM_REFCAT, fontsize=3.5, zorder=4)
            # Pin the view to the stamp's own edges: constant figure footprint
            # regardless of pixel count, and no autoscale from the overlays.
            ax.set_xlim(-_edge, _edge)
            ax.set_ylim(-_edge, _edge)

            inner_frac = row["inner_frac"]
            outer_frac = row["outer_frac"]
            outer_sector_minmax = row["outer_sector_minmax_frac"]
            snr = row["snr"]
            if_str = f"if={inner_frac:.3f}" if np.isfinite(inner_frac) else "if=?"
            of_str = f"of={outer_frac:.3f}" if np.isfinite(outer_frac) else "of=?"
            osm_str = f"osm={outer_sector_minmax:.3f}" if np.isfinite(outer_sector_minmax) else "osm=?"
            snr_str = f"snr={snr:.0f}" if np.isfinite(snr) else "snr=?"
            sid = row["id"]
            sid_str = f"id={sid}" if sid != 0 else ""
            _text_color = _COLOR_REJECTED if rejected else "black"

            _flags = [
                name
                for name, val in (
                    ("sat", row["rejected_sat"]),
                    ("inner", row["rejected_inner_frac"]),
                    ("outer", row["rejected_outer_frac"]),
                    ("snr", row["rejected_snr"]),
                )
                if val
            ]
            rej_str = f"[{'|'.join(_flags)}]" if _flags else ""
            # Bottom-anchored just above the axes, so the block grows upward and
            # never overlaps the stamp -- clearance is independent of stamp size.
            # (Top-anchoring inside the axes hung the text down over the image;
            # at stampSize 167 it overlapped by ~1pt, and worse for larger stamps.)
            ax.annotate(
                f"{snr_str}  {rej_str}\n{if_str}  {of_str}  {osm_str}\n{sid_str}",
                xy=(0.05, 1.00),
                xycoords="axes fraction",
                xytext=(0, 1.0),
                textcoords="offset points",
                fontsize=_STAMP_TEXT_FONTSIZE,
                va="bottom",
                ha="left",
                color=_text_color,
                bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"),
                zorder=6,
                annotation_clip=False,
            )

        for row_idx, (det_name, acc_rows, rej_rows) in enumerate(dets_with_data):
            sm = det_meta.get(det_name, {})
            scatter_val = sm.get("scatter_arcsec", float("nan"))
            scatter_str = f'{scatter_val:.3f}"' if np.isfinite(scatter_val) else "N/A"

            ax_stats = fig.add_subplot(gs[row_idx, 0])
            ax_stats.axis("off")
            lines = [
                f"{det_name}",
                f"donuts: {len(acc_rows)}",
                f"isr:    {sm.get('isr_run', float('nan')):.3f}s",
                f"bkg:    {sm.get('bkg_run', float('nan')):.3f}s",
                f"diam:   {sm.get('diam_run', float('nan')):.3f}s",
                f"detect: {sm.get('blind_detect_run', float('nan')):.3f}s",
                f"wcs:    {sm.get('wcs_refit_run', float('nan')):.3f}s  ({scatter_str})",
                f"select: {sm.get('catalog_select_run', float('nan')):.3f}s",
            ]
            if sm.get("wcs_refit_error"):
                lines.append(f"WCS ERR: {sm['wcs_refit_error'][:40]}")
            if sm.get("cat_select_error"):
                lines.append(f"CAT ERR: {sm['cat_select_error'][:40]}")
            ax_stats.text(
                0.05, 0.95, "\n".join(lines), transform=ax_stats.transAxes,
                fontsize=6, va="top", family="monospace",
            )

            for col_idx in range(STAMPS_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_ACCEPTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(acc_rows):
                    continue
                _draw_stamp(ax, acc_rows[col_idx])

            ax_sp = fig.add_subplot(gs[row_idx, COL_SPACER])
            ax_sp.axis("off")

            for col_idx in range(REJECTED_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_REJECTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(rej_rows):
                    continue
                _draw_stamp(ax, rej_rows[col_idx], rejected=True)

        ax_legend = fig.add_subplot(gs[n_dets, :])
        ax_legend.axis("off")
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor=_COLOR_PHOTO_REFCAT, markersize=6,
                label=f"photo refcat ({photo_filter_label})",
            ),
            Line2D(
                [0], [0], marker="+", color=_COLOR_ASTROM_REFCAT, markersize=6, linestyle="none",
                label=f"astrom refcat ({astrom_filter_label})",
            ),
        ]
        ax_legend.legend(
            handles=legend_handles, loc="center", ncol=2, fontsize=7,
            frameon=False, handletextpad=0.5, columnspacing=2.0,
        )

        fname = f"donut_diag_{visit_id}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        self.log.info("Saved diagnostic plot: %s", fname)

    def _saveWfDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a WF diagnostic PNG modelled on the AOS donut-fits layout.

        Layout: 2×2 grid of corners (R00, R04, R40, R44).
        Within each corner: one row per fit result.
        Each row: intra data|model|resid|zk_bar  extra data|model|resid|zk_bar.
        Zernike bars are vertical, ±1 µm, no tick labels.

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.
        """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        visit_id = meta["visit_id"]
        refcat_elapsed = meta["refcat_elapsed"]
        butler_elapsed = meta["butler_elapsed"]
        butler_times = meta["butler_times"]
        cutout_elapsed = meta["cutout_elapsed"]
        danish_elapsed = meta["danish_elapsed"]
        noll_cfg = meta["noll_indices"]
        wf_mode = meta["wf_mode"]
        ZK_MIN, ZK_MAX = 4, 28

        # Reconstruct wf_results-like list from QTable by grouping on "group" column.
        # Include only rows with a valid fit (fit_mode != "" and group >= 0).
        groups: dict[int, list] = {}
        for row in catalog:
            fm = row["fit_mode"]
            grp = row["group"]
            if grp < 0 or fm == "":
                continue
            if grp not in groups:
                groups[grp] = []
            groups[grp].append(row)

        plottable = []
        for grp_idx, rows in sorted(groups.items()):
            first = rows[0]
            # Determine if this group has a real model (any non-NaN model_img).
            has_model = any(
                not np.all(np.isnan(np.array(r["model_img"]))) for r in rows
            )
            if not has_model:
                continue
            det_names = list(dict.fromkeys(r["det_name"] for r in rows))
            mode = first["fit_mode"]
            success = bool(first["fit_success"])
            elapsed = first["fit_elapsed"]
            nfev = first["fit_nfev"]
            fwhm = first["fit_fwhm"]
            zk_by_noll = {j: first[f"Z{j}_dev"] for j in range(4, _ZK_JMAX + 1)}

            donuts_out = []
            for r in rows:
                model_arr = np.array(r["model_img"])
                donuts_out.append({
                    "donut_id": r["id"],
                    "det_name": r["det_name"],
                    "defocal": r["defocal"],
                    "img": np.array(r["wf_img"]),
                    "model_img": model_arr if not np.all(np.isnan(model_arr)) else None,
                    "blend_frac": r["blend_frac"],
                    "elapsed": elapsed,
                    "nfev": nfev,
                    "fwhm": fwhm,
                    "success": success,
                })
            plottable.append({
                "mode": mode,
                "det_names": det_names,
                "success": success,
                "fit_info": {"elapsed": elapsed, "nfev": nfev, "fwhm": fwhm},
                "donuts": donuts_out,
                "zk_by_noll": zk_by_noll,
            })

        # Candidate donuts that no fit consumed (paired-mode surplus: no partner
        # on the other detector, so ``group`` is -1 and ``fit_mode`` empty). They
        # have no model or Zernikes, but their binned stamp is still worth
        # seeing, so carry them as data-only single-donut records.
        unfitted = []
        for row in catalog:
            if row["group"] >= 0 or not row["candidate"]:
                continue
            img = np.array(row["wf_img"])
            if np.all(np.isnan(img)):
                continue
            unfitted.append({
                "mode": wf_mode,
                "det_names": [row["det_name"]],
                "success": False,
                "fit_info": {
                    "elapsed": float("nan"), "nfev": 0, "fwhm": float("nan"),
                },
                "donuts": [{
                    "donut_id": row["id"],
                    "det_name": row["det_name"],
                    "defocal": row["defocal"],
                    "img": img,
                    "model_img": None,
                    "blend_frac": row["blend_frac"],
                }],
                "zk_by_noll": {},
            })

        if not plottable and not unfitted:
            self.log.info("No WF results with model images; skipping WF diagnostic plot.")
            return

        _CORNERS = list(CORNER_PAIRS)

        def _corner_of(r):
            for s in r["det_names"]:
                if str(s) in CORNER_BY_DET_NAME:
                    return CORNER_BY_DET_NAME[str(s)]
            return _CORNERS[0]

        # 4-stop diverging colormap: blue → white (zero) → vermillion.
        # Anchors: -vmax=blue, -vmax/10=sky blue, 0=white, +vmax=vermillion.
        # Normalised positions over [-vmax, vmax]: 0.0, 0.45, 0.5, 1.0.
        def _hex_to_rgb(h):
            return tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))

        _cmap_bwr = LinearSegmentedColormap.from_list(
            "bwr_donut",
            list(zip(
                [0.0, 0.45, 0.5, 1.0],
                [_hex_to_rgb(h) for h in (
                    _COLOR_CMAP_NEG, _COLOR_CMAP_MID, "#FFFFFF", _COLOR_CMAP_POS,
                )],
            )),
        )
        _cmap_bwr_sym = LinearSegmentedColormap.from_list(
            "bwr_donut_sym",
            list(zip(
                [0.0, 0.5, 1.0],
                [_hex_to_rgb(h) for h in (_COLOR_CMAP_NEG, "#FFFFFF", _COLOR_CMAP_POS)],
            )),
        )

        def _draw_stamp(ax, img, cmap, vmin, vmax, label=""):
            ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if label:
                ax.set_title(label, fontsize=5, pad=1)

        def _draw_bar(ax, zk_by_noll, inset_label=""):
            """Vertical bar chart of Zernikes in µm, ±1 µm, no tick labels.

            Always spans ZK_MIN..ZK_MAX regardless of the configured
            ``nollIndices`` so plots stay comparable across configs; indices
            that were not fitted plot as zero rather than dropping out.
            """
            bar_noll = [j for j in range(ZK_MIN, ZK_MAX + 1)]
            vals = [
                v if np.isfinite(v := zk_by_noll.get(j, 0.0)) else 0.0
                for j in bar_noll
            ]
            ax.bar(bar_noll, vals, color="k", width=0.8)
            ax.axhline(0, color="k", linewidth=0.4)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(ZK_MIN - 0.5, ZK_MAX + 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for j in [4, 11, 22]:   # spherical (m=0)
                ax.axvspan(j - 0.5, j + 0.5, color="#000000", alpha=0.15, ec="none")
            for j in [7, 16]:       # coma (m=1)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_COMA, alpha=0.35, ec="none")
            for j in [5, 12, 23]:   # astigmatism (m=2)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_ASTIGMATISM, alpha=0.25, ec="none")
            for j in [9, 18]:       # trefoil (m=3)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_TREFOIL, alpha=0.25, ec="none")
            for j in [14, 25]:      # quadrafoil (m=4)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_QUADRAFOIL, alpha=0.25, ec="none")
            ax.axvspan(19.5, 21.5, color=_COLOR_PENTAFOIL, alpha=0.25, ec="none")   # pentafoil (m=5)
            ax.axvspan(26.5, 28.5, color=_COLOR_HEXAFOIL, alpha=0.25, ec="none")   # hexafoil (m=6)
            if inset_label:
                ax.text(0.03, 0.97, inset_label, transform=ax.transAxes, fontsize=4,
                        va="top", ha="left",
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6))

        by_corner: dict[str, list] = {c: [] for c in _CORNERS}
        for r in plottable:
            by_corner[_corner_of(r)].append(r)

        unfitted_by_corner: dict[str, list] = {c: [] for c in _CORNERS}
        for r in unfitted:
            unfitted_by_corner[_corner_of(r)].append(r)


        def _explode(r):
            """Split a group record into one record per donut, sharing group fields."""
            return [{**r, "donuts": [d]} for d in r.get("donuts", [])]

        def _pair_up(records):
            """Lay single-donut records out as (intra, extra) plot rows.

            Only for modes whose groups carry no intra/extra pairing of their
            own: the pairing here is cosmetic, so rows are matched by position
            and the shorter side padded with None to keep every donut visible.
            """
            intras = [r for r in records if r["donuts"][0].get("defocal") == "intra"]
            extras = [r for r in records if r["donuts"][0].get("defocal") == "extra"]
            return [
                (intras[i] if i < len(intras) else None,
                 extras[i] if i < len(extras) else None)
                for i in range(max(len(intras), len(extras)))
            ]

        row_pairs: dict[str, list[tuple]] = {}
        for corner, corner_results in by_corner.items():
            if wf_mode == "paired":
                # The group *is* an intra/extra pair, so it supplies both halves
                # of the row; the donut of each defocal type is picked out below.
                fit_rows = [(r, r) for r in corner_results]
            else:
                # These groups don't pair donuts, so flatten to one record per
                # donut and pair for layout only. Exploding is a no-op for
                # "unpaired" (one donut per group already).
                fit_rows = _pair_up(
                    [s for r in corner_results for s in _explode(r)]
                )
            # Surplus donuts have no partner by construction, so they lay out
            # positionally below the fitted rows, one side of each row blank.
            row_pairs[corner] = fit_rows + _pair_up(unfitted_by_corner[corner])

        CELL = 1.0
        ROW_H = 1.0
        HPAD = 0.08
        # Always lay out maxDonuts rows per corner, padding short corners with
        # blank rows, so figure dimensions and axes positions depend only on
        # config -- not on how many donuts a given mode happened to fit. This
        # makes plots for the same exposure blinkable across fitting modes.
        # Never fewer rows than any corner actually has, so a corner with more
        # fits than maxDonuts grows the layout rather than losing rows.
        max_rows = max(
            [meta["max_donuts"], 1]
            + [len(v) for v in row_pairs.values()]
        )
        for corner, pairs in row_pairs.items():
            row_pairs[corner] = pairs + [(None, None)] * (max_rows - len(pairs))

        corner_w = 10 * CELL
        fig_w = 2 * corner_w + 0.3
        fig_h = 2 * max_rows * ROW_H + 0.4

        fig = Figure(figsize=(fig_w, fig_h))
        outer = GridSpec(2, 2, figure=fig, hspace=HPAD, wspace=0.06,
                         left=0.01, right=0.99, top=0.94, bottom=0.01)
        corner_pos = {"R00": (0, 0), "R40": (0, 1), "R04": (1, 0), "R44": (1, 1)}

        for corner in _CORNERS:
            pairs = row_pairs[corner]
            grow, gcol = corner_pos[corner]
            inner = GridSpecFromSubplotSpec(
                max_rows, 8, subplot_spec=outer[grow, gcol],
                hspace=0.0, wspace=0.0,
                width_ratios=[1, 1, 1, 2, 1, 1, 1, 2],
            )
            sw1 = f"{corner}_SW1"
            sw0 = f"{corner}_SW0"

            def _rec_info(r):
                if r is None:
                    return float("nan"), 0, False, float("nan")
                fi = r.get("fit_info", {})
                return (fi.get("elapsed", float("nan")), fi.get("nfev", 0),
                        r.get("success", False), fi.get("fwhm", float("nan")))

            for row_idx, (r_intra, r_extra) in enumerate(pairs):
                if r_intra is not None:
                    intra_rec = next(
                        (d for d in r_intra.get("donuts", []) if d["defocal"] == "intra"), None
                    )
                    elapsed_i, nfev_i, success_i, fwhm_i = _rec_info(r_intra)
                    zk_by_noll_i = r_intra.get(
                        "zk_by_noll", {j: float("nan") for j in range(4, _ZK_JMAX + 1)}
                    )
                else:
                    intra_rec = None
                    elapsed_i, nfev_i, success_i, fwhm_i = _rec_info(None)
                    zk_by_noll_i = {j: float("nan") for j in range(4, _ZK_JMAX + 1)}

                if r_extra is not None:
                    extra_rec = next(
                        (d for d in r_extra.get("donuts", []) if d["defocal"] == "extra"), None
                    )
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(r_extra)
                    zk_by_noll_e = r_extra.get(
                        "zk_by_noll", {j: float("nan") for j in range(4, _ZK_JMAX + 1)}
                    )
                else:
                    extra_rec = None
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(None)
                    zk_by_noll_e = {j: float("nan") for j in range(4, _ZK_JMAX + 1)}

                intra_img = intra_rec["img"] if intra_rec else None
                intra_mod = intra_rec["model_img"] if intra_rec else None
                intra_sid = intra_rec["donut_id"] if intra_rec else None
                intra_blend = intra_rec.get("blend_frac", float("nan")) if intra_rec else float("nan")

                extra_img = extra_rec["img"] if extra_rec else None
                extra_mod = extra_rec["model_img"] if extra_rec else None
                extra_sid = extra_rec["donut_id"] if extra_rec else None
                extra_blend = extra_rec.get("blend_frac", float("nan")) if extra_rec else float("nan")

                def _bar_label(elapsed, nfev, success):
                    status = "x0" if nfev == 0 else ("ok" if success else "fail")
                    return f"t={elapsed:.1f}s {status} nfev={nfev}"

                intra_label = _bar_label(elapsed_i, nfev_i, success_i)
                extra_label = _bar_label(elapsed_e, nfev_e, success_e)
                intra_hdr = f"intra {sw1}" if row_idx == 0 else ""
                extra_hdr = f"extra {sw0}" if row_idx == 0 else ""

                def _triplet_and_bar(col_start, data, model, det_hdr, label,
                                     sid, fwhm, zk_by_noll, blend_frac_val=float("nan")):
                    if data is not None:
                        vmax = np.nanpercentile(np.abs(data), 99) or 1.0
                        has_model = model is not None
                        resid = (data - model) if has_model else None
                        vmax_r = (np.nanpercentile(np.abs(resid), 99) or 1.0) if has_model else 1.0
                        for ci, (img, cmap, vmin, vmx) in enumerate([
                            (data, _cmap_bwr, -vmax, vmax),
                            (model if has_model else None, _cmap_bwr, -vmax, vmax),
                            (resid, _cmap_bwr_sym, -vmax_r, vmax_r),
                        ]):
                            ax = fig.add_subplot(inner[row_idx, col_start + ci])
                            lbl = det_hdr if ci == 0 else ""
                            if img is None:
                                ax.axis("off")
                                if lbl:
                                    ax.set_title(lbl, fontsize=5, pad=1)
                                continue
                            _draw_stamp(ax, img, cmap, vmin, vmx, label=lbl)
                            ann_kw = dict(transform=ax.transAxes, fontsize=4, color="k",
                                         va="top", ha="left",
                                         bbox=dict(boxstyle="square,pad=0.1",
                                                   fc="white", ec="none", alpha=0.6))
                            if ci == 0 and sid is not None:
                                ax.text(0.02, 0.98, f"id={sid}", **ann_kw)
                            if ci == 1 and np.isfinite(fwhm):
                                ax.text(0.02, 0.98, f"blur={fwhm:.2f}arcsec", **ann_kw)
                            if ci == 2 and np.isfinite(blend_frac_val):
                                ax.text(0.02, 0.98, f"blend={blend_frac_val:.3f}", **ann_kw)
                        ax_bar = fig.add_subplot(inner[row_idx, col_start + 3])
                        if has_model:
                            _draw_bar(ax_bar, zk_by_noll, inset_label=label)
                        else:
                            ax_bar.axis("off")
                    else:
                        for ci in range(4):
                            ax = fig.add_subplot(inner[row_idx, col_start + ci])
                            ax.axis("off")
                            if ci == 0 and det_hdr:
                                ax.set_title(det_hdr, fontsize=5, pad=1)

                _triplet_and_bar(0, intra_img, intra_mod, intra_hdr, intra_label,
                                 intra_sid, fwhm_i, zk_by_noll_i, intra_blend)
                _triplet_and_bar(4, extra_img, extra_mod, extra_hdr, extra_label,
                                 extra_sid, fwhm_e, zk_by_noll_e, extra_blend)

        proc_total = refcat_elapsed + cutout_elapsed + danish_elapsed
        bt = butler_times or {}
        butler_line = (
            "  ".join(f"{k}={v:.1f}s" for k, v in bt.items() if v > 0.0)
            or f"total={butler_elapsed:.1f}s"
        )
        fig.suptitle(
            f"WF fits  visit={visit_id}  mode={wf_mode}\n"
            f"butler.get:  {butler_line}\n"
            f"refcat={refcat_elapsed:.1f}s  cutout={cutout_elapsed:.1f}s  "
            f"danish={danish_elapsed:.1f}s  proc_total={proc_total:.1f}s",
            fontsize=7,
        )
        fname = f"wf_diag_{visit_id}.png"
        # No bbox_inches="tight" here: it crops to drawn content, so the output
        # size would still shift with the number of populated rows even though
        # the layout is padded to max_rows. A fixed canvas keeps every plot for
        # a given config pixel-comparable.
        fig.savefig(fname, dpi=300)
        self.log.info("Saved WF diagnostic plot: %s", fname)


# Register plotTask after DonutBlitzPlotTask is defined
DonutBlitzMonolithTaskConfig.plotTask = pexConfig.ConfigurableField(
    target=DonutBlitzPlotTask,
    doc='Subtask that generates diagnostic plots for a blitz visit.',
)
