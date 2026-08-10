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
import dataclasses
import logging
import multiprocessing as mp
import signal
import sys
import time
from typing import Any

import batoid
import danish
import galsim
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import numpy as np
from astropy.table import QTable
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.afw.image import Exposure
from lsst.fgcmcal.utilities import lookupStaticCalibrations
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
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.utils import binArray, getTaskInstrument
from lsst.utils.timer import timeMethod
from scipy.optimize import least_squares
from scipy.signal import correlate
from scipy.stats import median_abs_deviation
from skimage.feature import peak_local_max

_CALIB_STORE: dict = {}  # populated in parent before fork; workers inherit via COW

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
CORNER_SENSOR_NAMES = frozenset(s for sw0, sw1 in CORNER_PAIRS.values() for s in (sw0, sw1))
# Sensor name -> corner, derived from CORNER_PAIRS rather than re-encoded.
CORNER_BY_SENSOR = {s: corner for corner, pair in CORNER_PAIRS.items() for s in pair}

# Display names for DonutSourceSelectorTask rejection reasons. Reasons absent
# here are shown verbatim. "faint_neighbor" folds into "blend" (both mean a
# neighbor spoiled this donut), and the selector's "field_dist" deliberately
# shares a label with the monolith's own field-distance check even though the
# thresholds differ slightly (selector maxFieldDist vs. detect maxFieldDist).
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


def _buildAnnularTemplate(radius: float, innerFrac: float) -> np.ndarray:
    """Return a binary annular stamp for cross-correlation donut detection."""
    r_int = int(radius)
    cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
    r = np.hypot(cx, cy)
    return np.where((r < radius) & (r >= radius * innerFrac), 1.0, 0.0)


def _detectPeaks(
    exposureTrim: Exposure,
    radius: float,
    obscuration: float,
    detectionBinning: int,
    peakMinDistanceFactor: float,
    peakExcludeBorderFactor: float,
) -> np.ndarray:
    """Detect donut centroids via annular template cross-correlation.

    The image is optionally binned, histogram-equalised, then cross-correlated
    with a binary annular template matched to the donut size.  Local maxima of
    the correlation map are returned as pixel coordinates in the un-binned
    frame.

    Parameters
    ----------
    exposureTrim : Exposure
        Science exposure (already trimmed to the sensor region of interest).
    radius : float
        Expected outer donut radius in pixels (un-binned).
    obscuration : float
        Central obscuration fraction (inner radius / outer radius).
    detectionBinning : int
        Pixel binning factor applied before correlation (1 = no binning).
    peakMinDistanceFactor : float
        Minimum peak separation as a multiple of the binned donut radius.
    peakExcludeBorderFactor : float
        Border exclusion width as a multiple of the binned donut radius.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing ``(row, col)`` centroids in
        un-binned pixel coordinates.
    """
    binning = detectionBinning
    radius_binned = radius / binning
    template = _buildAnnularTemplate(radius_binned, innerFrac=obscuration)

    if binning > 1:
        binnedImg = afwMath.binImage(exposureTrim.image, binning)
        arr = binnedImg.array
    else:
        arr = exposureTrim.image.array

    heq = np.digitize(arr, np.nanquantile(arr, np.linspace(0, 1, 256)))
    det = correlate(heq.astype(float), template, mode="same")
    peaks = peak_local_max(
        det,
        min_distance=int(peakMinDistanceFactor * radius_binned),
        exclude_border=int(peakExcludeBorderFactor * radius_binned),
    )

    if binning > 1:
        peaks = peaks * binning

    return peaks


def _measureFlux(
    peaks: np.ndarray,
    exposureTrim: Exposure,
    radius: float,
    obscuration: float,
    apertureOuterMarginFrac: float,
    apertureInnerBufferFrac: float,
    bkgAnnulusInnerFrac: float,
    bkgAnnulusOuterFrac: float,
) -> QTable:
    """Measure aperture flux and per-pixel noise for each detected donut.

    For each peak a local background is estimated from an annular region
    (inner pupil + outer sky), subtracted, then flux is summed over the
    main annular aperture.  Per-pixel noise is estimated from the IQR of
    first-differences in the background region.

    Parameters
    ----------
    peaks : np.ndarray
        ``(N, 2)`` array of ``(row, col)`` centroids from `_detectPeaks`.
    exposureTrim : Exposure
        Science exposure in un-binned pixel coordinates.
    radius : float
        Expected outer donut radius in pixels.
    obscuration : float
        Central obscuration fraction (inner radius / outer radius).
    apertureOuterMarginFrac : float
        Outer edge of the main photometric aperture, as a multiple of
        ``radius``.
    apertureInnerBufferFrac : float
        Inner edge of the background/blend-check region inside the
        obscuration, as a multiple of ``radius * obscuration``.
    bkgAnnulusInnerFrac : float
        Inner edge of the outer background/blend-check annulus, as a
        multiple of ``radius``.
    bkgAnnulusOuterFrac : float
        Outer edge of the outer background/blend-check annulus, as a
        multiple of ``radius``; also sets the cutout half-width.

    Returns
    -------
    QTable
        One row per peak with columns ``centroid_x``, ``centroid_y``,
        ``flux``, ``inner_flux``, ``outer_flux``, ``outer_sector_max_frac``,
        ``std``, and ``snr``.  Peaks that fall too close to the image border
        have ``nan`` values.
    """
    arr = exposureTrim.image.array
    half = int(radius * bkgAnnulusOuterFrac)

    gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
    r = np.hypot(gx, gy)
    sector_angle = np.arctan2(gy, gx)

    main_mask = (r < radius * apertureOuterMarginFrac) & (r > radius * obscuration)
    inner_mask = r < radius * obscuration * apertureInnerBufferFrac
    outer_mask = (r > radius * bkgAnnulusInnerFrac) & (r < radius * bkgAnnulusOuterFrac)
    bkg_mask = inner_mask | outer_mask
    n_main = np.sum(main_mask)
    outer_sector_masks = [
        outer_mask & (sector_angle >= -np.pi + k * np.pi / 4) & (sector_angle < -np.pi + (k + 1) * np.pi / 4)
        for k in range(8)
    ]

    flux_list, inner_flux_list, outer_flux_list, std_list = [], [], [], []
    outer_sector_max_list = []

    for row, col in zip(peaks[:, 0], peaks[:, 1]):
        rmin, rmax = int(round(row)) - half, int(round(row)) + half + 1
        cmin, cmax = int(round(col)) - half, int(round(col)) + half + 1

        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            flux_list.append(np.nan)
            inner_flux_list.append(np.nan)
            outer_flux_list.append(np.nan)
            outer_sector_max_list.append(np.nan)
            std_list.append(np.nan)
            continue

        stamp = arr[rmin:rmax, cmin:cmax]
        bkg = np.nanmedian(stamp[bkg_mask])
        stamp_sub = stamp - bkg

        flux = float(np.sum(stamp_sub[main_mask]))
        flux_list.append(flux)
        inner_flux_list.append(float(np.sum(stamp_sub[inner_mask])))
        outer_flux_list.append(float(np.sum(stamp_sub[outer_mask])))
        if flux != 0:
            sector_fluxes = [float(np.sum(stamp_sub[m])) / flux for m in outer_sector_masks]
            outer_sector_max_list.append(float(max(abs(f) for f in sector_fluxes)))
        else:
            outer_sector_max_list.append(np.nan)

        diff = (stamp_sub - np.roll(stamp_sub, 1, axis=0))[bkg_mask]
        q75, q25 = np.nanpercentile(diff, [75, 25])
        std_list.append(float((q75 - q25) / 1.349 / np.sqrt(2)))

    table = QTable()
    table["centroid_x"] = np.array(peaks[:, 1], dtype=float)
    table["centroid_y"] = np.array(peaks[:, 0], dtype=float)
    table["flux"] = np.array(flux_list, dtype=float)
    table["inner_flux"] = np.array(inner_flux_list, dtype=float)
    table["outer_flux"] = np.array(outer_flux_list, dtype=float)
    table["outer_sector_max_frac"] = np.array(outer_sector_max_list, dtype=float)
    table["std"] = np.array(std_list, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["snr"] = (table["flux"] / table["std"]) / np.sqrt(n_main)
    return table


def _blindDetect(
    exposure: Exposure,
    detect_cfg: dict,
    bkg_config: Any,
    radius: float,
    obscuration: float,
) -> QTable:
    """Subtract background and detect donuts via annular template cross-correlation.

    Runs `SubtractBackgroundTask` in-place on ``exposure``, erodes the border
    by ``edgeMargin`` pixels, then calls `_detectPeaks`.  Flux measurement is
    deferred to `_cutStamps`.

    Parameters
    ----------
    exposure : Exposure
        Science exposure; background is subtracted in-place.
    detect_cfg : dict
        Detection config keys: ``edgeMargin``, ``detectionBinning``,
        ``peakMinDistanceFactor``, ``peakExcludeBorderFactor``.
    bkg_config : Any
        Config object for `SubtractBackgroundTask`.
    radius : float
        Expected outer donut radius in pixels.
    obscuration : float
        Central obscuration fraction (inner radius / outer radius).

    Returns
    -------
    QTable
        Columns ``centroid_x``, ``centroid_y`` in full-exposure pixel
        coordinates.  Empty table if no peaks are found.
    """
    SubtractBackgroundTask(config=bkg_config).run(exposure=exposure)

    trimmedBBox = exposure.getBBox().erodedBy(detect_cfg["edgeMargin"])
    exposureTrim = exposure[trimmedBBox].clone()

    peaks = _detectPeaks(
        exposureTrim,
        radius,
        obscuration,
        detect_cfg["detectionBinning"],
        detect_cfg["peakMinDistanceFactor"],
        detect_cfg["peakExcludeBorderFactor"],
    )

    empty = QTable(names=["centroid_x", "centroid_y"], dtype=[float, float])

    if len(peaks) == 0:
        return empty

    xOffset = trimmedBBox.getMinX()
    yOffset = trimmedBBox.getMinY()
    return QTable(
        {
            "centroid_x": peaks[:, 1] + xOffset,
            "centroid_y": peaks[:, 0] + yOffset,
        }
    )


def _buildAfwSourceCat(blindDetections: QTable, wcs) -> afwTable.SourceCatalog:
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

    sourceCat.reserve(len(blindDetections))
    for i, row in enumerate(blindDetections):
        x, y = float(row["centroid_x"]), float(row["centroid_y"])
        sky = wcs.pixelToSky(x, y)
        src = sourceCat.addNew()
        src.set(sourceIdKey, i)
        src.set(sourceRAKey, sky.getRa())
        src.set(sourceDecKey, sky.getDec())
        src.set(sourceCentroidKey, lsst.geom.Point2D(x, y))

    if not sourceCat.isContiguous():
        sourceCat = sourceCat.copy(deep=True)
    return sourceCat


def _buildFakeExposure(
    detector,
    wcs,
    visitInfo,
    filterLabel,
) -> afwImage.ExposureF:
    """Build a pixel-free ExposureF carrying only geometry and metadata.

    AstrometryTask reads bbox, wcs, filter.bandLabel, visitInfo.date, and
    visitInfo.getExposureTime() from the exposure — no pixel data required.
    """
    bbox = detector.getBBox()
    fake = afwImage.ExposureF(bbox)
    fake.setWcs(wcs)
    fake.setDetector(detector)
    fake.getInfo().setVisitInfo(visitInfo)
    fake.setFilter(filterLabel)
    return fake


def _refitWcs(
    blindDetections: QTable,
    postIsr: Exposure,
    astrom_cfg: dict,
) -> tuple:
    """Attempt to refit the WCS using the pre-loaded astrometry refcat.

    Parameters
    ----------
    blindDetections : QTable
        Centroid table from `_blindDetect` (columns ``centroid_x``,
        ``centroid_y``).
    postIsr : Exposure
        Post-ISR science exposure supplying the detector, WCS, and visit info.
    astrom_cfg : dict
        Astrometry config keys: ``minSourcesForWcsFit``, ``maxFitScatter``,
        ``astrom_task_config``, ``astrom_ref_obj_loader``.

    Returns
    -------
    refitted_wcs : `lsst.afw.geom.SkyWcs` or None
        Refitted WCS, or ``None`` if the fit was skipped, failed, or scatter
        exceeded the threshold.  ``None`` signals callers to fall back to
        blind detections and skip the photometry refcat.
    scatter_arcsec : float or None
        RMS on-sky scatter of the fit in arcseconds, or ``None`` when no fit
        was attempted or an exception was raised.
    error_str : str or None
        Human-readable reason for failure, or ``None`` on success.
    """
    wcs = postIsr.getWcs()
    detector = postIsr.getDetector()
    sensor_name = detector.getName()
    astrom_load_result = _CALIB_STORE.get("sensor_refcats", {}).get(sensor_name)

    if len(blindDetections) < astrom_cfg["minSourcesForWcsFit"] or astrom_load_result is None:
        return None, None, None

    visitInfo = postIsr.getInfo().getVisitInfo()
    filterLabel = postIsr.getFilter()
    try:
        astromTask = AstrometryTask(config=astrom_cfg["astrom_task_config"])
        # Need a refObjLoader set even when passing load_result, for getMetadataBox later.
        # Since load_result is passed, loadPixelBox is never called.
        astromTask.setRefObjLoader(astrom_cfg["astrom_ref_obj_loader"])
        afwCat = _buildAfwSourceCat(blindDetections, wcs)
        fakeExp = _buildFakeExposure(detector, wcs, visitInfo, filterLabel)
        astromResult = astromTask.solve(
            exposure=fakeExp,
            sourceCat=afwCat,
            load_result=astrom_load_result,
        )
        scatter_arcsec = astromResult.scatterOnSky.asArcseconds()
        if scatter_arcsec < astrom_cfg["maxFitScatter"]:
            return fakeExp.getWcs(), scatter_arcsec, None
        else:
            return (
                None,
                scatter_arcsec,
                f'scatter {scatter_arcsec:.2f}" >= {astrom_cfg["maxFitScatter"]}"',
            )
    except Exception as e:
        return None, None, str(e)


def _selectFromPhotoCat(
    refitted_wcs,
    postIsr: Exposure,
    sensor_name: str,
    astrom_cfg: dict,
) -> tuple:
    """Select donut positions from the pre-loaded photometry refcat.

    Reprojects the refcat through ``refitted_wcs``, runs
    `DonutSourceSelectorTask`, and builds diagnostic overlay arrays.  Only
    executes the catalog path when ``refitted_wcs`` is not ``None``; otherwise
    all outputs are ``None`` and the caller falls back to blind detections.

    Parameters
    ----------
    refitted_wcs : `lsst.afw.geom.SkyWcs` or None
        Refitted WCS from `_refitWcs`.  If ``None`` the function returns
        immediately with all ``None`` outputs.
    postIsr : Exposure
        Post-ISR science exposure supplying the detector geometry.
    sensor_name : str
        Detector name used to look up the pre-loaded refcats in
        ``_CALIB_STORE``.
    astrom_cfg : dict
        Config keys used here: ``donut_selector_config``,
        ``resolvedPhotoFilterName``, ``astromRefFilter``,
        ``saveDiagnosticPlot``.

    Returns
    -------
    catalog_centroids : tuple or None
        ``(centroid_x, centroid_y, source_ids)`` arrays for selected sources,
        or ``None`` if the catalog path was skipped or failed.
    sel_rejected_centroids : tuple or None
        ``(centroid_x, centroid_y, flux, source_ids, rejection_reasons)`` for
        the brightest selector-rejected sources (for diagnostic display), or
        ``None``.
    all_photo_cat : tuple or None
        ``(x, y, mag)`` arrays for the full photometry refcat projected through
        ``refitted_wcs`` (for diagnostic overplotting), or ``None``.
    all_astrom_cat : tuple or None
        ``(x, y, mag)`` arrays for the full astrometry refcat projected through
        ``refitted_wcs`` (for diagnostic overplotting), or ``None``.
    error_str : str or None
        Human-readable error from the catalog selection step, or ``None``.
    """
    # One shared load per sensor: astrometry reads its fluxField from it, donut
    # selection reads the per-band flux column off the same catalog.
    load_result = _CALIB_STORE.get("sensor_refcats", {}).get(sensor_name)
    photo_load_result = load_result
    astrom_load_result = load_result
    detector = postIsr.getDetector()
    save_diag = astrom_cfg.get("saveDiagnosticPlot", True)

    catalog_centroids = None
    sel_rejected_centroids = None
    all_photo_cat = None
    all_astrom_cat = None
    cat_select_error = None
    sel_rejected_refcat = None
    sel_rejection_reasons = np.array([], dtype=object)
    filterName = astrom_cfg.get("resolvedPhotoFilterName", "")

    if photo_load_result is not None and refitted_wcs is not None:
        try:
            # Reproject catalog sky coords through the refitted WCS so stamp
            # centroids are consistent with the corrected pointing.
            refCat = photo_load_result.refCat.copy(deep=True)
            afwTable.updateRefCentroids(refitted_wcs, refCat)
            if not refCat.isContiguous():
                refCat = refCat.copy(deep=True)

            donutSelectorTask = DonutSourceSelectorTask(config=astrom_cfg["donut_selector_config"])
            donutSelection = donutSelectorTask.run(refCat, detector, filterName)
            sel_mask = np.array(donutSelection.selected, dtype=bool)
            refSelection = refCat[sel_mask]
            sel_rejected_refcat = refCat[~sel_mask]
            sel_rejection_reasons = np.array(donutSelection.rejectionReasons)[~sel_mask]

            if len(refSelection) > 0:
                catalog_centroids = (
                    np.array(refSelection["centroid_x"]),
                    np.array(refSelection["centroid_y"]),
                    np.array(refSelection["id"]),
                )

            if save_diag:
                with np.errstate(invalid="ignore", divide="ignore"):
                    _flux = np.array(refCat[f"{filterName}_flux"])
                    _mag = -2.5 * np.log10(_flux) + 31.4
                all_photo_cat = (
                    np.array(refCat["centroid_x"]),
                    np.array(refCat["centroid_y"]),
                    _mag,
                )
        except Exception as e:
            cat_select_error = str(e)
            catalog_centroids = None
            all_photo_cat = None

    # Selector-rejected sources: top REJECTED_CANDIDATES by flux for display.
    REJECTED_CANDIDATES = 10
    if save_diag and sel_rejected_refcat is not None and len(sel_rejected_refcat) > 0:
        try:
            _rrej_flux = np.array(sel_rejected_refcat[f"{filterName}_flux"])
            _rrej_order = np.argsort(_rrej_flux)[::-1][:REJECTED_CANDIDATES]
            sel_rejected_centroids = (
                np.array(sel_rejected_refcat["centroid_x"])[_rrej_order],
                np.array(sel_rejected_refcat["centroid_y"])[_rrej_order],
                _rrej_flux[_rrej_order],
                np.array(sel_rejected_refcat["id"])[_rrej_order],
                sel_rejection_reasons[_rrej_order],
            )
        except Exception:
            pass

    # Astrometry refcat overlay (centroids via refitted WCS).
    if astrom_load_result is not None and save_diag and refitted_wcs is not None:
        try:
            _astrom_cat = astrom_load_result.refCat.copy(deep=True)
            afwTable.updateRefCentroids(refitted_wcs, _astrom_cat)
            _flux_field = f"{astrom_cfg['astromRefFilter']}_flux"
            with np.errstate(invalid="ignore", divide="ignore"):
                _astrom_mag = -2.5 * np.log10(np.array(_astrom_cat[_flux_field])) + 31.4
            all_astrom_cat = (
                np.array(_astrom_cat["centroid_x"]),
                np.array(_astrom_cat["centroid_y"]),
                _astrom_mag,
            )
        except Exception:
            pass

    return (
        catalog_centroids,
        sel_rejected_centroids,
        all_photo_cat,
        all_astrom_cat,
        cat_select_error,
    )


def _cutStamps(
    postIsr: Exposure,
    sensor_name: str,
    blindDetections: QTable,
    catalog_centroids: tuple | None,
    sel_rejected_centroids: tuple | None,
    all_photo_cat: tuple | None,
    all_astrom_cat: tuple | None,
    astrom_cfg: dict,
    radius: float,
    obscuration: float,
) -> tuple:
    """Measure image fluxes, apply quality cuts, and cut postISR stamps.

    Uses ``catalog_centroids`` (refcat path) when available; falls back to
    ``blindDetections`` otherwise.  Calls `_measureFlux` once on the full
    ``postIsr`` image, then filters on SAT mask, inner/outer flux fraction,
    and SNR before cutting stamps.

    Parameters
    ----------
    postIsr : Exposure
        Background-subtracted post-ISR science exposure.
    sensor_name : str
        Detector name, stored in each output donut dict.
    blindDetections : QTable
        Centroid table from `_blindDetect`.  Used as fallback when
        ``catalog_centroids`` is ``None``.
    catalog_centroids : tuple or None
        ``(centroid_x, centroid_y, source_ids)`` from `_selectFromPhotoCat`,
        or ``None`` to trigger the blind-detection fallback.
    sel_rejected_centroids : tuple or None
        Selector-rejected sources from `_selectFromPhotoCat`, passed through
        to the rejected-stamp diagnostic display.
    all_photo_cat : tuple or None
        ``(x, y, mag)`` photometry refcat overlay for diagnostic stamps.
    all_astrom_cat : tuple or None
        ``(x, y, mag)`` astrometry refcat overlay for diagnostic stamps.
    astrom_cfg : dict
        Config dict supplying ``detect_cfg`` (``maxDonuts``, ``stampSize``,
        ``minStampSnr``, ``innerFluxFractionCut``, ``outerFluxFractionCut``,
        ``edgeMargin``) and SAT-mask lookup.
    radius : float
        Expected outer donut radius in pixels.
    obscuration : float
        Central obscuration fraction (inner radius / outer radius).

    Returns
    -------
    donuts : list of dict
        Accepted donut stamp dicts, sorted brightest-first and capped at
        ``maxDonuts``.
    rejected_donuts : list of dict
        Rejected donut stamp dicts (SAT, field-distance, flux-fraction, SNR,
        and selector-rejected sources), sorted brightest-first.  Uncapped;
        callers that render a fixed number of columns (e.g. the diagnostic
        plot) are responsible for slicing.
    """
    detect_cfg = astrom_cfg["detect_cfg"]
    maxDonuts = detect_cfg["maxDonuts"]
    detector = postIsr.getDetector()
    band = postIsr.filter.bandLabel
    visit_id = postIsr.getInfo().getVisitInfo().id
    det_id = detector.getId()
    n_quarter = detector.getOrientation().getNQuarter()
    stampSize = detect_cfg["stampSize"]
    half = stampSize // 2
    apertureOuterMarginFrac = detect_cfg["apertureOuterMarginFrac"]
    apertureInnerBufferFrac = detect_cfg["apertureInnerBufferFrac"]
    bkgAnnulusInnerFrac = detect_cfg["bkgAnnulusInnerFrac"]
    bkgAnnulusOuterFrac = detect_cfg["bkgAnnulusOuterFrac"]

    # --- Resolve centroid list ---
    if catalog_centroids is not None:
        cat_cx, cat_cy, source_ids = catalog_centroids
    else:
        if len(blindDetections) == 0:
            return [], []
        cat_cx = np.array(blindDetections["centroid_x"])
        cat_cy = np.array(blindDetections["centroid_y"])
        source_ids = np.arange(len(cat_cx), dtype=np.int64)

    # --- Flux measurement and quality selection ---
    peaks = np.column_stack([cat_cy, cat_cx])
    measTable = _measureFlux(
        peaks,
        postIsr,
        radius,
        obscuration,
        apertureOuterMarginFrac,
        apertureInnerBufferFrac,
        bkgAnnulusInnerFrac,
        bkgAnnulusOuterFrac,
    )
    valid_mask = np.isfinite(measTable["flux"]) & (measTable["flux"] > 0)
    measTable = measTable[valid_mask]
    source_ids = source_ids[valid_mask]
    if len(measTable) == 0:
        return [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        innerOk = np.abs(measTable["inner_flux"] / measTable["flux"]) < detect_cfg["innerFracThreshold"]
        outerOk = np.abs(measTable["outer_flux"] / measTable["flux"]) < detect_cfg["outerFracThreshold"]
    snrOk = measTable["snr"] > detect_cfg["snrThreshold"]
    qual_mask = innerOk & outerOk & snrOk
    measTable = measTable[qual_mask]
    source_ids = source_ids[qual_mask]
    if len(measTable) == 0:
        return [], []
    flux_arr = np.array(measTable["flux"])
    inner_frac_arr = np.array(measTable["inner_flux"]) / flux_arr
    outer_frac_arr = np.array(measTable["outer_flux"]) / flux_arr
    outer_sector_max_arr = np.array(measTable["outer_sector_max_frac"])
    order = np.argsort(flux_arr)[::-1][:maxDonuts]
    centroid_x = np.array(measTable["centroid_x"])[order]
    centroid_y = np.array(measTable["centroid_y"])[order]
    flux_arr = flux_arr[order]
    inner_frac_arr = inner_frac_arr[order]
    outer_frac_arr = outer_frac_arr[order]
    outer_sector_max_arr = outer_sector_max_arr[order]
    source_ids = source_ids[order]

    # --- Precompute annular masks ---
    arr = postIsr.image.array
    mask_arr = postIsr.mask.array
    sat_bit = postIsr.mask.getPlaneBitMask("SAT")

    # Grid matches the cut stamp below: 2*half+1 px, centroid pixel exactly centered.
    _sgy, _sgx = np.mgrid[-half : half + 1, -half : half + 1]
    _sr = np.hypot(_sgx, _sgy)
    _s_main = (_sr < radius * apertureOuterMarginFrac) & (_sr > radius * obscuration)
    _s_bkg = (_sr < radius * obscuration * apertureInnerBufferFrac) | (
        (_sr > radius * bkgAnnulusInnerFrac) & (_sr < radius * bkgAnnulusOuterFrac)
    )
    _s_n_main = int(np.sum(_s_main))

    # Fallback annular masks: only needed for selector-rejected centroids,
    # which never pass through _measureFlux and so have no precomputed
    # inner_frac/outer_frac/outer_sector_max to reuse.
    _mhalf = int(radius * bkgAnnulusOuterFrac)
    _gy, _gx = np.mgrid[-_mhalf : _mhalf + 1, -_mhalf : _mhalf + 1]
    _r = np.hypot(_gx, _gy)
    _main_mask = (_r < radius * apertureOuterMarginFrac) & (_r > radius * obscuration)
    _inner_mask = _r < radius * obscuration * apertureInnerBufferFrac
    _outer_mask = (_r > radius * bkgAnnulusInnerFrac) & (_r < radius * bkgAnnulusOuterFrac)
    _sector_angle = np.arctan2(_gy, _gx)

    def _cut_stamp_dict(
        cx_f,
        cy_f,
        flux_val,
        source_id_val,
        inner_frac=None,
        outer_frac=None,
        outer_sector_max=None,
        reject_reason=None,
        blind_cx=None,
        blind_cy=None,
    ):
        """Cut one stamp and compute metrics. Returns dict or None on failure.

        ``inner_frac``/``outer_frac``/``outer_sector_max`` should be passed
        in (already computed by `_measureFlux`) whenever available; they are
        only recomputed here as a fallback for selector-rejected centroids,
        which bypass `_measureFlux` entirely.
        """
        # Odd-sized cut (2*half+1) so the centroid pixel sits exactly at the
        # stamp center. An even cut would offset the donut half a pixel from the
        # stamp's geometric center, and would force the crop-by-1 fallback in
        # _prep_donut_for_danish (Danish requires an odd stamp).
        cx, cy = int(round(float(cx_f))), int(round(float(cy_f)))
        rmin, rmax = cy - half, cy + half + 1
        cmin, cmax = cx - half, cx + half + 1
        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            return None
        saturated = bool(np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit))
        stamp = np.array(arr[rmin:rmax, cmin:cmax])
        stamp_ccs = np.rot90(stamp, k=-n_quarter).T

        if inner_frac is None or outer_frac is None or outer_sector_max is None:
            mmin_r, mmax_r = cy - _mhalf, cy + _mhalf + 1
            mmin_c, mmax_c = cx - _mhalf, cx + _mhalf + 1
            with np.errstate(invalid="ignore", divide="ignore"):
                if mmin_r >= 0 and mmax_r <= arr.shape[0] and mmin_c >= 0 and mmax_c <= arr.shape[1]:
                    mpatch = arr[mmin_r:mmax_r, mmin_c:mmax_c]
                    bkg = float(np.nanmedian(mpatch[_inner_mask | _outer_mask]))
                    mpatch_sub = mpatch - bkg
                    mflux = float(np.sum(mpatch_sub[_main_mask]))
                    inner_frac = (
                        float(np.sum(mpatch_sub[_inner_mask]) / mflux) if mflux != 0 else float("nan")
                    )
                    outer_frac = (
                        float(np.sum(mpatch_sub[_outer_mask]) / mflux) if mflux != 0 else float("nan")
                    )
                    if mflux != 0:
                        _sector_fluxes = [
                            float(
                                np.sum(
                                    mpatch_sub[
                                        _outer_mask
                                        & (_sector_angle >= -np.pi + k * np.pi / 4)
                                        & (_sector_angle < -np.pi + (k + 1) * np.pi / 4)
                                    ]
                                )
                            )
                            / mflux
                            for k in range(8)
                        ]
                        outer_sector_max = float(max(abs(f) for f in _sector_fluxes))
                    else:
                        outer_sector_max = float("nan")
                else:
                    inner_frac = outer_frac = outer_sector_max = float("nan")

        with np.errstate(invalid="ignore", divide="ignore"):
            _s_bkg_pix = stamp[_s_bkg]
            _s_bkg_std = float(np.nanstd(_s_bkg_pix)) if np.any(_s_bkg) else float("nan")
            _s_bkg_med = float(np.nanmedian(_s_bkg_pix)) if np.any(_s_bkg) else 0.0
            _s_signal = float(np.sum((stamp - _s_bkg_med)[_s_main]))
            stamp_snr = (
                _s_signal / (_s_bkg_std * np.sqrt(_s_n_main))
                if _s_bkg_std > 0 and _s_n_main > 0
                else float("nan")
            )

        def _nearby(cat_tuple):
            # Offsets are relative to the *rounded* centroid (cx, cy), since
            # that is the pixel the stamp is cut around and the one that lands
            # at the stamp center. Using the float centroid would leave symbols
            # up to half a pixel off the features they mark.
            if cat_tuple is None:
                return []
            cat_x, cat_y, cat_mag = cat_tuple
            return [
                (float(sx) - cx, float(sy) - cy, float(sm))
                for sx, sy, sm in zip(cat_x, cat_y, cat_mag)
                if abs(float(sx) - cx) <= half and abs(float(sy) - cy) <= half
            ]

        _fa = detector.transform([lsst.geom.Point2D(float(cx_f), float(cy_f))], PIXELS, FIELD_ANGLE)[0]
        _field_dist_deg = np.degrees(np.hypot(_fa[0], _fa[1]))

        _nearby_photo_list = _nearby(all_photo_cat)
        _neighbor_dists = [np.hypot(dx, dy) for dx, dy, _ in _nearby_photo_list if np.hypot(dx, dy) >= 1.0]

        _catalog_centroid_offset_px = (
            float(np.hypot(cx_f - blind_cx, cy_f - blind_cy))
            if blind_cx is not None and blind_cy is not None
            else float("nan")
        )

        return dict(
            sensor=sensor_name,
            stamp=stamp_ccs,
            fa_x_ccs=float(_fa[1]),
            fa_y_ccs=float(_fa[0]),
            flux=float(flux_val),
            band=band,
            det_id=det_id,
            visit_id=visit_id,
            centroid_x_raw=float(cx_f),
            centroid_y_raw=float(cy_f),
            source_id=int(source_id_val),
            inner_frac=inner_frac,
            outer_frac=outer_frac,
            outer_sector_max=outer_sector_max,
            field_dist_deg=_field_dist_deg,
            donut_radius=radius,
            obscuration=obscuration,
            snr=stamp_snr,
            bkg_level=_s_bkg_med,
            bkg_std=_s_bkg_std,
            nearest_neighbor_dist_px=(float(min(_neighbor_dists)) if _neighbor_dists else float("nan")),
            n_neighbors_in_stamp=len(_neighbor_dists),
            catalog_centroid_offset_px=_catalog_centroid_offset_px,
            n_quarter=n_quarter,
            nearby_photo=_nearby_photo_list,
            nearby_astrom=_nearby(all_astrom_cat),
            reject_reasons=[reject_reason] if reject_reason else [],
            saturated=saturated,
        )

    inner_thr = detect_cfg["innerFracThreshold"]
    outer_thr = detect_cfg["outerFracThreshold"]
    max_field_dist = detect_cfg["maxFieldDist"]
    min_stamp_snr = detect_cfg["minStampSnr"]

    def _append_reject_reasons(d):
        if d["saturated"]:
            d["reject_reasons"].append("SAT")
        if d["field_dist_deg"] > max_field_dist:
            d["reject_reasons"].append("field_dist")
        if np.isfinite(d["inner_frac"]) and abs(d["inner_frac"]) > inner_thr:
            d["reject_reasons"].append("inner_frac")
        if np.isfinite(d["outer_frac"]) and abs(d["outer_frac"]) > outer_thr:
            d["reject_reasons"].append("outer_frac")
        if np.isfinite(d["snr"]) and d["snr"] < min_stamp_snr:
            d["reject_reasons"].append("snr")

    # Match each centroid to the nearest blind detection for catalog_centroid_offset_px.
    _blind_cx = np.array(blindDetections["centroid_x"]) if len(blindDetections) > 0 else np.empty(0)
    _blind_cy = np.array(blindDetections["centroid_y"]) if len(blindDetections) > 0 else np.empty(0)
    _match_tol = radius * 0.5

    def _nearest_blind(cx_f, cy_f):
        if len(_blind_cx) == 0:
            return None, None
        dists = np.hypot(_blind_cx - cx_f, _blind_cy - cy_f)
        idx = int(np.argmin(dists))
        return (float(_blind_cx[idx]), float(_blind_cy[idx])) if dists[idx] <= _match_tol else (None, None)

    # --- Single pass over centroids: accept or quality-reject ---
    donuts = []
    rejected_donuts_pre = []
    for i, (cx_f, cy_f) in enumerate(zip(centroid_x, centroid_y)):
        _b_cx, _b_cy = _nearest_blind(cx_f, cy_f)
        d = _cut_stamp_dict(
            cx_f,
            cy_f,
            flux_arr[i],
            source_ids[i],
            inner_frac=float(inner_frac_arr[i]),
            outer_frac=float(outer_frac_arr[i]),
            outer_sector_max=float(outer_sector_max_arr[i]),
            blind_cx=_b_cx,
            blind_cy=_b_cy,
        )
        if d is None:
            continue
        _append_reject_reasons(d)
        if d["reject_reasons"]:
            rejected_donuts_pre.append(d)
            continue
        donuts.append(d)

    # --- Rejected-stamp collection (selector-rejected sources added in) ---
    rejected_donuts = list(rejected_donuts_pre)
    if sel_rejected_centroids is not None:
        rrej_x, rrej_y, rrej_flux, rrej_ids, rrej_reasons = sel_rejected_centroids
        for cx_f, cy_f, flux_val, sid, sel_reason in zip(rrej_x, rrej_y, rrej_flux, rrej_ids, rrej_reasons):
            d = _cut_stamp_dict(cx_f, cy_f, flux_val, sid, reject_reason=sel_reason or "selector")
            if d is not None:
                _append_reject_reasons(d)
                rejected_donuts.append(d)
    rejected_donuts.sort(key=lambda d: d["flux"], reverse=True)

    return donuts, rejected_donuts


def _getCutouts(sensor_name: str, t_dispatch: float) -> dict:
    """Run ISR, blind detection, WCS refit, catalog selection, and stamp cutting.

    Orchestrates the full per-sensor cutout pipeline in a worker process.
    All inputs are read from the module-level ``_CALIB_STORE`` dict, which is
    populated by the parent process before forking.

    Parameters
    ----------
    sensor_name : str
        Detector name; used to look up per-sensor calibrations in
        ``_CALIB_STORE``.
    t_dispatch : float
        ``time.time()`` timestamp at which the task was dispatched from the
        parent, used to measure dispatch-to-arrival latency.

    Returns
    -------
    dict
        Keys: ``sensor``, ``catalog`` (accepted donut dicts),
        ``rejected_catalog``, ``scatter_arcsec``, ``wcs_refit_error``,
        ``cat_select_error``, and timing floats ``dispatch_to_arrival``,
        ``task_init``, ``isr_run``, ``blind_detect_run``, ``wcs_refit_run``,
        ``catalog_select_run``, ``stamp_cut_run``.
    """
    t_arrival = time.time()
    entry = _CALIB_STORE[sensor_name]

    t0 = time.perf_counter()
    isr_task = IsrTaskLSST(config=_CALIB_STORE["isr_config"])
    t1 = time.perf_counter()
    postIsr = isr_task.run(
        entry["raw"],
        ptc=entry["ptc"],
        flat=entry["flat"],
        linearizer=entry["linearizer"],
        crosstalk=entry["crosstalk"],
    ).exposure
    t2 = time.perf_counter()

    camera = _CALIB_STORE["camera"]
    detect_cfg = _CALIB_STORE["detect_cfg"]
    bkg_config = _CALIB_STORE["bkg_config"]

    camName = camera.getName()
    detectorName = postIsr.getDetector().getName()
    instrument = getTaskInstrument(camName, detectorName, detect_cfg["instConfigFile"])
    radius = instrument.donutRadius

    astrom_cfg = _CALIB_STORE["astrom_cfg"]
    obscuration = instrument.obscuration

    blindDetections = _blindDetect(
        postIsr,
        detect_cfg,
        bkg_config,
        radius,
        obscuration,
    )
    t3 = time.perf_counter()

    refitted_wcs, scatter_arcsec, wcs_err = _refitWcs(blindDetections, postIsr, astrom_cfg)
    t4 = time.perf_counter()

    (
        catalog_centroids,
        sel_rejected_centroids,
        all_photo_cat,
        all_astrom_cat,
        cat_err,
    ) = _selectFromPhotoCat(
        refitted_wcs,
        postIsr,
        sensor_name,
        astrom_cfg,
    )
    t5 = time.perf_counter()

    donuts, rejected_donuts = _cutStamps(
        postIsr,
        sensor_name,
        blindDetections,
        catalog_centroids,
        sel_rejected_centroids,
        all_photo_cat,
        all_astrom_cat,
        astrom_cfg,
        radius,
        obscuration,
    )
    t6 = time.perf_counter()

    return {
        "sensor": sensor_name,
        "catalog": donuts,
        "dispatch_to_arrival": t_arrival - t_dispatch,
        "task_init": t1 - t0,
        "isr_run": t2 - t1,
        "blind_detect_run": t3 - t2,
        "wcs_refit_run": t4 - t3,
        "catalog_select_run": t5 - t4,
        "stamp_cut_run": t6 - t5,
        "rejected_catalog": rejected_donuts,
        "scatter_arcsec": scatter_arcsec,
        "wcs_refit_error": wcs_err,
        "cat_select_error": cat_err,
    }


def _run_cutout_worker(args: tuple) -> dict:
    sensor_name, t_dispatch = args
    return _getCutouts(sensor_name, t_dispatch)


class _WfFitTimeoutError(Exception):
    pass


@contextlib.contextmanager
def _fit_timeout(seconds):
    """SIGALRM-based timeout context manager (Unix only)."""

    def _handler(_signum, _frame):
        raise _WfFitTimeoutError(f"WF fit exceeded {seconds:.0f}s timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


_ZK_LEN = 79  # Noll-indexed array length; index j holds Zernike j; indices 0-3 always 0.


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
            bkg_val = float(np.ravel(bkgs[donut_idx])[0])
        else:
            bkg_val = float(bkgs[0])
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


def _fit_bkg_val(fit_params, donut_idx: int) -> float:
    """Extract the fitted background constant for one donut (ADU/pixel)."""
    if fit_params is None:
        return float("nan")
    bkgs = fit_params.get("bkgs") or fit_params.get("bkg")
    if bkgs is None or len(bkgs) == 0 or donut_idx >= len(bkgs):
        return float("nan")
    if np.ndim(bkgs[0]) > 0:
        raveled = np.ravel(bkgs[donut_idx])
        return float(raveled[0]) if len(raveled) > 0 else float("nan")
    return float(bkgs[0])


def _blend_frac(
    img: np.ndarray,
    model_img_bkg_free: np.ndarray,
    bkg_std: float,
    faint_frac: float = 0.05,
    sig_thresh: float = 2.0,
) -> float:
    """Fraction of model flux found as significant residual in model-faint pixels.

    A blending out-of-focus source produces a donut-shaped imprint in the residual
    at locations where the forward model predicts near-zero flux. This metric
    integrates that signal and normalises by the total model flux, so it is
    comparable across SNR levels.

    model_img_bkg_free should be the forward-model image with the fitted
    background removed, so that faint_mask correctly identifies pixels where
    the donut optics model predicts near-zero signal.
    """
    if img is None or model_img_bkg_free is None or not np.isfinite(bkg_std) or bkg_std <= 0:
        return float("nan")
    model_peak = float(np.nanmax(model_img_bkg_free))
    total_model_flux = float(np.sum(model_img_bkg_free[model_img_bkg_free > 0]))
    if model_peak <= 0 or total_model_flux <= 0:
        return float("nan")
    resid = img - model_img_bkg_free
    faint_mask = model_img_bkg_free < faint_frac * model_peak
    sig_mask = np.abs(resid) > sig_thresh * bkg_std
    return float(np.sum(np.abs(resid[faint_mask & sig_mask]))) / total_model_flux


def _residual_rms(
    img: np.ndarray,
    model_img: np.ndarray,
    radius: float,
    obscuration: float,
    apertureOuterMarginFrac: float,
) -> float:
    """RMS of (data - model) over the main donut annulus."""
    if img is None or model_img is None:
        return float("nan")
    half = img.shape[0] // 2
    gy, gx = (
        np.mgrid[-half : half + 1, -half : half + 1]
        if img.shape[0] % 2 == 1
        else np.mgrid[-half:half, -half:half]
    )
    r = np.hypot(gx, gy)
    main_mask = (r < radius * apertureOuterMarginFrac) & (r > radius * obscuration)
    if main_mask.shape != img.shape:
        sz = min(main_mask.shape[0], img.shape[0])
        main_mask = main_mask[:sz, :sz]
        img = img[:sz, :sz]
        model_img = model_img[:sz, :sz]
    resid = img - model_img
    n = int(np.sum(main_mask))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean(resid[main_mask] ** 2)))


def _dense_intrinsic(donut: dict) -> np.ndarray:
    """Return length-79 intrinsic Zernike array in metres (0.0 where unavailable)."""
    out = np.zeros(_ZK_LEN)
    raw = donut.get("intrinsic_zk")  # µm, Noll 4-78
    if raw is not None:
        for idx in range(min(len(raw), _ZK_LEN - 4)):
            out[idx + 4] = float(raw[idx]) * 1e-6
    return out


def _dense_dev(zk_dev: np.ndarray, nollIndices) -> np.ndarray:
    """Return length-79 deviation array in metres (np.nan where not fitted)."""
    out = np.full(_ZK_LEN, np.nan)
    out[0:4] = 0.0
    for k, j in enumerate(nollIndices):
        if k < len(zk_dev):
            out[int(j)] = float(zk_dev[k])
    return out


def _zk_cols(suffix: str, zk: np.ndarray) -> dict:
    """Return ``{f"Z{j}_{suffix}": value_um}`` catalog columns for Noll 4-78.

    ``zk`` is a length-``_ZK_LEN`` Noll-indexed array in metres; values are
    converted to µm, with NaN passed through as NaN.
    """
    return {
        f"Z{j}_{suffix}": (float(zk[j]) * 1e6 if not np.isnan(zk[j]) else float("nan"))
        for j in range(4, _ZK_LEN)
    }


def _build_loss_fn():
    """Return a danish loss function from wf_cfg, or None for standard chi-squared."""
    alpha = _CALIB_STORE["wf_cfg"].get("systematicLossAlpha", 0.0)
    return danish.systematic_loss(alpha) if alpha != 0.0 else None


@dataclasses.dataclass
class _WfGroup:
    donuts: list  # donut dicts, ordered; each carries its own "sensor" key
    group_id: str
    mode: str  # "paired" | "unpaired" | "full_detector" | "full_corner"


def _mode_groups_are_pairs(mode) -> bool:
    """Does one group of this mode hold an intra/extra pair of its own?

    True only for ``"paired"``, whose groups are built as ``[extra, intra]``.
    Every other mode groups donuts without regard to defocal type: one donut
    per group for ``"unpaired"``, all of a sensor's or corner's donuts for
    ``"full_detector"``/``"full_corner"``.

    This is the single place that answer lives.  ``_build_wf_groups`` creates
    that structure and the WF diagnostic plot consumes it, so both ask here
    rather than each testing the mode string themselves.
    """
    return mode == "paired"


def _build_wf_groups(mode, results_by_sensor, wf_cfg):
    """Build _WfGroup list from per-sensor catalogs, matching the mode dispatch logic.

    Groups for ``"paired"`` hold one extra/intra pair; all other modes group
    without regard to defocal type -- the invariant reported by
    `_mode_groups_are_pairs`.

    Returns (groups, unmatched_donuts).
    """
    groups = []
    unmatched_donuts = []
    if _mode_groups_are_pairs(mode):
        for _corner, (sw0, sw1) in CORNER_PAIRS.items():
            extra_donuts = sorted(results_by_sensor.get(sw0, []), key=lambda d: d["snr"], reverse=True)
            intra_donuts = sorted(results_by_sensor.get(sw1, []), key=lambda d: d["snr"], reverse=True)
            for extra, intra in zip(extra_donuts, intra_donuts):
                gid = f"{extra['source_id']}_{intra['source_id']}"
                groups.append(_WfGroup(donuts=[extra, intra], group_id=gid, mode="paired"))
            n_pairs = min(len(extra_donuts), len(intra_donuts))
            unmatched_donuts.extend(extra_donuts[n_pairs:])
            unmatched_donuts.extend(intra_donuts[n_pairs:])
    elif mode == "unpaired":
        for sensor_donuts in results_by_sensor.values():
            for d in sensor_donuts:
                groups.append(_WfGroup(donuts=[d], group_id=str(d["source_id"]), mode="unpaired"))
    elif mode == "full_detector":
        for sensor_name, sensor_donuts in results_by_sensor.items():
            groups.append(_WfGroup(donuts=sensor_donuts, group_id=sensor_name, mode="full_detector"))
    else:  # full_corner
        for corner, (sw0, sw1) in CORNER_PAIRS.items():
            all_donuts = results_by_sensor.get(sw0, []) + results_by_sensor.get(sw1, [])
            groups.append(_WfGroup(donuts=all_donuts, group_id=corner, mode="full_corner"))
    return groups, unmatched_donuts


def _build_wf_factory(instrument):
    """Build a Danish donut factory from _CALIB_STORE["wf_cfg"] and instrument."""
    wf_cfg = _CALIB_STORE["wf_cfg"]
    factory_class = danish.DonutTriangleFactory if wf_cfg["triangleMode"] else danish.DonutFactory
    factory_kwargs = {}
    if wf_cfg["doAoiThroughput"]:
        factory_kwargs["bandpass_filter"] = wf_cfg["band"]
        alt_rad = wf_cfg["boresight_alt_rad"]
        if np.isfinite(alt_rad) and alt_rad > 0:
            factory_kwargs["airmass"] = float(np.clip(round(1.0 / np.sin(alt_rad), 1), 1.0, 2.5))
        else:
            factory_kwargs["airmass"] = 1.2
    return factory_class(
        R_outer=instrument.radius,
        R_inner=instrument.radius * instrument.obscuration,
        mask_params=instrument.maskParams,
        focal_length=instrument.focalLength,
        pixel_scale=instrument.pixelSize * wf_cfg["binning"],
        spider_angle=wf_cfg["rtp_deg"],
        **factory_kwargs,
    )


def _prep_donut_for_danish(donut: dict, instrument) -> tuple:
    """Prepare a donut dict for Danish fitting.

    Bins and crops the stamp to an odd pixel size, estimates background noise,
    computes the reference Zernike array ``zk_ref`` from ``batoid.zernikeTA``
    (with optional measured-intrinsics correction), and extracts the field
    angle.

    Parameters
    ----------
    donut : dict
        Donut record with keys: ``stamp`` (2-D array), ``det_id``, ``band``,
        ``fa_x_ccs``, ``fa_y_ccs`` (field angles in radians), and optionally
        ``intrinsic_zk`` (µm, Noll 4–78).
    instrument : object
        Instrument config object providing ``focalLength``, ``defocalOffset``,
        ``donutRadius``, and ``obscuration``.

    Returns
    -------
    img : np.ndarray
        ``(npix, npix)`` float stamp, binned and cropped to odd size.
    angle_rad : np.ndarray
        ``[fa_x_ccs, fa_y_ccs]`` field angle in radians.
    zk_ref : np.ndarray
        Reference Zernike array in metres, shape ``(79,)``, Noll-indexed.
        Equals ``W_TA_defoc`` at uncalibrated indices and
        ``W_TA_defoc + (W_meas - zk_opd_foc)`` at calibrated indices.
    bkg_var : float
        Background variance estimate (``bkg_std ** 2``) from pixel-difference
        MAD of the stamp.
    bkg_std : float
        Background standard deviation estimate.
    """
    wf_cfg = _CALIB_STORE["wf_cfg"]
    binning = wf_cfg["binning"]
    det_id = donut["det_id"]
    defocalSign = +1 if det_id in _EXTRA_FOCAL_DET_IDS else -1

    img = donut["stamp"].astype(float)
    if binning > 1:
        img = binArray(img, binning)
    if img.shape[0] % 2 == 0:
        img = img[:-1, :-1]
    diff = (img[1:] - img[:-1]).ravel()
    bkg_std = float(median_abs_deviation(diff, scale="normal") / np.sqrt(2.0))

    band = donut["band"]
    wavelength_by_band = wf_cfg["wavelength_by_band"]
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
        jmax=78,
        eps=eps,
        focal_length=instrument.focalLength,
        nrad=nrad,
        naz=int(2 * np.pi * nrad / (1 - eps)),
    )
    # W_TA_defoc: off-axis + nominal intrinsics + defocus in one call
    zk_ref = (
        batoid.zernikeTA(
            telescope_dz,
            donut["fa_x_ccs"],
            donut["fa_y_ccs"],
            wavelength,
            **zernikeTA_kwargs,
        )
        * wavelength
    )  # meters, shape (79,)

    # Replace nominal on-axis model (zk_opd_foc) with measured intrinsics (W_meas)
    # for calibrated indices.
    intrinsic_zk = donut.get("intrinsic_zk")
    if intrinsic_zk is not None:
        zk_opd_foc = (
            batoid.zernikeTA(
                telescope,
                donut["fa_x_ccs"],
                donut["fa_y_ccs"],
                wavelength,
                **zernikeTA_kwargs,
            )
            * wavelength
        )  # meters
        calib_noll = wf_cfg["calib_noll_indices"]
        for i, j in enumerate(calib_noll):
            if i < len(intrinsic_zk) and int(j) < 79:
                zk_ref[int(j)] += float(intrinsic_zk[i]) * 1e-6 - zk_opd_foc[int(j)]

    angle_rad = np.array([donut["fa_x_ccs"], donut["fa_y_ccs"]])
    return img, angle_rad, zk_ref, bkg_std**2, bkg_std


_log = logging.getLogger(__name__)

_DZ_MODEL_KEYS = ("fluxes", "dxs", "dys", "fwhm", "wavefront_params", "bkgs")


def _run_lstsq_fit(model, x0, bounds, imgs, variances, timeout, wf_cfg, label):
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
    wf_cfg : dict
        WF config dict supplying ``wfInitialGuessOnly``, ``lstsqKwargs``,
        ``nollIndices``.
    label : str
        Short description used in log messages (e.g. ``"paired extra=3 intra=7"``).

    Returns
    -------
    zk_dev : np.ndarray
        Fitted deviation Zernikes aligned to ``nollIndices`` (NaN on failure).
    params : dict or None
        Unpacked model parameters, or ``None`` on failure.
    model_imgs : list or None
        Model images from ``model.model(...)``, or ``None`` on failure.
    success : bool
        ``True`` if the fit converged without error or timeout.
    fit_info : dict
        Timing, convergence, and parameter summary.
    """
    nollIndices = wf_cfg["nollIndices"]
    t0 = time.perf_counter()
    params = None
    if wf_cfg["wfInitialGuessOnly"]:
        try:
            params = model.unpack_params(x0)
            zk_dev = np.zeros(len(nollIndices))
            model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
            elapsed = time.perf_counter() - t0
            success = True
            fit_info = dict(
                elapsed=elapsed,
                nfev=0,
                cost=float("nan"),
                optimality=float("nan"),
                njev=0,
                status=0,
                message="x0 only",
                fluxes=list(params["fluxes"]),
                dxs=list(params["dxs"]),
                dys=list(params["dys"]),
                fwhm=float(params["fwhm"]),
            )
            _log.info("WF %s (x0 only)", label)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            zk_dev = np.full(len(nollIndices), np.nan)
            model_imgs = None
            success = False
            fit_info = dict(elapsed=elapsed, error=str(exc))
            _log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
    else:
        galsim.errors.raise_fft_size_error = True
        try:
            with _fit_timeout(timeout):
                result = least_squares(
                    model.chi,
                    jac=model.jac,
                    x0=x0,
                    args=(imgs, variances),
                    bounds=bounds,
                    **wf_cfg["lstsqKwargs"],
                )
            elapsed = time.perf_counter() - t0
            params = model.unpack_params(result.x)
            zk_dev = np.array(params["wavefront_params"])
            model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
            success = bool(result.success)
            fit_info = dict(
                elapsed=elapsed,
                nfev=int(result.nfev),
                cost=float(result.cost),
                optimality=float(result.optimality),
                njev=int(result.njev),
                status=int(result.status),
                message=str(result.message),
                fluxes=list(params["fluxes"]),
                dxs=list(params["dxs"]),
                dys=list(params["dys"]),
                fwhm=float(params["fwhm"]),
            )
            _log.info(
                "WF %s success=%s nfev=%d elapsed=%.1fs",
                label,
                success,
                result.nfev,
                elapsed,
            )
        except _WfFitTimeoutError:
            elapsed = time.perf_counter() - t0
            zk_dev = np.full(len(nollIndices), np.nan)
            model_imgs = None
            success = False
            fit_info = dict(elapsed=elapsed, error=f"timeout after {timeout:.0f}s")
            _log.warning("WF %s TIMED OUT after %.1fs", label, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            zk_dev = np.full(len(nollIndices), np.nan)
            model_imgs = None
            success = False
            fit_info = dict(elapsed=elapsed, error=str(exc))
            _log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
    return zk_dev, params, model_imgs, success, fit_info


def _wf_worker(group: _WfGroup) -> dict:
    """Unified wavefront fitting worker for all modes.

    Fits all donuts in ``group`` jointly with a single ``DZMultiDonutModel``,
    sharing one wavefront solution across the group.  Replaces the four
    mode-specific workers (_wf_paired_worker etc.).
    """
    wf_cfg = _CALIB_STORE["wf_cfg"]
    nollIndices = wf_cfg["nollIndices"]
    instrument = _CALIB_STORE["instrument"]
    all_donuts = group.donuts
    n = len(all_donuts)

    if not all_donuts:
        return {
            "mode": group.mode,
            "group_id": group.group_id,
            "group_size": 0,
            "zk_dev": np.full(len(nollIndices), np.nan),
            "success": False,
            "fit_info": {},
            "donuts": [],
            "model_imgs": None,
            "imgs": [],
            "sensors": [],
        }

    t_setup0 = time.perf_counter()
    factory = _build_wf_factory(instrument)
    preps = [_prep_donut_for_danish(d, instrument) for d in all_donuts]
    imgs = [p[0] for p in preps]
    thxs = [p[1][0] for p in preps]
    thys = [p[1][1] for p in preps]
    zk_refs = [p[2] for p in preps]
    sky_lvl = [p[3] for p in preps]
    bkg_stds = [p[4] for p in preps]
    dz_terms = [(1, int(j)) for j in nollIndices]

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
        bkg_order=wf_cfg["bkgOrder"],
        loss_fn=_build_loss_fn(),
    )
    fluxes_init = [float(np.clip(np.sum(img), 1e3, 1e9)) for img in imgs]
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
    timeout = wf_cfg["wfFitTimeoutPerDonut"] * n
    _setup_elapsed = float(time.perf_counter() - t_setup0)
    label = f"{group.mode} group={group.group_id} n={n}"
    _log.info("WF %s npix=%d setup=%.2fs", label, npix, _setup_elapsed)

    zk_dev, params, model_imgs, success, fit_info = _run_lstsq_fit(
        model, x0, bounds, imgs, sky_lvl, timeout, wf_cfg, label
    )
    zk_dev_dense = _dense_dev(zk_dev, nollIndices)
    zk_norm_um = float(np.sqrt(np.nansum(zk_dev_dense[4:] ** 2)) * 1e6)
    fit_mode = wf_cfg.get("wfEstimationMode", group.mode)
    _fit_elapsed = float(fit_info.get("elapsed", float("nan")))
    _fit_nfev = int(fit_info.get("nfev", 0))
    _fit_cost = float(fit_info.get("cost", float("nan")))
    _fit_fwhm = float(fit_info.get("fwhm", float("nan")))
    _dxs = fit_info.get("dxs", [float("nan")] * n)
    _dys = fit_info.get("dys", [float("nan")] * n)
    _fluxes = fit_info.get("fluxes", [float("nan")] * n)

    donuts_out = []
    for i, d in enumerate(all_donuts):
        defocal = "intra" if int(d["det_id"]) in _INTRA_FOCAL_DET_IDS else "extra"
        _img = imgs[i] if i < len(imgs) else None
        _mimg = model_imgs[i] if (model_imgs is not None and i < len(model_imgs)) else None
        donuts_out.append(
            {
                "donut_id": int(d["source_id"]),
                "sensor": d["sensor"],
                "defocal": defocal,
                "zk_dev": zk_dev_dense,
                "zk_intrinsic": _dense_intrinsic(d),
                "img": _img,
                "model_img": _mimg,
                "fit_success": success,
                "fit_elapsed": _fit_elapsed,
                "setup_elapsed": _setup_elapsed,
                "fit_nfev": _fit_nfev,
                "fit_cost": _fit_cost,
                "fit_dx": float(_dxs[i]) if i < len(_dxs) else float("nan"),
                "fit_dy": float(_dys[i]) if i < len(_dys) else float("nan"),
                "fit_flux": float(_fluxes[i]) if i < len(_fluxes) else float("nan"),
                "fit_fwhm": _fit_fwhm,
                "fit_residual_rms": _residual_rms(
                    _img,
                    _mimg,
                    instrument.donutRadius,
                    instrument.obscuration,
                    _CALIB_STORE["detect_cfg"]["apertureOuterMarginFrac"],
                ),
                "blend_frac": _blend_frac(
                    _img,
                    (
                        _bkg_free_model(_mimg, model, params, i, wf_cfg["bkgOrder"])
                        if _mimg is not None
                        else None
                    ),
                    bkg_stds[i] if i < len(bkg_stds) else float("nan"),
                ),
                "fit_bkg": _fit_bkg_val(params, i),
                "zk_norm_um": zk_norm_um,
                "group_id": group.group_id,
                "group_size": n,
                "fit_mode": fit_mode,
            }
        )
    return {
        "mode": group.mode,
        "fit_mode": fit_mode,
        "group_id": group.group_id,
        "group_size": n,
        "zk_dev": zk_dev,
        "success": success,
        "fit_info": fit_info,
        "donuts": donuts_out,
        "model_imgs": model_imgs,
        "imgs": imgs,
        "sensors": [d["sensor"] for d in all_donuts],
    }



class DonutBlitzMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzMonolithTask."""

    raws = connectionTypes.Input(
        doc="Raw corner wavefront sensor exposures (all 8 sensors).",
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera geometry.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    ptc = connectionTypes.PrerequisiteInput(
        name="ptc",
        storageClass="PhotonTransferCurveDataset",
        doc="Photon transfer curve calibration, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        lookupFunction=lookupStaticCalibrations,
    )
    flat = connectionTypes.PrerequisiteInput(
        name="flat",
        storageClass="ExposureF",
        doc="Flat field calibration, one per detector.",
        dimensions=["instrument", "detector", "physical_filter"],
        isCalibration=True,
        multiple=True,
        lookupFunction=lookupStaticCalibrations,
    )
    linearizer = connectionTypes.PrerequisiteInput(
        name="linearizer",
        storageClass="Linearizer",
        doc="Linearity correction, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        lookupFunction=lookupStaticCalibrations,
    )
    crosstalk = connectionTypes.PrerequisiteInput(
        name="crosstalk",
        storageClass="CrosstalkCalib",
        doc="Crosstalk coefficients, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        lookupFunction=lookupStaticCalibrations,
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
        lookupFunction=lookupStaticCalibrations,  # type: ignore
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
    astromTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=AstrometryTask,
        doc="Astrometry subtask for WCS fitting.",
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask,
        doc="Donut source selector subtask.",
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
    innerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed |inner_flux / flux| for a candidate to be kept.",
        dtype=float,
        default=0.1,
    )
    outerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed |outer_flux / flux| for a candidate to be kept.",
        dtype=float,
        default=0.1,
    )
    snrThreshold: pexConfig.Field = pexConfig.Field(
        doc="Minimum signal-to-noise ratio for a candidate to be kept.",
        dtype=float,
        default=100.0,
    )
    maxFieldDist: pexConfig.Field = pexConfig.Field(
        doc="Maximum distance from the center of the focal plane in degrees.",
        dtype=float,
        default=1.725,
    )
    minStampSnr: pexConfig.Field = pexConfig.Field(
        doc="Minimum per-stamp SNR for a donut to be accepted for WF estimation.",
        dtype=float,
        default=500.0,
    )
    stampSize: pexConfig.Field = pexConfig.Field(
        doc=(
            "Side length in pixels of the square stamp cut around each donut "
            "centroid. Stamps are cut odd (2*(stampSize//2)+1) so the centroid "
            "pixel is exactly centered.\n\n"
            "Danish requires an odd stamp after binning, and `binArray` "
            "truncates before binning, so the fitted size is stampSize//binning. "
            "That is odd for binning 1, 2 and 3 when stampSize = 3 or 11 (mod "
            "12), and additionally for binning 4 when stampSize = 15 or 23 (mod "
            "24). The default 167 (= 23 mod 24) satisfies all four, giving "
            "167/83/55/41 px. Sizes failing this still work -- "
            "_prep_donut_for_danish crops by one pixel -- but that shifts the "
            "donut off center by half a pixel.\n\n"
            "Must also be large enough to contain the main photometric annulus: "
            "stampSize/2 >= donutRadius * apertureOuterMarginFrac."
        ),
        dtype=int,
        default=167,
    )
    maxDonuts: pexConfig.Field = pexConfig.Field(
        doc="Maximum number of donuts to return per sensor, sorted by flux descending.",
        dtype=int,
        default=8,
    )
    maxFitScatter: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed on-sky scatter (arcsec) for WCS refit to be accepted.",
        dtype=float,
        default=1.0,
    )
    minSourcesForWcsFit: pexConfig.Field = pexConfig.Field(
        doc="Minimum number of blind detections required to attempt WCS refit.",
        dtype=int,
        default=3,
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
            "full_detector": "Dispatch all donuts on each sensor as one work unit (8 fits per visit).",
        },
        default="paired",
    )
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
            # "tr_solver": "'lsmr'",
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


class DonutBlitzMonolithTask(pipeBase.PipelineTask):
    """Monolithic WEP task for corner wavefront sensors.

    Runs ISR, blind donut detection, WCS refit, catalog-based donut
    selection, and stamp cutting on all 8 corner sensor raws in parallel
    using a multiprocessing pool.  Reference catalogs are loaded in the parent
    process before forking and inherited by workers via copy-on-write.
    """

    ConfigClass = DonutBlitzMonolithTaskConfig
    _DefaultName = "donutBlitzMonolithTask"
    config: DonutBlitzMonolithTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("plotTask")
        self.makeSubtask("isrTask")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("astromTask")
        self.makeSubtask("donutSelector")
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
                _ANSI_YELLOW,
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
        camera = butlerQC.get(inputRefs.camera)
        t2 = time.perf_counter()
        ptc = butlerQC.get(inputRefs.ptc)
        t3 = time.perf_counter()
        flat = butlerQC.get(inputRefs.flat)
        t4 = time.perf_counter()
        linearizer = butlerQC.get(inputRefs.linearizer)
        t5 = time.perf_counter()
        crosstalk = butlerQC.get(inputRefs.crosstalk)
        t6 = time.perf_counter()
        refCat = butlerQC.get(inputRefs.refCat)
        t7 = time.perf_counter()
        intrinsicZernikes = butlerQC.get(inputRefs.intrinsicZernikes)
        t8 = time.perf_counter()
        self.log.info(
            _colorize(
                "butlerQC.get timing: raws=%.3fs camera=%.3fs ptc=%.3fs flat=%.3fs"
                " linearizer=%.3fs crosstalk=%.3fs refCat=%.3fs"
                " intrinsicZernikes=%.3fs total=%.3fs",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
            t6 - t5,
            t7 - t6,
            t8 - t7,
            t8 - t0,
        )
        butler_elapsed = t8 - t0
        butler_times = dict(
            raws=t1 - t0,
            camera=t2 - t1,
            ptc=t3 - t2,
            flat=t4 - t3,
            linearizer=t5 - t4,
            crosstalk=t6 - t5,
            refCat=t7 - t6,
            intrinsicZernikes=t8 - t7,
        )
        outputs = self.run(
            raws=raws,
            camera=camera,
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
        t9 = time.perf_counter()
        self.log.info("run() execution: %.3fs", t9 - t8)
        butlerQC.put(outputs.blitzResults, outputRefs.blitzResults)

    @timeMethod
    def run(
        self,
        raws: list,
        camera: Camera,
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
        camera : lsst.afw.cameraGeom.Camera
        ptc : list of lsst.ip.isr.PhotonTransferCurveDataset
        flat : list of lsst.afw.image.ExposureF
        linearizer : list of lsst.ip.isr.Linearizer
        crosstalk : list of lsst.ip.isr.CrosstalkCalib
        refCat : list of DeferredDatasetHandle or SimpleCatalog
            Shards used for both WCS fitting and donut selection, loaded once
            per sensor.  The WCS fit reads ``astromRefFilter`` (resolved as the
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
                _ANSI_YELLOW,
                enabled=self._colorLogEnabled,
            ),
            numCores,
            butler_elapsed,
        )
        t_run0 = time.perf_counter()

        rawByName = {exp.getDetector().getName(): exp for exp in raws}
        ptcByName = {p._detectorName: p for p in ptc}
        flatByName = {f.getDetector().getName(): f for f in flat}
        linearizerByName = {lin._detectorName: lin for lin in linearizer}
        crosstalkByName = {ct._detectorName: ct for ct in crosstalk}

        missing = CORNER_SENSOR_NAMES - rawByName.keys()
        if missing:
            raise RuntimeError(f"Missing corner sensor raws: {sorted(missing)}")

        if intrinsicZernikes:
            self.log.info("Loaded %d intrinsic Zernike calibration(s).", len(intrinsicZernikes))
        else:
            self.log.warning("No intrinsic Zernike calibrations provided.")
        self.intrinsicZernikes = list(intrinsicZernikes) if intrinsicZernikes else []
        intrinsicZernikesByName = {
            camera[iz.getMetadata()["LSST BUTLER DATAID DETECTOR"]].getName(): iz
            for iz in self.intrinsicZernikes
        }

        detect_cfg = dict(
            instConfigFile=self.config.instConfigFile,
            edgeMargin=self.config.edgeMargin,
            detectionBinning=self.config.detectionBinning,
            peakMinDistanceFactor=self.config.peakMinDistanceFactor,
            peakExcludeBorderFactor=self.config.peakExcludeBorderFactor,
            apertureOuterMarginFrac=self.config.apertureOuterMarginFrac,
            apertureInnerBufferFrac=self.config.apertureInnerBufferFrac,
            bkgAnnulusInnerFrac=self.config.bkgAnnulusInnerFrac,
            bkgAnnulusOuterFrac=self.config.bkgAnnulusOuterFrac,
            innerFracThreshold=self.config.innerFracThreshold,
            outerFracThreshold=self.config.outerFracThreshold,
            snrThreshold=self.config.snrThreshold,
            minStampSnr=self.config.minStampSnr,
            maxFieldDist=self.config.maxFieldDist,
            stampSize=self.config.stampSize,
            maxDonuts=self.config.maxDonuts,
        )

        example_band = next(iter(rawByName.values())).filter.bandLabel
        photo_filter_name = example_band
        if self.config.photoRefFilter is not None:
            photo_filter_name = self.config.photoRefFilter
        elif self.config.photoRefFilterPrefix is not None:
            photo_filter_name = f"{self.config.photoRefFilterPrefix}_{example_band}"

        refcat_handles = list(refCat)

        # Thin proxy to count how many shards loadPixelBox actually fetches.
        class _CountingHandle:
            def __init__(self, h, cnt):
                self._h = h
                self._cnt = cnt

            def __getattr__(self, name):
                return getattr(self._h, name)

            def get(self, *args, **kwargs):
                self._cnt.append(1)
                return self._h.get(*args, **kwargs)

        refcat_fetched: list = []

        def _make_loader(handles, counter=None):
            if not handles:
                return None
            wrapped = [_CountingHandle(h, counter) for h in handles] if counter is not None else handles
            loader = ReferenceObjectLoader(
                dataIds=[h.dataId for h in handles],
                refCats=wrapped,
            )
            loader.config.pixelMargin = 300  # extra tolerance for uncertain WCS
            return loader

        # One dataset, one loader, one load per sensor, shared by both consumers.
        #
        # No anyFilterMapsToThis: that alias exists to redirect *every* filter
        # request to a single column, for refcats that lack the filter being
        # asked for. the_monster does carry astromRefFilter, so the alias is
        # unnecessary here -- and actively harmful to sharing, since it would
        # also redirect the photometry lookup and silently feed astrometry
        # fluxes to donut selection. Without it, loadPixelBox resolves
        # fluxField from filterName for AstrometryTask, while the per-band
        # {prefix}_{band}_flux columns stay readable for donut selection.
        refcat_loader = _make_loader(refcat_handles, refcat_fetched)

        t_refcat0 = time.perf_counter()
        sensor_refcats: dict = {}
        for name, raw in rawByName.items():
            raw_wcs = raw.getWcs()
            raw_bbox = raw.getBBox()
            raw_epoch = raw.getInfo().getVisitInfo().date.toAstropy()
            load_result = None
            if refcat_loader is not None:
                try:
                    load_result = refcat_loader.loadPixelBox(
                        bbox=raw_bbox,
                        wcs=raw_wcs,
                        filterName=self.config.astromRefFilter,
                        epoch=raw_epoch,
                    )
                except Exception as exc:
                    self.log.warning("Failed to load refcat for %s: %s", name, exc)
            sensor_refcats[name] = load_result
        self.log.info(
            "Refcat load (loadPixelBox): %d/%d shards  (%.3fs)",
            len(refcat_fetched),
            len(refcat_handles),
            time.perf_counter() - t_refcat0,
        )
        t_refcat_elapsed = time.perf_counter() - t_refcat0

        # Stub loader: AstrometryTask.solve() calls refObjLoader.getMetadataBox()
        # unconditionally even when load_result is pre-supplied. That method is
        # pure geometry -- it never accesses catalog data, dataId.region, or the
        # flux aliases -- so the stub needs no anyFilterMapsToThis.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.pixelMargin = 0

        astrom_cfg = dict(
            astrom_task_config=self.astromTask.config,
            astrom_ref_obj_loader=astrom_stub_loader,
            detect_cfg=detect_cfg,
            maxFitScatter=self.config.maxFitScatter,
            minSourcesForWcsFit=self.config.minSourcesForWcsFit,
            donut_selector_config=self.donutSelector.config,
            astromRefFilter=self.config.astromRefFilter,
            photoRefFilter=self.config.photoRefFilter,
            photoRefFilterPrefix=self.config.photoRefFilterPrefix,
            resolvedPhotoFilterName=photo_filter_name,
            saveDiagnosticPlot=self.config.savePlots,
        )

        _CALIB_STORE.clear()
        _CALIB_STORE["isr_config"] = self.isrTask.config
        _CALIB_STORE["bkg_config"] = self.subtractBackground.config
        _CALIB_STORE["camera"] = camera
        _CALIB_STORE["detect_cfg"] = detect_cfg
        _CALIB_STORE["astrom_cfg"] = astrom_cfg
        _CALIB_STORE["sensor_refcats"] = sensor_refcats
        for name in CORNER_SENSOR_NAMES:
            missing_calib = [
                k for k, d in [("ptc", ptcByName), ("flat", flatByName),
                                ("linearizer", linearizerByName), ("crosstalk", crosstalkByName)]
                if name not in d
            ]
            if missing_calib:
                raise RuntimeError(
                    f"Missing calibration(s) for sensor {name}: {missing_calib}"
                )
            _CALIB_STORE[name] = dict(
                raw=rawByName[name],
                ptc=ptcByName[name],
                flat=flatByName[name],
                linearizer=linearizerByName[name],
                crosstalk=crosstalkByName[name],
            )

        # WF estimation config — read by WF worker functions after fork.
        example_visitInfo = next(iter(rawByName.values())).getInfo().getVisitInfo()
        boresight_rot_rad = example_visitInfo.boresightRotAngle.asRadians()
        boresight_par_rad = example_visitInfo.boresightParAngle.asRadians()
        boresight_alt_rad = example_visitInfo.boresightAzAlt.getLatitude().asRadians()
        rtp_deg = (
            (np.degrees(boresight_par_rad - boresight_rot_rad - np.pi / 2) + 180) % 360 - 180
            if self.config.modelSpiderShadows
            else None
        )
        example_instrument = getTaskInstrument(
            camera.getName(),
            next(iter(rawByName.keys())),
            self.config.instConfigFile,
        )
        wavelength_by_band = {bl.value: wl for bl, wl in example_instrument.wavelength.items()}
        lstsq_kwargs_parsed = {k: ast.literal_eval(v) for k, v in self.config.lstsqKwargs.items()}
        _CALIB_STORE["instrument"] = example_instrument
        _CALIB_STORE["wf_cfg"] = dict(
            nollIndices=np.array(list(self.config.nollIndices)),
            lstsqKwargs=lstsq_kwargs_parsed,
            binning=self.config.binning,
            bkgOrder=self.config.bkgOrder,
            modelSpiderShadows=self.config.modelSpiderShadows,
            doAoiThroughput=self.config.doAoiThroughput,
            systematicLossAlpha=self.config.systematicLossAlpha,
            triangleMode=self.config.triangleMode,
            rtp_deg=rtp_deg,
            boresight_alt_rad=boresight_alt_rad,
            band=example_band,
            calib_noll_indices=np.arange(4, 79),
            wavelength_by_band=wavelength_by_band,
            wfFitTimeoutPerDonut=self.config.wfFitTimeoutPerDonut,
            wfInitialGuessOnly=self.config.wfInitialGuessOnly,
            wfEstimationMode=self.config.wfEstimationMode,
        )
        # Telescope is band- and quantum-fixed; build once here and share via
        # COW instead of reloading "LSST_{band}.yaml" per donut in workers.
        _telescope = batoid.Optic.fromYaml(f"LSST_{example_band}.yaml")
        _CALIB_STORE["telescope"] = _telescope
        _CALIB_STORE["telescope_extra"] = _telescope.withLocallyShiftedOptic(
            "Detector", [0, 0, example_instrument.defocalOffset]
        )
        _CALIB_STORE["telescope_intra"] = _telescope.withLocallyShiftedOptic(
            "Detector", [0, 0, -example_instrument.defocalOffset]
        )

        cutout_args = sorted(CORNER_SENSOR_NAMES)

        self.log.info(
            "Running cutout workers on %d corner sensors with %d core(s)",
            len(cutout_args),
            numCores,
        )
        t_cutout0 = time.perf_counter()
        if numCores == 1:
            t_dispatch = time.time()
            results = [_run_cutout_worker((arg, t_dispatch)) for arg in cutout_args]
        else:
            t_pool0 = time.perf_counter()
            with mp.get_context("fork").Pool(processes=numCores) as pool:
                t_pool1 = time.perf_counter()
                t_dispatch = time.time()
                results = pool.map(_run_cutout_worker, [(arg, t_dispatch) for arg in cutout_args])
            t_pool2 = time.perf_counter()
            self.log.info(
                _colorize(
                    "Cutout pipeline: pool create: %.3fs, pool.map: %.3fs",
                    _ANSI_BOLD,
                    _ANSI_GREEN,
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
                "  %s: dispatch=%.3fs  init=%.3fs  isr=%.3fs"
                "  detect=%.3fs  wcs=%.3fs (scatter=%s)"
                "  select=%.3fs  cut=%.3fs  donuts=%d",
                r["sensor"],
                r["dispatch_to_arrival"],
                r["task_init"],
                r["isr_run"],
                r["blind_detect_run"],
                r["wcs_refit_run"],
                scatter_str,
                r["catalog_select_run"],
                r.get("stamp_cut_run", 0.0),
                len(r["catalog"]),
            )
            if r["wcs_refit_error"]:
                self.log.warning("  %s: WCS refit failed: %s", r["sensor"], r["wcs_refit_error"])
            if r["cat_select_error"]:
                self.log.warning(
                    "  %s: catalog selection failed: %s",
                    r["sensor"],
                    r["cat_select_error"],
                )
            donuts.extend(r["catalog"])

        # Annotate each accepted donut with realized intrinsic Zernikes.
        for r in results:
            calib = intrinsicZernikesByName.get(r["sensor"])
            for d in r["catalog"]:
                if calib is not None:
                    d["intrinsic_zk"] = np.squeeze(
                        calib.getIntrinsicZernikes(
                            np.degrees(d["fa_x_ccs"]),
                            np.degrees(d["fa_y_ccs"]),
                        )
                    )
                else:
                    d["intrinsic_zk"] = None

        # WF dispatch
        mode = self.config.wfEstimationMode
        results_by_sensor = {r["sensor"]: r["catalog"] for r in results}
        groups, unmatched_donuts = _build_wf_groups(mode, results_by_sensor, _CALIB_STORE["wf_cfg"])

        self.log.info("WF dispatch (%s): %d work unit(s)", mode, len(groups))
        t_wf0 = time.perf_counter()
        if not groups:
            wf_results = []
        elif numCores == 1 or len(groups) == 1:
            wf_results = [_wf_worker(g) for g in groups]
        else:
            n_workers = min(numCores, len(groups))
            with mp.get_context("fork").Pool(processes=n_workers) as wf_pool:
                wf_results = wf_pool.map(_wf_worker, groups)
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
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            butler_elapsed,
            t_refcat_elapsed,
            t_cutout1 - t_cutout0,
            t_wf1 - t_wf0,
            t_plot0 - t_run0,
        )
        visit_ids = {raw.getInfo().getVisitInfo().id for raw in raws}
        visit_ids.discard(0)
        visit_str = "_".join(str(v) for v in sorted(visit_ids))

        catalog = self._buildCatalog(
            results=results,
            wf_results=wf_results,
            donuts=donuts,
            unmatched_donuts=unmatched_donuts,
            visit_str=visit_str,
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

        return pipeBase.Struct(donuts=donuts, wf_results=wf_results, blitzResults=catalog)

    def _buildCatalog(
        self,
        results: list,
        wf_results: list,
        donuts: list,
        unmatched_donuts: list,
        visit_str: str = "",
        run_elapsed: float = 0.0,
        refcat_elapsed: float = 0.0,
        butler_elapsed: float = 0.0,
        butler_times: dict | None = None,
        cutout_elapsed: float = 0.0,
        danish_elapsed: float = 0.0,
        photo_filter_name: str = "",
        astrom_filter_name: str = "",
    ) -> QTable:
        """Build a per-donut QTable covering accepted, rejected, and unmatched donuts.

        Parameters
        ----------
        results : list
            Per-sensor cutout dicts from ``_getCutouts`` (supplies rejected donuts
            and per-sensor metadata).
        wf_results : list
            Per-fit WF result dicts from the WF worker pool.
        donuts : list
            Accepted donut dicts.
        unmatched_donuts : list
            Donut dicts with no intra/extra partner (paired mode only).
        visit_str : str
            Visit identifier string.

        Returns
        -------
        QTable
            One row per donut (accepted + rejected + unmatched).  Array columns
            (``stamp``, ``model_img``, ``wf_img``) are zero-padded to a common
            shape.  Visit-level and per-sensor scalars are stored in
            ``table.meta``.
        """
        import json

        # Build lookup: (source_id, sensor) -> (wf donut entry, group index).
        # Key on both fields to handle the same refcat star on SW0 and SW1.
        wf_by_id: dict = {}
        for group_idx, r in enumerate(wf_results):
            for wd in r.get("donuts", []):
                wf_by_id[(int(wd["donut_id"]), str(wd["sensor"]))] = (wd, group_idx)

        # Build lookup: sensor -> per-sensor metadata from cutout results.
        sensor_meta: dict = {}
        rejected_by_sensor: dict = {}
        for r in results:
            sname = str(r["sensor"])
            sensor_meta[sname] = {
                "scatter_arcsec": (
                    float(r["scatter_arcsec"]) if r["scatter_arcsec"] is not None else float("nan")
                ),
                "wcs_refit_error": str(r.get("wcs_refit_error") or ""),
                "cat_select_error": str(r.get("cat_select_error") or ""),
                "isr_run": float(r.get("isr_run", float("nan"))),
                "blind_detect_run": float(r.get("blind_detect_run", float("nan"))),
                "wcs_refit_run": float(r.get("wcs_refit_run", float("nan"))),
                "catalog_select_run": float(r.get("catalog_select_run", float("nan"))),
            }
            rejected_by_sensor[sname] = r.get("rejected_catalog", [])

        _KNOWN_REASONS = ("snr", "inner_frac", "outer_frac", "SAT", "field_dist")

        def _reject_reason_str(reasons):
            """Comma-join reject reasons in their display form.

            Order is preserved and duplicates are dropped, so a selector
            "faint_neighbor" collapsing onto an existing "blend" (or the
            selector's "field_dist" onto the monolith's) shows only once.
            """
            out: list[str] = []
            for rr in reasons:
                disp = _SELECTOR_REASON_DISP.get(rr, rr)
                if disp not in out:
                    out.append(disp)
            return ",".join(out)

        def _encode_nearby(entries):
            return json.dumps([[round(dx, 1), round(dy, 1), round(mag, 2)] for dx, dy, mag in entries])

        # Collect all donuts: accepted, rejected (per-sensor), unmatched.
        all_donuts = []
        for d in donuts:
            all_donuts.append((d, True))
        for r in results:
            for d in rejected_by_sensor.get(str(r["sensor"]), []):
                all_donuts.append((d, False))
        for d in unmatched_donuts:
            all_donuts.append((d, False))

        if not all_donuts:
            return QTable()

        # Determine common stamp shape for padding.
        max_ny = max(d["stamp"].shape[0] for d, _ in all_donuts)
        max_nx = max(d["stamp"].shape[1] for d, _ in all_donuts)
        # wf/model images are binned and cropped to odd size (see
        # _prep_donut_for_danish). Derive from the actual odd cut size,
        # 2*(stampSize//2)+1, not from stampSize itself.
        _cut = 2 * (self.config.stampSize // 2) + 1
        _binned = _cut // self.config.binning
        max_wf_ny = max_wf_nx = _binned if _binned % 2 == 1 else _binned - 1

        def _pad(arr, ny, nx):
            out = np.full((ny, nx), np.nan, dtype=float)
            out[: arr.shape[0], : arr.shape[1]] = arr
            return out

        def _wd_field(wd, key, caster, default):
            return caster(wd[key]) if wd is not None else default

        rows = []
        for d, accepted in all_donuts:
            sid = int(d["source_id"])
            wd, grp = wf_by_id.get((sid, str(d["sensor"])), (None, -1))

            zk_dev = wd["zk_dev"] if wd is not None else np.full(_ZK_LEN, np.nan)
            zk_int = wd["zk_intrinsic"] if wd is not None else np.full(_ZK_LEN, np.nan)

            reject_reasons = d.get("reject_reasons", [])
            stamp = _pad(d["stamp"].astype(float), max_ny, max_nx)

            wf_img_raw = wd.get("img") if wd is not None else None
            wf_img = (
                _pad(wf_img_raw.astype(float), max_wf_ny, max_wf_nx)
                if wf_img_raw is not None
                else np.full((max_wf_ny, max_wf_nx), np.nan, dtype=float)
            )

            row = {
                # --- identity ---
                "visit_id": int(d["visit_id"]),
                "det_id": int(d["det_id"]),
                "sensor": str(d["sensor"]),
                "source_id": sid,
                "defocal": _wd_field(wd, "defocal", str, ""),
                "band": str(d["band"]),
                "accepted": bool(accepted),
                # --- geometry ---
                "centroid_x_raw": float(d["centroid_x_raw"]),
                "centroid_y_raw": float(d["centroid_y_raw"]),
                "fa_x_ccs": float(d["fa_x_ccs"]),
                "fa_y_ccs": float(d["fa_y_ccs"]),
                "field_dist_deg": float(np.degrees(np.hypot(d["fa_x_ccs"], d["fa_y_ccs"]))),
                "n_quarter": int(d.get("n_quarter", 0)),
                # --- nearby refcat sources (JSON-encoded, pixel offsets + magnitudes) ---
                "nearby_photo": _encode_nearby(d.get("nearby_photo", [])),
                "nearby_astrom": _encode_nearby(d.get("nearby_astrom", [])),
                # --- selection metrics ---
                "flux": float(d["flux"]),
                "snr": float(d["snr"]),
                "inner_frac": float(d["inner_frac"]),
                "outer_frac": float(d["outer_frac"]),
                "outer_sector_max": float(d["outer_sector_max"]),
                "bkg_level": float(d["bkg_level"]),
                "bkg_std": float(d["bkg_std"]),
                "nearest_neighbor_dist_px": float(d["nearest_neighbor_dist_px"]),
                "n_neighbors_in_stamp": int(d["n_neighbors_in_stamp"]),
                "catalog_centroid_offset_px": float(d["catalog_centroid_offset_px"]),
                "rejected_snr": bool("snr" in reject_reasons),
                "rejected_inner_frac": bool("inner_frac" in reject_reasons),
                "rejected_outer_frac": bool("outer_frac" in reject_reasons),
                "rejected_sat": bool("SAT" in reject_reasons or d.get("saturated", False)),
                "rejected_field_dist": bool("field_dist" in reject_reasons),
                "rejected_selector": bool(any(rr not in _KNOWN_REASONS for rr in reject_reasons)),
                "reject_reasons": _reject_reason_str(reject_reasons),
                # --- fit results ---
                "fit_mode": _wd_field(wd, "fit_mode", str, ""),
                "group": int(grp),
                "group_size": _wd_field(wd, "group_size", int, 0),
                "fit_success": _wd_field(wd, "fit_success", bool, False),
                "fit_elapsed": _wd_field(wd, "fit_elapsed", float, float("nan")),
                "setup_elapsed": _wd_field(wd, "setup_elapsed", float, float("nan")),
                "fit_nfev": _wd_field(wd, "fit_nfev", int, 0),
                "fit_cost": _wd_field(wd, "fit_cost", float, float("nan")),
                "fit_dx": _wd_field(wd, "fit_dx", float, float("nan")),
                "fit_dy": _wd_field(wd, "fit_dy", float, float("nan")),
                "fit_flux": _wd_field(wd, "fit_flux", float, float("nan")),
                "fit_fwhm": _wd_field(wd, "fit_fwhm", float, float("nan")),
                "fit_bkg": _wd_field(wd, "fit_bkg", float, float("nan")),
                "fit_residual_rms": _wd_field(wd, "fit_residual_rms", float, float("nan")),
                "blend_frac": _wd_field(wd, "blend_frac", float, float("nan")),
                "zk_norm_um": _wd_field(wd, "zk_norm_um", float, float("nan")),
                # --- per-Noll Zernikes (µm), Noll 4–78 ---
                **_zk_cols("dev", zk_dev),
                **_zk_cols("intrinsic", zk_int),
                # --- embedded images ---
                "stamp": stamp,
                "wf_img": wf_img,
                "model_img": (
                    _pad(wd["model_img"].astype(float), max_wf_ny, max_wf_nx)
                    if (wd is not None and wd.get("model_img") is not None)
                    else np.full((max_wf_ny, max_wf_nx), np.nan, dtype=float)
                ),
            }
            rows.append(row)

        table = QTable(rows)
        table.meta["visit_str"] = visit_str
        table.meta["run_elapsed"] = float(run_elapsed)
        table.meta["refcat_elapsed"] = float(refcat_elapsed)
        table.meta["butler_elapsed"] = float(butler_elapsed)
        table.meta["butler_times"] = dict(butler_times or {})
        table.meta["cutout_elapsed"] = float(cutout_elapsed)
        table.meta["danish_elapsed"] = float(danish_elapsed)
        table.meta["photo_filter_name"] = str(photo_filter_name)
        table.meta["astrom_filter_name"] = str(astrom_filter_name)
        table.meta["noll_indices"] = list(_CALIB_STORE.get("wf_cfg", {}).get("nollIndices", []))
        table.meta["sensor_meta"] = sensor_meta
        _first_d = all_donuts[0][0]
        table.meta["donut_radius"] = _first_d["donut_radius"]
        table.meta["obscuration"] = _first_d["obscuration"]
        _detect_cfg = _CALIB_STORE["detect_cfg"]
        table.meta["aperture_outer_margin_frac"] = _detect_cfg["apertureOuterMarginFrac"]
        table.meta["aperture_inner_buffer_frac"] = _detect_cfg["apertureInnerBufferFrac"]
        table.meta["bkg_annulus_inner_frac"] = _detect_cfg["bkgAnnulusInnerFrac"]
        table.meta["bkg_annulus_outer_frac"] = _detect_cfg["bkgAnnulusOuterFrac"]
        table.meta["max_donuts"] = int(_detect_cfg["maxDonuts"])
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

    def run(self, catalog: QTable) -> None:
        """Generate donut and WF diagnostic plots from the blitzResults catalog.

        Parameters
        ----------
        catalog : QTable
            Per-donut table as produced by
            ``DonutBlitzMonolithTask._buildCatalog``.  Visit-level and
            per-sensor metadata are in ``catalog.meta``.
        """
        self._saveDonutDiagnosticPlot(catalog)
        self._saveWfDiagnosticPlot(catalog)

    def _saveDonutDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a single diagnostic PNG with one section per sensor.

        Layout per sensor:
          - Left column: stats text (timing, WCS scatter, donut count, errors)
          - Remaining columns: donut stamps (up to maxDonuts), each annotated
            with flux and field angle

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.  Per-sensor metadata is in
            ``catalog.meta["sensor_meta"]``; visit-level scalars are in
            ``catalog.meta``.
        """
        import json

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
        visit_str = meta["visit_str"]
        sensor_meta = meta["sensor_meta"]

        # Group rows by sensor; split into accepted and rejected.
        sensor_col = np.asarray(catalog["sensor"], dtype=str)
        sensors_with_data = []
        for sensor in sorted(set(sensor_col.tolist())):
            sensor_rows = catalog[sensor_col == sensor]
            acc = sensor_rows[sensor_rows["accepted"]]
            rej = sensor_rows[~sensor_rows["accepted"]]
            if len(acc) > 0 or len(rej) > 0:
                sensors_with_data.append((sensor, acc, rej))

        n_sensors = len(sensors_with_data)
        if n_sensors == 0:
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
        fig_h = n_sensors * ROW_H + LEGEND_H + SUPTITLE_H

        fig = Figure(figsize=(fig_w, fig_h), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
        butler_str = f"  butler={butler_elapsed:.1f}s" if butler_elapsed > 0 else ""
        fig.suptitle(
            f"DonutBlitz diagnostics  visit={visit_str}"
            f"  refcat={refcat_elapsed:.1f}s{butler_str}  run={run_elapsed:.1f}s",
            fontsize=9,
        )

        w_stats = STATS_COL_W / STAMP_COL_W
        w_spacer = SPACER_W / STAMP_COL_W
        gs = GridSpec(
            n_sensors + 1,
            N_COLS,
            figure=fig,
            height_ratios=[ROW_H] * n_sensors + [LEGEND_H],
            width_ratios=[w_stats] + [1] * STAMPS_PER_ROW + [w_spacer] + [1] * REJECTED_PER_ROW,
        )
        COL_ACCEPTED_START = 1
        COL_SPACER = 1 + STAMPS_PER_ROW
        COL_REJECTED_START = COL_SPACER + 1

        _dr = catalog.meta["donut_radius"]
        _ob = catalog.meta["obscuration"]
        _stamp_dr = _dr if np.isfinite(_dr) else None
        _stamp_ob = _ob if np.isfinite(_ob) else None
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
            # extent bounds the pixel *edges*, so the center pixel spans
            # [-0.5, +0.5] and offset 0 lands on the centroid pixel's center --
            # matching the annulus circles drawn at (0, 0) and the refcat
            # symbols placed at raw pixel offsets. Correct because stamps are
            # cut odd; an even stamp has no center pixel to align to.
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

            dr, ob = _stamp_dr, _stamp_ob
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

            nq = int(row["n_quarter"]) % 4

            def _xform(dx, dy):
                """Map a raw-pixel offset to stamp display coords.

                Must mirror the stamp transform in `_cutStamps`,
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

            for dx, dy, mag in json.loads(str(row["nearby_photo"])):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "o", ms=6, mfc="none", mec=_COLOR_PHOTO_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty + 3, f"{mag:.2f}", color=_COLOR_PHOTO_REFCAT, fontsize=3.5, zorder=4)
            for dx, dy, mag in json.loads(str(row["nearby_astrom"])):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "+", ms=6, mec=_COLOR_ASTROM_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty - 5, f"{mag:.2f}", color=_COLOR_ASTROM_REFCAT, fontsize=3.5, zorder=4)

            # Pin the view to the stamp's own edges: constant figure footprint
            # regardless of pixel count, and no autoscale from the overlays.
            ax.set_xlim(-_edge, _edge)
            ax.set_ylim(-_edge, _edge)

            inner_frac = float(row["inner_frac"])
            outer_frac = float(row["outer_frac"])
            outer_sector_max = float(row["outer_sector_max"])
            snr = float(row["snr"])
            if_str = f"if={inner_frac:.3f}" if np.isfinite(inner_frac) else "if=?"
            of_str = f"of={outer_frac:.3f}" if np.isfinite(outer_frac) else "of=?"
            osm_str = f"osm={outer_sector_max:.3f}" if np.isfinite(outer_sector_max) else "osm=?"
            snr_str = f"snr={snr:.0f}" if np.isfinite(snr) else "snr=?"
            sid = int(row["source_id"])
            sid_str = f"id={sid}" if sid != 0 else ""
            _text_color = _COLOR_REJECTED if rejected else "black"
            _reasons = str(row["reject_reasons"]) if rejected else ""
            rej_str = f"[{_reasons}]" if _reasons else ""
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

        for row_idx, (sensor, acc_rows, rej_rows) in enumerate(sensors_with_data):
            sm = sensor_meta.get(sensor, {})
            scatter_val = sm.get("scatter_arcsec", float("nan"))
            scatter_str = f'{scatter_val:.3f}"' if np.isfinite(scatter_val) else "N/A"

            ax_stats = fig.add_subplot(gs[row_idx, 0])
            ax_stats.axis("off")
            lines = [
                f"{sensor}",
                f"donuts: {len(acc_rows)}",
                f"isr:    {sm.get('isr_run', float('nan')):.2f}s",
                f"detect: {sm.get('blind_detect_run', float('nan')):.2f}s",
                f"wcs:    {sm.get('wcs_refit_run', float('nan')):.2f}s  ({scatter_str})",
                f"select: {sm.get('catalog_select_run', float('nan')):.2f}s",
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

        ax_legend = fig.add_subplot(gs[n_sensors, :])
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

        fname = f"donut_diag_{visit_str}.png"
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
        visit_str = meta["visit_str"]
        refcat_elapsed = meta["refcat_elapsed"]
        butler_elapsed = meta["butler_elapsed"]
        butler_times = meta["butler_times"]
        cutout_elapsed = meta["cutout_elapsed"]
        danish_elapsed = meta["danish_elapsed"]
        noll_cfg = meta["noll_indices"]
        ZK_MIN, ZK_MAX = 4, 28

        # Reconstruct wf_results-like list from QTable by grouping on "group" column.
        # Include only rows with a valid fit (fit_mode != "" and group >= 0).
        groups: dict[int, list] = {}
        for row in catalog:
            fm = str(row["fit_mode"])
            grp = int(row["group"])
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
            sensors = list(dict.fromkeys(str(r["sensor"]) for r in rows))
            mode = str(first["fit_mode"])
            success = bool(first["fit_success"])
            elapsed = float(first["fit_elapsed"])
            nfev = int(first["fit_nfev"])
            fwhm = float(first["fit_fwhm"])
            zk_by_noll = {j: float(first[f"Z{j}_dev"]) for j in range(4, 79)}

            donuts_out = []
            for r in rows:
                model_arr = np.array(r["model_img"])
                donuts_out.append({
                    "donut_id": int(r["source_id"]),
                    "sensor": str(r["sensor"]),
                    "defocal": str(r["defocal"]),
                    "img": np.array(r["wf_img"]),
                    "model_img": model_arr if not np.all(np.isnan(model_arr)) else None,
                    "blend_frac": float(r["blend_frac"]),
                    "elapsed": elapsed,
                    "nfev": nfev,
                    "fwhm": fwhm,
                    "success": success,
                })
            plottable.append({
                "mode": mode,
                "sensors": sensors,
                "success": success,
                "fit_info": {"elapsed": elapsed, "nfev": nfev, "fwhm": fwhm},
                "donuts": donuts_out,
                "zk_by_noll": zk_by_noll,
            })

        if not plottable:
            self.log.info("No WF results with model images; skipping WF diagnostic plot.")
            return

        _CORNERS = list(CORNER_PAIRS)

        def _corner_of(r):
            # Sensor names are detector.getName() ("R00_SW0"), but tolerate an
            # instrument prefix ("LSSTCam_R00_SW0") by matching on the suffix.
            for s in r.get("sensors", []):
                for sensor, corner in CORNER_BY_SENSOR.items():
                    if s and str(s).endswith(sensor):
                        return corner
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
            mode = corner_results[0].get("mode") if corner_results else None
            if _mode_groups_are_pairs(mode):
                # The group *is* an intra/extra pair, so it supplies both halves
                # of the row; the donut of each defocal type is picked out below.
                row_pairs[corner] = [(r, r) for r in corner_results]
            else:
                # These groups don't pair donuts, so flatten to one record per
                # donut and pair for layout only. Exploding is a no-op for
                # "unpaired" (one donut per group already).
                row_pairs[corner] = _pair_up(
                    [s for r in corner_results for s in _explode(r)]
                )

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
            [int(meta.get("max_donuts", 0) or 0), 1]
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
                    zk_by_noll_i = r_intra.get("zk_by_noll", {j: float("nan") for j in range(4, 79)})
                else:
                    intra_rec = None
                    elapsed_i, nfev_i, success_i, fwhm_i = _rec_info(None)
                    zk_by_noll_i = {j: float("nan") for j in range(4, 79)}

                if r_extra is not None:
                    extra_rec = next(
                        (d for d in r_extra.get("donuts", []) if d["defocal"] == "extra"), None
                    )
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(r_extra)
                    zk_by_noll_e = r_extra.get("zk_by_noll", {j: float("nan") for j in range(4, 79)})
                else:
                    extra_rec = None
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(None)
                    zk_by_noll_e = {j: float("nan") for j in range(4, 79)}

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
                    return f"t={elapsed:.1f}s {status} n={nfev}"

                intra_label = _bar_label(elapsed_i, nfev_i, success_i)
                extra_label = _bar_label(elapsed_e, nfev_e, success_e)
                intra_hdr = f"intra {sw1}" if row_idx == 0 else ""
                extra_hdr = f"extra {sw0}" if row_idx == 0 else ""

                def _triplet_and_bar(col_start, data, model, sensor_hdr, label,
                                     sid, fwhm, zk_by_noll, blend_frac_val=float("nan")):
                    if data is not None:
                        vmax = float(np.nanpercentile(np.abs(data), 99)) or 1.0
                        has_model = model is not None
                        resid = (data - model) if has_model else None
                        vmax_r = (float(np.nanpercentile(np.abs(resid), 99)) or 1.0) if has_model else 1.0
                        for ci, (img, cmap, vmin, vmx) in enumerate([
                            (data, _cmap_bwr, -vmax, vmax),
                            (model if has_model else None, _cmap_bwr, -vmax, vmax),
                            (resid, _cmap_bwr_sym, -vmax_r, vmax_r),
                        ]):
                            ax = fig.add_subplot(inner[row_idx, col_start + ci])
                            lbl = sensor_hdr if ci == 0 else ""
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
                            if ci == 0 and sensor_hdr:
                                ax.set_title(sensor_hdr, fontsize=5, pad=1)

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
        first_mode = plottable[0]["mode"] if plottable else ""
        fig.suptitle(
            f"WF fits  visit={visit_str}  mode={first_mode}\n"
            f"butler.get:  {butler_line}\n"
            f"refcat={refcat_elapsed:.1f}s  cutout={cutout_elapsed:.1f}s  "
            f"danish={danish_elapsed:.1f}s  proc_total={proc_total:.1f}s",
            fontsize=7,
        )
        fname = f"wf_diag_{visit_str}.png"
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
