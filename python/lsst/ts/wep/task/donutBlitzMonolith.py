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
]

import multiprocessing as mp
import time
from typing import Any

import numpy as np
from astropy.table import QTable
from scipy.signal import correlate
from skimage.feature import peak_local_max

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.afw.cameraGeom import Camera, FIELD_ANGLE, PIXELS
from lsst.afw.image import Exposure
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import MagnitudeLimit, ReferenceObjectLoader, SubtractBackgroundTask
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.base import InputQuantizedConnection, OutputQuantizedConnection, QuantumContext
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import donutCatalogToAstropy
from lsst.ts.wep.utils import getTaskInstrument
from lsst.utils.timer import timeMethod

_CALIB_STORE: dict = {}  # populated in parent before fork; workers inherit via COW

CORNER_SENSOR_NAMES = frozenset(
    [
        "R00_SW0",
        "R00_SW1",
        "R04_SW0",
        "R04_SW1",
        "R40_SW0",
        "R40_SW1",
        "R44_SW0",
        "R44_SW1",
    ]
)


def _buildAnnularTemplate(radius: float, innerFrac: float) -> np.ndarray:
    r_int = int(radius)
    cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
    r = np.hypot(cx, cy)
    return np.where((r < radius) & (r >= radius * innerFrac), 1.0, 0.0)


def _detectPeaks(
    exposureTrim: Exposure,
    donutRadius: float,
    obscuration: float,
    detectionBinning: int,
    peakMinDistanceFactor: float,
    peakExcludeBorderFactor: float,
) -> np.ndarray:
    binning = detectionBinning
    radius_binned = donutRadius / binning
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
    donutRadius: float,
    obscuration: float,
) -> QTable:
    arr = exposureTrim.image.array
    radius = donutRadius
    half = int(radius * 1.4)

    gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
    r = np.hypot(gx, gy)

    main_mask = (r < radius * 1.05) & (r > radius * obscuration)
    inner_mask = r < radius * obscuration * 0.67
    outer_mask = (r > radius * 1.25) & (r < radius * 1.4)
    bkg_mask = inner_mask | outer_mask
    n_main = np.sum(main_mask)

    flux_list, inner_flux_list, outer_flux_list, std_list = [], [], [], []

    for row, col in zip(peaks[:, 0], peaks[:, 1]):
        rmin, rmax = row - half, row + half + 1
        cmin, cmax = col - half, col + half + 1

        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            flux_list.append(np.nan)
            inner_flux_list.append(np.nan)
            outer_flux_list.append(np.nan)
            std_list.append(np.nan)
            continue

        stamp = arr[rmin:rmax, cmin:cmax]
        bkg = np.nanmedian(stamp[bkg_mask])
        stamp_sub = stamp - bkg

        flux_list.append(float(np.sum(stamp_sub[main_mask])))
        inner_flux_list.append(float(np.sum(stamp_sub[inner_mask])))
        outer_flux_list.append(float(np.sum(stamp_sub[outer_mask])))

        diff = (stamp_sub - np.roll(stamp_sub, 1, axis=0))[bkg_mask]
        q75, q25 = np.nanpercentile(diff, [75, 25])
        std_list.append(float((q75 - q25) / 1.349 / np.sqrt(2)))

    table = QTable()
    table["centroid_x"] = np.array(peaks[:, 1], dtype=float)
    table["centroid_y"] = np.array(peaks[:, 0], dtype=float)
    table["flux"] = np.array(flux_list, dtype=float)
    table["inner_flux"] = np.array(inner_flux_list, dtype=float)
    table["outer_flux"] = np.array(outer_flux_list, dtype=float)
    table["std"] = np.array(std_list, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["snr"] = (table["flux"] / table["std"]) / np.sqrt(n_main)
    return table



def _blindDetect(
    exposure: Exposure,
    detect_cfg: dict,
    bkg_config: Any,
    donutRadius: float,
    obscuration: float,
) -> QTable:
    """Run background subtraction and blind cross-correlation detection.

    Returns a QTable with centroid_x/y, flux, inner_flux, outer_flux, std,
    snr — all in full-exposure pixel coordinates.  Returns an empty table
    if nothing passes cuts.
    """
    SubtractBackgroundTask(config=bkg_config).run(exposure=exposure)

    trimmedBBox = exposure.getBBox().erodedBy(detect_cfg["edgeMargin"])
    exposureTrim = exposure[trimmedBBox].clone()

    peaks = _detectPeaks(
        exposureTrim,
        donutRadius,
        obscuration,
        detect_cfg["detectionBinning"],
        detect_cfg["peakMinDistanceFactor"],
        detect_cfg["peakExcludeBorderFactor"],
    )

    empty = QTable(
        names=["centroid_x", "centroid_y", "flux", "inner_flux", "outer_flux", "std", "snr"],
        dtype=[float] * 7,
    )

    if len(peaks) == 0:
        return empty

    measTable = _measureFlux(peaks, exposureTrim, donutRadius, obscuration)

    # Apply only SNR cut here so the WCS fitter sees as many sources as possible.
    # Inner/outer fraction cuts (donut-quality checks) are applied after WCS refit
    # in _refitWcsAndSelect, when they are used to decide which stamps to cut.
    validFlux = np.isfinite(measTable["flux"]) & (measTable["flux"] > 0)
    snOk = measTable["snr"] > detect_cfg["snrThreshold"]
    measTable = measTable[validFlux & snOk]

    if len(measTable) == 0:
        return empty

    xOffset = trimmedBBox.getMinX()
    yOffset = trimmedBBox.getMinY()
    measTable["centroid_x"] = np.array(measTable["centroid_x"]) + xOffset
    measTable["centroid_y"] = np.array(measTable["centroid_y"]) + yOffset

    return measTable


def _buildAfwSourceCat(blindDetections: QTable, wcs) -> afwTable.SourceCatalog:
    """Convert blind-detect QTable into a minimal afwTable.SourceCatalog
    suitable for AstrometryTask.run().
    """
    sourceSchema = afwTable.SourceTable.makeMinimalSchema()
    measBase.SingleFrameMeasurementTask(schema=sourceSchema)
    for c in ["ra", "dec"]:
        name = f"coord_{c}Err"
        if name not in sourceSchema.getNames():
            sourceSchema.addField(
                afwTable.Field["F"](name=name, doc=f"position err in {c}", units="rad"),
            )

    sourceCat = afwTable.SourceCatalog(sourceSchema)
    sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
    sourceIdKey = sourceSchema["id"].asKey()
    sourceRAKey = sourceSchema["coord_ra"].asKey()
    sourceDecKey = sourceSchema["coord_dec"].asKey()
    sourceInstFluxKey = sourceSchema["slot_ApFlux_instFlux"].asKey()
    sourceInstFluxErrKey = sourceSchema["slot_ApFlux_instFluxErr"].asKey()
    sourceRaErrKey = sourceSchema["coord_raErr"].asKey()
    sourceDecErrKey = sourceSchema["coord_decErr"].asKey()

    sourceCat.reserve(len(blindDetections))
    for i, row in enumerate(blindDetections):
        x, y = float(row["centroid_x"]), float(row["centroid_y"])
        sky = wcs.pixelToSky(x, y)
        src = sourceCat.addNew()
        src.set(sourceIdKey, i)
        ra = sky.getRa()
        dec = sky.getDec()
        src.set(sourceRAKey, ra)
        src.set(sourceDecKey, dec)
        src.set(sourceRaErrKey, lsst.geom.Angle(abs(ra.asRadians()) * 0.01))
        src.set(sourceDecErrKey, lsst.geom.Angle(abs(dec.asRadians()) * 0.01))
        src.set(sourceCentroidKey, lsst.geom.Point2D(x, y))
        flux = float(row["flux"])
        src.set(sourceInstFluxKey, flux)
        src.set(sourceInstFluxErrKey, abs(flux) * 0.01)

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


def _refitWcsAndSelect(
    postIsr: Exposure,
    blindDetections: QTable,
    astrom_cfg: dict,
    sensor_name: str,
    donutRadius: float,
    obscuration: float,
) -> list[dict]:
    """Refit WCS using the pre-loaded astrometry refcat, then select donuts
    from the pre-loaded photometry refcat and cut stamps.

    The refcat spatial filtering is done in the parent before forking, so no
    ReferenceObjectLoader is constructed here. The parent stores per-sensor
    pipeBase.Struct(refCat, fluxField) objects in _CALIB_STORE["sensor_refcats"].

    Falls back to blind-detect stamp cutting on any error.
    """
    detector = postIsr.getDetector()
    wcs = postIsr.getWcs()
    visitInfo = postIsr.getInfo().getVisitInfo()
    filterLabel = postIsr.getFilter()
    band = postIsr.filter.bandLabel
    visit_id = visitInfo.id
    det_id = detector.getId()
    n_quarter = detector.getOrientation().getNQuarter()

    detect_cfg = astrom_cfg["detect_cfg"]
    stampSize = detect_cfg["stampSize"]
    maxDonuts = detect_cfg["maxDonuts"]
    maxFitScatter = astrom_cfg["maxFitScatter"]

    sensor_refcats = _CALIB_STORE.get("sensor_refcats", {}).get(sensor_name, {})
    astrom_load_result = sensor_refcats.get("astrom")   # pipeBase.Struct(refCat, fluxField)
    photo_load_result = sensor_refcats.get("photo")     # pipeBase.Struct(refCat, fluxField)

    # --- WCS refit ---
    t0 = time.perf_counter()
    refitted_wcs = wcs
    scatter_arcsec = None
    wcs_refit_error = None
    if len(blindDetections) >= astrom_cfg["minSourcesForWcsFit"] and astrom_load_result is not None:
        try:
            astromTask = AstrometryTask(config=astrom_cfg["astrom_task_config"])
            # Need a refObjLoader set even when passing load_result, for getMetadataBox later.
            # Use a no-op loader stub by setting a loader with the pre-loaded catalog;
            # since load_result is passed, loadPixelBox is never called.
            astromTask.setRefObjLoader(astrom_cfg["astrom_ref_obj_loader"])

            afwCat = _buildAfwSourceCat(blindDetections, wcs)
            fakeExp = _buildFakeExposure(detector, wcs, visitInfo, filterLabel)

            astromResult = astromTask.solve(
                exposure=fakeExp,
                sourceCat=afwCat,
                load_result=astrom_load_result,
            )
            scatter_arcsec = astromResult.scatterOnSky.asArcseconds()
            if scatter_arcsec < maxFitScatter:
                refitted_wcs = fakeExp.getWcs()
            else:
                refitted_wcs = wcs
        except Exception as e:
            wcs_refit_error = str(e)
            refitted_wcs = wcs
    t1 = time.perf_counter()

    # --- Catalog-based selection from photo refcat ---
    t2 = time.perf_counter()
    catalog_centroids = None  # (centroid_x, centroid_y, source_flux) arrays
    cat_select_error = None
    sel_rejected_refcat = None
    if photo_load_result is not None:
        try:
            filterName = astrom_cfg["resolvedPhotoFilterName"]

            # Reproject catalog sky coords through the refitted WCS so stamp
            # centroids are consistent with the corrected pointing.
            refCat = photo_load_result.refCat.copy(deep=True)
            afwTable.updateRefCentroids(refitted_wcs, refCat)
            if not refCat.isContiguous():
                refCat = refCat.copy(deep=True)

            donutSelectorTask = (
                DonutSourceSelectorTask(config=astrom_cfg["donut_selector_config"])
                if astrom_cfg["doDonutSelection"]
                else None
            )

            if donutSelectorTask is None:
                refSelection = refCat
                blendCentersX = [[]] * len(refCat)
                blendCentersY = [[]] * len(refCat)
                sel_rejected_refcat = refCat[np.zeros(len(refCat), dtype=bool)]  # empty
            else:
                donutSelection = donutSelectorTask.run(refCat, detector, filterName)
                sel_mask = np.array(donutSelection.selected, dtype=bool)
                refSelection = refCat[sel_mask]
                blendCentersX = donutSelection.blendCentersX
                blendCentersY = donutSelection.blendCentersY
                sel_rejected_refcat = refCat[~sel_mask]

            filterList = list(astrom_cfg["catalogFilterList"])
            if filterName not in filterList:
                filterList.append(filterName)
            sortFilterIdx = filterList.index(filterName)

            catalog = donutCatalogToAstropy(
                refSelection, filterList, blendCentersX, blendCentersY,
                sortFilterIdx=sortFilterIdx,
            )
            if len(catalog) > 0:
                # Mirror the flux sort donutCatalogToAstropy applies so source
                # IDs stay aligned with the sorted centroid/flux arrays.
                ref_flux = np.array(refSelection[f"{filterName}_flux"])
                flux_sort = np.argsort(ref_flux)[::-1]
                sorted_ids = np.array(refSelection["id"])[flux_sort]
                catalog_centroids = (
                    np.array(catalog["centroid_x"]),
                    np.array(catalog["centroid_y"]),
                    np.array(catalog[f"{filterName}_flux"]),
                    sorted_ids,
                )

            # Build full-refcat lookup arrays for overplotting (all photo-refcat sources
            # in bbox, including sources rejected by the donut selector).
            if astrom_cfg.get("saveDiagnosticPlot", True):
                with np.errstate(invalid="ignore", divide="ignore"):
                    _ps1_flux = np.array(refCat[f"{filterName}_flux"])
                    _ps1_mag = -2.5 * np.log10(_ps1_flux) + 31.4
                all_photo = (
                    np.array(refCat["centroid_x"]),
                    np.array(refCat["centroid_y"]),
                    _ps1_mag,
                )
            else:
                all_photo = None
        except Exception as e:
            cat_select_error = str(e)
            catalog_centroids = None
            all_photo = None
    else:
        all_photo = None

    # Build astrom-refcat lookup arrays for overplotting (centroids via refitted WCS).
    if astrom_load_result is not None and astrom_cfg.get("saveDiagnosticPlot", True):
        try:
            _astrom_cat = astrom_load_result.refCat.copy(deep=True)
            afwTable.updateRefCentroids(refitted_wcs, _astrom_cat)
            _astrom_flux_field = f"{astrom_cfg['astromRefFilter']}_flux"
            with np.errstate(invalid="ignore", divide="ignore"):
                _astrom_flux = np.array(_astrom_cat[_astrom_flux_field])
                _astrom_mag = -2.5 * np.log10(_astrom_flux) + 31.4
            all_astrom = (
                np.array(_astrom_cat["centroid_x"]),
                np.array(_astrom_cat["centroid_y"]),
                _astrom_mag,
            )
        except Exception:
            all_astrom = None
    else:
        all_astrom = None
    t3 = time.perf_counter()

    t_stamp0 = time.perf_counter()
    # --- Stamp cutting ---
    arr = postIsr.image.array
    mask_arr = postIsr.mask.array
    sat_bit = postIsr.mask.getPlaneBitMask("SAT")
    half = stampSize // 2

    # Selector-rejected photo-refcat sources: cut stamps for up to 2 brightest for display.
    # Only needed when the diagnostic plot is enabled.
    REJECTED_DISP = 2
    sel_rejected_centroids = None  # (centroid_x, centroid_y, flux_arr, source_ids)
    if astrom_cfg.get("saveDiagnosticPlot", True) and sel_rejected_refcat is not None:
        try:
            _rrej = sel_rejected_refcat
            if len(_rrej) > 0:
                _rrej_flux = np.array(_rrej[f"{filterName}_flux"])
                _rrej_order = np.argsort(_rrej_flux)[::-1][:REJECTED_DISP]
                sel_rejected_centroids = (
                    np.array(_rrej["centroid_x"])[_rrej_order],
                    np.array(_rrej["centroid_y"])[_rrej_order],
                    _rrej_flux[_rrej_order],
                    np.array(_rrej["id"])[_rrej_order],
                )
        except Exception:
            pass

    if catalog_centroids is not None:
        centroid_x, centroid_y, flux_arr, source_ids = catalog_centroids
        # Sort by flux descending, cap at maxDonuts
        order = np.argsort(flux_arr)[::-1][:maxDonuts]
        centroid_x = centroid_x[order]
        centroid_y = centroid_y[order]
        flux_arr = flux_arr[order]
        source_ids = source_ids[order]
        use_catalog = True
    else:
        # Fall back to blind detections; apply donut-quality cuts now.
        if len(blindDetections) == 0:
            return [], [], scatter_arcsec, t1 - t0, t3 - t2, 0.0, wcs_refit_error, cat_select_error, 0
        with np.errstate(invalid="ignore", divide="ignore"):
            innerOk = np.abs(blindDetections["inner_flux"] / blindDetections["flux"]) < detect_cfg["innerFracThreshold"]
            outerOk = np.abs(blindDetections["outer_flux"] / blindDetections["flux"]) < detect_cfg["outerFracThreshold"]
        blindDetections = blindDetections[innerOk & outerOk]
        if len(blindDetections) == 0:
            return [], [], scatter_arcsec, t1 - t0, t3 - t2, 0.0, wcs_refit_error, cat_select_error, 0
        fluxArr = np.array(blindDetections["flux"])
        order = np.argsort(fluxArr)[::-1][:maxDonuts]
        centroid_x = np.array(blindDetections["centroid_x"])[order]
        centroid_y = np.array(blindDetections["centroid_y"])[order]
        flux_arr = fluxArr[order]
        source_ids = np.zeros(len(order), dtype=np.int64)
        use_catalog = False

    # Convert catalog pixel coords through refitted WCS to get field angles
    fieldXY = detector.transform(
        [lsst.geom.Point2D(x, y) for x, y in zip(centroid_x, centroid_y)],
        PIXELS,
        FIELD_ANGLE,
    )

    # Precompute annular masks for inner/outer flux measurement on postISR stamps.
    # These are the same geometry as _measureFlux but applied to the postISR image.
    _mhalf = int(donutRadius * 1.4)
    _gy, _gx = np.mgrid[-_mhalf:_mhalf + 1, -_mhalf:_mhalf + 1]
    _r = np.hypot(_gx, _gy)
    _main_mask = (_r < donutRadius * 1.05) & (_r > donutRadius * obscuration)
    _inner_mask = _r < donutRadius * obscuration * 0.67
    _outer_mask = (_r > donutRadius * 1.25) & (_r < donutRadius * 1.4)

    # Stamp-sized annular masks for SNR (constant per sensor, hoisted out of loop).
    _sgy, _sgx = np.mgrid[-half:half, -half:half]
    _sr = np.hypot(_sgx, _sgy)
    _s_main = (_sr < donutRadius * 1.05) & (_sr > donutRadius * obscuration)
    _s_bkg = (_sr < donutRadius * obscuration * 0.67) | ((_sr > donutRadius * 1.25) & (_sr < donutRadius * 1.4))
    _s_n_main = int(np.sum(_s_main))

    def _cut_stamp_dict(cx_f, cy_f, flux_val, source_id_val, reject_reason=None):
        """Cut one stamp and compute metrics. Returns dict or None on failure."""
        cx, cy = int(round(float(cx_f))), int(round(float(cy_f)))
        rmin, rmax = cy - half, cy + half
        cmin, cmax = cx - half, cx + half
        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            return None
        saturated = bool(np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit))
        if saturated and reject_reason is None:
            return None  # accepted path skips; caller handles counting
        stamp = np.array(arr[rmin:rmax, cmin:cmax])
        stamp_ccs = np.rot90(stamp, k=n_quarter).T

        mmin_r, mmax_r = cy - _mhalf, cy + _mhalf + 1
        mmin_c, mmax_c = cx - _mhalf, cx + _mhalf + 1
        with np.errstate(invalid="ignore", divide="ignore"):
            if mmin_r >= 0 and mmax_r <= arr.shape[0] and mmin_c >= 0 and mmax_c <= arr.shape[1]:
                mpatch = arr[mmin_r:mmax_r, mmin_c:mmax_c]
                bkg = float(np.nanmedian(mpatch[_inner_mask | _outer_mask]))
                mpatch_sub = mpatch - bkg
                mflux = float(np.sum(mpatch_sub[_main_mask]))
                inner_frac = float(np.sum(mpatch_sub[_inner_mask]) / mflux) if mflux != 0 else float("nan")
                outer_frac = float(np.sum(mpatch_sub[_outer_mask]) / mflux) if mflux != 0 else float("nan")
            else:
                inner_frac = outer_frac = float("nan")

        with np.errstate(invalid="ignore", divide="ignore"):
            _s_bkg_pix = stamp[_s_bkg]
            _s_bkg_std = float(np.nanstd(_s_bkg_pix)) if np.any(_s_bkg) else float("nan")
            _s_bkg_med = float(np.nanmedian(_s_bkg_pix)) if np.any(_s_bkg) else 0.0
            _s_signal = float(np.sum((stamp - _s_bkg_med)[_s_main]))
            stamp_snr = _s_signal / (_s_bkg_std * np.sqrt(_s_n_main)) if _s_bkg_std > 0 and _s_n_main > 0 else float("nan")

        def _nearby(cat_tuple):
            if cat_tuple is None:
                return []
            cat_x, cat_y, cat_mag = cat_tuple
            return [(float(sx) - float(cx_f), float(sy) - float(cy_f), float(sm))
                    for sx, sy, sm in zip(cat_x, cat_y, cat_mag)
                    if abs(float(sx) - float(cx_f)) <= half and abs(float(sy) - float(cy_f)) <= half]

        _fa = detector.transform([lsst.geom.Point2D(float(cx_f), float(cy_f))], PIXELS, FIELD_ANGLE)[0]
        _field_dist_deg = np.degrees(np.hypot(_fa[0], _fa[1]))
        if reject_reason is None and _field_dist_deg > detect_cfg["maxFieldDist"]:
            return None
        return dict(
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
            snr=stamp_snr,
            n_quarter=n_quarter,
            nearby_photo=_nearby(all_photo),
            nearby_astrom=_nearby(all_astrom),
            reject_reason=reject_reason,
            saturated=saturated,
        )

    inner_thr = detect_cfg["innerFracThreshold"]
    outer_thr = detect_cfg["outerFracThreshold"]
    donuts = []
    rejected_donuts_pre = []  # flux_frac rejects collected during accepted-stamp loop
    sat_rejected = 0
    flux_frac_rejected = 0
    for i, (cx_f, cy_f) in enumerate(zip(centroid_x, centroid_y)):
        cx, cy = int(round(float(cx_f))), int(round(float(cy_f)))
        rmin, rmax = cy - half, cy + half
        cmin, cmax = cx - half, cx + half
        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            continue
        if np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit):
            sat_rejected += 1
            continue
        d = _cut_stamp_dict(cx_f, cy_f, flux_arr[i], source_ids[i])
        if d is None:
            continue
        if (np.isfinite(d["inner_frac"]) and abs(d["inner_frac"]) > inner_thr) or \
                (np.isfinite(d["outer_frac"]) and abs(d["outer_frac"]) > outer_thr):
            d["reject_reason"] = "flux_frac"
            flux_frac_rejected += 1
            rejected_donuts_pre.append(d)
            continue
        donuts.append(d)

    # Rejected donuts for display: flux_frac-rejected + SAT-rejected +
    # selector-rejected PS1 sources, all sorted bright-to-faint, capped at REJECTED_DISP.
    rejected_donuts = list(rejected_donuts_pre)
    for i, (cx_f, cy_f) in enumerate(zip(centroid_x, centroid_y)):
        cx, cy = int(round(float(cx_f))), int(round(float(cy_f)))
        rmin, rmax = cy - half, cy + half
        cmin, cmax = cx - half, cx + half
        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            continue
        if np.any(mask_arr[rmin:rmax, cmin:cmax] & sat_bit):
            d = _cut_stamp_dict(cx_f, cy_f, flux_arr[i], source_ids[i], reject_reason="SAT")
            if d is not None:
                rejected_donuts.append(d)
    if sel_rejected_centroids is not None:
        rrej_x, rrej_y, rrej_flux, rrej_ids = sel_rejected_centroids
        for cx_f, cy_f, flux_val, sid in zip(rrej_x, rrej_y, rrej_flux, rrej_ids):
            d = _cut_stamp_dict(cx_f, cy_f, flux_val, sid, reject_reason="selector")
            if d is not None:
                rejected_donuts.append(d)
    # Sort all rejected bright-to-faint, keep top REJECTED_DISP.
    rejected_donuts.sort(key=lambda d: d["flux"], reverse=True)
    rejected_donuts = rejected_donuts[:REJECTED_DISP]

    t_stamp1 = time.perf_counter()
    return donuts, rejected_donuts, scatter_arcsec, t1 - t0, t3 - t2, t_stamp1 - t_stamp0, wcs_refit_error, cat_select_error, sat_rejected, flux_frac_rejected


def _getStamps(sensor_name: str, t_dispatch: float) -> dict:
    """Run ISR, blind detection, WCS refit, catalog selection, stamp cutting."""
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
    donutRadius = instrument.donutRadius

    if donutRadius < 5:
        return {
            "sensor": sensor_name, "catalog": [],
            "dispatch_to_arrival": t_arrival - t_dispatch,
            "task_init": t1 - t0, "isr_run": t2 - t1,
            "blind_detect_run": 0.0, "wcs_refit_run": 0.0,
            "catalog_select_run": 0.0, "stamp_cut_run": 0.0,
            "rejected_catalog": [], "scatter_arcsec": None,
            "wcs_refit_error": None, "cat_select_error": None,
            "sat_rejected": 0, "flux_frac_rejected": 0,
        }

    blindDetections = _blindDetect(
        postIsr, detect_cfg, bkg_config, donutRadius, instrument.obscuration,
    )
    t3 = time.perf_counter()

    donuts, rejected_donuts, scatter_arcsec, t_wcs, t_select, t_stamp_cut, wcs_err, cat_err, sat_rejected, flux_frac_rejected = _refitWcsAndSelect(
        postIsr,
        blindDetections,
        _CALIB_STORE["astrom_cfg"],
        sensor_name,
        donutRadius,
        instrument.obscuration,
    )
    t4 = time.perf_counter()

    return {
        "sensor": sensor_name,
        "catalog": donuts,
        "dispatch_to_arrival": t_arrival - t_dispatch,
        "task_init": t1 - t0,
        "isr_run": t2 - t1,
        "blind_detect_run": t3 - t2,
        "wcs_refit_run": t_wcs,
        "catalog_select_run": t_select,
        "stamp_cut_run": t_stamp_cut,
        "rejected_catalog": rejected_donuts,
        "scatter_arcsec": scatter_arcsec,
        "wcs_refit_error": wcs_err,
        "cat_select_error": cat_err,
        "sat_rejected": sat_rejected,
        "flux_frac_rejected": flux_frac_rejected,
    }


def _run_stamp_worker(args: tuple) -> dict:
    sensor_name, t_dispatch = args
    return _getStamps(sensor_name, t_dispatch)


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
    astromRefCat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog for WCS fitting.",
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
    )
    photoRefCat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog for donut selection.",
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
    )

    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """Adjust quanta to only keep calibs for raws that are present."""



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
        doc=(
            "Integer factor by which to bin the image before running the "
            "cross-correlation detection step."
        ),
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
    stampSize: pexConfig.Field = pexConfig.Field(
        doc="Side length in pixels of the square stamp cut around each donut centroid.",
        dtype=int,
        default=160,
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
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether to run DonutSourceSelectorTask after catalog selection.",
        dtype=bool,
        default=True,
    )
    astromRefFilter: pexConfig.Field = pexConfig.Field(
        doc="Filter name to use when querying the astrometry reference catalog.",
        dtype=str,
        default="phot_g_mean",
    )
    photoRefFilter: pexConfig.Field = pexConfig.Field(
        doc=(
            "Explicit filter name to use in photometry reference catalog "
            "(e.g. 'phot_g_mean'). Overrides photoRefFilterPrefix when set."
        ),
        dtype=str,
        optional=True,
    )
    photoRefFilterPrefix: pexConfig.Field = pexConfig.Field(
        doc=(
            "Filter prefix for the photometry reference catalog. "
            "Combined with the exposure band label as '{prefix}_{band}' "
            "(e.g. 'monster_ComCam' → 'monster_ComCam_g'). "
            "Used when photoRefFilter is not set."
        ),
        dtype=str,
        default="monster_ComCam",
    )
    catalogFilterList: pexConfig.ListField = pexConfig.ListField(
        dtype=str,
        doc="Filters from the photometry reference catalog to include in the donut catalog.",
        default=[
            "phot_g_mean", "phot_bp_mean", "phot_rp_mean",
            "monster_ComCam_u", "monster_ComCam_g", "monster_ComCam_r",
            "monster_ComCam_i", "monster_ComCam_z", "monster_ComCam_y",
        ],
    )
    saveDiagnosticPlot: pexConfig.Field = pexConfig.Field(
        doc=(
            "Save a diagnostic PNG summarising each visit. "
            "Disable in production to skip plot-only computation "
            "(refcat overplot lookups, rejected-stamp cutting)."
        ),
        dtype=bool,
        default=True,
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
        self.makeSubtask("isrTask")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("astromTask")
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")
        self._pool: mp.Pool | None = None
        self._pool_size: int = 0

    def __del__(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def _get_pool(self, numCores: int) -> mp.Pool:
        if self._pool is None or self._pool_size != numCores:
            if self._pool is not None:
                self._pool.terminate()
            self._pool = mp.Pool(processes=numCores)
            self._pool_size = numCores
        return self._pool

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        raw_det_ids = {ref.dataId["detector"] for ref in inputRefs.raws}
        inputRefs.ptc = [r for r in inputRefs.ptc if r.dataId["detector"] in raw_det_ids]
        inputRefs.flat = [r for r in inputRefs.flat if r.dataId["detector"] in raw_det_ids]
        inputRefs.linearizer = [r for r in inputRefs.linearizer if r.dataId["detector"] in raw_det_ids]
        inputRefs.crosstalk = [r for r in inputRefs.crosstalk if r.dataId["detector"] in raw_det_ids]

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
        astromRefCat = butlerQC.get(inputRefs.astromRefCat)
        t7 = time.perf_counter()
        photoRefCat = butlerQC.get(inputRefs.photoRefCat)
        t8 = time.perf_counter()
        self.log.info(
            "butlerQC.get timing: raws=%.3fs camera=%.3fs ptc=%.3fs flat=%.3fs"
            " linearizer=%.3fs crosstalk=%.3fs astromRefCat=%.3fs photoRefCat=%.3fs"
            " total=%.3fs",
            t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t8-t0,
        )
        outputs = self.run(
            raws=raws, camera=camera, ptc=ptc, flat=flat,
            linearizer=linearizer, crosstalk=crosstalk,
            astromRefCat=astromRefCat, photoRefCat=photoRefCat,
            numCores=butlerQC.resources.num_cores,
        )
        t9 = time.perf_counter()
        self.log.info("run() execution: %.3fs", t9 - t8)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(
        self,
        raws: list,
        camera: Camera,
        ptc: list,
        flat: list,
        linearizer: list,
        crosstalk: list,
        astromRefCat: list,
        photoRefCat: list,
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
        astromRefCat : list of DeferredDatasetHandle or SimpleCatalog
            Shards for WCS fitting.
        photoRefCat : list of DeferredDatasetHandle or SimpleCatalog
            Shards for donut selection.
        numCores : int
        """
        t_run0 = time.perf_counter()
        rawByName = {exp.getDetector().getName(): exp for exp in raws}
        ptcByName = {p._detectorName: p for p in ptc}
        flatByName = {f.getDetector().getName(): f for f in flat}
        linearizerByName = {lin._detectorName: lin for lin in linearizer}
        crosstalkByName = {ct._detectorName: ct for ct in crosstalk}

        missing = CORNER_SENSOR_NAMES - rawByName.keys()
        if missing:
            raise RuntimeError(f"Missing corner sensor raws: {sorted(missing)}")

        detect_cfg = dict(
            instConfigFile=self.config.instConfigFile,
            edgeMargin=self.config.edgeMargin,
            detectionBinning=self.config.detectionBinning,
            peakMinDistanceFactor=self.config.peakMinDistanceFactor,
            peakExcludeBorderFactor=self.config.peakExcludeBorderFactor,
            innerFracThreshold=self.config.innerFracThreshold,
            outerFracThreshold=self.config.outerFracThreshold,
            snrThreshold=self.config.snrThreshold,
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

        astrom_handles = list(astromRefCat)
        photo_handles = list(photoRefCat)

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

        astrom_fetched: list = []
        photo_fetched: list = []

        def _make_loader(handles, any_filter_maps_to=None, counter=None):
            if not handles:
                return None
            wrapped = [_CountingHandle(h, counter) for h in handles] if counter is not None else handles
            loader = ReferenceObjectLoader(
                dataIds=[h.dataId for h in handles],
                refCats=wrapped,
            )
            loader.config.pixelMargin = 300  # extra tolerance for uncertain WCS
            if any_filter_maps_to is not None:
                loader.config.anyFilterMapsToThis = any_filter_maps_to
            return loader

        t_setup = time.perf_counter()
        self.log.info("run() setup (dicts, detect_cfg): %.3fs", t_setup - t_run0)

        astrom_loader = _make_loader(astrom_handles, self.config.astromRefFilter, astrom_fetched)
        photo_loader = _make_loader(photo_handles, counter=photo_fetched)

        t_refcat0 = time.perf_counter()
        sensor_refcats: dict = {}
        for name, raw in rawByName.items():
            raw_wcs = raw.getWcs()
            raw_bbox = raw.getBBox()
            raw_epoch = raw.getInfo().getVisitInfo().date.toAstropy()
            astrom_load = None
            if astrom_loader is not None:
                try:
                    astrom_load = astrom_loader.loadPixelBox(
                        bbox=raw_bbox, wcs=raw_wcs,
                        filterName=self.config.astromRefFilter, epoch=raw_epoch,
                    )
                except Exception as exc:
                    self.log.warning("Failed to load astrom refcat for %s: %s", name, exc)
            photo_load = None
            if photo_loader is not None:
                try:
                    photo_load = photo_loader.loadPixelBox(
                        bbox=raw_bbox, wcs=raw_wcs,
                        filterName=photo_filter_name, epoch=raw_epoch,
                    )
                except Exception as exc:
                    self.log.warning("Failed to load photo refcat for %s: %s", name, exc)
            sensor_refcats[name] = dict(astrom=astrom_load, photo=photo_load)
        self.log.info(
            "Refcat load (loadPixelBox): astrom=%d/%d shards  photo=%d/%d shards  (%.3fs)",
            len(astrom_fetched), len(astrom_handles),
            len(photo_fetched), len(photo_handles),
            time.perf_counter() - t_refcat0,
        )

        # Stub loader: AstrometryTask.solve() calls refObjLoader.getMetadataBox()
        # unconditionally even when load_result is pre-supplied. That method is
        # pure geometry and never accesses catalog data or dataId.region.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.anyFilterMapsToThis = self.config.astromRefFilter
        astrom_stub_loader.config.pixelMargin = 0

        astrom_cfg = dict(
            astrom_task_config=self.astromTask.config,
            astrom_ref_obj_loader=astrom_stub_loader,
            detect_cfg=detect_cfg,
            maxFitScatter=self.config.maxFitScatter,
            minSourcesForWcsFit=self.config.minSourcesForWcsFit,
            doDonutSelection=self.config.doDonutSelection,
            donut_selector_config=self.donutSelector.config if self.config.doDonutSelection else None,
            astromRefFilter=self.config.astromRefFilter,
            photoRefFilter=self.config.photoRefFilter,
            photoRefFilterPrefix=self.config.photoRefFilterPrefix,
            resolvedPhotoFilterName=photo_filter_name,
            catalogFilterList=list(self.config.catalogFilterList),
            saveDiagnosticPlot=self.config.saveDiagnosticPlot,
        )

        t_calib_store0 = time.perf_counter()
        _CALIB_STORE.clear()
        _CALIB_STORE["isr_config"] = self.isrTask.config
        _CALIB_STORE["bkg_config"] = self.subtractBackground.config
        _CALIB_STORE["camera"] = camera
        _CALIB_STORE["detect_cfg"] = detect_cfg
        _CALIB_STORE["astrom_cfg"] = astrom_cfg
        _CALIB_STORE["sensor_refcats"] = sensor_refcats
        for name in CORNER_SENSOR_NAMES:
            _CALIB_STORE[name] = dict(
                raw=rawByName[name],
                ptc=ptcByName[name],
                flat=flatByName[name],
                linearizer=linearizerByName[name],
                crosstalk=crosstalkByName[name],
            )
        t_calib_store1 = time.perf_counter()
        self.log.info("_CALIB_STORE population: %.3fs", t_calib_store1 - t_calib_store0)

        isr_args = sorted(CORNER_SENSOR_NAMES)

        self.log.info(
            "Running ISR+WCS+selection on %d corner sensors with %d core(s)",
            len(isr_args), numCores,
        )
        if numCores == 1:
            t_dispatch = time.time()
            results = [_run_stamp_worker((arg, t_dispatch)) for arg in isr_args]
        else:
            t_pool0 = time.perf_counter()
            pool = self._get_pool(numCores)
            t_pool1 = time.perf_counter()
            t_dispatch = time.time()
            results = pool.map(_run_stamp_worker, [(arg, t_dispatch) for arg in isr_args])
            t_pool2 = time.perf_counter()
            self.log.info(
                "Pool get: %.3fs, pool.map: %.3fs", t_pool1 - t_pool0, t_pool2 - t_pool1
            )

        donuts = []
        for r in results:
            scatter_str = f"{r['scatter_arcsec']:.3f}\"" if r["scatter_arcsec"] is not None else "N/A"
            self.log.info(
                "  %s: dispatch_to_arrival=%.3fs  task_init=%.3fs  isr=%.3fs"
                "  blind_detect=%.3fs  wcs_refit=%.3fs (scatter=%s)"
                "  cat_select=%.3fs  stamp_cut=%.3fs  donuts=%d  sat_rej=%d  frac_rej=%d",
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
                r.get("sat_rejected", 0),
                r.get("flux_frac_rejected", 0),
            )
            if r["wcs_refit_error"]:
                self.log.warning("  %s: WCS refit failed: %s", r["sensor"], r["wcs_refit_error"])
            if r["cat_select_error"]:
                self.log.warning("  %s: catalog selection failed: %s", r["sensor"], r["cat_select_error"])
            donuts.extend(r["catalog"])

        t_plot0 = time.perf_counter()
        self.log.info("run() pre-plot elapsed: %.1fs", t_plot0 - t_run0)
        if self.config.saveDiagnosticPlot:
            self._saveDiagnosticPlots(
                results,
                run_elapsed=t_plot0 - t_run0,
                photo_filter_name=photo_filter_name,
                astrom_filter_name=self.config.astromRefFilter,
            )
            self.log.info("Diagnostic plot: %.3fs", time.perf_counter() - t_plot0)
        return pipeBase.Struct(donuts=donuts)

    def _saveDiagnosticPlots(self, results: list, run_elapsed: float = 0.0,
                             photo_filter_name: str = "photo", astrom_filter_name: str = "astrom") -> None:
        """Save a single diagnostic PNG with one section per sensor.

        Layout per sensor:
          - Left column: stats text (timing, WCS scatter, donut count, errors)
          - Remaining columns: donut stamps (up to maxDonuts), each annotated
            with flux and field angle
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        STAMPS_PER_ROW = 8   # max accepted stamps per sensor row
        REJECTED_PER_ROW = 2 # rejected stamp columns (right side, separated by spacer)
        STAMP_COL_W = 1.8    # inches per stamp column
        STATS_COL_W = 2.8    # inches for the stats text column
        ROW_H = 2.2          # inches per sensor row

        active = [r for r in results if r["catalog"] or r.get("rejected_catalog")]
        n_sensors = len(active)
        if n_sensors == 0:
            return

        LEGEND_H = 0.35      # inches for the legend strip at the bottom
        SPACER_W = 0.15      # narrow spacer column between accepted and rejected

        # Total columns: stats | accepted(8) | spacer | rejected(2)
        N_COLS = 1 + STAMPS_PER_ROW + 1 + REJECTED_PER_ROW
        fig_w = STATS_COL_W + (STAMPS_PER_ROW + REJECTED_PER_ROW) * STAMP_COL_W + SPACER_W
        fig_h = n_sensors * ROW_H + LEGEND_H + 0.25

        visit_ids = {d["visit_id"] for r in active for d in r["catalog"]}
        visit_str = ", ".join(str(v) for v in sorted(visit_ids))

        photo_filter_label = photo_filter_name  # resolved name, passed in
        astrom_filter_label = astrom_filter_name

        fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
        fig.suptitle(
            f"DonutBlitz diagnostics  visit={visit_str}  run={run_elapsed:.1f}s",
            fontsize=9,
        )

        # width_ratios: stats | accepted(8×1) | spacer | rejected(2×1)
        w_stats = STATS_COL_W / STAMP_COL_W
        w_spacer = SPACER_W / STAMP_COL_W
        gs = GridSpec(
            n_sensors + 1, N_COLS,
            figure=fig,
            height_ratios=[ROW_H] * n_sensors + [LEGEND_H],
            width_ratios=[w_stats] + [1] * STAMPS_PER_ROW + [w_spacer] + [1] * REJECTED_PER_ROW,
            hspace=0.05,
            wspace=0.05,
        )
        # Column index helpers
        COL_ACCEPTED_START = 1
        COL_SPACER = 1 + STAMPS_PER_ROW
        COL_REJECTED_START = COL_SPACER + 1

        for row_idx, r in enumerate(active):
            sensor = r["sensor"]
            catalog = r["catalog"]
            scatter_str = f"{r['scatter_arcsec']:.3f}\"" if r["scatter_arcsec"] is not None else "N/A"

            # Stats panel
            ax_stats = fig.add_subplot(gs[row_idx, 0])
            ax_stats.axis("off")
            lines = [
                f"{sensor}",
                f"donuts: {len(catalog)}  sat_rej: {r.get('sat_rejected', 0)}  frac_rej: {r.get('flux_frac_rejected', 0)}",
                f"isr:    {r['isr_run']:.2f}s",
                f"detect: {r['blind_detect_run']:.2f}s",
                f"wcs:    {r['wcs_refit_run']:.2f}s  ({scatter_str})",
                f"select: {r['catalog_select_run']:.2f}s",
            ]
            if r["wcs_refit_error"]:
                lines.append(f"WCS ERR: {r['wcs_refit_error'][:40]}")
            if r["cat_select_error"]:
                lines.append(f"CAT ERR: {r['cat_select_error'][:40]}")
            ax_stats.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=6, va="top", family="monospace",
            )

            def _draw_stamp(ax, donut, rejected=False):
                stamp = donut["stamp"]
                h_px = stamp.shape[0] // 2
                vmin, vmax = np.nanpercentile(stamp, [1, 99])
                ax.imshow(stamp, origin="lower", vmin=vmin, vmax=vmax,
                          cmap="gray", aspect="equal",
                          extent=[-h_px, h_px, -h_px, h_px])

                if rejected:
                    ax.plot([-h_px, h_px], [-h_px, h_px], color="red", lw=1.5,
                            transform=ax.transData, zorder=5)
                    ax.plot([-h_px, h_px], [h_px, -h_px], color="red", lw=1.5,
                            transform=ax.transData, zorder=5)

                nq = donut.get("n_quarter", 0)
                def _xform(dx, dy):
                    r, c = dy, dx
                    for _ in range(nq % 4):
                        r, c = -c, r
                    return c, r

                for dx, dy, mag in donut.get("nearby_photo", []):
                    tx, ty = _xform(dx, dy)
                    ax.plot(tx, ty, "o", ms=6, mfc="none", mec="cyan", mew=0.8, zorder=3)
                    if np.isfinite(mag):
                        ax.text(tx + 2, ty + 2, f"{mag:.1f}", color="cyan",
                                fontsize=3.5, zorder=4)
                for dx, dy, mag in donut.get("nearby_astrom", []):
                    tx, ty = _xform(dx, dy)
                    ax.plot(tx, ty, "+", ms=6, mec="red", mew=0.8, zorder=3)
                    if np.isfinite(mag):
                        ax.text(tx + 2, ty - 4, f"{mag:.1f}", color="red",
                                fontsize=3.5, zorder=4)

                inner_frac = donut.get("inner_frac", float("nan"))
                outer_frac = donut.get("outer_frac", float("nan"))
                snr = donut.get("snr", float("nan"))
                if_str = f"if={inner_frac:.3f}" if np.isfinite(inner_frac) else "if=?"
                of_str = f"of={outer_frac:.3f}" if np.isfinite(outer_frac) else "of=?"
                snr_str = f"snr={snr:.0f}" if np.isfinite(snr) else "snr=?"
                sid = donut.get("source_id", 0)
                sid_str = f"id={sid}" if sid != 0 else ""
                rej_str = f"[{donut['reject_reason']}]" if donut.get("reject_reason") else ""
                ax.set_title(
                    f"{donut['flux']:.0f}  {snr_str}  {rej_str}\n"
                    f"({donut['fa_x_ccs']:.3f},{donut['fa_y_ccs']:.3f})\n"
                    f"{if_str}  {of_str}\n"
                    f"{sid_str}",
                    fontsize=4,
                    color="orangered" if rejected else "black",
                )

            # Accepted stamp panels
            for col_idx in range(STAMPS_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_ACCEPTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(catalog):
                    continue
                _draw_stamp(ax, catalog[col_idx])

            # Spacer column header (label "rejected" above first rejected stamp)
            ax_sp = fig.add_subplot(gs[row_idx, COL_SPACER])
            ax_sp.axis("off")

            # Rejected stamp panels
            rejected = r.get("rejected_catalog", [])
            for col_idx in range(REJECTED_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_REJECTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(rejected):
                    continue
                _draw_stamp(ax, rejected[col_idx], rejected=True)

        # Legend strip spanning the full figure width.
        ax_legend = fig.add_subplot(gs[n_sensors, :])
        ax_legend.axis("off")
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="cyan", markersize=6, label=f"photo refcat ({photo_filter_label})"),
            Line2D([0], [0], marker="+", color="red", markersize=6, linestyle="none",
                   label=f"astrom refcat ({astrom_filter_label})"),
        ]
        ax_legend.legend(
            handles=legend_handles, loc="center", ncol=2,
            fontsize=7, frameon=False, handletextpad=0.5, columnspacing=2.0,
        )

        fname = "donut_diag.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.log.info("Saved diagnostic plot: %s", fname)
