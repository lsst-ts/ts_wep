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

"""Per-detector cutout pipeline run in the fork-based worker pool.

All inputs come from `lsst.ts.wep.blitz.utils._CALIB_STORE`, which the parent
process populates before forking; nothing here imports the subtasks it runs.
"""

__all__ = []

import logging
import time

import numpy as np
from astropy.table import QTable

import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
from lsst.afw.geom import SkyWcs

from .utils import (
    _ANSI_BOLD,
    _ANSI_YELLOW,
    _CALIB_STORE,
    _colorize,
    _resolveDonutRadius,
)

_log = logging.getLogger(__name__)


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
