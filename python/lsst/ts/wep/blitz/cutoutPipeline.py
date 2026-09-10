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
    _REFCAT_COLUMNS,
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


def _cutout_one_exposure(
    raw,
    calibs: dict,
    refcat_load_result,
    det_name: str,
    maxFitScatter: float,
    astromRefFilter: str,
    photoRefFilter: str,
) -> dict:
    """Run ISR, background subtraction, blind detection, WCS refit, catalog
    selection, and stamp cutting on one exposure of one detector.

    Takes its per-exposure inputs explicitly so both modes can drive it: corner
    mode has one exposure per detector and reads them from ``_CALIB_STORE`` (see
    `_cutoutPipeline`), while full-array mode calls this twice per detector with
    the intra and extra exposures of a pair.

    The *subtasks* are still read from the module-level ``_CALIB_STORE``, which
    the parent populates before forking. That is deliberate: they are identical
    for every call and shared by copy-on-write, so threading them through as
    arguments would buy nothing and cost a pickle.

    Parameters
    ----------
    raw : lsst.afw.image.Exposure
        The raw exposure to process.
    calibs : dict
        Per-detector calibrations, keyed ``ptc``, ``flat``, ``linearizer``,
        ``crosstalk``.
    refcat_load_result
        Pre-loaded reference catalog for this detector, or None to skip the WCS
        refit and refcat-based selection.
    det_name : str
        Detector name, for logging and the returned record.
    maxFitScatter : float
        Maximum acceptable astrometric scatter, in arcseconds, for the refit WCS
        to be used.
    astromRefFilter : str
        Reference catalog flux column prefix used for astrometry.
    photoRefFilter : str
        Reference catalog flux column prefix used for donut selection.

    Returns
    -------
    dict
        Keys: ``det_name``, ``catalog`` (accepted donuts), ``rejected_catalog``,
        ``scatter_arcsec``, ``wcs_refit_error``, ``cat_select_error``,
        ``selection_source``, ``pair_path``, ``n_quarter`` (detector
        orientation), ``wcs`` (the WCS actually used, or
        None), and one timing float per stage, keyed as
        `lsst.ts.wep.blitz.utils._CUTOUT_STAGE_KEYS` lists -- this function is
        where those keys are defined, and everything that reports them takes
        the order from there.  A stage that never ran is NaN, not 0.0.

        ``selection_source`` and ``wcs`` exist because full-array mode has to
        decide how to pair donuts between the two exposures: an exact refcat-id
        match is only available when both exposures selected from the refcat.

        ``pair_path`` is not decided here -- it is the grouping stage's, and both
        modes overwrite it once they know which pairing algorithm ran.  It is
        seeded with ``"n/a"`` rather than left absent so that every result
        reaching `build_donut_catalog` carries a meaningful string, including the
        results of a full-array worker that died before it reached grouping.
    """
    # --- ISR ---
    t0 = time.perf_counter()
    isr_task = _CALIB_STORE["isr_task"]
    postIsr = isr_task.run(
        raw,
        ptc=calibs["ptc"],
        flat=calibs["flat"],
        linearizer=calibs["linearizer"],
        crosstalk=calibs["crosstalk"],
    ).exposure

    # Detector orientation, reported per detector so a consumer can undo the CCS
    # stamp rotation without loading the camera model. Read here because both
    # returns below carry it, including the one that bails out before selection.
    n_quarter = postIsr.getDetector().getOrientation().getNQuarter()

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
            "isr_run": t1 - t0,
            "bkg_run": t2 - t1,
            "diam_run": t3 - t2,
            "blind_detect_run": time.perf_counter() - t3,
            # NaN, not 0.0: these three never ran, and reporting them as zero
            # makes a detector that bailed out here read as one whose WCS refit
            # and selection were instantaneous.
            "wcs_refit_run": float("nan"),
            "catalog_select_run": float("nan"),
            "stamp_cut_run": float("nan"),
            "rejected_catalog": [],
            "scatter_arcsec": None,
            "wcs_refit_error": "No blind detections",
            "cat_select_error": "",
            # No selector ran at all on this detector, which is distinct from the
            # selector running and rejecting everything ("blind_failed").
            "selection_source": "no_detections",
            "pair_path": "n/a",
            "n_quarter": n_quarter,
            "wcs": None,
        }

    # --- astrometry ---
    t4 = time.perf_counter()
    astrom_task = _CALIB_STORE["astrom_task"]
    detector = postIsr.getDetector()
    refcat_handle = refcat_load_result
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
        if scatter_arcsec < maxFitScatter:
            wcs = postIsr.getWcs()
        else:
            wcs_err = f'scatter {scatter_arcsec:.2f}" >= {maxFitScatter}"'
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
            refcat = refcat_handle.refCat.copy(deep=True)
            afwTable.updateRefCentroids(wcs, refcat)
            # Much quicker to just copy the keys we need than convert the whole table to
            # astropy
            keys = [
                "id",
                "coord_ra", "coord_dec",
                "centroid_x", "centroid_y",
                f"{photoRefFilter}_flux", f"{astromRefFilter}_flux"
            ]
            refcat = QTable({k: np.array(refcat[k]) for k in keys})
            # The refcat source id is the donut id from here on: it is what
            # `Donut.donut_id` and the catalog's `donut_id` column carry on the
            # refcat path (blind detection supplies its own 1..N counter under
            # the same name).
            refcat.rename_column("id", "donut_id")
            refcat["photo_flux"] = refcat[f"{photoRefFilter}_flux"]
            refcat["astrom_flux"] = refcat[f"{astromRefFilter}_flux"]
            with np.errstate(invalid="ignore", divide="ignore"):
                refcat["photo_mag"] = -2.5 * np.log10(refcat["photo_flux"]) + 31.4
                refcat["astrom_mag"] = -2.5 * np.log10(refcat["astrom_flux"]) + 31.4
            result = donut_selector.run(refcat, detector, photoRefFilter)
            selections = result.sourceCat
            selection_source = "refcat"
        except Exception as exc:
            cat_err = str(exc)
            refcat = None  # don't leave a partially-built refcat around

    # If the refcat path didn't produce a selection, run the blind detections
    # through the same selector.  If that also fails, then exit gracefully.
    if selection_source != "refcat":
        # Every table reaching `CutDonutStampsTask` carries the refcat-provenance
        # columns, so that a blind-path donut is a row with NaN values rather than
        # a row the consumer has to test the schema for.  Both branches below
        # derive `selections` from `blindDetections` -- the selector returns a row
        # subset, and the failure branch an empty slice -- so filling them here
        # covers both.  Mutating in place is safe: `blindDetections` is built
        # fresh per detector, and this path is the only one that reads it again.
        for column in _REFCAT_COLUMNS:
            blindDetections[column] = np.full(len(blindDetections), np.nan)
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
        donutRadius=donutRadius
    )

    t7 = time.perf_counter()

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Annulus
    # from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS

    # fig, ax = plt.subplots(figsize=(10, 5))
    # vmin, vmax = np.nanquantile(postIsr.image.array, [0.01, 0.99])
    # ax.imshow(postIsr.image.array, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # ax.set_xlim(0, postIsr.image.array.shape[1])
    # ax.set_ylim(0, postIsr.image.array.shape[0])
    # ax.scatter(refcat["centroid_x"], refcat["centroid_y"], s=20, edgecolor="cyan", facecolor="none")
    # ax.scatter(blindDetections["centroid_x"], blindDetections["centroid_y"], s=50, edgecolor="blue", facecolor="none")
    # ax.scatter(selections["centroid_x"], selections["centroid_y"], s=80, edgecolor="red", facecolor="none")
    # ax.scatter(
    #     [d.x_det for d in cut_result.donuts],
    #     [d.y_det for d in cut_result.donuts],
    #     s=110, edgecolor="yellow", facecolor="none"
    # )
    # for d in cut_result.donuts:
    #     ax.annotate(
    #         f"{d.snr:.1f}",
    #         (d.x_det, d.y_det),
    #         xytext=(10, 10),
    #         textcoords="offset points", color="yellow", fontsize=10,
    #         annotation_clip=True,
    #     )
    # for d in cut_result.rejected_donuts:
    #     ax.annotate(
    #         f"{d.snr:.1f}",
    #         (d.x_det, d.y_det),
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
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.suptitle(f"Detector: {det_name}")
    # plt.show()

    return {
        "det_name": det_name,
        "catalog": cut_result.donuts,
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
        "selection_source": selection_source,
        # Overwritten by the grouping stage; see the Returns note above.
        "pair_path": "n/a",
        "n_quarter": n_quarter,
        "wcs": wcs,
    }


def _cutoutPipeline(det_name: str, t_dispatch: float) -> dict:
    """Corner-mode entry point: one exposure per detector, all from _CALIB_STORE.

    Parameters
    ----------
    det_name : str
        Detector name; used to look up the raw and its calibrations in
        ``_CALIB_STORE``.
    t_dispatch : float
        ``time.time()`` timestamp at which the task was dispatched from the
        parent, used to measure dispatch-to-arrival latency.

    Returns
    -------
    dict
        As `_cutout_one_exposure`, plus ``dispatch_to_arrival``.
    """
    t_arrival = time.time()
    entry = _CALIB_STORE[det_name]
    result = _cutout_one_exposure(
        raw=entry["raw"],
        calibs=entry,
        refcat_load_result=_CALIB_STORE["det_refcats"].get(det_name),
        det_name=det_name,
        maxFitScatter=_CALIB_STORE["maxFitScatter"],
        astromRefFilter=_CALIB_STORE["astromRefFilter"],
        photoRefFilter=_CALIB_STORE["photoRefFilter"],
    )
    result["dispatch_to_arrival"] = t_arrival - t_dispatch
    return result


def _run_cutout_worker(args: tuple) -> dict:
    det_name, t_dispatch = args
    return _cutoutPipeline(det_name, t_dispatch)
