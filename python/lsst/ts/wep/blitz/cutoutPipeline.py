# This file is part of ts_wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Per-detector cutout pipeline run in the fork-based worker pool.

All inputs come from `lsst.ts.wep.blitz.utils._COW_STORE`, which the parent
process populates before forking; nothing here imports the subtasks it runs.
"""

__all__ = []

import logging
import time

import numpy as np
from astropy.table import QTable

import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.afw.geom import SkyWcs

from .dataStructures import CutoutResult, DetectorView, SourceSet
from .utils import (
    _ANSI_BOLD,
    _ANSI_YELLOW,
    _COW_STORE,
    _REFCAT_COLUMNS,
    IsrCalibs,
    _colorize,
    _resolve_donut_radius,
)

_log = logging.getLogger(__name__)

# Factor the `DetectorView` image is binned by.
_VIEW_BINNING = 4


def _build_afw_source_cat(blitz_detections: QTable, wcs: SkyWcs) -> afwTable.SourceCatalog:
    """Convert blitz-detect QTable into a minimal afwTable.SourceCatalog
    suitable for AstrometryTask.run().
    """
    schema = afwTable.SourceTable.makeMinimalSchema()
    measBase.SingleFrameMeasurementTask(schema=schema)

    source_cat = afwTable.SourceCatalog(schema)
    centroid_key = afwTable.Point2DKey(schema["slot_Centroid"])
    id_key = schema["id"].asKey()
    ra_key = schema["coord_ra"].asKey()
    dec_key = schema["coord_dec"].asKey()

    # reserve() allocates the records as one block, which is what makes the
    # finished catalog contiguous -- AstrometryTask requires that. Without it,
    # addNew() spills into further blocks past ~100 records and the catalog
    # would need an explicit copy(deep=True) to compact it.
    source_cat.reserve(len(blitz_detections))
    for i, row in enumerate(blitz_detections):
        x, y = row["centroid_x"], row["centroid_y"]
        sky = wcs.pixelToSky(x, y)
        src = source_cat.addNew()
        src.set(id_key, i)
        src.set(ra_key, sky.getRa())
        src.set(dec_key, sky.getDec())
        src.set(centroid_key, lsst.geom.Point2D(x, y))

    return source_cat


def _source_set(table: QTable | None, mag_column: str) -> SourceSet | None:
    """One stage's centroids as a `SourceSet`, or None if the stage never ran.

    Parameters
    ----------
    table : QTable or None
        A stage's output, carrying ``centroid_x``, ``centroid_y`` and
        ``donut_id``.  None propagates to None: that is what distinguishes a
        stage that did not run from one that ran and selected nothing.
    mag_column : str
        Magnitude column to carry, or ``""`` for a stage that has none (blitz
        detection), which yields all-NaN.  A named column that is absent also
        yields all-NaN rather than raising, since the blitz path NaN-fills the
        refcat columns anyway (see `_REFCAT_COLUMNS`).
    """
    if table is None:
        return None
    n = len(table)
    if mag_column and mag_column in table.colnames:
        mag = np.asarray(table[mag_column], dtype=np.float32)
    else:
        mag = np.full(n, np.nan, dtype=np.float32)
    return SourceSet(
        x_det=np.asarray(table["centroid_x"], dtype=float),
        y_det=np.asarray(table["centroid_y"], dtype=float),
        donut_id=np.asarray(table["donut_id"]),
        mag=mag,
    )


def _field_distance_grid(detector, bbox, binning: int, shape: tuple[int, int]) -> np.ndarray:
    """Field distance [deg] at binned pixel centers, shaped like `shape`."""
    try:
        mapping = detector.getTransform(PIXELS, FIELD_ANGLE).getMapping()
        # Binned pixel j covers un-binned [b*j, b*j + b - 1], so its center is
        # at b*j + (b - 1)/2 -- the same half-pixel offset the plot's overlays
        # use.
        offset = (binning - 1) / 2
        ys = np.arange(shape[0]) * binning + offset + bbox.getMinY()
        xs = np.arange(shape[1]) * binning + offset + bbox.getMinX()
        grid_x, grid_y = np.meshgrid(xs, ys)
        field = mapping.applyForward(np.vstack([grid_x.ravel(), grid_y.ravel()]))
        return np.degrees(np.hypot(field[0], field[1])).reshape(shape).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - decoration only; never fail a cutout
        _log.warning("Field-distance grid unavailable for %s: %s", detector.getName(), exc)
        return np.full(shape, np.nan, dtype=np.float32)


def _detector_view(
    post_isr,
    refcat: QTable | None,
    blitz_detections: QTable | None,
    selections: QTable | None,
) -> DetectorView:
    """Bundle one detector's binned pixels and overlay sources for the plot.

    Called at every return of `_cutout_one_exposure` that has an image to
    bundle, including the no-detections one -- a detector that found nothing is
    exactly the one worth looking at.

    The ``.copy()`` matters: without it the array is a view onto the binned
    image's buffer, whose base would be pickled along with it.
    """
    bbox = post_isr.getBBox()
    binned = afwMath.binImage(post_isr.image, _VIEW_BINNING)
    image = binned.getArray().astype(np.float32, copy=True)
    return DetectorView(
        image=image,
        binning=_VIEW_BINNING,
        bbox_min=(bbox.getMinX(), bbox.getMinY()),
        bbox_shape=(bbox.getHeight(), bbox.getWidth()),
        # The refcat is None off the refcat path, and `_source_set` keeps it
        # that way -- an empty overlay would imply a catalog was consulted.
        refcat=_source_set(refcat, "photo_mag"),
        detections=_source_set(blitz_detections, ""),
        selections=_source_set(selections, "photo_mag"),
        field_dist=_field_distance_grid(post_isr.getDetector(), bbox, _VIEW_BINNING, image.shape),
        # Off the live subtask config, not the class default: corner mode
        # overrides maxFieldDist, so the two differ.
        max_field_dist_deg=float(_COW_STORE.select_task.config.maxFieldDist),
    )


def _cutout_one_exposure(
    raw,
    calibs: IsrCalibs,
    refcat_load_result,
    det_name: str,
    max_fit_scatter: float,
    astrom_ref_filter: str,
    photo_ref_filter: str,
) -> CutoutResult:
    """Run ISR, background subtraction, blitz detection, WCS refit, catalog
    selection, and stamp cutting on one exposure of one detector.

    Takes its per-exposure inputs explicitly so both modes can drive it: corner
    mode has one exposure per detector and reads them from ``_COW_STORE``
    (see `_cutout_corner_detector`), while full-array mode calls this twice per
    detector with the intra and extra exposures of a pair.

    The *subtasks* are still read from the module-level ``_COW_STORE``, which
    the parent populates before forking. That is deliberate: they are identical
    for every call and shared by copy-on-write, so threading them through as
    arguments would buy nothing and cost a pickle.

    Parameters
    ----------
    raw : lsst.afw.image.Exposure
        The raw exposure to process.
    calibs : `lsst.ts.wep.blitz.utils.IsrCalibs`
        This detector's materialized calibrations. Corner mode takes them
        straight from the store; a full-array worker resolves its own handles
        into one of these first.
    refcat_load_result
        Pre-loaded reference catalog for this detector, or None to skip the WCS
        refit and refcat-based selection.
    det_name : str
        Detector name, for logging and the returned record.
    max_fit_scatter : float
        Maximum acceptable astrometric scatter, in arcseconds, for the refit
        WCS to be used.
    astrom_ref_filter : str
        Reference catalog flux column prefix used for astrometry.
    photo_ref_filter : str
        Reference catalog flux column prefix used for donut selection.

    Returns
    -------
    `lsst.ts.wep.blitz.dataStructures.CutoutResult`
        This detector's donuts, provenance and per-stage timings.  See that
        class for what each field means.
    """
    # --- ISR ---
    t0 = time.perf_counter()
    isr_task = _COW_STORE.isr_task
    post_isr = isr_task.run(
        raw,
        ptc=calibs.ptc,
        flat=calibs.flat,
        linearizer=calibs.linearizer,
        crosstalk=calibs.crosstalk,
    ).exposure

    # Detector orientation, reported per detector so a consumer can undo the
    # CCS stamp rotation without loading the camera model. Read here because
    # both returns below carry it, including the one that bails out before
    # selection.
    n_quarter = post_isr.getDetector().getOrientation().getNQuarter()

    # --- background subtraction ---
    t1 = time.perf_counter()
    bkg_task = _COW_STORE.bkg_task
    bkg_task.run(exposure=post_isr)

    # --- detect diameter ---
    t2 = time.perf_counter()
    diam_task = _COW_STORE.diam_task
    donut_diameter = diam_task.run(post_isr).diameter
    donut_radius = _resolve_donut_radius(donut_diameter / 2 if donut_diameter is not None else None)

    # --- blitz detection ---
    t3 = time.perf_counter()
    detect_task = _COW_STORE.detect_task
    blitz_detections = detect_task.run(post_isr, donut_radius=donut_radius).detections

    if len(blitz_detections) == 0:
        result = CutoutResult.no_detections(
            det_name=det_name,
            n_quarter=n_quarter,
            isr_run=t1 - t0,
            bkg_run=t2 - t1,
            diam_run=t3 - t2,
            detect_run=time.perf_counter() - t3,
        )
        # Built even here: a detector that detected nothing is the one a reader
        # most wants to see the image of. No selection ran, so only the image
        # and the (empty) detections are meaningful.
        if _COW_STORE.build_detector_view:
            result.view = _detector_view(post_isr, None, blitz_detections, None)
        return result

    # --- astrometry ---
    t4 = time.perf_counter()
    astrom_task = _COW_STORE.astrom_task
    detector = post_isr.getDetector()
    scatter_arcsec = None
    wcs = None
    wcs_err = ""
    try:
        astrom_result = astrom_task.solve(
            exposure=post_isr,
            sourceCat=_build_afw_source_cat(blitz_detections, post_isr.getWcs()),
            load_result=refcat_load_result,
        )
        scatter_arcsec = astrom_result.scatterOnSky.asArcseconds()
        if scatter_arcsec < max_fit_scatter:
            wcs = post_isr.getWcs()
        else:
            wcs_err = f'scatter {scatter_arcsec:.2f}" >= {max_fit_scatter}"'
    except Exception as exc:
        wcs_err = f"astrometry solve failed: {type(exc).__name__}: {exc}"
        _log.warning(
            _colorize(
                "Astrometry solve failed for %s; falling back to blitz detections: %s",
                _ANSI_BOLD,
                _ANSI_YELLOW,
            ),
            det_name,
            wcs_err,
        )

    # --- catalog selection ---
    t5 = time.perf_counter()
    selections = blitz_detections
    refcat = None
    cat_err = ""
    selection_source = None
    select_task = _COW_STORE.select_task

    if wcs is not None:
        try:
            refcat = refcat_load_result.refCat.copy(deep=True)
            afwTable.updateRefCentroids(wcs, refcat)
            # Much quicker to just copy the keys we need than convert the whole
            # table to astropy
            keys = [
                "id",
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                f"{photo_ref_filter}_flux",
                f"{astrom_ref_filter}_flux",
            ]
            refcat = QTable({k: np.array(refcat[k]) for k in keys})
            # The refcat source id is the donut id from here on: it is what
            # `Donut.donut_id` and the catalog's `donut_id` column carry on the
            # refcat path (blitz detection supplies its own 1..N counter under
            # the same name).
            refcat.rename_column("id", "donut_id")
            refcat["photo_flux"] = refcat[f"{photo_ref_filter}_flux"]
            refcat["astrom_flux"] = refcat[f"{astrom_ref_filter}_flux"]
            with np.errstate(invalid="ignore", divide="ignore"):
                refcat["photo_mag"] = -2.5 * np.log10(refcat["photo_flux"]) + 31.4
                refcat["astrom_mag"] = -2.5 * np.log10(refcat["astrom_flux"]) + 31.4
            # `sourceCat` stays camelCase: it is `DonutSourceSelectorTask`'s
            # own Struct field, not a blitz name to convert.
            selections = select_task.run(refcat, detector, photo_ref_filter).sourceCat
            selection_source = "refcat"
        except Exception as exc:
            cat_err = str(exc)
            refcat = None  # don't leave a partially-built refcat around

    # If the refcat path didn't produce a selection, run the blitz detections
    # through the same selector.  If that also fails, then exit gracefully.
    if selection_source != "refcat":
        # Every table reaching `CutDonutStampsTask` carries the
        # refcat-provenance columns, so that a blitz-path donut is a row with
        # NaN values rather than a row the consumer has to test the schema for.
        # Both branches below derive `selections` from `blitz_detections` --
        # the selector returns a row subset, and the failure branch an empty
        # slice -- so filling them here covers both.  Mutating in place is
        # safe: `blitz_detections` is built fresh per detector, and this path
        # is the only one that reads it again.
        for column in _REFCAT_COLUMNS:
            blitz_detections[column] = np.full(len(blitz_detections), np.nan)
        try:
            selections = select_task.run(blitz_detections, detector, "").sourceCat
            selection_source = "blitz_selected"
        except Exception as exc:
            cat_err = cat_err or str(exc)
            _log.warning(
                "Donut selector failed on blitz detections for %s; dropping detector's donuts: %s",
                det_name,
                exc,
            )
            selections = blitz_detections[:0]  # empty; flows through to empty catalog
            selection_source = "blitz_failed"
    _log.info("Donut selection path: %s (%d sources)", selection_source, len(selections))

    # --- stamp cutting ---
    t6 = time.perf_counter()
    measure_task = _COW_STORE.measure_task
    candidates = measure_task.run(
        post_isr,
        selections,
        donut_radius=donut_radius,
    ).measurements
    cut_task = _COW_STORE.cut_task
    cut_result = cut_task.run(post_isr, candidates, refcat, donut_radius=donut_radius)

    t7 = time.perf_counter()

    return CutoutResult(
        det_name=det_name,
        catalog=cut_result.donuts,
        rejected_catalog=cut_result.rejected_donuts,
        isr_run=t1 - t0,
        bkg_run=t2 - t1,
        diam_run=t3 - t2,
        detect_run=t4 - t3,
        wcs_refit_run=t5 - t4,
        catalog_select_run=t6 - t5,
        stamp_cut_run=t7 - t6,
        scatter_arcsec=scatter_arcsec,
        wcs_refit_error=wcs_err,
        cat_select_error=cat_err,
        selection_source=selection_source,
        n_quarter=n_quarter,
        wcs=wcs,
        # `pair_path` keeps its default; the grouping stage overwrites it.
        # `refcat` is None wherever the refcat path did not produce a
        # selection, which is what makes that overlay absent rather than empty.
        view=(
            _detector_view(post_isr, refcat, blitz_detections, selections)
            if _COW_STORE.build_detector_view
            else None
        ),
    )


def _cutout_corner_detector(args: tuple) -> CutoutResult:
    """Corner-mode entry point: one exposure per detector, from _COW_STORE.

    Takes its arguments as one tuple because that is what `_fork_map` hands a
    work unit, and is a module-level function so it is picklable by name.

    Parameters
    ----------
    args : tuple
        ``(det_name, t_dispatch)``.  Only the detector name crosses the pickle
        boundary; the raw and its calibrations are looked up in ``_COW_STORE``.
        ``t_dispatch`` is the ``time.time()`` timestamp at which the unit was
        dispatched from the parent, used to measure dispatch-to-arrival
        latency.

    Returns
    -------
    `lsst.ts.wep.blitz.dataStructures.CutoutResult`
        As `_cutout_one_exposure`, with ``dispatch_to_arrival`` filled in.
    """
    det_name, t_dispatch = args
    t_arrival = time.time()
    entry = _COW_STORE.corner_detectors[det_name]
    result = _cutout_one_exposure(
        raw=entry.raw,
        calibs=entry.calibs,
        refcat_load_result=_COW_STORE.det_refcats.get(det_name),
        det_name=det_name,
        max_fit_scatter=_COW_STORE.max_fit_scatter,
        astrom_ref_filter=_COW_STORE.astrom_ref_filter,
        photo_ref_filter=_COW_STORE.photo_ref_filter,
    )
    result.dispatch_to_arrival = t_arrival - t_dispatch
    return result
