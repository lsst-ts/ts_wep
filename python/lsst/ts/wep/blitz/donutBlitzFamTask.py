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

"""The full-array-mode (FAM) blitz pipeline task.

Where `DonutBlitzMonolithTask` processes one visit's 8 corner wavefront sensors,
this processes the 189 science detectors of an intra/extra **exposure pair** -- one
quantum per ``group``.

To be frugal with memory, the task avoids loading the entire focal plane at once.
So here the parent resolves *deferred* handles only and each forked worker does
its own butler I/O, frees the pixels, and fits.  See `famPipeline` for the worker
and for why children must reset the DB connection pool.
"""

__all__ = [
    "DonutBlitzFamTaskConnections",
    "DonutBlitzFamTaskConfig",
    "DonutBlitzFamTask",
]

import logging
import multiprocessing as mp
import time
from collections import Counter
from typing import Any

import batoid
import numpy as np
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.sphgeom
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import (
    MagnitudeLimit,
    ReferenceObjectLoader,
    SubtractBackgroundTask,
)
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.base import (
    InputQuantizedConnection,
    NoWorkFound,
    OutputQuantizedConnection,
    QuantumContext,
)
from lsst.ts.wep.task.donutDetectDiameterTask import DonutDetectDiameterTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask

from .blindDetectTask import BlindDetect
from .catalogBuilder import CatalogOptions, build_donut_catalog
from .cutDonutStampsTask import CutDonutStampsTask
from .famPipeline import _fam_detector_worker, _fam_pool_initializer
from .measureDonutCandidatesTask import MeasureDonutCandidatesTask
from .utils import (
    _ANSI_BOLD,
    _ANSI_CYAN,
    _ANSI_GREEN,
    _CALIB_STORE,
    _CUTOUT_STAGE_KEYS,
    _INSTRUMENT,
    _colorize,
    _defocal_radial_scale,
    _resolveColorLogEnabled,
    _telescope_for_offsets,
)
from .wavefrontFittingTask import WavefrontFittingTask

# Detector purpose, from the `detector` dimension record. FAM fits the science
# array only: on a +/-1.5 mm pair the corner wavefront sensors sit at 0 and
# -/+3 mm, so half of them are in focus and half are too large for stampSize.
_SCIENCE_PURPOSE = "SCIENCE"

# The refcat lookup function below is called by the graph builder, not by the
# task, so it has no `self.log`.
_log = logging.getLogger(__name__)

# On-sky radius, in degrees, of the circle used to cut the reference catalog down
# to the shards the focal plane can actually reach. Science detector corners reach
# 2.05 deg; the extra covers the loaders' 300 px pixelMargin (~0.02 deg) and any
# pointing error, since over-including a shard costs one wasted file read and
# under-including one would silently lose donuts.
_FOCAL_PLANE_SEARCH_RADIUS_DEG = 2.2

# htm7, matching the reference catalog's own dimension.
_REFCAT_HTM_LEVEL = 7

# Pipeline stages reported per detector and aggregated over them, in the order they
# run: the shared cutout stages, spliced in from `_CUTOUT_STAGE_KEYS` so this
# and corner mode cannot disagree about them, wrapped in the four this task adds.
# `dispatch` and `io` have no corner-mode counterpart: a FAM worker waits for a pool
# slot and then does its own butler reads, where corner mode's parent has already
# loaded every pixel before it forks. `fit` likewise, because FAM fits inside the
# same worker rather than in a second pool the parent summarises separately.
_STAGE_KEYS = (
    "dispatch",
    "io",
    *_CUTOUT_STAGE_KEYS,
    "fit",
    "wall",
)

# How many stragglers to name. With 189 detectors over a handful of cores the slow
# tail sets the wall clock, and the aggregate mean cannot show whether the pool was
# starved at the end or one CCD was pathological.
_N_SLOWEST = 5


def _detector_stage_times(r: dict) -> dict[str, float]:
    """Per-stage elapsed times for one detector, in `_STAGE_KEYS` order.

    A FAM worker runs the cutout pipeline once per exposure, so the seven cutout
    stages are **summed over the pair** -- the two halves are not independently
    interesting, and summing keeps the row comparable to ``wall``.  ``io`` is
    likewise already summed by the worker across the raw reads and both
    per-exposure calibration reads.

    NaN propagates deliberately: a detector that failed part way through has no
    meaningful stage total, and reporting the one exposure that did finish would
    read as a suspiciously fast detector rather than a broken one.

    Parameters
    ----------
    r : dict
        One `_fam_detector_worker` result.

    Returns
    -------
    dict [str, float]
        Seconds per stage, NaN for anything the worker never reached.
    """
    results = r.get("results") or []
    stages = {
        "dispatch": r.get("dispatch_to_arrival", float("nan")),
        "io": r.get("io_run", float("nan")),
        "fit": r.get("fit_run", float("nan")),
        "wall": r.get("worker_wall", float("nan")),
    }
    for label, key in _CUTOUT_STAGE_KEYS.items():
        stages[label] = (
            np.sum([res.get(key, float("nan")) for res in results])
            if results
            else float("nan")
        )
    return {key: stages[key] for key in _STAGE_KEYS}


def _mean_std_max(values: list[float]) -> tuple[float, float, float]:
    """Mean, standard deviation, and max of ``values``, ignoring NaN.

    Returns all-NaN rather than warning when nothing is finite, which is the
    all-detectors-failed case.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return (float("nan"),) * 3
    return (
        np.nanmean(array),
        np.nanstd(array),
        np.nanmax(array),
    )


def _lookup_refcat_shards(datasetType, registry, quantumDataId, collections):
    """Find the reference catalog shards this group's focal plane overlaps.

    A ``PrerequisiteInput`` lookup function, called once per quantum during graph
    generation. It exists because this task's quantum is dimensioned
    ``(instrument, group, physical_filter)`` and **carries no spatial region**:
    ``group`` is what makes the quantum an exposure *pair*, but only
    ``visit``/``exposure`` are spatial, so the default spatial lookup has nothing
    to constrain against and returns all 131072 htm7 shards on the sky instead of
    the ~48 the field actually covers

    Parameters
    ----------
    datasetType : `lsst.daf.butler.DatasetType`
        The reference catalog dataset type, dimensioned ``htm7``.
    registry : `lsst.daf.butler.Registry`
        Registry to query. Available here because this runs at graph-generation
        time, unlike `runQuantum` under ``run-qbb``.
    quantumDataId : `lsst.daf.butler.DataCoordinate`
        The quantum's data ID: instrument, group, physical_filter.
    collections : `~collections.abc.Sequence` [ `str` ]
        Input collections to search.

    Returns
    -------
    refs : `list` [ `lsst.daf.butler.DatasetRef` ]
        The overlapping shards, or every shard if the pointing could not be
        determined -- degrading to the default behavior rather than silently
        dropping the reference catalog.

    Notes
    -----
    The region comes from the boresight on the group's ``exposure`` records plus
    `_FOCAL_PLANE_SEARCH_RADIUS_DEG`, and `lsst.sphgeom.HtmPixelization.envelope`
    turns it straight into shard indices -- so nothing enumerates the 131072.
    """
    instrument = quantumDataId["instrument"]
    group = quantumDataId["group"]

    records = list(
        registry.queryDimensionRecords(
            "exposure",
            where="instrument=:instrument and exposure.group=:group",
            bind={"instrument": instrument, "group": group},
        )
    )
    pixelization = lsst.sphgeom.HtmPixelization(_REFCAT_HTM_LEVEL)
    ranges = lsst.sphgeom.RangeSet()
    for record in records:
        if record.tracking_ra is None or record.tracking_dec is None:
            continue
        ranges = ranges | pixelization.envelope(
            lsst.sphgeom.Circle(
                lsst.sphgeom.UnitVector3d(
                    lsst.sphgeom.LonLat.fromDegrees(
                        record.tracking_ra, record.tracking_dec
                    )
                ),
                lsst.sphgeom.Angle.fromDegrees(_FOCAL_PLANE_SEARCH_RADIUS_DEG),
            )
        )
    shards = [index for begin, end in ranges.ranges() for index in range(begin, end)]

    if not shards:
        _log.warning(
            "group=%s: no boresight on the exposure records, so the reference "
            "catalog could not be narrowed; falling back to every shard.",
            group,
        )
        return list(
            registry.queryDatasets(datasetType, collections=collections, findFirst=True)
        )

    return list(
        registry.queryDatasets(
            datasetType,
            collections=collections,
            where="htm7 in (:shards)",
            bind={"shards": shards},
            findFirst=True,
        )
    )


class DonutBlitzFamTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "group", "physical_filter"),  # type: ignore
):
    """Pipeline connections for DonutBlitzFamTask.

    Every pixel and calibration input is ``deferLoad=True``: the parent resolves
    handles and reads nothing, so its memory stays flat regardless of how many
    detectors the quantum covers.

    Notes
    -----
    ``group`` is what makes the quantum a *pair*: ``exposure`` implies ``group``,
    so a group-dimensioned quantum receives both exposures across all detectors.

    ``physical_filter`` is *required*, not incidental. ``group`` does not imply it
    (only ``exposure``/``visit`` do), so without it the filter-dependent
    prerequisites arrive once per filter -- 8 ``flat`` and 6 ``intrinsicZernikes``
    refs per detector rather than 1.  It also carries ``band`` into the quantum
    data ID for free.

    What it does *not* fix is the reference catalog: ``group`` carries no spatial
    region either, so the htm7 lookup is unconstrained and every shard on the sky
    arrives -- so ``refCat`` carries a `lookupFunction` (`_lookup_refcat_shards`)
    that narrows it at graph-build time, where a registry is available.
    """

    raws = connectionTypes.Input(
        doc=(
            "Raws of both exposures in this group, all detectors.  Deferred: the "
            "workers read them, not the parent."
        ),
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    ptc = connectionTypes.PrerequisiteInput(
        name="ptc",
        storageClass="PhotonTransferCurveDataset",
        doc="Photon transfer curve calibration, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )
    flat = connectionTypes.PrerequisiteInput(
        name="flat",
        storageClass="ExposureF",
        doc="Flat field calibration, one per detector.",
        dimensions=["instrument", "detector", "physical_filter"],
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )
    linearizer = connectionTypes.PrerequisiteInput(
        name="linearizer",
        storageClass="Linearizer",
        doc="Linearity correction, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )
    crosstalk = connectionTypes.PrerequisiteInput(
        name="crosstalk",
        storageClass="CrosstalkCalib",
        doc="Crosstalk coefficients, one per detector.",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )
    refCat = connectionTypes.PrerequisiteInput(
        doc=(
            "Reference catalog for both WCS fitting and donut selection.  Narrowed "
            "to the shards this group's focal plane overlaps by "
            "`_lookup_refcat_shards`, since a group-dimensioned quantum has no "
            "region for the default spatial lookup to use.  Every surviving shard "
            "handle then goes to every worker; `ReferenceObjectLoader` intersects "
            "each shard's region with the detector's search box, so a worker still "
            "reads only the ~2 that overlap it."
        ),
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
        lookupFunction=_lookup_refcat_shards,
    )
    intrinsicZernikes = connectionTypes.PrerequisiteInput(
        doc="Intrinsic Zernike calibration, one per detector.",
        dimensions=("detector", "instrument", "physical_filter"),
        storageClass="IsrCalib",
        name="intrinsicZernikes",
        multiple=True,
        isCalibration=True,
        minimum=0,
        deferLoad=True,
    )
    famResults = connectionTypes.Output(
        doc=(
            "Per-donut catalog for the pair, covering both exposures: selection "
            "metrics, fit results, Zernikes, and optionally stamp/model images.  "
            "Same schema as corner mode's donutBlitzResults, but a separate "
            "dataset type: a dataset type carries one dimension set and one "
            "meaning, and a future analysis of the ~3 mm donuts on the corner "
            "sensors of these same exposures would collide with a shared one."
        ),
        name="donutBlitzFamResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        # The group spans both visits of the pair; only the extra-focal one is
        # written, so this must tolerate an unfilled predicted ref.
        multiple=True,
    )


class DonutBlitzFamTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzFamTaskConnections,  # type: ignore
):
    """Configuration for DonutBlitzFamTask."""

    isrTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="ISR subtask run on each science sensor exposure.",
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
        doc="Blind donut detection subtask run on each exposure.",
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
    wfFittingTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=WavefrontFittingTask,
        doc="Wavefront fitting subtask using Danish algorithm.",
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
            "set."
        ),
        dtype=str,
        default="monster_ComCam",
    )
    detectorOffset: pexConfig.Field = pexConfig.Field(
        doc=(
            "Magnitude of the detector-plane z shift that defocuses each "
            "exposure, in metres.  Signed per exposure: +offset extra-focal, "
            "-offset intra-focal.  Zero by default -- full-array mode defocuses "
            "by moving the whole camera, see cameraOffset."
        ),
        dtype=float,
        default=0.0,
    )
    cameraOffset: pexConfig.Field = pexConfig.Field(
        doc=(
            "Magnitude of the camera z shift that defocuses each exposure, in "
            "metres.  Signed per exposure as detectorOffset is.  This is the "
            "full-array default; the nested detector moves with the camera, so "
            "the sign convention agrees with corner mode's."
        ),
        dtype=float,
        default=_INSTRUMENT.defocalOffset,
    )
    m2Offset: pexConfig.Field = pexConfig.Field(
        doc=(
            "Magnitude of the M2 z shift that defocuses each exposure, in "
            "metres.  Signed per exposure as detectorOffset is.  For data taken "
            "by moving M2 rather than the camera."
        ),
        dtype=float,
        default=0.0,
    )
    pairMatchTolerance: pexConfig.Field = pexConfig.Field(
        doc=(
            "Spatial donut pairing tolerance, as a fraction of the donut radius. "
            "Only used on the fallback path -- when both exposures selected from "
            "the reference catalog, the same star has the same id and is matched "
            "exactly.  Applied after the radial defocus shift is divided out, so "
            "one value is valid across the whole focal plane."
        ),
        dtype=float,
        default=0.25,
    )
    saveStamps: pexConfig.Field = pexConfig.Field(
        doc=(
            "Include the un-binned `stamp` image column in the output catalog. "
            "Off by default: at ~10k rows per pair it is the difference between "
            "a ~30 MB table and a multi-GB one.  Turn it on for pilot runs."
        ),
        dtype=bool,
        default=False,
    )
    saveWfImages: pexConfig.Field = pexConfig.Field(
        doc=(
            "Include the binned `wf_img` and `model_img` columns.  Off by "
            "default for the same reason as saveStamps.  With this on and "
            "saveStamps off, donuts that no fit consumed get a NaN wf_img."
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
        doc=(
            "Wavefront estimation dispatch mode.  Every work unit lives on one "
            "detector, so all four are computed inside that detector's worker."
        ),
        dtype=str,
        allowed={
            "paired": (
                "One star, both sides of focus.  The only mode that associates "
                "donuts between the two exposures, and so the only one where a "
                "donut can end up unmatched."
            ),
            "unpaired": "One star, one side of focus; each donut fit alone.",
            "full_detector": (
                "Every donut on the detector from one exposure, as one work unit: "
                "two per detector."
            ),
            "full_detector_pair": (
                "Every donut on the detector from both exposures, as one joint "
                "work unit.  Does not associate donuts -- each carries its own "
                "defocal offsets, so the fit already knows which side of focus "
                "each one is on."
            ),
        },
        default="paired",
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
        self.donutSelector.useCustomMagLimit = True
        # Load-bearing, not a tuning knob: this is what keeps every selected
        # donut inside the LsstCam.yaml maskParams validity domain
        # (thetaMax: 1.85 deg) even though science detector corners reach
        # 2.05 deg.  Eight detectors lie entirely outside it and yield nothing by
        # design -- 189 detectors are read, 181 produce donuts.
        self.donutSelector.maxFieldDist = 1.725
        self.donutSelector.sourceLimit = 40
        self.donutSelector.allowFluxless = True

        # A science detector is ~2x the area of a corner sensor, plus we're less
        # constrained for time here.
        self.cutStampsTask.maxDonuts = 20

        # 189 detectors x up to 40 groups each is ~10k lines of per-group
        # narration, which buries the per-detector summaries `_runWorkers` logs
        # instead. Those carry the same information in aggregate; failures and
        # timeouts still warn.
        self.wfFittingTask.logPerGroup = False


class DonutBlitzFamTask(pipeBase.PipelineTask):
    """Full-array-mode WEP task: one intra/extra exposure pair, 189 detectors.

    One quantum per ``group``.  `runQuantum` resolves deferred handles, works out
    which exposure is which side of focus, restricts to the science array, and
    forks a pool of workers -- one detector each, both exposures -- that do their
    own butler I/O and their own Danish fits.  Results are assembled into the same
    catalog schema corner mode emits and written against the extra-focal visit.
    """

    ConfigClass = DonutBlitzFamTaskConfig
    _DefaultName = "donutBlitzFamTask"
    config: DonutBlitzFamTaskConfig

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
        self._colorLogEnabled = _resolveColorLogEnabled(self.config.colorLog)

    @property
    def _extraFocalOffsets(self) -> tuple[float, float, float]:
        """Optic z shifts of the extra-focal exposure, ordered as `_OFFSET_OPTICS`."""
        return (
            +self.config.detectorOffset,
            +self.config.cameraOffset,
            +self.config.m2Offset,
        )

    @property
    def _intraFocalOffsets(self) -> tuple[float, float, float]:
        """Optic z shifts of the intra-focal exposure, ordered as `_OFFSET_OPTICS`."""
        # `-o if o else 0.0` rather than plain negation, to keep an unused
        # component as 0.0 instead of -0.0 in logs and store keys.
        return tuple(-o if o else 0.0 for o in self._extraFocalOffsets)  # type: ignore[return-value]

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        t_start = time.perf_counter()
        group = butlerQC.quantum.dataId["group"]

        # --- 1. the group must be a clean intra/extra pair ---
        # Load-bearing: on a typical night most groups are singletons and a
        # handful of non-cwfs groups hold three exposures, so a query that only
        # constrained the day would otherwise reach here with the wrong shape.
        raws_by_exp: dict[int, list] = {}
        for ref in inputRefs.raws:
            raws_by_exp.setdefault(ref.dataId["exposure"], []).append(ref)
        exposures = sorted(raws_by_exp)
        if len(exposures) != 2:
            raise NoWorkFound(
                f"group={group} holds {len(exposures)} exposure(s) {exposures}, "
                "not an intra/extra pair; skipping."
            )

        # --- 2. which exposure is which side of focus ---
        # Ascending exposure id: intra first.  But cross-check per quantum
        # rather than trusting the convention silently.
        intra_exp, extra_exp = exposures
        self._checkDefocalOrder(raws_by_exp, intra_exp, extra_exp, group)

        # --- 3. science array only ---
        # Done task-side so a data query that forgot to exclude the corner
        # wavefront sensors is still correct rather than quietly wrong.
        det_purpose: dict[int, str] = {}
        for ref in inputRefs.raws:
            record = ref.dataId.records.get("detector")
            det_purpose[ref.dataId["detector"]] = (
                str(record.purpose) if record is not None else _SCIENCE_PURPOSE
            )
        dets_by_exp = {
            exp: {ref.dataId["detector"] for ref in refs}
            for exp, refs in raws_by_exp.items()
        }
        all_dets = dets_by_exp[intra_exp] | dets_by_exp[extra_exp]
        det_ids = sorted(
            d for d in all_dets if det_purpose.get(d) == _SCIENCE_PURPOSE
        )
        # A detector missing from one side of the pair has nothing to be paired
        # with, so drop it here rather than failing inside a worker.
        both_sides = dets_by_exp[intra_exp] & dets_by_exp[extra_exp]
        one_sided = [d for d in det_ids if d not in both_sides]
        if one_sided:
            self.log.warning(
                "Dropping %d detector(s) present in only one exposure of the "
                "pair: %s",
                len(one_sided),
                one_sided,
            )
        det_ids = [d for d in det_ids if d in both_sides]
        n_skipped = len(all_dets) - len(det_ids)
        if not det_ids:
            raise NoWorkFound(
                f"group={group}: no science detectors present in both exposures."
            )
        self.log.info(
            _colorize(
                "DonutBlitzFamTask.runQuantum() group=%s intra=%d extra=%d "
                "on %d science detector(s) (%d other/one-sided skipped)",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            group,
            intra_exp,
            extra_exp,
            len(det_ids),
            n_skipped,
        )

        # --- 4. resolve handles; no pixels are read in this process ---
        t_resolve0 = time.perf_counter()
        raw_handles: dict[int, dict[int, Any]] = {d: {} for d in det_ids}
        for exp, refs in raws_by_exp.items():
            for ref in refs:
                det = ref.dataId["detector"]
                if det in raw_handles:
                    raw_handles[det][exp] = butlerQC.get(ref)
        calib_handles = {
            name: self._calibHandlesByDetector(butlerQC, inputRefs, name, raw_handles)
            for name in ("ptc", "flat", "linearizer", "crosstalk", "intrinsicZernikes")
        }
        refcat_handles = list(butlerQC.get(inputRefs.refCat))
        t_resolve = time.perf_counter() - t_resolve0

        missing = {
            name: sorted(set(det_ids) - set(handles))
            for name, handles in calib_handles.items()
        }
        for name in ("ptc", "flat", "linearizer", "crosstalk"):
            if missing[name]:
                raise RuntimeError(
                    f"Missing {name} calibration for detector(s) {missing[name]}"
                )
        if missing["intrinsicZernikes"]:
            self.log.warning(
                "No intrinsic Zernike calibration for %d detector(s); their "
                "donuts are referenced against the nominal design optics at "
                "their own field angle rather than measured intrinsics, so "
                "their fitted deviations absorb any real static aberration the "
                "design does not have.",
                len(missing["intrinsicZernikes"]),
            )
        if not refcat_handles:
            self.log.warning(
                "No reference catalog shards provided; every detector will fall "
                "back to blind detection and spatial donut pairing."
            )

        band = str(butlerQC.quantum.dataId["band"])

        # The rest of the exposure metadata is read as a *component* off one raw
        # handle -- header only, no pixels. Everything downstream that needs the
        # rotator angle is quantum-wide, so reading it once here is both cheapest
        # and the only way to keep it consistent across workers.
        visit_info = raw_handles[det_ids[0]][extra_exp].get(component="visitInfo")
        if visit_info.id != extra_exp:
            self.log.warning(
                "Raw visitInfo.id=%d does not match its exposure id %d; the "
                "catalog's visit_id column follows visitInfo.",
                visit_info.id,
                extra_exp,
            )

        # rotTelPos, wrapped to (-pi, pi]. Always computed: the CCS -> OCS
        # Zernike rotation in the catalog needs it, whereas spider shadows are
        # opt-in.
        rtp_rad = (
            visit_info.boresightParAngle.asRadians()
            - visit_info.boresightRotAngle.asRadians()
            - np.pi / 2
            + np.pi
        ) % (2 * np.pi) - np.pi
        rtp_deg = (
            np.degrees(rtp_rad)
            if self.wfFittingTask.config.modelSpiderShadows
            else None
        )
        boresight_alt_rad = visit_info.boresightAzAlt.getLatitude().asRadians()

        photo_filter_name = (
            self.config.photoRefFilter
            if self.config.photoRefFilter is not None
            else f"{self.config.photoRefFilterPrefix}_{band}"
        )

        # --- 5. populate the store the workers inherit by copy-on-write ---
        self._populateCalibStore(
            det_ids=det_ids,
            raw_handles=raw_handles,
            calib_handles=calib_handles,
            refcat_handles=refcat_handles,
            intra_exp=intra_exp,
            extra_exp=extra_exp,
            band=band,
            rtp_deg=rtp_deg,
            boresight_alt_rad=boresight_alt_rad,
            photo_filter_name=photo_filter_name,
        )

        # --- 6. fork ---
        num_cores = butlerQC.resources.num_cores
        results = self._runWorkers(det_ids, num_cores)

        # --- 7. assemble and write ---
        catalog = self._buildCatalog(
            results=results,
            visit_id=extra_exp,
            rtp_rad=rtp_rad,
            photo_filter_name=photo_filter_name,
            run_elapsed=time.perf_counter() - t_start,
            butler_elapsed=t_resolve,
        )
        catalog.meta["group"] = str(group)
        catalog.meta["intra_exposure"] = intra_exp
        catalog.meta["extra_exposure"] = extra_exp

        # Keyed to the extra-focal visit, 1:1 with groups. The intra ref is
        # predicted but deliberately left unproduced.
        visit_refs = {ref.dataId["visit"]: ref for ref in outputRefs.famResults}
        target = visit_refs.get(extra_exp)
        if target is None:
            raise RuntimeError(
                f"No predicted output ref for the extra-focal visit {extra_exp}; "
                f"predicted {sorted(visit_refs)}."
            )
        butlerQC.put(Table(catalog), target)

    def _calibHandlesByDetector(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        name: str,
        raw_handles: dict,
    ) -> dict:
        """Resolve one calibration connection to one handle per detector.

        Raises rather than silently keeping the last ref if a detector has more
        than one. Declaring ``physical_filter`` on the quantum is what guarantees
        it does not: without it, ``flat`` and ``intrinsicZernikes`` arrive once per
        filter (8 and 6 per detector on the reference night), and a plain
        detector-keyed dict would quietly flat-field with an arbitrary band.  That
        failure produces plausible-looking wrong numbers rather than an error, so
        it is worth an assertion rather than trust.
        """
        handles: dict[int, Any] = {}
        for ref in getattr(inputRefs, name):
            det = ref.dataId["detector"]
            if det not in raw_handles:
                continue
            if det in handles:
                raise RuntimeError(
                    f"Multiple {name} refs for detector {det} in one quantum "
                    f"({dict(ref.dataId.mapping)}); the quantum's dimensions are "
                    "not narrowing this calibration to one per detector."
                )
            handles[det] = butlerQC.get(ref)
        return handles

    def _checkDefocalOrder(
        self, raws_by_exp: dict, intra_exp: int, extra_exp: int, group: Any
    ) -> None:
        """Warn if ``observation_reason`` contradicts the intra-first convention.

        The side of focus is assigned from the exposure id alone, because
        ``observation_reason`` is free-form and cannot be relied on.  But when it
        *is* populated and disagrees, every Zernike in the output has the wrong
        sign, so it is worth saying so loudly.
        """
        reasons = {}
        for exp in (intra_exp, extra_exp):
            record = raws_by_exp[exp][0].dataId.records.get("exposure")
            reasons[exp] = (getattr(record, "observation_reason", "") or "").lower()
        if "intra" in reasons[extra_exp] or "extra" in reasons[intra_exp]:
            self.log.warning(
                "group=%s: observation_reason contradicts the intra-first "
                "convention (exposure %d reason=%r taken as intra, %d reason=%r "
                "taken as extra). Zernike signs depend on this.",
                group,
                intra_exp,
                reasons[intra_exp],
                extra_exp,
                reasons[extra_exp],
            )

    def _populateCalibStore(
        self,
        det_ids: list[int],
        raw_handles: dict,
        calib_handles: dict,
        refcat_handles: list,
        intra_exp: int,
        extra_exp: int,
        band: str,
        rtp_deg: float | None,
        boresight_alt_rad: float | None,
        photo_filter_name: str,
    ) -> None:
        """Fill `_CALIB_STORE` with everything the workers need.

        Subtasks, handles, and the pre-built batoid telescopes all go in here so
        the children inherit them by copy-on-write instead of receiving them
        through a pickle.
        """
        # AstrometryTask.solve() calls refObjLoader.getMetadataBox()
        # unconditionally even when load_result is pre-supplied, and that method
        # is pure geometry -- it never touches catalog data. So a stub satisfies
        # it, and each worker builds its own real loader.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.pixelMargin = 0
        self.astromTask.setRefObjLoader(astrom_stub_loader)

        _CALIB_STORE.clear()
        _CALIB_STORE["isr_task"] = self.isrTask
        _CALIB_STORE["bkg_task"] = self.subtractBackground
        _CALIB_STORE["detect_diameter_task"] = self.detectDiameter
        _CALIB_STORE["blind_detect_task"] = self.blindDetect
        _CALIB_STORE["astrom_task"] = self.astromTask
        _CALIB_STORE["donut_selector_task"] = self.donutSelector
        _CALIB_STORE["measure_candidates_task"] = self.measureCandidatesTask
        _CALIB_STORE["cut_stamps_task"] = self.cutStampsTask
        _CALIB_STORE["wf_fitting_task"] = self.wfFittingTask
        _CALIB_STORE["wfEstimationMode"] = self.config.wfEstimationMode
        _CALIB_STORE["maxFitScatter"] = self.config.maxFitScatter
        _CALIB_STORE["astromRefFilter"] = self.config.astromRefFilter
        _CALIB_STORE["photoRefFilter"] = photo_filter_name
        _CALIB_STORE["pairMatchTolerance"] = self.config.pairMatchTolerance
        _CALIB_STORE["saveStamps"] = self.config.saveStamps
        _CALIB_STORE["saveWfImages"] = self.config.saveWfImages
        _CALIB_STORE["band"] = band
        _CALIB_STORE["rtp_deg"] = rtp_deg
        _CALIB_STORE["boresight_alt_rad"] = boresight_alt_rad
        _CALIB_STORE["intra_exposure"] = intra_exp
        _CALIB_STORE["extra_exposure"] = extra_exp
        _CALIB_STORE["offsets_by_exposure"] = {
            intra_exp: self._intraFocalOffsets,
            extra_exp: self._extraFocalOffsets,
        }
        _CALIB_STORE["refcat_handles"] = refcat_handles
        _CALIB_STORE["detectors"] = {
            det: {
                "raws": raw_handles[det],
                **{
                    name: calib_handles[name].get(det)
                    for name in (
                        "ptc",
                        "flat",
                        "linearizer",
                        "crosstalk",
                        "intrinsicZernikes",
                    )
                },
            }
            for det in det_ids
        }
        # `QuantumContext` does not expose its butler, but a resolved deferred
        # handle does -- and the fork pool initializer needs it to reset the
        # inherited connection pool. See `_fam_pool_initializer`.
        _CALIB_STORE["butler"] = getattr(
            raw_handles[det_ids[0]][extra_exp], "butler", None
        )

        # Telescope is band- and quantum-fixed. Build the base and both defocused
        # variants here so workers only ever look them up; the radial scales they
        # need for spatial pairing memoise into the same store.
        _CALIB_STORE["telescope"] = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
        for offsets in (self._intraFocalOffsets, self._extraFocalOffsets):
            _telescope_for_offsets(offsets)
            _defocal_radial_scale(offsets)

    def _runWorkers(self, det_ids: list[int], num_cores: int) -> list[dict]:
        """Run `_fam_detector_worker` over every detector, forking if asked to.

        ``num_cores`` comes from the execution environment (``pipetask
        -n/--cores-per-quantum``, default 1), never from config, matching
        `DonutBlitzMonolithTask`.  One core runs inline with no pool at all.
        """
        t0 = time.perf_counter()
        if num_cores == 1:
            self.log.info("Running %d detector(s) inline", len(det_ids))
            t_dispatch = time.time()
            results = [_fam_detector_worker((d, t_dispatch)) for d in det_ids]
        else:
            n_workers = min(num_cores, len(det_ids))
            self.log.info(
                "Forking %d worker(s) over %d detector(s)", n_workers, len(det_ids)
            )
            # Unlike the monolith's pools these workers read from the butler, so
            # the initializer is mandatory, not defensive: children sharing the
            # parent's inherited psycopg2 SSL socket corrupt it.
            # chunksize=1 is the streaming knob -- work-stealing keeps at most
            # n_workers detectors' pixels resident at once.
            with mp.get_context("fork").Pool(
                processes=n_workers, initializer=_fam_pool_initializer
            ) as pool:
                t_dispatch = time.time()
                results = list(
                    pool.imap_unordered(
                        _fam_detector_worker,
                        [(d, t_dispatch) for d in det_ids],
                        chunksize=1,
                    )
                )
        elapsed = time.perf_counter() - t0

        # A skip is an expected outcome (a dead CCD), a failure is not; keeping
        # them apart stops the routine ones from training the eye to ignore the
        # log line that matters.
        skipped = [r for r in results if r.get("skipped")]
        failures = [r for r in results if r["error"] and not r.get("skipped")]
        if skipped:
            self.log.info(
                "%d detector(s) skipped with no work: %s",
                len(skipped),
                ", ".join(f"{r['det_name'] or r['det_id']}" for r in skipped),
            )
        for r in failures:
            self.log.warning(
                "detector %s (%s) failed: %s",
                r["det_id"],
                r["det_name"] or "?",
                r["error"],
            )
        # A pairing path silently falling back to spatial matching across the
        # whole focal plane is a real condition worth seeing in the logs, not
        # something to discover later in the Zernikes.
        self._logWorkerSummaries(results)

        paths = Counter(r["pair_path"] for r in results if not r["error"])
        self.log.info(
            _colorize(
                "Workers done in %.1fs: %d/%d detectors ok (%d skipped), %d donut(s), "
                "%d fit group(s), pairing paths %s",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            elapsed,
            # Skipped detectors are neither ok nor failed: they contributed no
            # rows, so counting them as ok overstates the yield.
            len(results) - len(failures) - len(skipped),
            len(results),
            len(skipped),
            sum(len(r["donuts"]) for r in results),
            sum(len(r["wf_results"]) for r in results),
            dict(paths),
        )
        return results

    def _logWorkerSummaries(self, results: list[dict]) -> None:
        """Log one summary line per detector, then aggregates over them.

        The per-detector line is modelled on `DonutBlitzMonolithTask`'s, with two
        differences that follow from FAM fusing the whole pipeline into one worker:
        it carries an ``io`` stage, because each worker does its own butler reads;
        and it carries the Danish ``fit`` result, because the fit happens in the
        same process rather than in a separate pool the parent can summarise on its
        own.  Between them they replace `WavefrontFittingTask`'s per-group lines,
        which `setDefaults` turns off here.

        Lines are sorted by detector name: `imap_unordered` returns detectors in
        completion order, which is neither reproducible between runs nor useful for
        finding a raft.  Skipped and failed detectors get a line too -- a truncated
        one, since the full error is already logged above -- so that a missing
        detector is visibly missing rather than absent.

        The aggregates cover only the detectors that produced results.
        """
        if not results:
            return

        def name_of(r: dict) -> str:
            return r["det_name"] or f"det{r['det_id']}"

        ok = []
        for r in sorted(results, key=name_of):
            name = name_of(r)
            stages = _detector_stage_times(r)

            if r.get("skipped") or r["error"]:
                self.log.info(
                    "  %s: %s wall=%.2fs -- %s",
                    name,
                    "SKIPPED" if r.get("skipped") else "FAILED",
                    stages["wall"],
                    r["error"][:120],
                )
                continue
            ok.append(r)

            # Cutout results are appended intra-first by the worker, so both of
            # these read intra/extra.
            scatter = "/".join(
                "N/A" if res.get("scatter_arcsec") is None else f'{res["scatter_arcsec"]:.2f}"'
                for res in r["results"]
            )
            donuts = "+".join(str(len(res["catalog"])) for res in r["results"])

            wf = r["wf_results"]
            if wf:
                n_ok = sum(bool(g.get("success")) for g in wf)
                sizes = [g.get("group_size", 0) for g in wf]
                # A timed-out or failed group has an empty fit_info, so nfev is
                # missing rather than zero.
                nfev_mean = _mean_std_max(
                    [g.get("fit_info", {}).get("nfev", np.nan) or np.nan for g in wf]
                )[0]
                fit = (
                    f"fit={stages['fit']:.1f}s ({n_ok}/{len(wf)} ok, "
                    f"n={np.mean(sizes):.1f}"
                    + (f", nfev={nfev_mean:.1f})" if np.isfinite(nfev_mean) else ")")
                )
            else:
                fit = f"fit={stages['fit']:.2f}s (no groups)"

            pieces = []
            for key in _STAGE_KEYS:
                if key in ("fit", "wall"):
                    continue
                piece = f"{key}={stages[key]:.2f}s"
                # Scatter belongs to the WCS refit, so it hangs off that stage
                # rather than standing as its own column, as in corner mode.
                pieces.append(f"{piece} (scatter={scatter})" if key == "wcs" else piece)
            pieces += [
                f"donuts={donuts}",
                f"pair={r['pair_path']}",
                fit,
                f"wall={stages['wall']:.2f}s",
            ]
            self.log.info("  %s: %s", name, "  ".join(pieces))

        if not ok:
            return
        self._logWorkerAggregates(ok)

    def _logWorkerAggregates(self, ok: list[dict]) -> None:
        """Log mean/std of every per-detector quantity over the detectors that ran.

        Split out from `_logWorkerSummaries` only for length; it is called with the
        detectors that neither failed nor skipped, since a NaN-filled row would
        otherwise widen every standard deviation with a number that means "absent"
        rather than "slow".
        """
        n = len(ok)
        stages = [_detector_stage_times(r) for r in ok]
        self.log.info(
            _colorize(
                "Per-detector timing (n=%d): %s",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            n,
            "  ".join(
                "{}={:.2f}+/-{:.2f}s".format(key, *_mean_std_max([s[key] for s in stages])[:2])
                for key in _STAGE_KEYS
            ),
        )

        donuts = [sum(len(res["catalog"]) for res in r["results"]) for r in ok]
        groups = [len(r["wf_results"]) for r in ok]
        scatters = [
            res["scatter_arcsec"]
            for r in ok
            for res in r["results"]
            if res.get("scatter_arcsec") is not None
        ]
        fits = [
            g.get("fit_info", {}).get("elapsed", np.nan) for r in ok for g in r["wf_results"]
        ]
        self.log.info(
            "Per-detector yield (n=%d): donuts=%.1f+/-%.1f  groups=%.1f+/-%.1f  "
            'scatter=%.2f+/-%.2f"  per-group fit=%.1f+/-%.1fs (max %.1fs)',
            n,
            *_mean_std_max(donuts)[:2],
            *_mean_std_max(groups)[:2],
            *_mean_std_max(scatters)[:2],
            *_mean_std_max(fits),
        )

        slowest = sorted(
            ok,
            key=lambda r: (
                r.get("worker_wall", 0.0) if np.isfinite(r.get("worker_wall", np.nan)) else 0.0
            ),
            reverse=True,
        )[:_N_SLOWEST]
        self.log.info(
            "Slowest %d detector(s): %s",
            len(slowest),
            ", ".join(
                f"{r['det_name'] or r['det_id']} {r.get('worker_wall', float('nan')):.1f}s"
                for r in slowest
            ),
        )

    def _buildCatalog(
        self,
        results: list[dict],
        visit_id: int,
        rtp_rad: float,
        photo_filter_name: str,
        run_elapsed: float,
        butler_elapsed: float,
    ) -> Any:
        """Flatten the per-detector worker results into the shared catalog schema."""
        cutout_results = [r for w in results for r in w["results"]]
        wf_results = [r for w in results for r in w["wf_results"]]
        donuts = [d for w in results for d in w["donuts"]]
        unmatched = [d for w in results for d in w["unmatched_donuts"]]

        n_ok = sum(r.get("success", False) for r in wf_results)
        self.log.info(
            "WF results (%s): %d/%d group(s) succeeded",
            self.config.wfEstimationMode,
            n_ok,
            len(wf_results),
        )

        return build_donut_catalog(
            results=cutout_results,
            wf_results=wf_results,
            donuts=donuts,
            unmatched_donuts=unmatched,
            visit_id=visit_id,
            options=self._catalogOptions(),
            run_elapsed=run_elapsed,
            butler_elapsed=butler_elapsed,
            cutout_elapsed=sum(
                w["cutout_run"] for w in results if np.isfinite(w["cutout_run"])
            ),
            danish_elapsed=sum(
                w["fit_run"] for w in results if np.isfinite(w["fit_run"])
            ),
            photo_filter_name=photo_filter_name,
            astrom_filter_name=self.config.astromRefFilter,
            rtp_rad=rtp_rad,
        )

    def _catalogOptions(self) -> CatalogOptions:
        """Gather the config-derived scalars the output catalog needs.

        Same function corner mode calls, deliberately: one schema for both modes
        rather than two that drift.  The image flags differ -- full-array mode has
        ~10k rows per pair, where they would dominate the file.
        """
        return CatalogOptions(
            stamp_size=self.cutStampsTask.config.stampSize,
            binning=self.wfFittingTask.config.binning,
            noll_indices=tuple(self.wfFittingTask.config.nollIndices),
            aperture_margin_frac=self.measureCandidatesTask.config.apertureMarginFrac,
            bkg_inner_disc_frac=self.measureCandidatesTask.config.bkgInnerDiscFrac,
            bkg_annulus_inner_frac=self.measureCandidatesTask.config.bkgAnnulusInnerFrac,
            bkg_annulus_outer_frac=self.measureCandidatesTask.config.bkgAnnulusOuterFrac,
            max_donuts=self.cutStampsTask.config.maxDonuts,
            wf_mode=self.config.wfEstimationMode,
            save_stamps=self.config.saveStamps,
            save_wf_images=self.config.saveWfImages,
        )
