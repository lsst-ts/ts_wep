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

"""The corner-wavefront-sensor blitz pipeline task."""

__all__ = [
    "DonutBlitzCornerConnections",
    "DonutBlitzCornerConfig",
    "DonutBlitzCornerTask",
]

import time
from dataclasses import dataclass
from typing import Any

import batoid
import numpy as np
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
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
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutDetectDiameterTask import DonutDetectDiameterTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.utils.timer import timeMethod

from .blitzDetect import BlitzDetectTask
from .catalogBuilder import _build_donut_catalog, _CatalogOptions, _CatalogTimings
from .cutDonutStamps import CutDonutStampsTask
from .cutoutPipeline import _cutout_corner_detector
from .dataStructures import CutoutResult, WfGroupResult
from .donutBlitzPlot import DonutBlitzPlotTask
from .forkPool import _dump_stacks_on_hang, _fork_map
from .measureDonutCandidates import MeasureDonutCandidatesTask
from .utils import (
    _ANSI_BOLD,
    _ANSI_CYAN,
    _ANSI_GREEN,
    _COW_STORE,
    _CUTOUT_STAGE_KEYS,
    _INSTRUMENT,
    _INTRA_FOCAL_DET_IDS,
    CORNER_DET_NAMES,
    CornerDetectorInputs,
    CowStore,
    IsrCalibs,
    _colorize,
    _resolve_color_log_enabled,
    _rot_tel_pos_rad,
)
from .wavefrontFitting import (
    WavefrontFittingTask,
    _build_wf_groups,
    _wf_fitting_worker,
)
from .zernikesTable import build_zernikes_tables


def _exposure_group(refs) -> str:
    """The butler ``group`` of the exposure these raw refs came from.

    Corner mode's quantum is visit-dimensioned, so unlike full-array mode it
    cannot read ``group`` off its own data ID.  The ``raws`` connection is
    exposure-dimensioned though, and ``exposure`` implies ``group``, so an
    expanded data ID carries it.  All the raws of one corner set share an
    exposure, so the first ref answers for all of them.

    Returns ``""`` when the data IDs are not expanded (``hasFull()`` False),
    which is the only way the implied dimension can be missing; the group is
    metadata for the output table, not something worth failing a quantum over.
    """
    for ref in refs:
        if ref.dataId.hasFull():
            return str(ref.dataId["group"])
    return ""


# Corner mode defocuses by shifting the detector plane inside the camera: SW0
# (extra-focal) sits at +defocalOffset, SW1 (intra-focal) at -defocalOffset.
# Ordered as `_OFFSET_OPTICS`: (detector, camera, m2). Full-array mode shifts
# the whole camera instead, which is why the offsets ride on each Donut rather
# than being assumed by the fitter.
_EXTRA_FOCAL_OFFSETS = (+_INSTRUMENT.defocalOffset, 0.0, 0.0)
_INTRA_FOCAL_OFFSETS = (-_INSTRUMENT.defocalOffset, 0.0, 0.0)

# The connection names `runQuantum` fetches, in the order it fetches them.
# These strings are three things at once, which is why the tuple is shared
# rather than written out per use: the `inputRefs` attributes to read, the keys
# of the ``butler_times`` breakdown that reaches the output catalog's meta (and
# so the plot's label), and the order of the timing log.  They stay camelCase
# because the connections are framework-named.
_BUTLER_INPUTS = (
    "raws",
    "ptc",
    "flat",
    "linearizer",
    "crosstalk",
    "refCat",
    "intrinsicZernikes",
)

# The calibrations narrowed to the detectors the raws actually cover, so a
# missing raw does not drag its calibrations through the fetch.  Excludes
# `raws`, which is what the detector set is read *from*, and `refCat`, a
# `PrerequisiteInput` dimensioned on htm7 shards rather than on detector and so
# carrying no per-detector refs to filter.
_PER_DETECTOR_INPUTS = tuple(name for name in _BUTLER_INPUTS if name not in ("raws", "refCat"))


@dataclass(frozen=True)
class _CornerInputs:
    """The per-detector inputs of one corner visit, indexed by detector.

    Built by `DonutBlitzCornerTask._indexInputs` from the flat lists `run()`
    receives, and valid by construction: every detector named here is a corner
    detector with a raw and a complete set of the four ISR calibrations.

    Deliberately **not** in `dataStructures.py`, unlike `CutoutResult` and its
    two siblings.  Those cross a fork boundary by design and are public; this
    one holds materialized `Exposure` objects and is only ever read in the
    parent process, so it stays module-private and out of ``__all__``.  It is
    the same distinction `cutDonutStamps._ExposureContext` draws.

    Attributes
    ----------
    det_names : tuple of str
        The detectors to process, sorted.  Drives everything downstream --
        `CORNER_DET_NAMES` is only ever the set this is checked *against*,
        since a partial corner set is a normal outcome.
    det_name_by_id : dict
        Detector id to name, covering exactly the raws supplied.  Only the
        intrinsic Zernike lookup needs it, calibrations being name-keyed.
    raw_by_name : dict
        The raw exposures.  Kept because three later stages read the band, the
        visit info and the visit id back off one of them.
    corner_detectors : dict
        What the cutout workers inherit, one `CornerDetectorInputs` per
        detector.
    """

    det_names: tuple[str, ...]
    det_name_by_id: dict[int, str]
    raw_by_name: dict[str, Any]
    corner_detectors: dict[str, CornerDetectorInputs]


class DonutBlitzCornerConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzCornerTask."""

    raws = connectionTypes.Input(
        doc=(
            "Raw corner wavefront sensor exposures. Any subset of the 8 corner "
            "detectors is processed; missing detectors are simply skipped."
        ),
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
    cornerResults = connectionTypes.Output(
        doc=(
            "Per-donut catalog containing selection metrics, fit results, Zernikes, "
            "stamp/model images, and all metadata needed to regenerate diagnostic plots."
        ),
        name="donutBlitzCornerResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )
    zernikes = connectionTypes.Output(
        doc=(
            "Per-corner Zernike table in the schema CalcZernikesTask emits, for "
            "consumers written against the non-blitz corner pipeline. One per corner, "
            "keyed on the extra-focal (SW0) detector. Carries only the deviation "
            "Zernikes; see lsst.ts.wep.blitz.zernikesTable."
        ),
        name="zernikes",
        storageClass="AstropyQTable",
        dimensions=("visit", "detector", "instrument"),
        # Only the extra-focal visits are ever written...h
        multiple=True,
    )

    def __init__(self, *, config: Any = None) -> None:
        super().__init__(config=config)
        if config is not None and not config.doZernikesOutput:
            del self.zernikes


class DonutBlitzCornerConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzCornerConnections,  # type: ignore
):
    """Configuration for DonutBlitzCornerTask."""

    isr: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="ISR subtask run on each corner wavefront sensor exposure.",
    )
    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Background subtraction subtask run before donut detection.",
    )
    measureDiameter: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutDetectDiameterTask,
        doc="Donut diameter detection subtask.",
    )
    blitzDetect: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=BlitzDetectTask,
        doc=("Blitz donut detection subtask run on each corner wavefront sensor exposure."),
    )
    astrometry: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=AstrometryTask,
        doc="Astrometry subtask for WCS fitting.",
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask,
        doc="Donut source selector subtask.",
    )
    measureCandidates: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=MeasureDonutCandidatesTask,
        doc="Donut candidate measurement subtask.",
    )
    cutStamps: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=CutDonutStampsTask,
        doc="Donut stamp cutting subtask.",
    )
    wavefrontFit: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=WavefrontFittingTask,
        doc="Wavefront fitting subtask using Danish algorithm.",
    )
    plot: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutBlitzPlotTask,
        doc="Subtask that generates diagnostic plots for a blitz visit.",
    )
    instConfigFile: pexConfig.Field[str] = pexConfig.Field[str](
        doc=(
            "Path to an instrument configuration file to override the default. "
            "If begins with 'policy:' the path is relative to the ts_wep policy "
            "directory. If not provided, the default instrument for the camera "
            "will be loaded."
        ),
        optional=True,
    )
    maxFitScatter: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Maximum allowed on-sky scatter (arcsec) for WCS refit to be accepted.",
        default=1.0,
    )
    astromRefFilter: pexConfig.Field[str] = pexConfig.Field[str](
        doc=(
            "Filter name to read from the reference catalog when fitting the "
            "WCS. Aliased over every filter via anyFilterMapsToThis, so it is "
            "what AstrometryTask resolves as its reference flux field."
        ),
        default="phot_g_mean",
    )
    photoRefFilter: pexConfig.Field[str] = pexConfig.Field[str](
        doc=(
            "Explicit filter name to read from the reference catalog for donut "
            "selection (e.g. 'phot_g_mean'). Overrides photoRefFilterPrefix "
            "when set."
        ),
        optional=True,
    )
    photoRefFilterPrefix: pexConfig.Field[str] = pexConfig.Field[str](
        doc=(
            "Filter prefix used for donut selection, combined with the exposure "
            "band label as '{prefix}_{band}'. Used when photoRefFilter is not "
            "set. The default matches the only per-band LSST-like fluxes "
            "present in the_monster_20250219 ('monster_ComCam_g' etc.); there "
            "are no monster_LSSTCam_* columns in any released refcat version, "
            "so override this once a matching set exists."
        ),
        default="monster_ComCam",
    )
    saveStamps: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Include the unbinned `stamp` image column in the output catalog. "
            "The diagnostic plots use it when present and fall back to the "
            "binned `wf_img` when not."
        ),
        default=True,
    )
    savePlots: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Generate diagnostic PNGs for each visit. "
            "Set False in production to skip plot generation and deliver "
            "Zernikes faster; plots can be generated later by calling "
            "plot.run() with the in-memory results."
        ),
        default=False,
    )
    unitTimeout: pexConfig.Field[float] = pexConfig.Field[float](
        doc=(
            "Seconds one detector's cutout, or one wavefront group's fit, may "
            "run before that worker is killed and its unit recorded as lost, "
            "so a single pathological unit costs its own detector or group "
            "instead of the whole quantum.  Cutout median is 1.9s and "
            "wavefront median 3.1s, and neither pool passed 26s across ~2900 "
            "visits, so the default leaves a crowded field room to be slow "
            "without being killed -- do not tune it down toward nominal.  "
            "None waits indefinitely, restoring the behaviour where only "
            "hangTimeout notices an overrun.  Note this bounds one unit and "
            "not the pool: units run in waves of num_cores, so a visit in "
            "which *every* unit times out can still reach hangTimeout and "
            "abort, which is the right outcome for what is by then a systemic "
            "failure rather than one bad detector."
        ),
        default=30.0,
        optional=True,
    )
    hangTimeout: pexConfig.Field[float] = pexConfig.Field[float](
        doc=(
            "Seconds either the cutout or the wavefront pool may run before "
            "the hang watchdog dumps stacks and aborts the quantum, so that a "
            "pool blocked forever fails loudly and gets retried instead of "
            "burning its walltime.  This is a backstop: unitTimeout is what "
            "bounds a single slow unit, and it is the only one of the two that "
            "can degrade gracefully, since the watchdog fires from a side "
            "thread with no way to know which unit is late and so can only "
            "kill the process.  What is left for it is a wedge outside the "
            "pools.  The default clears the worst case unitTimeout admits at "
            "its own default, so the per-unit path resolves first: 8 corner "
            "detectors in waves of num_cores is 4 waves at 2 cores, 4x30s = "
            "120s < 180s.  Raise it alongside unitTimeout, never below it, and "
            "check the galactic-bulge visits since cutout time is what scales "
            "with reference density."
        ),
        default=180.0,
    )
    colorLog: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Colorize select log messages with ANSI escape codes. If None "
            "(the default), color is enabled only when stdout is an "
            "interactive terminal."
        ),
        default=None,
        optional=True,
    )
    doZernikesOutput: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Also emit the per-corner `zernikes` table for legacy consumers. "
            "Off by default: the per-donut `donutBlitzCornerResults` catalog "
            "is the primary output and carries strictly more information."
        ),
        default=False,
    )
    combineZernikes: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=CombineZernikesSigmaClipTask,
        doc=(
            "How to combine the per-group Zernikes into the `average` row of "
            "the `zernikes` table. Only used when doZernikesOutput is True."
        ),
    )
    doBlurClip: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Sigma clip donuts whose fitted blur (fwhm) is an outlier from the "
            "`zernikes` average. See blurClipMinRows."
        ),
        default=True,
    )
    blurClipMinRows: pexConfig.Field[int] = pexConfig.Field[int](
        doc=(
            "Minimum number of data rows for blur clipping to run. Below this "
            "it is skipped, because mad_std over one or two samples cannot flag "
            "anything and the clip would still replace the configured "
            "combineZernikes average with an unweighted mean. Relevant for the "
            "joint-fit modes, which yield one row per corner (full_corner) or "
            "two (full_detector)."
        ),
        default=3,
    )
    wfEstimationMode: pexConfig.ChoiceField[str] = pexConfig.ChoiceField[str](
        doc="Wavefront estimation dispatch mode.",
        allowed={
            "paired": "Pair donuts from SW0/SW1 by SNR rank and dispatch as intra/extra pairs.",
            "unpaired": "Dispatch individual donuts independently.",
            "full_corner": (
                "Dispatch all donuts from a corner (SW0+SW1, whichever are present) as one work unit."
            ),
            "full_detector": "Dispatch all donuts on each detector as one work unit (8 fits per visit).",
        },
        default="paired",
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.isr.doAmpOffset = False
        self.isr.ampOffset.doApplyAmpOffset = False
        self.isr.doBrighterFatter = False
        self.isr.doSaturation = True
        self.isr.doStandardStatistics = False
        self.isr.doInterpolate = False
        self.isr.doVariance = False
        self.isr.doDeferredCharge = False
        self.isr.doDefect = False
        self.isr.doApplyGains = True
        self.isr.doBias = False
        self.isr.doFlat = True
        self.isr.doDark = False
        self.isr.doLinearize = True
        self.isr.doSuspect = False
        self.isr.doSetBadRegions = False
        self.isr.doBootstrap = False
        self.isr.doCrosstalk = True
        self.isr.crosstalk.doQuadraticCrosstalkCorrection = False
        self.isr.doITLEdgeBleedMask = False
        self.isr.qa.saveStats = False

        self.astrometry.wcsFitter.retarget(FitAffineWcsTask)
        self.astrometry.doMagnitudeOutlierRejection = False
        self.astrometry.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 18
        self.astrometry.referenceSelector.magLimit = magLimit
        self.astrometry.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astrometry.sourceSelector["science"].doRequirePrimary = False
        self.astrometry.sourceSelector["science"].doIsolated = False
        self.astrometry.sourceSelector["science"].doSignalToNoise = False
        self.astrometry.sourceSelector["science"].doCentroidErrorLimit = False
        self.astrometry.maxIter = 5
        self.astrometry.matcher.maxOffsetPix = 1000

        # Cap the references handed to the pattern matcher.  Essential for
        # keeping the cost to refit the WCS near the galactic bulge.
        self.astrometry.matcher.maxRefObjects = 2048

        # Monster refcat uses full filter names (e.g. phot_g_mean), not band
        # labels, so the default mag-limit policy lookup by band would fail.
        # Use custom mag limits instead.
        self.donutSelector.useCustomMagLimit = True
        self.donutSelector.maxFieldDist = 1.725
        self.donutSelector.sourceLimit = 40
        self.donutSelector.allowFluxless = True


class DonutBlitzCornerTask(pipeBase.PipelineTask):
    """Blitz WEP task for the corner wavefront sensors.

    Runs ISR, blitz donut detection, WCS refit, catalog-based donut
    selection, and stamp cutting on whichever corner detector raws are present,
    in parallel using a multiprocessing pool.  Reference catalogs are loaded in
    the parent process before forking and inherited by workers via
    copy-on-write.
    """

    ConfigClass = DonutBlitzCornerConfig
    _DefaultName = "donutBlitzCorner"
    config: DonutBlitzCornerConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("measureDiameter")
        self.makeSubtask("blitzDetect")
        self.makeSubtask("astrometry")
        self.makeSubtask("donutSelector")
        self.makeSubtask("measureCandidates")
        self.makeSubtask("cutStamps")
        self.makeSubtask("wavefrontFit")
        self.makeSubtask("plot")
        if self.config.doZernikesOutput:
            self.makeSubtask("combineZernikes")
        self._colorLogEnabled = _resolve_color_log_enabled(self.config.colorLog)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        self.log.info(
            _colorize(
                "DonutBlitzCornerTask.runQuantum() on exposure %d",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            inputRefs.raws[0].dataId["exposure"],
        )
        raw_det_ids = {ref.dataId["detector"] for ref in inputRefs.raws}
        for attr in _PER_DETECTOR_INPUTS:
            refs = getattr(inputRefs, attr)
            setattr(inputRefs, attr, [r for r in refs if r.dataId["detector"] in raw_det_ids])

        # Fetched one dataset type at a time, and timed that way, because which
        # input dominates the I/O is the thing worth knowing: a slow refcat
        # shard load and a slow raw read call for different fixes.
        fetched = {}
        butler_times = {}
        for name in _BUTLER_INPUTS:
            t_start = time.perf_counter()
            fetched[name] = butlerQC.get(getattr(inputRefs, name))
            butler_times[name] = time.perf_counter() - t_start
        butler_elapsed = sum(butler_times.values())

        self.log.info(
            _colorize(
                "butlerQC.get timing: %s total=%.3fs",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            " ".join(f"{name}={butler_times[name]:.3f}s" for name in _BUTLER_INPUTS),
            butler_elapsed,
        )
        t_run0 = time.perf_counter()
        outputs = self.run(
            raws=fetched["raws"],
            ptc=fetched["ptc"],
            flat=fetched["flat"],
            linearizer=fetched["linearizer"],
            crosstalk=fetched["crosstalk"],
            ref_cat=fetched["refCat"],
            intrinsic_zernikes=fetched["intrinsicZernikes"],
            butler_elapsed=butler_elapsed,
            butler_times=butler_times,
            num_cores=butlerQC.resources.num_cores,
            exposure_group=_exposure_group(inputRefs.raws),
            instrument=str(butlerQC.quantum.dataId["instrument"]),
        )
        self.log.info("run() execution: %.3fs", time.perf_counter() - t_run0)
        butlerQC.put(outputs.cornerResults, outputRefs.cornerResults)

        if self.config.doZernikesOutput:
            # A ref is predicted for every corner detector the query
            # covers, but only the extra-focal ones are keys here: the
            # intra-focal refs and any corner that contributed no fits
            # are deliberately left unwritten.
            refs = {ref.dataId["detector"]: ref for ref in outputRefs.zernikes}
            for det_id, table in outputs.zernikes.items():
                ref = refs.get(det_id)
                if ref is None:
                    self.log.warning(
                        "No predicted output ref for zernikes on detector %d; not written.", det_id
                    )
                    continue
                butlerQC.put(table, ref)

    @timeMethod
    def run(
        self,
        raws: list,
        ptc: list,
        flat: list,
        linearizer: list,
        crosstalk: list,
        ref_cat: list,
        intrinsic_zernikes: list | None = None,
        butler_elapsed: float = 0.0,
        butler_times: dict | None = None,
        num_cores: int = 1,
        exposure_group: str = "",
        instrument: str = "",
    ) -> pipeBase.Struct:
        """Run ISR, WCS refit, catalog selection, and stamp cutting on the
        corner raws that are present, in parallel.

        Parameters
        ----------
        raws : list of lsst.afw.image.Exposure
            Corner wavefront sensor raws.  Any subset of the 8 corner detectors
            is accepted; processing covers exactly the detectors supplied here.
            Calibrations must be complete for every detector present.
        ptc : list of lsst.ip.isr.PhotonTransferCurveDataset
        flat : list of lsst.afw.image.ExposureF
        linearizer : list of lsst.ip.isr.Linearizer
        crosstalk : list of lsst.ip.isr.CrosstalkCalib
        ref_cat : list of DeferredDatasetHandle or SimpleCatalog
            Shards used for both WCS fitting and donut selection, loaded once
            per detector.  The WCS fit reads ``astromRefFilter`` (resolved as
            the load's ``fluxField``) and donut selection reads the per-band
            ``photoRefFilter``/``photoRefFilterPrefix`` column off the same
            catalog.
        intrinsic_zernikes : list of IntrinsicZernikes, optional
            One calibration per corner detector.  None or empty when absent.
        butler_elapsed : float, optional
            Total butlerQC.get() wall time in seconds, for logging and plot.
        butler_times : dict, optional
            Per-dataset butlerQC.get() times keyed by dataset type name.
        num_cores : int
        exposure_group : str, optional
            Butler ``group`` of the corner exposure, for output meta.
        instrument : str, optional
            Butler ``instrument`` dimension, for output meta.  Taken from the
            dataId rather than ``visit_info.instrumentLabel`` because the
            dimension is authoritative where the header field can be blank.
        """
        self.log.info(
            _colorize(
                "DonutBlitzCornerTask.run() with %d cores, butler elapsed=%.3fs",
                _ANSI_BOLD,
                _ANSI_GREEN,
                enabled=self._colorLogEnabled,
            ),
            num_cores,
            butler_elapsed,
        )
        t_run0 = time.perf_counter()

        inputs = self._indexInputs(raws, ptc, flat, linearizer, crosstalk)
        intrinsic_zernikes_by_name = self._indexIntrinsics(intrinsic_zernikes, inputs.det_name_by_id)

        band = next(iter(inputs.raw_by_name.values())).filter.bandLabel
        if self.config.photoRefFilter is not None:
            photo_filter_name = self.config.photoRefFilter
        else:
            photo_filter_name = f"{self.config.photoRefFilterPrefix}_{band}"

        t_stage0 = time.perf_counter()
        det_refcats = self._loadRefcats(ref_cat, inputs.raw_by_name)
        refcat_elapsed = time.perf_counter() - t_stage0

        visit_info = next(iter(inputs.raw_by_name.values())).getInfo().getVisitInfo()
        boresight_alt_rad = visit_info.boresightAzAlt.getLatitude().asRadians()
        rtp_rad = _rot_tel_pos_rad(visit_info)
        rtp_deg = np.degrees(rtp_rad) if self.wavefrontFit.config.modelSpiderShadows else None

        self._populateCowStore(inputs, det_refcats, band, photo_filter_name)

        t_stage0 = time.perf_counter()
        results = self._runCutoutPool(inputs.det_names, num_cores)
        cutout_elapsed = time.perf_counter() - t_stage0
        donuts = self._logCutoutSummaries(results)
        self._annotateDonuts(results, intrinsic_zernikes_by_name)

        mode = self.config.wfEstimationMode
        results_by_det = {r.det_name: r.catalog for r in results}
        groups, unmatched_donuts, pair_path = _build_wf_groups(
            mode, results_by_det, band, rtp_deg, boresight_alt_rad
        )
        # Stamp the pairing path on every cutout result: those are what reach
        # _build_donut_catalog, so this is what gets pairing provenance into
        # the persisted table.  Full-array mode does the same in its worker.
        for r in results:
            r.pair_path = pair_path

        t_stage0 = time.perf_counter()
        wf_results = self._runWfPool(groups, num_cores)
        danish_elapsed = time.perf_counter() - t_stage0

        t_plot0 = time.perf_counter()
        run_elapsed = t_plot0 - t_run0
        self.log.info(
            _colorize(
                "Timing summary: butler=%.1fs  refcat=%.1fs  cutout=%.1fs  danish=%.1fs  total=%.1fs",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            butler_elapsed,
            refcat_elapsed,
            cutout_elapsed,
            danish_elapsed,
            run_elapsed,
        )
        visit_id = next(iter(raws)).getInfo().getVisitInfo().id

        catalog = _build_donut_catalog(
            results=results,
            wf_results=wf_results,
            donuts=donuts,
            unmatched_donuts=unmatched_donuts,
            visit_id=visit_id,
            options=self._catalogOptions(),
            intra_visit_id=visit_id,
            extra_visit_id=visit_id,
            exposure_group=exposure_group,
            timings=_CatalogTimings(
                run_elapsed=run_elapsed,
                refcat_elapsed=refcat_elapsed,
                butler_elapsed=butler_elapsed,
                butler_times=butler_times or {},
                cutout_elapsed=cutout_elapsed,
                danish_elapsed=danish_elapsed,
            ),
            photo_filter_name=photo_filter_name,
            astrom_filter_name=self.config.astromRefFilter,
            rtp_rad=rtp_rad,
            mode="corner",
            # Corner mode has one exposure holding both sides of focus, so the
            # single visit_info read above for the boresight angles is also the
            # observation record for the whole table.
            visit_info=visit_info,
            instrument=instrument,
        )

        if self.config.savePlots:
            self.plot.run(catalog)
            self.log.info("Diagnostic plot: %.3fs", time.perf_counter() - t_plot0)

        # Built from the finished catalog rather than from wf_results, so the
        # two outputs cannot disagree about what was fit.
        zernikes = {}
        if self.config.doZernikesOutput:
            zernikes = build_zernikes_tables(
                catalog,
                noll_indices=self.wavefrontFit.config.nollIndices,
                wf_mode=mode,
                combine_zernikes=self.combineZernikes,
                visit_id=visit_id,
                cam_name=instrument,
                visit_info=visit_info,
                do_blur_clip=self.config.doBlurClip,
                blur_clip_min_rows=self.config.blurClipMinRows,
                log=self.log,
            )

        return pipeBase.Struct(
            donuts=donuts,
            wfResults=wf_results,
            cornerResults=Table(catalog),
            zernikes=zernikes,
        )

    def _indexInputs(
        self,
        raws: list,
        ptc: list,
        flat: list,
        linearizer: list,
        crosstalk: list,
    ) -> _CornerInputs:
        """Index the flat input lists by detector name and validate them.

        Corner mode's counterpart to full-array mode's handle resolution, and
        necessarily a different shape: this task's butler I/O all happens in
        the parent, so what there is to index here is materialized exposures
        and calibrations rather than deferred handles.

        Raises
        ------
        RuntimeError
            If a non-corner raw is supplied, if no raws are, or if any detector
            with a raw is missing one of the four ISR calibrations.  A
            *partial* corner set is not an error -- see below.
        """
        det_name_by_id = {}
        raw_by_name = {}
        for exp in raws:
            det = exp.getDetector()
            det_name_by_id[det.getId()] = det.getName()
            raw_by_name[det.getName()] = exp
        flat_by_name = {f.getDetector().getName(): f for f in flat}
        # `metadata["DET_NAME"]` is the only public way to ask an `IsrCalib`
        # which detector it belongs to -- there is no `getDetectorName()`.
        ptc_by_name = {p.metadata["DET_NAME"]: p for p in ptc}
        linearizer_by_name = {lin.metadata["DET_NAME"]: lin for lin in linearizer}
        crosstalk_by_name = {ct.metadata["DET_NAME"]: ct for ct in crosstalk}

        # Process whichever corner raws arrived. A partial set is normal
        # (dropped image, per-detector butler gap) and there is no reason to
        # throw away the corners that did arrive, so this is a warning rather
        # than an abort. `det_names` -- not CORNER_DET_NAMES -- drives
        # everything downstream.
        unexpected = raw_by_name.keys() - CORNER_DET_NAMES
        if unexpected:
            raise RuntimeError(f"Non-corner detector raws supplied: {sorted(unexpected)}")
        if not raw_by_name:
            raise RuntimeError("No corner detector raws supplied.")
        det_names = tuple(sorted(raw_by_name))
        missing = CORNER_DET_NAMES - raw_by_name.keys()
        if missing:
            self.log.warning(
                "Processing %d/%d corner detectors; no raw for: %s",
                len(det_names),
                len(CORNER_DET_NAMES),
                sorted(missing),
            )

        corner_detectors = {}
        for name in det_names:
            missing_calib = [
                k
                for k, d in [
                    ("ptc", ptc_by_name),
                    ("flat", flat_by_name),
                    ("linearizer", linearizer_by_name),
                    ("crosstalk", crosstalk_by_name),
                ]
                if name not in d
            ]
            if missing_calib:
                raise RuntimeError(f"Missing calibration(s) for detector {name}: {missing_calib}")
            # Materialized, not deferred: corner mode does its butler I/O in
            # the parent, so the workers use these objects directly.
            corner_detectors[name] = CornerDetectorInputs(
                raw=raw_by_name[name],
                calibs=IsrCalibs(
                    ptc=ptc_by_name[name],
                    flat=flat_by_name[name],
                    linearizer=linearizer_by_name[name],
                    crosstalk=crosstalk_by_name[name],
                ),
            )

        return _CornerInputs(
            det_names=det_names,
            det_name_by_id=det_name_by_id,
            raw_by_name=raw_by_name,
            corner_detectors=corner_detectors,
        )

    def _indexIntrinsics(self, intrinsic_zernikes: list | None, det_name_by_id: dict[int, str]) -> dict:
        """Index the intrinsic Zernike calibrations by detector name.

        The calibrations are the one input arriving keyed by detector *id*
        rather than name, which is why they are indexed apart from the rest.
        """
        if intrinsic_zernikes:
            self.log.info("Loaded %d intrinsic Zernike calibration(s).", len(intrinsic_zernikes))
        else:
            self.log.warning("No intrinsic Zernike calibrations provided.")
        # det_name_by_id only covers the raws present, so a calibration for a
        # detector we are not processing is dropped rather than raising.
        # runQuantum already filters these, but run() is also called directly.
        by_name = {}
        for iz in intrinsic_zernikes or []:
            iz_det_id = iz.getMetadata()["LSST BUTLER DATAID DETECTOR"]
            iz_det_name = det_name_by_id.get(iz_det_id)
            if iz_det_name is None:
                self.log.debug(
                    "Ignoring intrinsic Zernike calibration for detector %s: no raw.",
                    iz_det_id,
                )
                continue
            by_name[iz_det_name] = iz
        return by_name

    def _loadRefcats(self, ref_cat: list, raw_by_name: dict) -> dict:
        """Load one refcat per detector, and stub the astrometry task.

        Done in the parent so the shards are loaded once and inherited by the
        cutout workers copy-on-write, which is the whole reason corner mode's
        workers need no butler.  A shard load that fails is a warning and a
        ``None`` entry, not an abort: that detector falls back to blitz
        detection, the same as when no refcat was supplied at all.
        """
        loader = None
        if not ref_cat:
            self.log.warning("No reference catalog shards provided; skipping WCS refit and donut selection.")
        else:
            self.log.info("Loading reference catalog shards for WCS refit and donut selection.")
            loader = ReferenceObjectLoader(
                dataIds=[h.dataId for h in ref_cat],
                refCats=ref_cat,
            )
            loader.config.pixelMargin = 300  # extra tolerance for uncertain WCS

        det_refcats: dict = {}
        for name, raw in raw_by_name.items():
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

        # Stub loader: AstrometryTask.solve() calls
        # refObjLoader.getMetadataBox() unconditionally even when load_result
        # is pre-supplied. That method is pure geometry -- it never accesses
        # catalog data, dataId.region, or the flux aliases.  Installed here
        # rather than with the store because it is the other half of handing
        # solve() a pre-loaded result.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.pixelMargin = 0
        self.astrometry.setRefObjLoader(astrom_stub_loader)

        return det_refcats

    def _populateCowStore(
        self,
        inputs: _CornerInputs,
        det_refcats: dict,
        band: str,
        photo_filter_name: str,
    ) -> None:
        """Fill `_COW_STORE` with everything the workers read.

        Named to match `DonutBlitzFamTask._populateCowStore` so the two modes'
        store population can be diffed.  Corner mode's is far shorter, because
        its workers inherit materialized exposures rather than handles and both
        its pools read the same store.
        """
        # The telescope is band- and quantum-fixed, and its 41 ms YAML load is
        # the part worth doing once here rather than per donut in a worker;
        # defocusing it costs 20 us, so the workers do that on demand.
        _COW_STORE.adopt(
            CowStore.for_corner(
                isr_task=self.isr,
                bkg_task=self.subtractBackground,
                diam_task=self.measureDiameter,
                detect_task=self.blitzDetect,
                astrom_task=self.astrometry,
                select_task=self.donutSelector,
                measure_task=self.measureCandidates,
                cut_task=self.cutStamps,
                wf_fit_task=self.wavefrontFit,
                wf_estimation_mode=self.config.wfEstimationMode,
                max_fit_scatter=self.config.maxFitScatter,
                astrom_ref_filter=self.config.astromRefFilter,
                photo_ref_filter=photo_filter_name,
                telescope=batoid.Optic.fromYaml(f"LSST_{band}.yaml"),
                corner_detectors=inputs.corner_detectors,
                det_refcats=det_refcats,
            )
        )

    def _runCutoutPool(self, det_names: tuple[str, ...], num_cores: int) -> list[CutoutResult]:
        """Cut stamps on every detector, forking if asked to.

        One work unit per detector.  A killed worker costs its detector and
        nothing else: `_fork_map` returns it under ``deaths`` and it becomes a
        `CutoutResult.dead` here, so the result list still covers every
        detector asked for.
        """
        self.log.info(
            "Running cutout workers on %d corner detectors with %d core(s)",
            len(det_names),
            num_cores,
        )
        if num_cores == 1:
            t_dispatch = time.time()
            return [_cutout_corner_detector((name, t_dispatch)) for name in det_names]

        t_pool0 = time.perf_counter()
        # Never more workers than detectors to process, matching the WF pool.
        # det_names is the detectors with raws, non-empty by _indexInputs.
        n_cutout_workers = min(num_cores, len(det_names))
        # Bare fork workers are safe here. Everything is preloaded in
        # runQuantum and inherited via COW. _fork_map ensures that one
        # killed worker does not take down the entire pool/quantum.
        t_dispatch = time.time()
        with _dump_stacks_on_hang(self.config.hangTimeout, "cutout pool", self.log):
            results, deaths = _fork_map(
                _cutout_corner_detector,
                [(name, t_dispatch) for name in det_names],
                n_cutout_workers,
                unit_timeout=self.config.unitTimeout,
            )
        for unit, reason in deaths:
            # A killed worker is a real fault, not a routine per-detector
            # failure, so it is logged at error level even though the visit
            # goes on without it.
            self.log.error("Cutout worker for detector %s died: %s", unit[0], reason)
            results.append(CutoutResult.dead(unit[0], reason))
        # One fork per detector, started as slots free up, so there is no
        # separate pool-creation phase left to time.
        self.log.info(
            _colorize(
                "Cutout pipeline: %d worker(s), %d/%d detector(s) returned, map: %.3fs",
                _ANSI_BOLD,
                _ANSI_CYAN,
                enabled=self._colorLogEnabled,
            ),
            n_cutout_workers,
            len(det_names) - len(deaths),
            len(det_names),
            time.perf_counter() - t_pool0,
        )
        return results

    def _logCutoutSummaries(self, results: list[CutoutResult]) -> list:
        """Log one stage-timing line per detector, **and** flatten the donuts.

        Not a pure logging method, unlike full-array mode's
        `DonutBlitzFamTask._logWorkerSummaries`: it returns the concatenated
        selected-donut list, because it is already walking every result in the
        order the catalog wants them.  Splitting the two apart would mean two
        walks and a reader hunting for where `donuts` is built.
        """
        donuts = []
        for r in results:
            scatter_str = f'{r.scatter_arcsec:.3f}"' if r.scatter_arcsec is not None else "N/A"
            # Stage columns come off `_CUTOUT_STAGE_KEYS` rather than being
            # spelled out, so a stage added to the cutout pipeline reaches this
            # line for free.  A stage the worker never reached prints as `nan`
            # by design: a missing stage should not read as a fast one.
            pieces = [f"dispatch={r.dispatch_to_arrival:.3f}s"]
            for label, key in _CUTOUT_STAGE_KEYS.items():
                piece = f"{label}={getattr(r, key):.3f}s"
                # Scatter belongs to the WCS refit, so it hangs off that stage.
                pieces.append(f"{piece} (scatter={scatter_str})" if label == "astrom" else piece)
            pieces.append(f"donuts={len(r.catalog)}")
            self.log.info("  %s: %s", r.det_name, "  ".join(pieces))
            if r.wcs_refit_error:
                self.log.warning("  %s: WCS refit failed: %s", r.det_name, r.wcs_refit_error)
            if r.cat_select_error:
                self.log.warning(
                    "  %s: catalog selection failed: %s",
                    r.det_name,
                    r.cat_select_error,
                )
            donuts.extend(r.catalog)
        return donuts

    def _annotateDonuts(self, results: list[CutoutResult], intrinsic_zernikes_by_name: dict) -> None:
        """Annotate each donut with its defocal offsets and intrinsics.

        Rejected donuts are annotated alongside the selected ones in both
        cases, and for the same reason: they get a row in the output catalog
        too.  The offsets are additionally required by
        `WavefrontFittingTask._prep_donut_for_danish` for anything it is
        handed, and the intrinsics are a function of field position rather than
        of whether a donut passed selection.  Full-array mode likewise
        annotates both lists.
        """
        for r in results:
            calib = intrinsic_zernikes_by_name.get(r.det_name)
            for d in r.catalog + r.rejected_catalog:
                # In corner mode the side of focus follows from the detector:
                # SW0 is extra-focal, SW1 intra-focal.
                d.defocal_offsets = (
                    _INTRA_FOCAL_OFFSETS if d.det_id in _INTRA_FOCAL_DET_IDS else _EXTRA_FOCAL_OFFSETS
                )
                if calib is not None:
                    d.intrinsic_zk = np.squeeze(
                        calib.getIntrinsicZernikes(
                            np.degrees(d.thx_ccs),
                            np.degrees(d.thy_ccs),
                        )
                    )
                else:
                    d.intrinsic_zk = None

    def _runWfPool(self, groups: list, num_cores: int) -> list[WfGroupResult]:
        """Fit every wavefront group, forking if asked to.

        A single group runs inline even on many cores: the fork would cost more
        than it saves.  As in the cutout pool a killed worker becomes a dead
        record rather than taking the quantum with it.
        """
        mode = self.config.wfEstimationMode
        self.log.info("WF dispatch (%s): %d work unit(s)", mode, len(groups))
        t_wf0 = time.perf_counter()
        if not groups:
            wf_results = []
        elif num_cores == 1 or len(groups) == 1:
            wf_results = [_wf_fitting_worker(g) for g in groups]
        else:
            n_workers = min(num_cores, len(groups))
            # _fork_map again ensures that one killed worker does not take down
            # the entire pool/quantum
            with _dump_stacks_on_hang(self.config.hangTimeout, "WF pool", self.log):
                wf_results, wf_deaths = _fork_map(
                    _wf_fitting_worker,
                    groups,
                    n_workers,
                    unit_timeout=self.config.unitTimeout,
                )
            n_zk = len(self.wavefrontFit.config.nollIndices)
            for group, reason in wf_deaths:
                self.log.error("WF worker for group %s died: %s", group.group_id, reason)
                wf_results.append(WfGroupResult.dead(group, reason, n_zk))
        elapsed_fits = [r.fit_elapsed for r in wf_results]
        self.log.info(
            "WF results (%s): %d/%d succeeded  wall=%.1fs  fit_total=%.1fs  fit_mean=%.1fs",
            mode,
            sum(r.success for r in wf_results),
            len(wf_results),
            time.perf_counter() - t_wf0,
            sum(e for e in elapsed_fits if not np.isnan(e)),
            np.nanmean(elapsed_fits) if elapsed_fits else float("nan"),
        )
        return wf_results

    def _catalogOptions(self) -> _CatalogOptions:
        """Gather the config-derived scalars the output catalog needs.

        Corner mode saves both image column sets by default: its row count is
        small enough that they cost tens of MB, and the diagnostic plots want
        them.
        """
        return _CatalogOptions(
            stamp_size=self.cutStamps.config.stampSize,
            binning=self.wavefrontFit.config.binning,
            noll_indices=tuple(self.wavefrontFit.config.nollIndices),
            aperture_margin_frac=self.measureCandidates.config.apertureMarginFrac,
            bkg_inner_disc_frac=self.measureCandidates.config.bkgInnerDiscFrac,
            bkg_annulus_inner_frac=self.measureCandidates.config.bkgAnnulusInnerFrac,
            bkg_annulus_outer_frac=self.measureCandidates.config.bkgAnnulusOuterFrac,
            max_donuts=self.cutStamps.config.maxDonuts,
            wf_mode=self.config.wfEstimationMode,
            save_stamps=self.config.saveStamps,
            save_wf_images=True,
            bkg_order=self.wavefrontFit.config.bkgOrder,
        )
