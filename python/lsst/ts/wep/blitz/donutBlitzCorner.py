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

"""The corner-wavefront-sensor blitz pipeline task."""

__all__ = [
    "DonutBlitzCornerConnections",
    "DonutBlitzCornerConfig",
    "DonutBlitzCornerTask",
]

import time
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
)
from .wavefrontFitting import (
    WavefrontFittingTask,
    _build_wf_groups,
    _wf_fitting_worker,
)


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
    saveStamps: pexConfig.Field = pexConfig.Field(
        doc=(
            "Include the unbinned `stamp` image column in the output catalog. "
            "The diagnostic plots use it when present and fall back to the "
            "binned `wf_img` when not."
        ),
        dtype=bool,
        default=True,
    )
    savePlots: pexConfig.Field = pexConfig.Field(
        doc=(
            "Generate diagnostic PNGs for each visit. "
            "Set False in production to skip plot generation and deliver "
            "Zernikes faster; plots can be generated later by calling "
            "plot.run() with the in-memory results."
        ),
        dtype=bool,
        default=False,
    )
    unitTimeout: pexConfig.Field = pexConfig.Field(
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
        dtype=float,
        default=30.0,
        optional=True,
    )
    hangTimeout: pexConfig.Field = pexConfig.Field(
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
        dtype=float,
        default=180.0,
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
            "full_corner": (
                "Dispatch all donuts from a corner (SW0+SW1, whichever are present) as one work unit."
            ),
            "full_detector": "Dispatch all donuts on each detector as one work unit (8 fits per visit).",
        },
        default="paired",
    )
    wavefrontFit: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=WavefrontFittingTask,
        doc="Wavefront fitting subtask using Danish algorithm.",
    )
    plot: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutBlitzPlotTask,
        doc="Subtask that generates diagnostic plots for a blitz visit.",
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

        det_name_by_id = {}
        raw_by_name = {}
        for exp in raws:
            det = exp.getDetector()
            det_name_by_id[det.getId()] = det.getName()
            raw_by_name[det.getName()] = exp
        ptc_by_name = {p._detectorName: p for p in ptc}
        flat_by_name = {f.getDetector().getName(): f for f in flat}
        linearizer_by_name = {lin._detectorName: lin for lin in linearizer}
        crosstalk_by_name = {ct._detectorName: ct for ct in crosstalk}

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
        det_names = sorted(raw_by_name)
        missing = CORNER_DET_NAMES - raw_by_name.keys()
        if missing:
            self.log.warning(
                "Processing %d/%d corner detectors; no raw for: %s",
                len(det_names),
                len(CORNER_DET_NAMES),
                sorted(missing),
            )

        if intrinsic_zernikes:
            self.log.info("Loaded %d intrinsic Zernike calibration(s).", len(intrinsic_zernikes))
        else:
            self.log.warning("No intrinsic Zernike calibrations provided.")
        self.intrinsicZernikes = list(intrinsic_zernikes) if intrinsic_zernikes else []
        # det_name_by_id only covers the raws present, so a calibration for a
        # detector we are not processing is dropped rather than raising.
        # runQuantum already filters these, but run() is also called directly.
        intrinsic_zernikes_by_name = {}
        for iz in self.intrinsicZernikes:
            iz_det_id = iz.getMetadata()["LSST BUTLER DATAID DETECTOR"]
            iz_det_name = det_name_by_id.get(iz_det_id)
            if iz_det_name is None:
                self.log.debug(
                    "Ignoring intrinsic Zernike calibration for detector %s: no raw.",
                    iz_det_id,
                )
                continue
            intrinsic_zernikes_by_name[iz_det_name] = iz

        band = next(iter(raw_by_name.values())).filter.bandLabel
        if self.config.photoRefFilter is not None:
            photo_filter_name = self.config.photoRefFilter
        else:
            photo_filter_name = f"{self.config.photoRefFilterPrefix}_{band}"

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

        t_refcat0 = time.perf_counter()
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
        t_refcat_elapsed = time.perf_counter() - t_refcat0

        # Stub loader: AstrometryTask.solve() calls
        # refObjLoader.getMetadataBox() unconditionally even when load_result
        # is pre-supplied. That method is pure geometry -- it never accesses
        # catalog data, dataId.region, or the flux aliases.
        astrom_stub_loader = ReferenceObjectLoader(dataIds=[], refCats=[])
        astrom_stub_loader.config.pixelMargin = 0
        self.astrometry.setRefObjLoader(astrom_stub_loader)

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

        visit_info = next(iter(raw_by_name.values())).getInfo().getVisitInfo()
        boresight_rot_rad = visit_info.boresightRotAngle.asRadians()
        boresight_par_rad = visit_info.boresightParAngle.asRadians()
        boresight_alt_rad = visit_info.boresightAzAlt.getLatitude().asRadians()
        # rotTelPos, wrapped to (-pi, pi]. Always computed: the CCS -> OCS
        # Zernike rotation in _buildCatalog needs it, whereas spider shadows
        # are opt-in.
        rtp_rad = (boresight_par_rad - boresight_rot_rad - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        rtp_deg = np.degrees(rtp_rad) if self.wavefrontFit.config.modelSpiderShadows else None

        # Everything the cutout and fit workers read, in one place. The
        # telescope is band- and quantum-fixed, and its 41 ms YAML load is the
        # part worth doing once here rather than per donut in a worker;
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
                corner_detectors=corner_detectors,
                det_refcats=det_refcats,
            )
        )

        cutout_args = det_names

        self.log.info(
            "Running cutout workers on %d corner detectors with %d core(s)",
            len(cutout_args),
            num_cores,
        )
        t_cutout0 = time.perf_counter()
        if num_cores == 1:
            t_dispatch = time.time()
            results = [_cutout_corner_detector((arg, t_dispatch)) for arg in cutout_args]
        else:
            t_pool0 = time.perf_counter()
            # Never more workers than detectors to process, matching the WF
            # pool below. cutout_args is the detectors with raws, non-empty by
            # the guard above.
            n_cutout_workers = min(num_cores, len(cutout_args))
            # Bare fork workers are safe here. Everything is preloaded in
            # runQuantum and inherited via COW. _fork_map ensures that one
            # killed worker does not take down the entire pool/quantum.
            t_dispatch = time.time()
            with _dump_stacks_on_hang(self.config.hangTimeout, "cutout pool", self.log):
                results, deaths = _fork_map(
                    _cutout_corner_detector,
                    [(arg, t_dispatch) for arg in cutout_args],
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
                len(cutout_args) - len(deaths),
                len(cutout_args),
                time.perf_counter() - t_pool0,
            )
        t_cutout1 = time.perf_counter()

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

        # Annotate the optic shifts that put each donut off focus. In corner
        # mode this follows from the detector: SW0 is extra-focal, SW1
        # intra-focal. Rejected donuts are annotated too -- they get a row in
        # the output catalog, and _prep_donut_for_danish requires the offsets
        # of anything it is handed.
        for r in results:
            for d in r.catalog + r.rejected_catalog:
                d.defocal_offsets = (
                    _INTRA_FOCAL_OFFSETS if d.det_id in _INTRA_FOCAL_DET_IDS else _EXTRA_FOCAL_OFFSETS
                )

        # Annotate every donut with realized intrinsic Zernikes, rejected ones
        # included: intrinsics are a function of field position, not of whether
        # the donut passed selection, and rejected donuts get catalog rows too.
        # Full-array mode already annotates both lists.
        for r in results:
            calib = intrinsic_zernikes_by_name.get(r.det_name)
            for d in r.catalog + r.rejected_catalog:
                if calib is not None:
                    d.intrinsic_zk = np.squeeze(
                        calib.getIntrinsicZernikes(
                            np.degrees(d.thx_ccs),
                            np.degrees(d.thy_ccs),
                        )
                    )
                else:
                    d.intrinsic_zk = None

        # WF dispatch
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
        t_wf1 = time.perf_counter()
        n_ok = sum(r.success for r in wf_results)
        elapsed_fits = [r.fit_elapsed for r in wf_results]
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
                run_elapsed=t_plot0 - t_run0,
                refcat_elapsed=t_refcat_elapsed,
                butler_elapsed=butler_elapsed,
                butler_times=butler_times or {},
                cutout_elapsed=t_cutout1 - t_cutout0,
                danish_elapsed=t_wf1 - t_wf0,
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

        return pipeBase.Struct(donuts=donuts, wfResults=wf_results, cornerResults=Table(catalog))

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
        )
