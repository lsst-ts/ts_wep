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

"""The monolithic corner-wavefront WEP pipeline task."""

__all__ = [
    "DonutBlitzMonolithTaskConnections",
    "DonutBlitzMonolithTaskConfig",
    "DonutBlitzMonolithTask",
]

import multiprocessing as mp
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

from .blindDetectTask import BlindDetect
from .catalogBuilder import CatalogOptions, build_donut_catalog
from .cutDonutStampsTask import CutDonutStampsTask
from .cutoutPipeline import _run_cutout_worker
from .donutBlitzPlotTask import DonutBlitzPlotTask
from .measureDonutCandidatesTask import MeasureDonutCandidatesTask
from .utils import (
    CORNER_DET_NAMES,
    _ANSI_BOLD,
    _ANSI_CYAN,
    _ANSI_GREEN,
    _CALIB_STORE,
    _INSTRUMENT,
    _INTRA_FOCAL_DET_IDS,
    _colorize,
    _resolveColorLogEnabled,
    _telescope_for_offsets,
)
from .wavefrontFittingTask import (
    WavefrontFittingTask,
    _build_wf_groups,
    _wf_fitting_worker,
)


# Corner mode defocuses by shifting the detector plane inside the camera: SW0
# (extra-focal) sits at +defocalOffset, SW1 (intra-focal) at -defocalOffset.
# Ordered as `_OFFSET_OPTICS`: (detector, camera, m2). Full-array mode shifts the
# whole camera instead, which is why the offsets ride on each Donut rather than
# being assumed by the fitter.
_EXTRA_FOCAL_OFFSETS = (+_INSTRUMENT.defocalOffset, 0.0, 0.0)
_INTRA_FOCAL_OFFSETS = (-_INSTRUMENT.defocalOffset, 0.0, 0.0)


class DonutBlitzMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzMonolithTask."""

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
            "full_corner": (
                "Dispatch all donuts from a corner (SW0+SW1, whichever are present) "
                "as one work unit."
            ),
            "full_detector": "Dispatch all donuts on each detector as one work unit (8 fits per visit).",
        },
        default="paired",
    )
    wfFittingTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=WavefrontFittingTask,
        doc="Wavefront fitting subtask using Danish algorithm.",
    )
    plotTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutBlitzPlotTask,
        doc="Subtask that generates diagnostic plots for a blitz visit.",
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
    selection, and stamp cutting on whichever corner detector raws are present,
    in parallel using a multiprocessing pool.  Reference catalogs are loaded in
    the parent process before forking and inherited by workers via
    copy-on-write.
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

        # Process whichever corner raws arrived. A partial set is normal (dropped
        # image, per-detector butler gap) and there is no reason to throw away the
        # corners that did arrive, so this is a warning rather than an abort.
        # `detNames` -- not CORNER_DET_NAMES -- drives everything downstream.
        unexpected = rawByName.keys() - CORNER_DET_NAMES
        if unexpected:
            raise RuntimeError(
                f"Non-corner detector raws supplied: {sorted(unexpected)}"
            )
        if not rawByName:
            raise RuntimeError("No corner detector raws supplied.")
        detNames = sorted(rawByName)
        missing = CORNER_DET_NAMES - rawByName.keys()
        if missing:
            self.log.warning(
                "Processing %d/%d corner detectors; no raw for: %s",
                len(detNames),
                len(CORNER_DET_NAMES),
                sorted(missing),
            )

        if intrinsicZernikes:
            self.log.info("Loaded %d intrinsic Zernike calibration(s).", len(intrinsicZernikes))
        else:
            self.log.warning("No intrinsic Zernike calibrations provided.")
        self.intrinsicZernikes = list(intrinsicZernikes) if intrinsicZernikes else []
        # detNameById only covers the raws present, so a calibration for a
        # detector we are not processing is dropped rather than raising.
        # runQuantum already filters these, but run() is also called directly.
        intrinsicZernikesByName = {}
        for iz in self.intrinsicZernikes:
            iz_det_id = iz.getMetadata()["LSST BUTLER DATAID DETECTOR"]
            iz_det_name = detNameById.get(iz_det_id)
            if iz_det_name is None:
                self.log.debug(
                    "Ignoring intrinsic Zernike calibration for detector %s: no raw.",
                    iz_det_id,
                )
                continue
            intrinsicZernikesByName[iz_det_name] = iz

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
        for name in detNames:
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
        # rotTelPos, wrapped to (-pi, pi]. Always computed: the CCS -> OCS Zernike
        # rotation in _buildCatalog needs it, whereas spider shadows are opt-in.
        rtp_rad = (boresight_par_rad - boresight_rot_rad - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        rtp_deg = np.degrees(rtp_rad) if self.wfFittingTask.config.modelSpiderShadows else None

        # Store task and mode for WF worker access
        _CALIB_STORE["wf_fitting_task"] = self.wfFittingTask
        _CALIB_STORE["wfEstimationMode"] = self.config.wfEstimationMode

        # Telescope is band- and quantum-fixed; build once here and share via
        # COW instead of reloading "LSST_{band}.yaml" per donut in workers.
        _telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
        _CALIB_STORE["telescope"] = _telescope
        # Pre-build both defocused telescopes so the workers only ever look them
        # up. Corner mode defocuses by moving the detector plane, hence the
        # (detector, camera, m2) triplets below.
        _CALIB_STORE.pop("telescope_by_offsets", None)
        for _offsets in (_EXTRA_FOCAL_OFFSETS, _INTRA_FOCAL_OFFSETS):
            _telescope_for_offsets(_offsets)

        cutout_args = detNames

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
            # below. cutout_args is the detectors with raws, non-empty by the
            # guard above.
            n_cutout_workers = min(numCores, len(cutout_args))
            # A bare fork Pool is safe here only because these workers never
            # touch the butler -- everything is preloaded in runQuantum and
            # inherited via COW. Children sharing the parent's inherited psycopg2
            # SSL socket corrupt it (~5-10% of 8-worker forks), so any worker
            # that starts reading from the butler needs `initializer=`
            # calling SqlRegistry.resetConnectionPool().
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

        # Annotate the defocal side and the optic shifts that produced it. In
        # corner mode this follows from the detector: SW0 is extra-focal, SW1
        # intra-focal. Rejected donuts are annotated too -- they get a row in the
        # output catalog, so they need a defocal label as much as accepted ones do.
        for r in results:
            for d in r["catalog"] + r.get("rejected_catalog", []):
                is_intra = d.det_id in _INTRA_FOCAL_DET_IDS
                d.defocal = "intra" if is_intra else "extra"
                d.defocal_offsets = (
                    _INTRA_FOCAL_OFFSETS if is_intra else _EXTRA_FOCAL_OFFSETS
                )

        # Annotate each accepted donut with realized intrinsic Zernikes.
        for r in results:
            calib = intrinsicZernikesByName.get(r["det_name"])
            for d in r["catalog"]:
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
            # Butler-free like the cutout pool above; same caveat applies.
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

        catalog = build_donut_catalog(
            results=results,
            wf_results=wf_results,
            donuts=donuts,
            unmatched_donuts=unmatched_donuts,
            visit_id=visit_id,
            options=self._catalogOptions(),
            run_elapsed=t_plot0 - t_run0,
            refcat_elapsed=t_refcat_elapsed,
            butler_elapsed=butler_elapsed,
            butler_times=butler_times or {},
            cutout_elapsed=t_cutout1 - t_cutout0,
            danish_elapsed=t_wf1 - t_wf0,
            photo_filter_name=photo_filter_name,
            astrom_filter_name=self.config.astromRefFilter,
            rtp_rad=rtp_rad,
        )

        if self.config.savePlots:
            self.plotTask.run(catalog)
            self.log.info("Diagnostic plot: %.3fs", time.perf_counter() - t_plot0)

        return pipeBase.Struct(
            donuts=donuts,
            wf_results=wf_results,
            blitzResults=Table(catalog)
        )

    def _catalogOptions(self) -> CatalogOptions:
        """Gather the config-derived scalars the output catalog needs.

        Corner mode saves both image column sets by default: its row count is
        small enough that they cost tens of MB, and the diagnostic plots want
        them.
        """
        return CatalogOptions(
            stamp_size=self.cutStampsTask.config.stampSize,
            binning=self.wfFittingTask.config.binning,
            noll_indices=tuple(self.wfFittingTask.config.nollIndices),
            aperture_outer_margin_frac=self.measureCandidatesTask.config.apertureOuterMarginFrac,
            aperture_inner_buffer_frac=self.measureCandidatesTask.config.apertureInnerBufferFrac,
            bkg_annulus_inner_frac=self.measureCandidatesTask.config.bkgAnnulusInnerFrac,
            bkg_annulus_outer_frac=self.measureCandidatesTask.config.bkgAnnulusOuterFrac,
            max_donuts=self.cutStampsTask.config.maxDonuts,
            wf_mode=self.config.wfEstimationMode,
            save_stamps=self.config.saveStamps,
            save_wf_images=True,
        )
