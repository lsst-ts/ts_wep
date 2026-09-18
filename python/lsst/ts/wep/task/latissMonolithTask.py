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

"""Self-contained LATISS wavefront estimation, from raws to Zernikes.

One quantum does everything ``latiss_wep_align.run_wep`` does, plus ISR:

    ISR -> QuickFrameMeasurement -> CutOutDonutsScienceSensorTask
        -> peak-normalize -> EstimateZernikesDanishTask

Detection is ``QuickFrameMeasurement`` rather than the ts_wep donut detection
tasks because a LATISS alignment exposure has one bright donut near the
boresight, which is what ``latiss_wep_align`` relies on. The one LATISS-
specific step before the fit is peak normalization: danish's flux and
sky-level parameters are scale dependent, and the modern cutout task returns
raw ADU (~1e5) where ts_wep <= 15.1.0 returned stamps with peak ~1. Without it
the stock fit stops after a handful of function evaluations. Everything else
is the stock ts_wep machinery. The background for these choices is on RSO-873.

``run`` is callable without a butler, so ``run_wep`` in ts_externalscripts can
call it directly.
"""

__all__ = [
    "LatissMonolithTaskConnections",
    "LatissMonolithTaskConfig",
    "LatissMonolithTask",
    "peakNormalize",
]

import copy
import logging
import warnings
from typing import Any, cast

import astropy.units as u
import numpy as np
from astropy.table import QTable

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from lsst.ip.isr import IsrTaskLSST
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
)
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable
from lsst.ts.wep.task.pairTask import ExposurePairer
from lsst.utils.timer import timeMethod

# LATISS plate scale, arcsec/pixel, measured astrometrically on DM-24592, from
# lsst.ts.observatory.control.constants.latiss_constants. Needed because
# ``run_wep`` compares its boresight distance in ARCSECONDS
# (``calculate_xy_offsets`` multiplies by this), so a threshold expressed in
# pixels would be ~10x too strict.
LATISS_PIXEL_SCALE = 0.09569

# Structured dtype for the paired (x, y) columns, matching calcZernikesTask so
# that the output table stays readable by the same donut_viz code.
pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])


def peakNormalize(stamps: DonutStamps) -> None:
    """Divide each stamp's ``wep_im`` by its peak, in place.

    Danish's flux and sky-level parameters are scale dependent, and the flux
    starting guess is clipped to [1e3, 1e8]: an unnormalized LATISS stamp sums
    to ~1e9, so the fit starts far off and stops early. Stamps with peak ~1,
    as ts_wep <= 15.1.0 produced, fit properly.

    Only ``wep_im`` is touched; ``stamp_im`` keeps the ADU pixels.

    Raises
    ------
    ValueError
        If a stamp has no finite positive peak.
    """
    for stamp in stamps:
        arr = np.asarray(stamp.wep_im.image, dtype=float)
        peak = float(np.nanmax(arr))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("stamp has no finite positive peak")
        stamp.wep_im.image = arr / peak


class LatissMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),  # type: ignore
):
    """Connections for LatissMonolithTask.

    One quantum per CWFS pair, keyed on the **extra-focal** visit, as the
    ts_wep paired tasks and ``donutBlitzMonolith`` are. The pairing happens at
    graph-build time in ``adjust_all_quanta``: the default graph gives every
    exposure its own quantum holding its own raw; the intra-focal raw is then
    moved into its partner's quantum and the intra quantum dropped. Keying on
    visit is what makes ``_log``/``_metadata`` per pair -- with a per-night or
    per-run quantum every pair in a long-lived output run (rapid analysis
    reuses ``LATISS/runs/quickLook/N``) would write the same dataId and the
    second pair would fail on the provenance datasets. ``visit`` also implies
    ``day_obs``, so the calibration lookup is time-bounded.
    """

    # Populated by pex_config; declared for adjust_all_quanta and mypy.
    config: Any

    raws = ct.Input(
        doc="Raw LATISS exposures of one CWFS pair: the extra-focal raw of this "
        "quantum's visit plus its intra-focal partner, attached by adjust_all_quanta.",
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    camera = ct.PrerequisiteInput(
        doc="Input camera geometry.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    # The full calibration set, matching what BestEffortIsr passes on the
    # summit.
    # Withholding these was the cause of the QFM mis-picks: with doDefect off,
    # the LATISS defect column at x=3795-3797 (y 6-1999) survived ISR at ~1.2e5
    # ADU against an image median of ~20, and since QuickFrameMeasurement ranks
    # candidates on a 70 px aperture flux -- which a solid column fills more
    # uniformly than a donut with a hole -- the column outranked the donut.
    # Enabling defects moved 38 of 60 previously-bad pair sides back on-axis
    # (median pick distance 2092 px -> 50 px) and broke none. See RSO-873.
    bias = ct.PrerequisiteInput(
        doc="Combined bias calibration frame.",
        name="bias",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    dark = ct.PrerequisiteInput(
        doc="Combined dark calibration frame.",
        name="dark",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    flat = ct.PrerequisiteInput(
        doc="Combined flat calibration frames, one per physical_filter.",
        name="flat",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
        multiple=True,
        minimum=0,
    )
    defects = ct.PrerequisiteInput(
        doc="Defect list; masks the bad LATISS column that otherwise outranks the donut.",
        name="defects",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    linearizer = ct.PrerequisiteInput(
        doc="Linearity correction calibration.",
        name="linearizer",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    crosstalk = ct.PrerequisiteInput(
        doc="Intra-detector crosstalk coefficients.",
        name="crosstalk",
        storageClass="CrosstalkCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    ptc = ct.PrerequisiteInput(
        doc="Photon transfer curve dataset. Required by IsrTaskLSST.",
        name="ptc",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    zernikes = ct.Output(
        doc="Zernike coefficients for the pair and an average row, with fit quality columns.",
        name="zernikes",
        storageClass="AstropyQTable",
        dimensions=("visit", "detector", "instrument"),
    )
    donutStampsExtra = ct.Output(
        doc="Extra-focal donut postage stamps.",
        name="donutStampsExtra",
        storageClass="StampsBase",
        dimensions=("visit", "detector", "instrument"),
    )
    donutStampsIntra = ct.Output(
        doc="Intra-focal donut postage stamps, stored under the extra-focal visit.",
        name="donutStampsIntra",
        storageClass="StampsBase",
        dimensions=("visit", "detector", "instrument"),
    )

    def __init__(self, *, config: Any | None = None) -> None:
        super().__init__(config=config)
        if config is not None and not config.doSaveStamps:
            del self.donutStampsExtra
            del self.donutStampsIntra

    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """Turn one-quantum-per-exposure into one-quantum-per-pair.

        Every exposure in the data query starts with its own quantum. An
        exposure whose ``observation_reason`` contains ``extra`` keeps its
        quantum and receives the raw of the intra-focal exposure taken
        immediately before it (``seq_num - 1`` on the same night, which is how
        ``latiss_wep_align`` takes the pair). All other quanta are removed:
        intra exposures once their raw has been re-homed, and unpaired extras
        (whose partner was not in the query or is not marked ``intra``), with
        a warning. This mirrors ``ReassignCwfsCutoutsFamTask`` for the LSSTCam
        FAM pipeline. A data query naming both exposures of the pair, e.g.
        ``exposure in (17, 18)``, is therefore all that is required; nothing
        needs to be said about which is which.
        """
        log = logging.getLogger(__name__)
        butler = adjuster.butler
        data_ids = list(adjuster.iter_data_ids())
        if not data_ids:
            return
        instrument = data_ids[0]["instrument"]
        # LATISS visit ids equal exposure ids (one exposure per visit), and
        # the exposure record carries the intra/extra observation_reason.
        records = {
            rec.id: rec
            for rec in butler.registry.queryDimensionRecords(
                "exposure",
                where="instrument = :inst AND exposure IN (:ids)",
                bind={"inst": instrument, "ids": [int(d["visit"]) for d in data_ids]},
            )
        }
        by_exposure = {int(d["visit"]): d for d in data_ids}

        def reason(exposure: int) -> str:
            rec = records.get(exposure)
            return (rec.observation_reason or "").lower() if rec is not None else ""

        # First pass: give each extra-focal quantum its partner's raw. This
        # must finish before any quantum is removed, because add_input reads
        # the raw from the intra quantum that still holds it.
        keep = set()
        for exposure, data_id in by_exposure.items():
            if "extra" not in reason(exposure):
                continue
            partner = exposure - 1
            if partner not in by_exposure or "intra" not in reason(partner):
                log.warning(
                    "Dropping extra-focal exposure %d: intra-focal partner %d is not in the data "
                    "query (or is not marked 'intra').",
                    exposure,
                    partner,
                )
                continue
            for raw_data_id in adjuster.get_inputs(by_exposure[partner])["raws"]:
                adjuster.add_input(data_id, "raws", raw_data_id)
            keep.add(exposure)

        # Second pass: everything that is not a completed pair goes.
        for exposure, data_id in by_exposure.items():
            if exposure not in keep:
                adjuster.remove_quantum(data_id)


class LatissMonolithTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=LatissMonolithTaskConnections,  # type: ignore
):
    """Configuration for LatissMonolithTask."""

    isrTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="ISR subtask run on each raw exposure.",
    )
    quickFrameMeasurement: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=QuickFrameMeasurementTask,
        doc="Finds the bright central donut. Used in place of donut detection: "
        "a LATISS alignment exposure has one bright donut near the boresight, "
        "and this is what latiss_wep_align uses.",
    )
    cutOutDonuts: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=CutOutDonutsScienceSensorTask,
        doc="Stamp cutout subtask. Deliberately the stock ts_wep task: a "
        "hand-rolled box cutout was tried and centres the donut measurably "
        "worse (inner_frac ~1.11, Z8 collapsing from +86 to +2 nm).",
    )
    pairer: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=ExposurePairer,
        doc="Task to pair up intra- and extra-focal exposures.",
    )
    estimateZernikes: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=EstimateZernikesDanishTask,
        doc="Zernike estimation subtask, run on the peak-normalized stamps. "
        "nollIndices and instConfigFile are taken from here.",
    )
    donutDiameter: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=228,
        doc="Donut diameter in pixels, passed to QuickFrameMeasurement and used "
        "as the stamp size. 228 is what latiss_wep_align derives for dz=0.8 "
        "(ceil(192*1.1*0.8/1.5/2)*2 * 2). Note the default stamp size of 160 "
        "is LSSTCam-sized and CLIPS an AuxTel donut, which is 194 px across.",
    )
    opticalModel: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="onAxis",
        doc="Optical model for the cutout masks. Must be 'onAxis' for AuxTel: "
        "there is no off-axis batoid fit for AuxTel, so the 'offAxis' default "
        "is wrong.",
    )
    maxDistanceFromBoresight: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=500.0,
        doc="Maximum distance in ARCSECONDS from the boresight for a detected "
        "donut to be accepted. When one side is out of bounds the other side's "
        "centroid is substituted, as latiss_wep_align does. The unit matters: "
        "run_wep's default of 500 goes through calculate_xy_offsets, which "
        "converts to arcsec, so 500 is ~5225 px -- most of the detector, i.e. "
        "a sanity check rather than a tight cut.",
    )
    maxChiSquare: pexConfig.Field = pexConfig.Field(
        dtype=float,
        optional=True,
        default=None,
        doc="If set, pairs whose danish reduced chi-square exceeds this are "
        "flagged used=False and excluded from the average. Fit quality tracks "
        "Z4 error, so this is a usable cut, but a good threshold depends on "
        "stamp size and seeing -- hence no default.",
    )
    doSaveStamps: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to write the donut stamps as outputs.",
    )

    def setDefaults(self) -> None:
        super().setDefaults()

        # LATISS ISR, configured to match BestEffortIsr -- which is what
        # ``latiss_wep_align.run_wep`` uses on the summit, where detection
        # works. BestEffortIsr cannot be used here directly: it builds its own
        # Butler from a repo string and writes to CURRENT_RUN, and its engine
        # QuickLookIsrTask raises "IsrTaskLSST requires a PTC" on a raw.
        # So this replicates its configuration instead, on the same underlying
        # IsrTaskLSST. Reference: summit_utils config/quickLookIsr.py.
        #
        # An earlier version of this task ran gains + overscan only, on the
        # premise that "there are no usable bias/dark/flat calibrations for
        # alignment sequences". That premise was wrong -- bias, dark, flat,
        # defects, linearizer, crosstalk and ptc are all present for LATISS in
        # LATISS/defaults -- and withholding them made QuickFrameMeasurement
        # centroid on a detector artifact rather than the donut. See the
        # ``defects`` connection above for the mechanism and the effect.
        self.isrTask.doSaturation = True  # "very important for roundness in qfm"
        self.isrTask.brighterFatterMaxIter = 2
        self.isrTask.doDeferredCharge = False  # no calibration for this yet
        self.isrTask.doBootstrap = False
        self.isrTask.doApplyGains = True
        self.isrTask.doSuspect = False
        self.isrTask.defaultSaturationSource = "CAMERAMODEL"

        # Departures from BestEffortIsr, forced by LATISS calib availability:
        # there is no bfKernel/bfGains for LATISS, and IsrTaskLSST raises
        # "Must supply an kernel if BF correction method is COULTON*" if told
        # do brighter-fatter without one.
        self.isrTask.doBrighterFatter = False
        # Not needed downstream, and cheaper to skip.
        self.isrTask.doVariance = False

        # Everything else -- defects, flat, linearize, crosstalk, bias, dark,
        # interpolation, NaN masking -- is left at the IsrTaskLSST default of
        # True, which is what BestEffortIsr also does. Do not disable these
        # without re-checking donut detection on the pairs listed in RSO-873.

        # AuxTel is defocused by moving M2, so the extra-focal exposure has the
        # *smaller* focusZ -- inverted relative to LSSTCam. -0.8 mm is what
        # pairTask already hardcodes for LATISS; set it explicitly so the
        # pairing does not silently change if that default moves.
        self.pairer.doOverrideSeparation = True
        self.pairer.overrideSeparation = -0.8

        # High elevation limit for AuxTel is 86.5 deg;
        # difference in rtp between two succesive exposures at high elevation
        # may exceed the default 1 degree
        self.pairer.rotationThreshold = 1.5
        # Z4-Z22, as latiss_wep_align fits; the ts_wep default runs to Z28.
        self.estimateZernikes.nollIndices = list(range(4, 23))
        # Scale each parameter by its Jacobian column, as the LSSTCam Danish
        # pipelines do. The AuxTel Z4 direction (3.4 um of defocus) is badly
        # scaled against the tens-of-nm higher orders, and with scipy's unit
        # scaling the same stamps converge to Z4 values 40 nm apart on two
        # conda environments (numpy 2.3 vs 2.4). With 'jac' they agree to 4 nm.
        # The loose ftol/xtol/gtol the LSSTCam pipelines add are NOT copied:
        # on simulated LATISS donuts they stop the fit at its starting point.
        self.estimateZernikes.lstsqKwargs = {"x_scale": "jac"}


class LatissMonolithTask(pipeBase.PipelineTask):
    """Estimate LATISS Zernikes from raws in a single quantum.

    Notes
    -----
    No ``intrinsicZernikes`` connection: LATISS has no such calibration, so
    the ``*_intrinsic`` and ``*_deviation`` columns are NaN by design.

    No refcat, astrometry or WCS-refit config either. A LATISS alignment
    exposure has one bright donut, so ``minSourcesForWcsFit=3`` is never met
    and the refit path is dead weight.
    """

    ConfigClass = LatissMonolithTaskConfig
    _DefaultName = "latissMonolithTask"
    config: LatissMonolithTaskConfig
    # Set by makeSubtask, so declared here for the type checker.
    isrTask: IsrTaskLSST
    quickFrameMeasurement: QuickFrameMeasurementTask
    pairer: ExposurePairer
    estimateZernikes: EstimateZernikesDanishTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = cast(LatissMonolithTaskConfig, self.config)

        self.makeSubtask("isrTask")
        self.makeSubtask("quickFrameMeasurement")
        self.makeSubtask("pairer")
        self.makeSubtask("estimateZernikes")

        # The cutout subtask needs the AuxTel-specific stamp geometry. Build
        # its config here rather than in setDefaults so donutDiameter stays the
        # single place the stamp size is set.
        cutOutConfig = CutOutDonutsScienceSensorTaskConfig()
        cutOutConfig.donutStampSize = config.donutDiameter
        cutOutConfig.opticalModel = config.opticalModel
        cutOutConfig.initialCutoutPadding = 40
        if config.estimateZernikes.instConfigFile is not None:
            cutOutConfig.instConfigFile = config.estimateZernikes.instConfigFile
        self.cutOutDonuts = CutOutDonutsScienceSensorTask(config=cutOutConfig)

        self.nollIndices = np.array(config.estimateZernikes.nollIndices, dtype=int)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Identify the pair's two raws, run the chain, write the outputs.

        The quantum holds exactly the two raws of one CWFS pair (see
        ``adjust_all_quanta``). Which is which is decided from ``focusZ`` via
        the pairer, not from the quantum's visit, so a mislabelled
        ``observation_reason`` cannot swap intra and extra.
        """
        camera = butlerQC.get(inputRefs.camera)

        # The calibrations IsrTaskLSST needs. Each is declared minimum=0 so a
        # missing product degrades rather than failing the quantum, but the
        # detection depends on `defects`: without it the LATISS bad column
        # outranks the real donut (see the connection docstring).
        isrCalibs = {}
        for name in ("bias", "dark", "defects", "linearizer", "crosstalk", "ptc"):
            ref = getattr(inputRefs, name, None)
            if ref is None:
                continue
            value = butlerQC.get(ref)
            # minimum=0 connections arrive as a possibly-empty list.
            if isinstance(value, list):
                value = value[0] if value else None
            if value is not None:
                isrCalibs[name] = value
        # `flat` is per physical_filter: the quantum's visit pins one, but the
        # connection is multiple=True, so take whichever resolved.
        flats = butlerQC.get(getattr(inputRefs, "flat", None) or [])
        if flats:
            isrCalibs["flat"] = flats[0]
        missing = {"defects", "linearizer", "crosstalk", "ptc", "flat"} - set(isrCalibs)
        if missing:
            self.log.warning("ISR calibrations not found, donut detection may suffer: %s", sorted(missing))

        # butlerQC.get resolves the deferred refs into DeferredDatasetHandles;
        # the refs themselves cannot be read from.
        rawHandles = dict(
            zip(
                (ref.dataId["exposure"] for ref in inputRefs.raws),
                butlerQC.get(inputRefs.raws),
            )
        )
        if len(rawHandles) != 2:
            raise pipeBase.NoWorkFound(
                f"Expected the two raws of one CWFS pair, got {sorted(rawHandles)}; "
                "was adjust_all_quanta applied?"
            )
        visitInfos = {expId: handle.get(component="visitInfo") for expId, handle in rawHandles.items()}
        pairs = self.pairer.run(visitInfos)
        if len(pairs) != 1:
            raise pipeBase.NoWorkFound(
                f"Exposures {sorted(rawHandles)} do not form an intra/extra pair by focusZ."
            )
        pair = pairs[0]
        self.log.info("Fitting pair: extra=%d intra=%d", pair.extra, pair.intra)

        # The quantum's visit is the exposure adjust_all_quanta took to be
        # extra-focal from its observation_reason; the pairer decides from
        # focusZ. If they disagree the outputs land under what the header
        # calls the extra-focal exposure, but the fit itself follows focusZ.
        # Say so, because downstream code assumes the two agree.
        quantumDataId = butlerQC.quantum.dataId
        if quantumDataId is None:
            raise RuntimeError("Quantum has no dataId; cannot determine the visit.")
        quantumVisit = int(quantumDataId["visit"])
        if quantumVisit != pair.extra:
            self.log.warning(
                "Quantum visit %d is labelled extra-focal in the exposure record, but by focusZ "
                "exposure %d is extra-focal and %d is intra-focal. Fitting by focusZ; the "
                "outputs are stored under visit %d.",
                quantumVisit,
                pair.extra,
                pair.intra,
                quantumVisit,
            )

        outputs = self.run(
            rawHandles[pair.extra].get(), rawHandles[pair.intra].get(), camera, isrCalibs=isrCalibs
        )

        butlerQC.put(outputs.zernikes, outputRefs.zernikes)
        if self.config.doSaveStamps:
            butlerQC.put(outputs.donutStampsExtra, outputRefs.donutStampsExtra)
            # The intra stamps share the pair's (extra-focal) dataId.
            butlerQC.put(outputs.donutStampsIntra, outputRefs.donutStampsIntra)

    @timeMethod
    def run(
        self,
        rawExtra: afwImage.Exposure,
        rawIntra: afwImage.Exposure,
        camera: lsst.afw.cameraGeom.Camera,
        doIsr: bool = True,
        isrCalibs: dict | None = None,
    ) -> pipeBase.Struct:
        """Run the full chain on one intra/extra pair.

        Parameters
        ----------
        rawExtra, rawIntra : `lsst.afw.image.Exposure`
            The pair. Raw if ``doIsr``, else already ISR-corrected.
        isrCalibs : `dict`, optional
            Calibrations forwarded to ``isrTask.run`` (``defects``, ``flat``,
            ``linearizer``, ``crosstalk``, ``ptc``, ``bias``, ``dark``). Only
            used when ``doIsr``. Callers outside a pipeline -- ``run_wep``, for
            instance -- may omit it, but donut detection is markedly worse
            without ``defects``.
        camera : `lsst.afw.cameraGeom.Camera`
            LATISS camera geometry.
        doIsr : bool, optional
            Set False to pass in exposures that are already ISR-corrected,
            e.g. from ``BestEffortIsr`` when called outside a pipeline.

        Returns
        -------
        `lsst.pipe.base.Struct`
            ``zernikes`` (`QTable`), ``donutStampsExtra``,
            ``donutStampsIntra``, and ``wfEstInfo``, the per-pair metadata
            dict from ``EstimateZernikesDanishTask``.
        """
        if doIsr:
            # IsrTaskLSST returns .exposure (IsrTask returned .outputExposure).
            # isrCalibs carries defects/flat/linearizer/crosstalk/ptc/bias;
            # withholding them is what caused QFM to centroid on a detector
            # artifact, so pass through whatever runQuantum found.
            calibs = isrCalibs or {}
            # IsrTaskLSST raises if doFlat is set and no flat is supplied. Some
            # LATISS alignment exposures use filters with no flat (e.g.
            # 'unknown~empty'); a fit without flat-fielding beats no fit.
            if self.isrTask.config.doFlat and "flat" not in calibs:
                self.log.warning("No flat for this pair; running ISR without flat correction.")
                isrConfig = copy.deepcopy(self.isrTask.config)
                isrConfig.doFlat = False
                isrTask = self.isrTask.__class__(config=isrConfig)
            else:
                isrTask = self.isrTask
            expExtra = isrTask.run(rawExtra, camera=camera, **calibs).exposure
            expIntra = isrTask.run(rawIntra, camera=camera, **calibs).exposure
        else:
            expExtra, expIntra = rawExtra, rawIntra

        catExtra, catIntra = self._detectDonuts(expExtra, expIntra)

        cutOutput = self.cutOutDonuts.run(
            [expExtra, expIntra],
            [catExtra, catIntra],
            camera,
        )
        stampsExtra = cutOutput.donutStampsExtra
        stampsIntra = cutOutput.donutStampsIntra
        if len(stampsExtra) == 0 or len(stampsIntra) == 0:
            raise pipeBase.NoWorkFound(
                f"Cutout produced {len(stampsExtra)} extra and {len(stampsIntra)} "
                "intra stamps; need at least one of each."
            )

        peakNormalize(stampsExtra)
        peakNormalize(stampsIntra)
        zkOutput = self.estimateZernikes.run(stampsExtra, stampsIntra)
        zkTable = self._makeZkTable(zkOutput.zernikes, zkOutput.wfEstInfo, stampsExtra, stampsIntra)

        return pipeBase.Struct(
            zernikes=zkTable,
            donutStampsExtra=stampsExtra,
            donutStampsIntra=stampsIntra,
            wfEstInfo=zkOutput.wfEstInfo,
        )

    def _detectDonuts(
        self,
        expExtra: afwImage.Exposure,
        expIntra: afwImage.Exposure,
    ) -> tuple[QTable, QTable]:
        """Find the bright central donut on each side.

        Uses QuickFrameMeasurement.

        Follows ``latiss_wep_align.run_wep``: if exactly one side's donut is
        too far from the boresight, that side borrows the other side's
        centroid; if both are, it is an error.
        """
        resExtra = self.quickFrameMeasurement.run(expExtra.clone(), donutDiameter=self.config.donutDiameter)
        resIntra = self.quickFrameMeasurement.run(expIntra.clone(), donutDiameter=self.config.donutDiameter)
        if not resExtra.success or not resIntra.success:
            raise RuntimeError(
                "QuickFrameMeasurement failed to find a centroid: "
                f"extra success={resExtra.success}, intra success={resIntra.success}."
            )

        # latiss_wep_align measures from latiss_constants.boresight, which is
        # the detector centre (to within 1 px, i.e. 0.1 arcsec).
        boresight = expExtra.getDetector().getBBox().getCenter()

        maxDist = self.config.maxDistanceFromBoresight
        outOfBounds = {}
        for side, res in (("extra", resExtra), ("intra", resIntra)):
            dx = res.brightestObjCentroid[0] - boresight.getX()
            dy = res.brightestObjCentroid[1] - boresight.getY()
            # Arcseconds, matching run_wep: it measures this distance with
            # calculate_xy_offsets, which applies the plate scale. Comparing
            # pixels against run_wep's 500 would be ~10x too strict.
            drPixels = float(np.hypot(dx, dy))
            dr = drPixels * LATISS_PIXEL_SCALE
            outOfBounds[side] = dr > maxDist
            self.log.info("%s-focal donut is %.1f arcsec (%.1f px) from the boresight.", side, dr, drPixels)

        if outOfBounds["extra"] and outOfBounds["intra"]:
            raise RuntimeError(f"Both detected donuts are further than {maxDist} arcsec from the boresight.")
        for side in ("extra", "intra"):
            if outOfBounds[side]:
                self.log.warning("%s-focal donut is out of bounds; using the other side's centroid.", side)

        # Substitute the in-bounds side for whichever side is out of bounds.
        srcExtra = (resExtra, expExtra) if not outOfBounds["extra"] else (resIntra, expIntra)
        srcIntra = (resIntra, expIntra) if not outOfBounds["intra"] else (resExtra, expExtra)

        return self._makeDonutCatalog(*srcExtra), self._makeDonutCatalog(*srcIntra)

    @staticmethod
    def _makeDonutCatalog(result: pipeBase.Struct, exposure: afwImage.Exposure) -> QTable:
        """Build a one-row donut catalog from a QuickFrameMeasurement result.

        Same construction as ``latiss_wep_align.get_donut_catalog``, so the
        cutout task sees exactly the input it does on the summit.
        """
        wcs = exposure.getWcs()
        ra, dec = wcs.pixelToSkyArray(
            result.brightestObjCentroidCofM[0],
            result.brightestObjCentroidCofM[1],
            degrees=False,
        )
        catalog = QTable()
        catalog["coord_ra"] = ra * u.rad
        catalog["coord_dec"] = dec * u.rad
        catalog["centroid_x"] = [result.brightestObjCentroidCofM[0]] * u.pixel
        catalog["centroid_y"] = [result.brightestObjCentroidCofM[1]] * u.pixel
        catalog["source_flux"] = [result.brightestObjApFlux70] * u.nJy
        catalog.meta["blend_centroid_x"] = ""
        catalog.meta["blend_centroid_y"] = ""
        catalog.sort("source_flux", reverse=True)
        return addVisitInfoToCatTable(exposure, catalog)

    def _makeZkTable(
        self,
        zernikes: np.ndarray,
        wfEstInfo: dict,
        stampsExtra: DonutStamps,
        stampsIntra: DonutStamps,
    ) -> QTable:
        """Assemble the output table.

        Schema follows ``CalcZernikesTask.initZkTable`` -- an ``average`` row
        first, then one row per pair, with ``Z<j>``/``Z<j>_intrinsic``/
        ``Z<j>_deviation`` in nm -- so donut_viz aggregation reads it
        unchanged. Intrinsic and deviation columns are NaN: LATISS has no
        intrinsic Zernike calibration.

        Adds four quality columns ``CalcZernikesTask`` keeps only in metadata:
        ``chi_square``, ``fwhm``, ``nfev`` and ``fit_success``. With a single
        pair per visit there is nothing to sigma-clip against, so these are
        the only handle on fit quality downstream.

        Parameters
        ----------
        zernikes : `np.ndarray`
            Shape (nPairs, nNoll), in microns, from
            ``EstimateZernikesDanishTask.run``.
        wfEstInfo : `dict`
            Per-pair metadata from the same call; each value is a list.
        """
        dtype: list[tuple] = [
            ("label", "<U12"),
            ("used", np.bool_),
            ("intra_field", pos2f_dtype),
            ("extra_field", pos2f_dtype),
            ("intra_centroid", pos2f_dtype),
            ("extra_centroid", pos2f_dtype),
            ("chi_square", "<f8"),
            ("fwhm", "<f4"),
            ("nfev", "<i4"),
            ("fit_success", np.bool_),
        ]
        for suffix in ("", "_intrinsic", "_deviation"):
            for j in self.nollIndices:
                dtype.append((f"Z{j}{suffix}", "<f4"))

        table = QTable(dtype=dtype)
        for col in ("intra_field", "extra_field"):
            table[col].unit = u.deg
        for col in ("intra_centroid", "extra_centroid"):
            table[col].unit = u.pixel
        table["fwhm"].unit = u.arcsec
        for suffix in ("", "_intrinsic", "_deviation"):
            for j in self.nollIndices:
                table[f"Z{j}{suffix}"].unit = u.nm

        maxChiSquare = self.config.maxChiSquare

        # Columns carrying units are Quantity in a QTable, so every value going
        # into one must be a Quantity too -- a bare float raises.
        def _fieldAndCentroid(stamp: Any) -> tuple:
            return (
                np.array(stamp.calcFieldXY(), dtype=pos2f_dtype) * u.deg,
                np.array(
                    (stamp.centroid_position.x, stamp.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel,
            )

        def _scalar(key: str, i: int, default: float = np.nan) -> float:
            values = wfEstInfo.get(key)
            if values is None or i >= len(values) or values[i] is None:
                return default
            return float(values[i])

        # Placeholder average row, filled in below once `used` is known.
        table.add_row({"label": "average", "used": True})

        zernikes = np.atleast_2d(np.asarray(zernikes, dtype=float))
        for i, (extra, intra) in enumerate(zip(stampsExtra, stampsIntra)):
            extraField, extraCentroid = _fieldAndCentroid(extra)
            intraField, intraCentroid = _fieldAndCentroid(intra)

            chiSquare = _scalar("chi_square", i)
            success = bool(_scalar("fit_success", i, default=0.0))
            used = success and np.isfinite(chiSquare)
            if used and maxChiSquare is not None and chiSquare > maxChiSquare:
                self.log.warning(
                    "Pair %d rejected: chi-square %.2f exceeds maxChiSquare %.2f.",
                    i + 1,
                    chiSquare,
                    maxChiSquare,
                )
                used = False

            row: dict = {
                "label": f"pair{i + 1}",
                "used": used,
                "intra_field": intraField,
                "extra_field": extraField,
                "intra_centroid": intraCentroid,
                "extra_centroid": extraCentroid,
                "chi_square": chiSquare,
                "fwhm": _scalar("fwhm", i) * u.arcsec,
                "nfev": int(_scalar("lstsq_nfev", i, default=0.0)),
                "fit_success": success,
            }
            # The estimator returns microns; the table is in nm.
            zk_nm = zernikes[i] * 1e3
            for k, j in enumerate(self.nollIndices):
                row[f"Z{j}"] = zk_nm[k] * u.nm
                row[f"Z{j}_intrinsic"] = np.nan * u.nm
                row[f"Z{j}_deviation"] = np.nan * u.nm
            table.add_row(row)

        pairRows = table[1:]
        usedRows = pairRows[pairRows["used"]]
        if len(usedRows) == 0:
            self.log.warning("No pairs passed quality cuts; average row will be NaN.")
        with warnings.catch_warnings():
            # An all-NaN slice is expected when every pair failed.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for j in self.nollIndices:
                values = usedRows[f"Z{j}"].to_value(u.nm) if len(usedRows) else np.array([np.nan])
                table[f"Z{j}"][0] = np.nanmean(values) * u.nm
                table[f"Z{j}_intrinsic"][0] = np.nan * u.nm
                table[f"Z{j}_deviation"][0] = np.nan * u.nm
            table["chi_square"][0] = np.nanmean(usedRows["chi_square"]) if len(usedRows) else np.nan
            table["fwhm"][0] = (
                np.nanmean(usedRows["fwhm"].to_value(u.arcsec)) if len(usedRows) else np.nan
            ) * u.arcsec
        table["nfev"][0] = int(np.nansum(pairRows["nfev"])) if len(pairRows) else 0
        table["fit_success"][0] = len(usedRows) > 0
        nanPair = np.array((np.nan, np.nan), dtype=pos2f_dtype)
        for col, unit in (
            ("intra_field", u.deg),
            ("extra_field", u.deg),
            ("intra_centroid", u.pixel),
            ("extra_centroid", u.pixel),
        ):
            table[col][0] = nanPair * unit

        table.meta = self._makeMetadata(stampsExtra, stampsIntra)
        # Same key CalcZernikesTask uses, minus the per-pixel model images,
        # which do not belong in table metadata.
        table.meta["estimatorInfo"] = {k: v for k, v in wfEstInfo.items() if k != "model_img"}
        return table

    def _makeMetadata(self, stampsExtra: DonutStamps, stampsIntra: DonutStamps) -> dict:
        """Build table metadata.

        Uses the ``CalcZernikesTask.createZkTableMetadata`` form.
        """
        meta: dict = {"intra": {}, "extra": {}}
        camName = None
        for key, stamps in (("intra", stampsIntra), ("extra", stampsExtra)):
            if not stamps.metadata:
                continue
            md = stamps.metadata
            meta[key] = {
                "det_name": md["DET_NAME"],
                "visit": md["VISIT"],
                "dfc_dist": md["DFC_DIST"],
                "band": md["BANDPASS"],
                "boresight_rot_angle_rad": md["BORESIGHT_ROT_ANGLE_RAD"],
                "boresight_par_angle_rad": md["BORESIGHT_PAR_ANGLE_RAD"],
                "boresight_alt_rad": md["BORESIGHT_ALT_RAD"],
                "boresight_az_rad": md["BORESIGHT_AZ_RAD"],
                "boresight_ra_rad": md["BORESIGHT_RA_RAD"],
                "boresight_dec_rad": md["BORESIGHT_DEC_RAD"],
                "mjd": md["MJD"],
            }
            if camName is None:
                camName = md["CAM_NAME"]

        nollList = [int(j) for j in self.nollIndices]
        meta["cam_name"] = camName
        meta["noll_indices"] = nollList
        meta["opd_columns"] = [f"Z{j}" for j in nollList]
        meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in nollList]
        meta["deviation_columns"] = [f"Z{j}_deviation" for j in nollList]
        meta["optical_model"] = self.config.opticalModel
        meta["donut_diameter"] = int(self.config.donutDiameter)
        meta["peak_normalized_stamps"] = True
        return meta
