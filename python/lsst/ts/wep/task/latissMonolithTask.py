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

"""Self-contained LATISS wavefront estimation, from raws to Zernikes.

One quantum does everything ``latiss_wep_align.run_wep`` does, plus ISR:

    ISR -> QuickFrameMeasurement -> CutOutDonutsScienceSensorTask -> danish

The point of doing it in one task is the last step. ``CalcZernikesTask`` ->
``EstimateZernikesDanishTask`` is unusable for AuxTel on modern ts_wep because
``_prepDanish`` gets two things wrong, both invisible for LSSTCam:

  1. It builds ``zkRef`` from ``instrument.getOffAxisCoeff``, which returns
     *transverse-aberration* Zernikes. ``AuxTel.yaml`` offsets M2 (0.8 mm)
     rather than the Detector, and transverse aberration -- unlike wavefront
     OPD -- picks up the M2->detector magnification, so the reference is 43.5x
     too large. We compute an OPD ``zkRef`` with ``batoid.zernike`` instead.
  2. Danish's ``sky_levels``/``fluxes`` are scale dependent, and the modern
     cutout task returns raw ADU (~1e5) where 15.1.0 returned peak ~1 stamps.
     We peak-normalize before fitting.

Together these are worth the difference between working and not: on 12 LATISS
pairs, stock 17.8.1 returns NaN on 11 ("Non-positive image flux"). With both
fixes all 12 fit, and on identical stamps agree with ts_wep 15.1.0 at Z4/Z8
correlation 0.953.

The fitting core lives in ``latissMonolith``; this module is the butler
plumbing around it. ``run`` is callable without a butler, so ``run_wep`` in
ts_externalscripts can call it directly.

See DM ticket RSO-873.
"""

__all__ = [
    "LatissMonolithTaskConnections",
    "LatissMonolithTaskConfig",
    "LatissMonolithTask",
]

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
from lsst.ip.isr.isrTask import IsrTask
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
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable
from lsst.ts.wep.task.latissMonolith import fit_latiss_danish
from lsst.ts.wep.task.pairTask import ExposurePairer
from lsst.ts.wep.utils import getTaskInstrument
from lsst.utils.timer import timeMethod

# Boresight on the LATISS detector, in pixels. Duplicated from
# lsst.ts.observatory.control.constants.latiss_constants so that ts_wep does
# not gain a dependency on ts_observatory_control.
LATISS_BORESIGHT_XY = (2036.5, 2000.5)

# Structured dtype for the paired (x, y) columns, matching calcZernikesTask so
# that the output table stays readable by the same donut_viz code.
pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])


class LatissMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "detector"),  # type: ignore
):
    """Connections for LatissMonolithTask.

    Dimensioned on detector, not visit: like
    ``CutOutDonutsScienceSensorTaskConnections`` this task consumes *two*
    exposures per fit and pairs them internally, so the quantum cannot be
    keyed on a single visit. LATISS has one detector, so in practice this is
    one quantum per run.
    """

    raws = ct.Input(
        doc="Raw LATISS exposures; paired into intra/extra by focusZ.",
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        minimum=2,
        deferLoad=True,
    )
    camera = ct.PrerequisiteInput(
        doc="Input camera geometry.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    zernikes = ct.Output(
        doc="Zernike coefficients per pair and averaged, with fit quality columns.",
        name="zernikes",
        storageClass="AstropyQTable",
        dimensions=("visit", "detector", "instrument"),
        multiple=True,
    )
    donutStampsExtra = ct.Output(
        doc="Extra-focal donut postage stamps.",
        name="donutStampsExtra",
        storageClass="StampsBase",
        dimensions=("visit", "detector", "instrument"),
        multiple=True,
    )
    donutStampsIntra = ct.Output(
        doc="Intra-focal donut postage stamps.",
        name="donutStampsIntra",
        storageClass="StampsBase",
        dimensions=("visit", "detector", "instrument"),
        multiple=True,
    )

    def __init__(self, *, config: Any | None = None) -> None:
        super().__init__(config=config)
        if config is not None and not config.doSaveStamps:
            del self.donutStampsExtra
            del self.donutStampsIntra


class LatissMonolithTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=LatissMonolithTaskConnections,  # type: ignore
):
    """Configuration for LatissMonolithTask."""

    isrTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=IsrTask,
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
    donutDiameter: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=228,
        doc="Donut diameter in pixels, passed to QuickFrameMeasurement and used "
        "as the stamp size. 228 is what latiss_wep_align derives for dz=0.8 "
        "(ceil(192*1.1*0.8/1.5/2)*2 * 2). Note the default stamp size of 160 "
        "is LSSTCam-sized and CLIPS an AuxTel donut, which is 194 px across.",
    )
    nollIndices: pexConfig.ListField = pexConfig.ListField(
        dtype=int,
        default=tuple(range(4, 23)),
        doc="Noll indices to estimate. Must be ascending, >= 4, with complete "
        "azimuthal pairs.",
    )
    opticalModel: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="onAxis",
        doc="Optical model for masks. Must be 'onAxis' for AuxTel: there is no "
        "off-axis batoid fit for AuxTel, so the 'offAxis' default is wrong.",
    )
    instConfigFile: pexConfig.Field = pexConfig.Field(
        dtype=str,
        optional=True,
        doc="Instrument config file override. Defaults to the camera's, i.e. "
        "policy:instruments/AuxTel.yaml for LATISS.",
    )
    startWithIntrinsic: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add the design intrinsic Zernikes to zkRef and to the reported sum, "
        "as ts_wep does (zkSum = zkFit + mean(zkStart)).",
    )
    lstsqKwargs: pexConfig.DictField = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={},
        doc="Extra keyword arguments for scipy.optimize.least_squares, except "
        "fun, x0, jac, or args. Empty means scipy's own tolerances, matching "
        "ts_wep 15.1.0.",
    )
    maxDistanceFromBoresight: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=500.0,
        doc="Maximum distance in pixels from the boresight for a detected donut "
        "to be accepted. When one side is out of bounds the other side's "
        "centroid is substituted, as latiss_wep_align does.",
    )
    maxFitCost: pexConfig.Field = pexConfig.Field(
        dtype=float,
        optional=True,
        default=None,
        doc="If set, pairs whose danish fit cost exceeds this are flagged "
        "used=False and excluded from the average. Fit cost predicts Z4 error "
        "at r=0.90 in simulation, so it is a usable quality cut, but a good "
        "threshold is stamp-size and seeing dependent -- hence no default. "
        "Cost is NOT comparable across differently-sized stamps.",
    )
    doSaveStamps: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to write the donut stamps as outputs.",
    )

    def setDefaults(self) -> None:
        super().setDefaults()

        # LATISS ISR: gains and overscan only. There are no usable bias/dark/
        # flat calibrations for these alignment sequences, which is why this
        # uses IsrTask rather than blitz's IsrTaskLSST. Mirrors
        # tests/testData/pipelineConfigs/testCalcZernikesLatissPipeline.yaml.
        self.isrTask.doApplyGains = True
        self.isrTask.doOverscan = True
        self.isrTask.overscan.fitType = "MEDIAN_PER_ROW"
        self.isrTask.doBias = False
        self.isrTask.doDark = False
        self.isrTask.doFlat = False
        self.isrTask.doFringe = False
        self.isrTask.doDefect = False
        self.isrTask.doLinearize = False
        self.isrTask.doCrosstalk = False
        self.isrTask.doBrighterFatter = False
        self.isrTask.doVariance = False
        self.isrTask.doNanMasking = False
        self.isrTask.doInterpolate = False

        # AuxTel is defocused by moving M2, so the extra-focal exposure has the
        # *smaller* focusZ -- inverted relative to LSSTCam. -0.8 mm is what
        # pairTask already hardcodes for LATISS; set it explicitly so the
        # pairing does not silently change if that default moves.
        self.pairer.doOverrideSeparation = True
        self.pairer.overrideSeparation = -0.8


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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = cast(LatissMonolithTaskConfig, self.config)

        self.makeSubtask("isrTask")
        self.makeSubtask("quickFrameMeasurement")
        self.makeSubtask("pairer")

        # The cutout subtask needs the AuxTel-specific stamp geometry. Build
        # its config here rather than in setDefaults so donutDiameter stays the
        # single place the stamp size is set.
        cutOutConfig = CutOutDonutsScienceSensorTaskConfig()
        cutOutConfig.donutStampSize = config.donutDiameter
        cutOutConfig.opticalModel = config.opticalModel
        cutOutConfig.initialCutoutPadding = 40
        if config.instConfigFile is not None:
            cutOutConfig.instConfigFile = config.instConfigFile
        self.cutOutDonuts = CutOutDonutsScienceSensorTask(config=cutOutConfig)

        self.nollIndices = np.array(sorted(config.nollIndices), dtype=int)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Pair the raws, then run one fit per pair.

        The raws are deferred so that pairing -- which needs only ``visitInfo``
        -- does not read pixels for exposures that turn out to be unpaired.
        """
        camera = butlerQC.get(inputRefs.camera)

        # butlerQC.get resolves the deferred refs into DeferredDatasetHandles;
        # the refs themselves cannot be read from.
        rawHandles = dict(
            zip(
                (ref.dataId["exposure"] for ref in inputRefs.raws),
                butlerQC.get(inputRefs.raws),
            )
        )
        visitInfos = {expId: handle.get(component="visitInfo") for expId, handle in rawHandles.items()}

        pairs = self.pairer.run(visitInfos)
        if len(pairs) == 0:
            raise pipeBase.NoWorkFound(
                f"No intra/extra pairs found among exposures {sorted(rawHandles)}."
            )
        self.log.info("Found %d intra/extra pair(s) among %d exposures.", len(pairs), len(rawHandles))

        # Outputs are keyed on the extra-focal visit, matching
        # CutOutDonutsScienceSensorTask's paired-mode convention.
        zernikeHandles = {ref.dataId["visit"]: ref for ref in outputRefs.zernikes}
        if self.config.doSaveStamps:
            extraHandles = {ref.dataId["visit"]: ref for ref in outputRefs.donutStampsExtra}
            intraHandles = {ref.dataId["visit"]: ref for ref in outputRefs.donutStampsIntra}

        for pair in pairs:
            self.log.info("Fitting pair: extra=%d intra=%d", pair.extra, pair.intra)
            rawExtra = rawHandles[pair.extra].get()
            rawIntra = rawHandles[pair.intra].get()

            outputs = self.run(rawExtra, rawIntra, camera)

            butlerQC.put(outputs.zernikes, zernikeHandles[pair.extra])
            if self.config.doSaveStamps:
                butlerQC.put(outputs.donutStampsExtra, extraHandles[pair.extra])
                # Intentionally the extra-focal id for the intra stamps, so a
                # pair's products share one dataId.
                butlerQC.put(outputs.donutStampsIntra, intraHandles[pair.extra])

    @timeMethod
    def run(
        self,
        rawExtra: afwImage.Exposure,
        rawIntra: afwImage.Exposure,
        camera: lsst.afw.cameraGeom.Camera,
        doIsr: bool = True,
    ) -> pipeBase.Struct:
        """Run the full chain on one intra/extra pair.

        Parameters
        ----------
        rawExtra, rawIntra : `lsst.afw.image.Exposure`
            The pair. Raw if ``doIsr``, else already ISR-corrected.
        camera : `lsst.afw.cameraGeom.Camera`
            LATISS camera geometry.
        doIsr : bool, optional
            Set False to pass in exposures that are already ISR-corrected,
            e.g. from ``BestEffortIsr`` when called outside a pipeline.

        Returns
        -------
        `lsst.pipe.base.Struct`
            ``zernikes`` (`QTable`), ``donutStampsExtra``, ``donutStampsIntra``,
            and the raw ``fitResults`` dicts from the danish fit.
        """
        if doIsr:
            expExtra = self.isrTask.run(rawExtra, camera=camera).outputExposure
            expIntra = self.isrTask.run(rawIntra, camera=camera).outputExposure
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

        fitResults = self._fitPairs(stampsExtra, stampsIntra)
        zkTable = self._makeZkTable(fitResults, stampsExtra, stampsIntra)

        return pipeBase.Struct(
            zernikes=zkTable,
            donutStampsExtra=stampsExtra,
            donutStampsIntra=stampsIntra,
            fitResults=fitResults,
        )

    def _detectDonuts(
        self,
        expExtra: afwImage.Exposure,
        expIntra: afwImage.Exposure,
    ) -> tuple[QTable, QTable]:
        """Find the bright central donut on each side, via QuickFrameMeasurement.

        Follows ``latiss_wep_align.run_wep``: if exactly one side's donut is
        too far from the boresight, that side borrows the other side's
        centroid; if both are, it is an error.
        """
        resExtra = self.quickFrameMeasurement.run(
            expExtra.clone(), donutDiameter=self.config.donutDiameter
        )
        resIntra = self.quickFrameMeasurement.run(
            expIntra.clone(), donutDiameter=self.config.donutDiameter
        )
        if not resExtra.success or not resIntra.success:
            raise RuntimeError(
                "QuickFrameMeasurement failed to find a centroid: "
                f"extra success={resExtra.success}, intra success={resIntra.success}."
            )

        maxDist = self.config.maxDistanceFromBoresight
        outOfBounds = {}
        for side, res in (("extra", resExtra), ("intra", resIntra)):
            dx = res.brightestObjCentroid[0] - LATISS_BORESIGHT_XY[0]
            dy = res.brightestObjCentroid[1] - LATISS_BORESIGHT_XY[1]
            dr = float(np.hypot(dx, dy))
            outOfBounds[side] = dr > maxDist
            self.log.info("%s-focal donut is %.1f px from the boresight.", side, dr)

        if outOfBounds["extra"] and outOfBounds["intra"]:
            raise RuntimeError(
                "Both detected donuts are further than "
                f"{maxDist} px from the boresight."
            )
        for side in ("extra", "intra"):
            if outOfBounds[side]:
                self.log.warning(
                    "%s-focal donut is out of bounds; using the other side's centroid.", side
                )

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

    def _fitPairs(self, stampsExtra: DonutStamps, stampsIntra: DonutStamps) -> list[dict]:
        """Fit each intra/extra stamp pair with danish.

        A failed fit becomes a result dict with NaN Zernikes rather than an
        exception, so one bad pair does not lose the others.
        """
        refStamp = stampsExtra[0]
        instrument = getTaskInstrument(
            refStamp.cam_name,
            refStamp.detector_name,
            self.config.instConfigFile,
        )

        results = []
        for i, (extra, intra) in enumerate(zip(stampsExtra, stampsIntra)):
            try:
                result = fit_latiss_danish(
                    extra,
                    intra,
                    instrument,
                    noll_indices=self.nollIndices,
                    optical_model=self.config.opticalModel,
                    start_with_intrinsic=self.config.startWithIntrinsic,
                    lstsq_kwargs=dict(self.config.lstsqKwargs),
                )
                self.log.info(
                    "Pair %d: cost=%.1f fwhm=%.2f nfev=%d Z4=%.1f nm",
                    i + 1,
                    result["cost"],
                    result["fwhm"] if result["fwhm"] is not None else np.nan,
                    result["nfev"],
                    result["zernikes_nm"].get(4, np.nan),
                )
            except Exception as exc:  # noqa: BLE001 -- one bad pair must not lose the rest
                self.log.warning("Pair %d failed to fit: %s", i + 1, exc)
                nan = np.full(len(self.nollIndices), np.nan)
                result = dict(
                    zk_sum=nan,
                    zk_fit=nan,
                    zernikes_nm={int(j): np.nan for j in self.nollIndices},
                    noll_indices=self.nollIndices,
                    fwhm=np.nan,
                    cost=np.nan,
                    nfev=0,
                    success=False,
                )
            results.append(result)

        return results

    def _makeZkTable(
        self,
        fitResults: list[dict],
        stampsExtra: DonutStamps,
        stampsIntra: DonutStamps,
    ) -> QTable:
        """Assemble the output table.

        Schema follows ``CalcZernikesTask.initZkTable`` -- an ``average`` row
        first, then one row per pair, with ``Z<j>``/``Z<j>_intrinsic``/
        ``Z<j>_deviation`` in nm -- so donut_viz aggregation reads it
        unchanged. Intrinsic and deviation columns are NaN: LATISS has no
        intrinsic Zernike calibration.

        Adds four columns ``CalcZernikesTask`` does not have: ``cost``,
        ``fwhm``, ``nfev`` and ``fit_success``. ``cost`` is the reason -- it
        predicts |Z4 error| at r=0.90 in simulation, making it the best
        available quality flag when there is no truth to compare against.
        """
        dtype: list[tuple] = [
            ("label", "<U12"),
            ("used", np.bool_),
            ("intra_field", pos2f_dtype),
            ("extra_field", pos2f_dtype),
            ("intra_centroid", pos2f_dtype),
            ("extra_centroid", pos2f_dtype),
            ("cost", "<f8"),
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

        maxCost = self.config.maxFitCost

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

        # Placeholder average row, filled in below once `used` is known.
        table.add_row({"label": "average", "used": True})

        for i, (result, extra, intra) in enumerate(zip(fitResults, stampsExtra, stampsIntra)):
            extraField, extraCentroid = _fieldAndCentroid(extra)
            intraField, intraCentroid = _fieldAndCentroid(intra)

            cost = float(result["cost"])
            used = bool(result["success"]) and np.isfinite(cost)
            if used and maxCost is not None and cost > maxCost:
                self.log.warning(
                    "Pair %d rejected: cost %.1f exceeds maxFitCost %.1f.", i + 1, cost, maxCost
                )
                used = False

            fwhm = result["fwhm"]
            row: dict = {
                "label": f"pair{i + 1}",
                "used": used,
                "intra_field": intraField,
                "extra_field": extraField,
                "intra_centroid": intraCentroid,
                "extra_centroid": extraCentroid,
                "cost": cost,
                "fwhm": (float(fwhm) if fwhm is not None else np.nan) * u.arcsec,
                "nfev": int(result["nfev"]),
                "fit_success": bool(result["success"]),
            }
            # zk_sum is in metres; the table is in nm.
            zk_nm = np.asarray(result["zk_sum"], dtype=float) * 1e9
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
            table["cost"][0] = np.nanmean(usedRows["cost"]) if len(usedRows) else np.nan
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
        return table

    def _makeMetadata(self, stampsExtra: DonutStamps, stampsIntra: DonutStamps) -> dict:
        """Build table metadata in ``CalcZernikesTask.createZkTableMetadata`` form."""
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
        # Flag the two AuxTel-specific corrections, so a table can be traced
        # back to whether it was fit with them.
        meta["opd_zk_ref"] = True
        meta["peak_normalized_stamps"] = True
        return meta
