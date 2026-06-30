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

import lsst.afw.math as afwMath
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.afw.cameraGeom import Camera, FIELD_ANGLE, PIXELS
from lsst.afw.image import Exposure
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.pipe.base import InputQuantizedConnection, OutputQuantizedConnection, QuantumContext
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


def _detectAndCutDonuts(
    exposure: Exposure,
    camera: Camera,
    detect_cfg: dict,
    bkg_config: Any,
) -> list[dict]:
    """Run background subtraction, detection, quality cuts, and stamp extraction.

    Returns a list of dicts, one per accepted donut, containing everything
    needed for danish fitting:
        stamp      : np.ndarray (stampSize x stampSize), CCS orientation
        fa_x_ccs   : field angle x in CCS frame (radians)
        fa_y_ccs   : field angle y in CCS frame (radians)
        flux       : float
        std        : float
        band       : str
        det_id     : int
        visit_id   : int
    """
    detector = exposure.getDetector()
    camName = camera.getName()
    detectorName = detector.getName()
    band = exposure.filter.bandLabel
    visit_id = exposure.visitInfo.id
    det_id = detector.getId()
    n_quarter = detector.getOrientation().getNQuarter()

    instrument = getTaskInstrument(camName, detectorName, detect_cfg["instConfigFile"])
    donutRadius = instrument.donutRadius

    if donutRadius < 5:
        return []

    SubtractBackgroundTask(config=bkg_config).run(exposure=exposure)

    trimmedBBox = exposure.getBBox().erodedBy(detect_cfg["edgeMargin"])
    exposureTrim = exposure[trimmedBBox].clone()

    peaks = _detectPeaks(
        exposureTrim,
        donutRadius,
        instrument.obscuration,
        detect_cfg["detectionBinning"],
        detect_cfg["peakMinDistanceFactor"],
        detect_cfg["peakExcludeBorderFactor"],
    )

    if len(peaks) == 0:
        return []

    measTable = _measureFlux(peaks, exposureTrim, donutRadius, instrument.obscuration)

    validFlux = np.isfinite(measTable["flux"]) & (measTable["flux"] > 0)
    measTable = measTable[validFlux]

    if len(measTable) > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            innerOk = np.abs(measTable["inner_flux"] / measTable["flux"]) < detect_cfg["innerFracThreshold"]
            outerOk = np.abs(measTable["outer_flux"] / measTable["flux"]) < detect_cfg["outerFracThreshold"]
        snOk = measTable["snr"] > detect_cfg["snrThreshold"]
        measTable = measTable[innerOk & outerOk & snOk]

    if len(measTable) == 0:
        return []

    # Add back the edge trim offset to get full-exposure pixel coords
    xOffset = trimmedBBox.getMinX()
    yOffset = trimmedBBox.getMinY()
    centroid_x = np.array(measTable["centroid_x"]) + xOffset
    centroid_y = np.array(measTable["centroid_y"]) + yOffset

    # Field angle cut and CCS conversion (swap x/y vs DVCS)
    fieldXY = detector.transform(
        [lsst.geom.Point2D(x, y) for x, y in zip(centroid_x, centroid_y)],
        PIXELS,
        FIELD_ANGLE,
    )
    fieldDist = np.array([np.degrees(np.hypot(p[0], p[1])) for p in fieldXY])
    keep = fieldDist <= detect_cfg["maxFieldDist"]

    # Sort kept donuts by flux descending, cap at maxDonuts
    fluxArr = np.array(measTable["flux"])[keep]
    fluxOrder = np.argsort(fluxArr)[::-1][: detect_cfg["maxDonuts"]]

    kept_indices = np.where(keep)[0][fluxOrder]

    arr = exposure.image.array
    half = detect_cfg["stampSize"] // 2
    donuts = []
    for i in kept_indices:
        cx, cy = int(round(centroid_x[i])), int(round(centroid_y[i]))
        rmin, rmax = cy - half, cy + half
        cmin, cmax = cx - half, cx + half
        if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
            continue
        stamp = np.array(arr[rmin:rmax, cmin:cmax])
        # Transform to CCS: rotate by n_quarter, then transpose
        stamp_ccs = np.rot90(stamp, k=n_quarter).T
        fa = fieldXY[i]
        donuts.append(dict(
            stamp=stamp_ccs,
            fa_x_ccs=float(fa[1]),   # CCS: x=DVCS y, y=DVCS x
            fa_y_ccs=float(fa[0]),
            flux=float(measTable["flux"][i]),
            std=float(measTable["std"][i]),
            band=band,
            det_id=det_id,
            visit_id=visit_id,
        ))
    return donuts


class DonutBlitzMonolithTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzMonolithTask.

    One quantum per visit; all 8 corner wavefront sensors are handled
    internally via multiple=True inputs.
    """

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
        default=0.005,
    )
    outerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed |outer_flux / flux| for a candidate to be kept.",
        dtype=float,
        default=0.005,
    )
    snrThreshold: pexConfig.Field = pexConfig.Field(
        doc="Minimum signal-to-noise ratio for a candidate to be kept.",
        dtype=float,
        default=100.0,
    )
    maxFieldDist: pexConfig.Field = pexConfig.Field(
        doc="Maximum distance from the center of the focal plane in degrees.",
        dtype=float,
        default=1.808,
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


class DonutBlitzMonolithTask(pipeBase.PipelineTask):
    """Monolithic WEP task for corner wavefront sensors.

    Runs ISR and donut detection on all 8 corner sensor raws in parallel
    using a multiprocessing pool, returning per-sensor donut catalogs.
    """

    ConfigClass = DonutBlitzMonolithTaskConfig
    _DefaultName = "donutBlitzMonolithTask"
    config: DonutBlitzMonolithTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("isrTask")
        self.makeSubtask("subtractBackground")
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

    @staticmethod
    def _run_isr_worker(args: tuple) -> dict:
        """Run ISR and donut detection for one sensor.

        Inputs are read from _CALIB_STORE, which was populated in the parent
        before forking. Only the sensor name and dispatch timestamp are pickled
        across the process boundary on the way in; only the donut catalog
        (a small QTable) is pickled on the way out.
        """
        sensor_name, t_dispatch = args
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

        catalog = _detectAndCutDonuts(
            postIsr,
            _CALIB_STORE["camera"],
            _CALIB_STORE["detect_cfg"],
            _CALIB_STORE["bkg_config"],
        )
        t3 = time.perf_counter()

        return {
            "sensor": sensor_name,
            "catalog": catalog,
            "dispatch_to_arrival": t_arrival - t_dispatch,
            "task_init": t1 - t0,
            "isr_run": t2 - t1,
            "detect_run": t3 - t2,
        }

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs, numCores=butlerQC.resources.num_cores)
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
        numCores: int = 1,
    ) -> pipeBase.Struct:
        """Run ISR and donut detection on all 8 corner raws in parallel.

        Parameters
        ----------
        raws : list of lsst.afw.image.Exposure
            Raw corner wavefront sensor exposures. Must contain all 8
            sensors (CORNER_SENSOR_NAMES).
        camera : lsst.afw.cameraGeom.Camera
            Camera geometry object.
        ptc : list of lsst.ip.isr.PhotonTransferCurveDataset
            One PTC dataset per detector.
        flat : list of lsst.afw.image.ExposureF
            One flat field per detector.
        linearizer : list of lsst.ip.isr.Linearizer
            One linearizer per detector.
        crosstalk : list of lsst.ip.isr.CrosstalkCalib
            One crosstalk calibration per detector.
        numCores : int
            Number of parallel processes.

        Returns
        -------
        lsst.pipe.base.Struct
            Struct with ``donutCatalogs``: list of per-sensor QTables,
            aligned with ``sorted(CORNER_SENSOR_NAMES)``.
        """
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

        _CALIB_STORE.clear()
        _CALIB_STORE["isr_config"] = self.isrTask.config
        _CALIB_STORE["bkg_config"] = self.subtractBackground.config
        _CALIB_STORE["camera"] = camera
        _CALIB_STORE["detect_cfg"] = detect_cfg
        for name in CORNER_SENSOR_NAMES:
            _CALIB_STORE[name] = dict(
                raw=rawByName[name],
                ptc=ptcByName[name],
                flat=flatByName[name],
                linearizer=linearizerByName[name],
                crosstalk=crosstalkByName[name],
            )

        isr_args = sorted(CORNER_SENSOR_NAMES)

        self.log.info("Running ISR+detection on %d corner sensors with %d core(s)", len(isr_args), numCores)
        if numCores == 1:
            t_dispatch = time.time()
            results = [self._run_isr_worker((arg, t_dispatch)) for arg in isr_args]
        else:
            t_pool0 = time.perf_counter()
            pool = self._get_pool(numCores)
            t_pool1 = time.perf_counter()
            t_dispatch = time.time()
            results = pool.map(self._run_isr_worker, [(arg, t_dispatch) for arg in isr_args])
            t_pool2 = time.perf_counter()
            self.log.info(
                "Pool get: %.3fs, pool.map: %.3fs", t_pool1 - t_pool0, t_pool2 - t_pool1
            )

        donuts = []
        for r in results:
            self.log.info(
                "  %s: dispatch_to_arrival=%.3fs  task_init=%.3fs  isr_run=%.3fs  detect_run=%.3fs  donuts=%d",
                r["sensor"],
                r["dispatch_to_arrival"],
                r["task_init"],
                r["isr_run"],
                r["detect_run"],
                len(r["catalog"]),
            )
            donuts.extend(r["catalog"])

        return pipeBase.Struct(donuts=donuts)
