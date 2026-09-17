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

"""Tests for LatissMonolithTask.

Split in two, because ``tests/testData/gen3TestRepo`` contains only LSSTCam
and LSSTComCam -- there is no LATISS data to run a self-contained pipeline
against:

* ``TestLatissMonolithTaskConfig`` needs no data at all. It checks the
  configuration, connections and table assembly, and therefore runs in CI.
* ``TestLatissMonolithTaskOnSky`` runs the real chain and is skipped unless
  ``/repo/main`` and a butler password are available, following
  ``test_calcZernikesTieTaskLatiss.py``.
"""

import os
import types
import unittest
from typing import Any

import astropy.units as u
import numpy as np
import pytest

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.utils.tests
from lsst.afw.cameraGeom import Camera
from lsst.afw.coord import Observatory
from lsst.daf.base import DateTime
from lsst.daf.butler import Butler
from lsst.geom import SpherePoint, degrees
from lsst.ts.wep.image import Image
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.ts.wep.task.latissMonolithTask import (
    LatissMonolithTask,
    LatissMonolithTaskConfig,
    LatissMonolithTaskConnections,
    peakNormalize,
)
from lsst.ts.wep.utils import DefocalType

# The pair used throughout: a BLOCK-T743 CWFS sequence on 20260625 with visit
# records defined. Note LATISS visits are not defined for the most recent
# nights even though the raws exist, so a pair must be chosen with care.
EXPOSURE_INTRA = 2026062500012
EXPOSURE_EXTRA = 2026062500013

NOLL_INDICES = list(range(4, 23))


class TestLatissMonolithTaskConfig(lsst.utils.tests.TestCase):
    """Configuration, connections and table assembly. Needs no butler."""

    def setUp(self) -> None:
        self.config = LatissMonolithTaskConfig()

    def testDefaultsAreAuxTelSpecific(self) -> None:
        """The AuxTel values that the fit depends on must be the defaults."""
        # onAxis is required: there is no off-axis batoid fit for AuxTel.
        self.assertEqual(self.config.opticalModel, "onAxis")
        # 228 px is what latiss_wep_align derives for dz=0.8. The ts_wep
        # default of 160 is LSSTCam-sized and clips a 194 px AuxTel donut.
        self.assertEqual(self.config.donutDiameter, 228)
        # Z4-Z22 as latiss_wep_align fits, not the ts_wep default of Z4-Z28.
        self.assertEqual(list(self.config.estimateZernikes.nollIndices), NOLL_INDICES)
        # The fit is the stock Danish task, not a private reimplementation.
        self.assertIs(self.config.estimateZernikes.target, EstimateZernikesDanishTask)
        # Jacobian scaling only: it makes the fit reproducible across numpy
        # versions; the LSSTCam pipelines' loose tolerances would stall it.
        self.assertEqual(dict(self.config.estimateZernikes.lstsqKwargs), {"x_scale": "jac"})

    def testPeakNormalize(self) -> None:
        """Only wep_im is rescaled, to peak 1; a flat stamp is an error."""
        stamp = _FakeStamp()
        stamp.wep_im = Image(
            image=np.arange(16, dtype=float).reshape(4, 4) * 1000.0,
            fieldAngle=(0.0, 0.0),
            defocalType=DefocalType.Extra,
            bandLabel="ref",
        )
        peakNormalize([stamp])
        self.assertAlmostEqual(float(stamp.wep_im.image.max()), 1.0)
        self.assertAlmostEqual(float(stamp.wep_im.image[0, 1]), 1.0 / 15.0)

        stamp.wep_im.image = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            peakNormalize([stamp])

    def testIsrAppliesTheFullCalibrationSet(self) -> None:
        """Regression guard: ISR must apply defects, flat, linearize, crosstalk
        (all IsrTaskLSST defaults).

        An earlier version of this task ran gains + overscan only, on the
        premise that LATISS alignment sequences have no usable calibrations.
        That premise was false -- bias, dark, flat, defects, linearizer,
        crosstalk and ptc are all present for LATISS in ``LATISS/defaults`` --
        and disabling them broke donut detection: the LATISS bad column at
        x=3795-3797 survived ISR at ~1.2e5 ADU against an image median of ~20,
        and since ``QuickFrameMeasurement`` ranks candidates on a 70 px
        aperture flux -- which a solid column fills more uniformly than a donut
        with a hole -- the column outranked the donut. Enabling defects moved
        38 of 60 previously-bad pair sides back on-axis and broke none.

        These are ``IsrTaskLSST`` defaults, so this test guards against someone
        re-disabling them rather than against a missing assignment.
        """
        for field in ("doDefect", "doFlat", "doLinearize", "doCrosstalk", "doInterpolate"):
            self.assertTrue(getattr(self.config.isrTask, field), field)
        self.assertTrue(self.config.isrTask.doApplyGains)
        self.assertTrue(self.config.isrTask.doSaturation)
        # No bfKernel/bfGains exist for LATISS, and IsrTaskLSST raises if asked
        # to do brighter-fatter without one.
        self.assertFalse(self.config.isrTask.doBrighterFatter)

    def testBoresightToleranceIsInArcsecNotPixels(self) -> None:
        """Regression guard: the unit of maxDistanceFromBoresight.

        ``run_wep`` measures this distance with ``calculate_xy_offsets``, which
        applies the 0.09569 arcsec/px plate scale, so its default of 500 means
        500 *arcsec* (~5225 px). Interpreting it as pixels makes the cut ~10x
        too strict and rejects perfectly usable pairs -- two of 23 in the first
        multi-pair run, both of which ``run_wep`` would have accepted.
        """
        from lsst.ts.wep.task.latissMonolithTask import LATISS_PIXEL_SCALE

        self.assertAlmostEqual(LATISS_PIXEL_SCALE, 0.09569)
        self.assertEqual(self.config.maxDistanceFromBoresight, 500.0)
        # 500 arcsec must be most of the 4072 px detector, not a tight cut.
        self.assertGreater(self.config.maxDistanceFromBoresight / LATISS_PIXEL_SCALE, 4000.0)

    def testPairerSeparationIsInvertedForLatiss(self) -> None:
        """AuxTel moves M2, so extra-focal has the SMALLER focusZ.

        This is inverted relative to LSSTCam. pairTask hardcodes the same value
        for LATISS, but the task pins it so the pairing cannot drift.
        """
        self.assertTrue(self.config.pairer.doOverrideSeparation)
        self.assertEqual(self.config.pairer.overrideSeparation, -0.8)

    def testConnections(self) -> None:
        """One quantum per pair, keyed on the extra-focal visit.

        The intra-focal raw is attached at graph-build time by
        adjust_all_quanta, so the quantum can be visit-keyed even though it
        consumes two exposures. That is what makes the _log/_metadata
        provenance datasets per pair: rapid analysis reuses one output run
        for a whole night, and a per-night or per-run quantum would make the
        second pair collide on them.
        """
        connections = LatissMonolithTaskConnections(config=self.config)
        self.assertEqual(set(connections.dimensions), {"instrument", "visit", "detector"})
        self.assertEqual(set(connections.inputs), {"raws"})
        # The full calibration set BestEffortIsr passes on the summit.
        # `defects` is the load-bearing one: without it the LATISS bad column
        # at x=3795-3797 outranks the real donut in QuickFrameMeasurement's
        # aperture flux, putting the "donut" ~2000 px off the boresight.
        self.assertEqual(
            set(connections.prerequisiteInputs),
            {"camera", "bias", "dark", "flat", "defects", "linearizer", "crosstalk", "ptc"},
        )
        self.assertEqual(set(connections.outputs), {"zernikes", "donutStampsExtra", "donutStampsIntra"})
        # One pair per quantum, so single outputs; the raws input is the only
        # multiple one (the pair's two exposures).
        for name in ("zernikes", "donutStampsExtra", "donutStampsIntra"):
            self.assertFalse(getattr(connections, name).multiple, name)
        self.assertTrue(connections.raws.multiple)
        # No intrinsicZernikes connection: LATISS has no such calibration.
        self.assertFalse(hasattr(connections, "intrinsicZernikes"))
        # And no refcat/astrometry: a LATISS exposure has one bright donut.
        self.assertFalse(hasattr(connections, "refCat"))

    def testDoSaveStampsRemovesStampOutputs(self) -> None:
        self.config.doSaveStamps = False
        connections = LatissMonolithTaskConnections(config=self.config)
        self.assertEqual(set(connections.outputs), {"zernikes"})

    def testStampSizePropagatesToCutoutSubtask(self) -> None:
        """donutDiameter is the single place the stamp size is set."""
        self.config.donutDiameter = 200
        task = LatissMonolithTask(config=self.config)
        self.assertEqual(task.cutOutDonuts.config.donutStampSize, 200)
        self.assertEqual(task.cutOutDonuts.config.opticalModel, "onAxis")
        self.assertEqual(task.cutOutDonuts.config.initialCutoutPadding, 40)

    def testZkTableSchemaAndFailedFit(self) -> None:
        """A failed fit becomes a NaN row that is excluded from the average.

        Inputs mimic ``EstimateZernikesDanishTask.run``: Zernikes in microns,
        shape (nPairs, nNoll), and a metadata dict of per-pair lists.
        """
        task = LatissMonolithTask(config=self.config)
        n = len(NOLL_INDICES)
        zernikes = np.vstack([np.full(n, 0.1), np.full(n, np.nan)])  # 0.1 um = 100 nm
        wfEstInfo = {
            "chi_square": [100.0, np.nan],
            "fwhm": [1.5, np.nan],
            "lstsq_nfev": [20, None],
            "fit_success": [True, False],
            "model_img": [object(), None],  # must not leak into table.meta
        }

        stamps = _FakeStamps([_FakeStamp(), _FakeStamp()])
        table = task._makeZkTable(zernikes, wfEstInfo, stamps, stamps)

        self.assertEqual(len(table), 3)  # average + 2 pairs
        self.assertEqual(list(table["label"]), ["average", "pair1", "pair2"])
        self.assertEqual(list(table["used"]), [True, True, False])
        self.assertEqual(list(table["fit_success"]), [True, True, False])
        self.assertEqual(list(table["nfev"]), [20, 20, 0])

        # The QA columns that CalcZernikesTask keeps only in metadata.
        for column in ("chi_square", "fwhm", "nfev", "fit_success"):
            self.assertIn(column, table.colnames)

        # 100 nm, and the failed pair must not drag the average.
        self.assertAlmostEqual(table["Z4"][0].to_value(u.nm), 100.0, places=3)
        self.assertAlmostEqual(table["Z4"][1].to_value(u.nm), 100.0, places=3)
        self.assertTrue(np.isnan(table["Z4"][2].to_value(u.nm)))

        # LATISS has no intrinsic Zernike calibration, so these are NaN by
        # design rather than by accident.
        self.assertTrue(np.all(np.isnan(table["Z4_intrinsic"].to_value(u.nm))))
        self.assertTrue(np.all(np.isnan(table["Z4_deviation"].to_value(u.nm))))

        self.assertEqual(table.meta["noll_indices"], NOLL_INDICES)
        self.assertEqual(table.meta["opd_columns"], [f"Z{j}" for j in NOLL_INDICES])
        self.assertTrue(table.meta["peak_normalized_stamps"])
        self.assertEqual(table.meta["estimatorInfo"]["fit_success"], [True, False])
        self.assertNotIn("model_img", table.meta["estimatorInfo"])

    def testMaxChiSquareRejectsPoorFits(self) -> None:
        self.config.maxChiSquare = 50.0
        task = LatissMonolithTask(config=self.config)
        zernikes = np.full((1, len(NOLL_INDICES)), 0.1)
        wfEstInfo = {"chi_square": [100.0], "fwhm": [1.5], "lstsq_nfev": [10], "fit_success": [True]}
        stamps = _FakeStamps([_FakeStamp()])
        table = task._makeZkTable(zernikes, wfEstInfo, stamps, stamps)
        # The fit succeeded but is rejected by the quality cut, so it is not
        # used and the average is NaN.
        self.assertTrue(table["fit_success"][1])
        self.assertFalse(table["used"][1])
        self.assertTrue(np.isnan(table["Z4"][0].to_value(u.nm)))


class _FakeStamp:
    """Minimal stand-in for DonutStamp, for table-assembly tests."""

    class _Point:
        x = 100.0
        y = 200.0

    centroid_position = _Point()
    detector_name = "RXX_S00"
    cam_name = "LATISS"
    # Assigned per test; only peakNormalize reads it.
    wep_im: Image

    def calcFieldXY(self) -> tuple[float, float]:
        return (0.0, 0.0)


class _FakeStamps(list):
    """A DonutStamps-like list carrying the metadata the table builder
    reads.
    """

    metadata: dict = {}


def _makeVisitInfo(exposureId: int, focusZ: float, mjd: float) -> afwImage.VisitInfo:
    """A VisitInfo with just what ExposurePairer reads."""
    return afwImage.VisitInfo(
        id=exposureId,
        focusZ=focusZ,
        date=DateTime(mjd, DateTime.MJD, DateTime.TAI),
        boresightRaDec=SpherePoint(10.0 * degrees, -30.0 * degrees),
        boresightRotAngle=0.0 * degrees,
        era=0.0 * degrees,
        observatory=Observatory(-70.75 * degrees, -30.24 * degrees, 2650.0),
        instrumentLabel="LATISS",
    )


class _FakeRef:
    """A hashable DatasetRef stand-in with the dataId runQuantum reads."""

    def __init__(self, exposure: int) -> None:
        self.dataId = {"exposure": exposure}


class _FakeQuantumContext:
    """Stands in for QuantumContext: hands out pre-made objects per ref."""

    def __init__(self, visit: int, values: dict) -> None:
        self.quantum = types.SimpleNamespace(dataId={"visit": visit})
        self._values = values
        self.puts: dict = {}

    def get(self, refs: Any) -> Any:
        if isinstance(refs, list):
            return [self._values[r] for r in refs]
        return self._values[refs]

    def put(self, value: Any, ref: Any) -> None:
        self.puts[ref] = value


class TestLatissMonolithTaskRunQuantum(lsst.utils.tests.TestCase):
    """runQuantum's pairing bookkeeping, with the fit itself stubbed out."""

    def _runQuantum(self, quantumVisit: int, focusZ: dict) -> list[str]:
        """Run runQuantum on two fake raws and return the WARNING messages."""
        config = LatissMonolithTaskConfig()
        config.doSaveStamps = False
        task = LatissMonolithTask(config=config)

        rawRefs = {e: _FakeRef(e) for e in focusZ}
        # DeferredDatasetHandle stand-ins: only visitInfo is read before run().
        handles = {}
        for i, (e, fz) in enumerate(sorted(focusZ.items())):
            vi = _makeVisitInfo(e, fz, 60000.0 + i * 40.0 / 86400.0)
            handles[e] = types.SimpleNamespace(get=lambda component=None, vi=vi, e=e: vi if component else e)
        values: dict[Any, Any] = {rawRefs[e]: handles[e] for e in focusZ}
        cameraRef = _FakeRef(-1)
        values[cameraRef] = "camera"
        butlerQC = _FakeQuantumContext(quantumVisit, values)
        inputRefs = types.SimpleNamespace(camera=cameraRef, raws=list(rawRefs.values()))
        outputRefs = types.SimpleNamespace(zernikes="zernikesRef")

        seen = {}

        def fakeRun(rawExtra: afwImage.Exposure, rawIntra: afwImage.Exposure,
                    camera: Camera, doIsr:bool=True,
                    isrCalibs: dict | None = None) -> pipeBase.Struct:
            seen["extra"], seen["intra"] = rawExtra, rawIntra
            return pipeBase.Struct(zernikes="table", wfEstInfo={})

        task.run = fakeRun  # type: ignore[method-assign]
        with self.assertLogs(task.log.name, level="INFO") as logs:
            task.runQuantum(butlerQC, inputRefs, outputRefs)  # type: ignore[arg-type]
        self.seen = seen
        self.puts = butlerQC.puts
        return [m for m in logs.output if m.startswith("WARNING:")]

    def testPairingFollowsFocusZ(self) -> None:
        """AuxTel extra-focal has the SMALLER focusZ; quantum visit agrees."""
        warnings = self._runQuantum(quantumVisit=18, focusZ={17: +0.8, 18: -0.8})
        self.assertEqual((self.seen["extra"], self.seen["intra"]), (18, 17))
        self.assertEqual(self.puts, {"zernikesRef": "table"})
        self.assertFalse([w for w in warnings if "labelled extra-focal" in w], warnings)

    def testWarnsWhenHeaderLabelDisagreesWithFocusZ(self) -> None:
        """Header says 18 is extra but focusZ says 17:
           fit by focusZ, and warn."""
        warnings = self._runQuantum(quantumVisit=18, focusZ={17: -0.8, 18: +0.8})
        self.assertEqual((self.seen["extra"], self.seen["intra"]), (17, 18))
        # Still written under the quantum's visit, and loudly.
        self.assertEqual(self.puts, {"zernikesRef": "table"})
        mismatch = [w for w in warnings if "labelled extra-focal" in w]
        self.assertEqual(len(mismatch), 1, warnings)
        self.assertIn("exposure 17 is extra-focal and 18 is intra-focal", mismatch[0])


@pytest.mark.skipif(
    not os.path.exists("/sdf/data/rubin/repo/main"),
    reason="requires access to data in /repo/main",
)
@pytest.mark.skipif(not os.getenv("PGPASSFILE"), reason="requires access to butler db")
class TestLatissMonolithTaskOnSky(lsst.utils.tests.TestCase):
    """Run the real ISR -> QFM -> cutout -> danish chain on one on-sky pair."""

    repoDir = "/sdf/data/rubin/repo/main"
    # Populated by setUpClass, so declared here for the type checker.
    butler: Butler
    camera: Camera
    rawIntra: afwImage.Exposure
    rawExtra: afwImage.Exposure
    result: pipeBase.Struct
    isrCalibs: dict

    @classmethod
    def setUpClass(cls) -> None:
        from lsst.obs.lsst import Latiss

        cls.butler = Butler.from_config(cls.repoDir, collections=["LATISS/defaults"])
        cls.camera = Latiss.getCamera()
        cls.rawIntra = cls.butler.get("raw", instrument="LATISS", exposure=EXPOSURE_INTRA, detector=0)
        cls.rawExtra = cls.butler.get("raw", instrument="LATISS", exposure=EXPOSURE_EXTRA, detector=0)

        # IsrTaskLSST needs a PTC (it reads gains from it), and donut detection
        # needs `defects` -- so a caller outside a pipeline must supply the
        # calibrations itself, exactly as ``run_wep`` will have to.
        dataId = {"instrument": "LATISS", "exposure": EXPOSURE_EXTRA, "detector": 0}
        cls.isrCalibs = {}
        for name in ("bias", "dark", "flat", "defects", "linearizer", "crosstalk", "ptc"):
            if cls.butler.exists(name, dataId):
                cls.isrCalibs[name] = cls.butler.get(name, dataId=dataId)

        task = LatissMonolithTask(config=LatissMonolithTaskConfig())
        cls.result = task.run(cls.rawExtra, cls.rawIntra, cls.camera, isrCalibs=cls.isrCalibs)

    def testStampsAreTheConfiguredSize(self) -> None:
        for stamps in (self.result.donutStampsExtra, self.result.donutStampsIntra):
            self.assertEqual(len(stamps), 1)
            self.assertEqual(stamps[0].stamp_im.image.array.shape, (228, 228))

    def testStampsArePeakNormalized(self) -> None:
        """wep_im is order unity for the fit; stamp_im keeps the ADU pixels.

        The peak is slightly below 1 because Danish's ``prepImage`` then
        subtracts the median background from ``wep_im`` in place.
        """
        for stamps in (self.result.donutStampsExtra, self.result.donutStampsIntra):
            peak = float(np.nanmax(stamps[0].wep_im.image))
            self.assertTrue(0.5 < peak <= 1.0, peak)
            self.assertGreater(float(np.nanmax(stamps[0].stamp_im.image.array)), 100.0)

    def testFitSucceeds(self) -> None:
        """The whole point: the fit converges rather than stopping early.

        Without peak normalization the stock fit on raw-ADU LATISS stamps
        returns after a handful of function evaluations.
        """
        table = self.result.zernikes
        pair = table[table["label"] == "pair1"]
        self.assertTrue(bool(pair["fit_success"][0]))
        self.assertTrue(np.isfinite(pair["Z4"][0].to_value(u.nm)))
        self.assertGreater(int(pair["nfev"][0]), 5)
        # And the fitted seeing must be physical.
        self.assertTrue(0.1 < pair["fwhm"][0].to_value(u.arcsec) < 5.0)

    def testZernikesAreOfPlausibleMagnitude(self) -> None:
        """LATISS low-order aberrations are hundreds of nm, not microns."""
        table = self.result.zernikes
        pair = table[table["label"] == "pair1"]
        for j in (4, 7, 8):
            self.assertLess(abs(pair[f"Z{j}"][0].to_value(u.nm)), 5000.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: types.ModuleType) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
