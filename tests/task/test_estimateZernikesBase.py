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

import multiprocessing as mp
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.coordinates import Angle

from lsst.ts.wep.estimation import ObservingConditions
from lsst.ts.wep.task.estimateZernikesBase import (
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
    estimate_zk_pair,
    estimate_zk_single,
)
from lsst.ts.wep.utils import WfAlgorithmName


class _ConcreteTask(EstimateZernikesBaseTask):
    """Minimal concrete subclass for testing the base class."""

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        return WfAlgorithmName.TIE


# The classes below are defined at module level (not as MagicMocks) so they can
# be pickled and sent to worker processes when numCores > 1.
class _FakeDonut:
    """Picklable donut stub with the attributes the estimator touches.

    A donut is flagged "bad" by giving it ``wep_im=None``; the fake
    estimator raises on such donuts to exercise the failure path.
    """

    def __init__(self, donut_id: int, bad: bool = False) -> None:
        self.donut_id = donut_id
        self.wep_im = None if bad else f"image-{donut_id}"


class _FakeDonutStamps(list):
    """List of donut stubs that also carries the metadata attribute
    ``_get_obs_conditions`` expects."""

    metadata = {
        "BORESIGHT_ROT_ANGLE_RAD": 0.1,
        "BORESIGHT_PAR_ANGLE_RAD": 0.3,
        "BORESIGHT_ALT_RAD": 1.0,
    }


class _FakeWfEstimator:
    """Picklable stand-in for WfEstimator.

    Returns zero Zernikes for good donuts and raises when the extra-focal
    image is None, mimicking a single failed fit among successes. A bad
    donut is therefore placed on the extra-focal side (in single-donut mode
    that side carries the only image, so the same rule applies).
    """

    def __init__(self, nollIndices: range = range(4, 23)) -> None:
        self.nollIndices = np.array(list(nollIndices))
        self.history: dict = {}

    def estimateZk(
        self, wepImExtra: object, wepImIntra: object = None, obs: object = None
    ) -> tuple[np.ndarray, dict]:
        if wepImExtra is None:
            raise ValueError("Cannot compute zernike with failed rays.")
        return np.zeros(len(self.nollIndices)), {"fit_success": True, "fwhm": 1.0}


class TestEstimateZernikesBaseConfig(unittest.TestCase):
    def testTimeoutDefault(self) -> None:
        config = EstimateZernikesBaseConfig()
        self.assertEqual(config.timeout, 600)

    def testTimeoutConfigurable(self) -> None:
        config = EstimateZernikesBaseConfig()
        config.timeout = 30
        self.assertEqual(config.timeout, 30)


class TestApplyToList(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _ConcreteTask()

    def testSingleCoreAppliesFunction(self) -> None:
        results = self.task._applyToList(lambda x: x * 2, [1, 2, 3], numCores=1)
        self.assertEqual(results, [2, 4, 6])

    def testSingleCoreEmptyArgs(self) -> None:
        results = self.task._applyToList(lambda x: x, [], numCores=1)
        self.assertEqual(results, [])

    def testMultiCoreReturnsResults(self) -> None:
        # Fake the pool.map_async(...).get(timeout=...) call chain without
        # spawning real processes. __enter__/__exit__ make the `with Pool()`
        # context manager work; return_value/False suppress no exceptions.
        mock_async = MagicMock()
        mock_async.get.return_value = [2, 4, 6]
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map_async.return_value = mock_async

        with patch("lsst.ts.wep.task.estimateZernikesBase.mp.Pool", return_value=mock_pool):
            results = self.task._applyToList(lambda x: x * 2, [1, 2, 3], numCores=2)

        self.assertEqual(results, [2, 4, 6])
        mock_async.get.assert_called_once_with(timeout=self.task.config.timeout)

    def testMultiCoreTimeoutReturnsEmpty(self) -> None:
        # side_effect makes .get() raise instead of return, exercising the
        # timeout-handling path without waiting for a real timeout.
        mock_async = MagicMock()
        mock_async.get.side_effect = mp.TimeoutError
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map_async.return_value = mock_async

        with patch("lsst.ts.wep.task.estimateZernikesBase.mp.Pool", return_value=mock_pool):
            results = self.task._applyToList(lambda x: x * 2, [1, 2, 3], numCores=2)

        self.assertEqual(results, [])

    def testMultiCoreTimeoutLogsError(self) -> None:
        mock_async = MagicMock()
        mock_async.get.side_effect = mp.TimeoutError
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map_async.return_value = mock_async

        with patch("lsst.ts.wep.task.estimateZernikesBase.mp.Pool", return_value=mock_pool):
            with self.assertLogs(level="ERROR") as cm:
                self.task._applyToList(lambda x: x * 2, [1, 2, 3], numCores=2)

        self.assertTrue(any("timed out" in msg for msg in cm.output))


class TestGetObsConditions(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _ConcreteTask()

    def _makeStamps(self, metadata: dict) -> MagicMock:
        # Only .metadata is needed; MagicMock auto-creates any other attribute
        # that gets touched so we don't need a real stamp/butler object.
        stamps = MagicMock()
        stamps.metadata = metadata
        return stamps

    def testNoneInputReturnsEmpty(self) -> None:
        result = self.task._get_obs_conditions(None)
        self.assertIsInstance(result, ObservingConditions)
        self.assertIsNone(result.rtp)
        self.assertIsNone(result.altitude)

    def testAllKeysPresent(self) -> None:
        rsp = 0.1
        q = 0.3
        alt = 1.0
        stamps = self._makeStamps(
            {
                "BORESIGHT_ROT_ANGLE_RAD": rsp,
                "BORESIGHT_PAR_ANGLE_RAD": q,
                "BORESIGHT_ALT_RAD": alt,
            }
        )
        result = self.task._get_obs_conditions(stamps)

        expected_rtp = Angle(q - rsp - np.pi / 2, "rad")
        expected_alt = Angle(alt, "rad")
        self.assertAlmostEqual(result.rtp.rad, expected_rtp.rad)
        self.assertAlmostEqual(result.altitude.rad, expected_alt.rad)

    def testMissingKeysYieldsNoneFields(self) -> None:
        stamps = self._makeStamps({})
        with self.assertLogs(level="WARNING") as cm:
            result = self.task._get_obs_conditions(stamps)

        self.assertIsNone(result.rtp)
        self.assertIsNone(result.altitude)
        # One warning per missing key
        self.assertEqual(sum("missing" in msg for msg in cm.output), 3)

    def testPartialMetadataNoneRtp(self) -> None:
        # altitude present but rsp/q missing → rtp cannot be computed
        stamps = self._makeStamps({"BORESIGHT_ALT_RAD": 0.8})
        with self.assertLogs(level="WARNING"):
            result = self.task._get_obs_conditions(stamps)

        self.assertIsNone(result.rtp)
        self.assertAlmostEqual(result.altitude.rad, 0.8)

    def _makeInstrument(self, maskParamsFile: str | None, maskParams: dict | None = None) -> MagicMock:
        # Only the attributes touched by _logMaskVersions are needed.
        inst = MagicMock()
        inst.maskParamsFile = maskParamsFile
        inst._maskParams = maskParams
        inst.name = "LsstCam"
        inst.configFile = "policy:instruments/LsstCam.yaml"
        inst.batoidModelName = "LSST_{band}"
        return inst

    def testLogMaskVersionsDanishAndBatoid(self) -> None:
        inst = self._makeInstrument("RubinObsc.yaml")
        with self.assertLogs(level="INFO") as cm:
            self.task._logMaskVersions(inst)

        self.assertTrue(any("Mask model: danish" in msg for msg in cm.output))
        self.assertTrue(any("maskParamsFile=RubinObsc.yaml" in msg for msg in cm.output))
        self.assertTrue(any("Batoid model: batoid" in msg for msg in cm.output))
        self.assertTrue(any("LSST_{band}" in msg for msg in cm.output))

        # The mask and Batoid models are also recorded in the task metadata.
        # The danish file is resolved (following the version symlink), so the
        # stored value starts with the danish prefix and the resolved name.
        self.assertTrue(self.task.metadata["maskModel"].startswith("danish:"))
        self.assertEqual(self.task.metadata["batoidModel"], "batoid:LSST_{band}")

    def testLogMaskVersionsExplicitOverride(self) -> None:
        # Explicit maskParams override any danish file.
        inst = self._makeInstrument("RubinObsc.yaml", maskParams={"M1": {}})
        with self.assertLogs(level="INFO") as cm:
            self.task._logMaskVersions(inst)

        self.assertTrue(any("overrides any danish file" in msg for msg in cm.output))
        self.assertFalse(any("resolved to" in msg for msg in cm.output))

        # Explicit maskParams are recorded as coming from the policy instrument
        # config file.
        self.assertEqual(self.task.metadata["maskModel"], "policy:instruments/LsstCam.yaml")
        self.assertEqual(self.task.metadata["batoidModel"], "batoid:LSST_{band}")

    def testLogMaskVersionsNoDanishFile(self) -> None:
        inst = self._makeInstrument(None)
        with self.assertLogs(level="INFO") as cm:
            self.task._logMaskVersions(inst)

        self.assertTrue(any("no danish file" in msg for msg in cm.output))
        self.assertTrue(any("Batoid model: batoid" in msg for msg in cm.output))

        # With no danish file, the mask model is recorded as the policy
        # instrument config file.
        self.assertEqual(self.task.metadata["maskModel"], "policy:instruments/LsstCam.yaml")
        self.assertEqual(self.task.metadata["batoidModel"], "batoid:LSST_{band}")


class TestEstimateZkFailureHandling(unittest.TestCase):
    """A single bad donut should not abort the whole task."""

    def _makeWfEstimator(self, nollIndices: range = range(4, 23)) -> MagicMock:
        wfEst = MagicMock()
        wfEst.nollIndices = np.array(list(nollIndices))
        return wfEst

    def _makeDonut(self, donut_id: int) -> MagicMock:
        donut = MagicMock()
        donut.donut_id = donut_id
        return donut

    def testPairFailureReturnsNaNsFlaggedAsFailure(self) -> None:
        wfEst = self._makeWfEstimator()
        wfEst.estimateZk.side_effect = ValueError(
            "Cannot compute zernike with Gaussian Quadrature with failed rays."
        )
        obs = ObservingConditions()
        args = (self._makeDonut(1), self._makeDonut(2), obs, wfEst)

        with self.assertLogs(level="WARNING") as cm:
            zk, zkMeta, history = estimate_zk_pair(args)

        self.assertEqual(len(zk), len(wfEst.nollIndices))
        self.assertTrue(np.all(np.isnan(zk)))
        self.assertFalse(zkMeta["fit_success"])
        self.assertEqual(history, {})
        self.assertTrue(any("failed" in msg for msg in cm.output))

    def testSingleFailureReturnsNaNsFlaggedAsFailure(self) -> None:
        wfEst = self._makeWfEstimator()
        wfEst.estimateZk.side_effect = ValueError("failed rays")
        obs = ObservingConditions()
        args = (self._makeDonut(1), obs, wfEst)

        with self.assertLogs(level="WARNING"):
            zk, zkMeta, history = estimate_zk_single(args)

        self.assertEqual(len(zk), len(wfEst.nollIndices))
        self.assertTrue(np.all(np.isnan(zk)))
        self.assertFalse(zkMeta["fit_success"])
        self.assertEqual(history, {})

    def testPairSuccessPassesThrough(self) -> None:
        wfEst = self._makeWfEstimator()
        expectedZk = np.arange(len(wfEst.nollIndices), dtype=float)
        wfEst.estimateZk.return_value = (expectedZk, {"fit_success": True})
        wfEst.history = {"foo": "bar"}
        obs = ObservingConditions()
        args = (self._makeDonut(1), self._makeDonut(2), obs, wfEst)

        zk, zkMeta, history = estimate_zk_pair(args)

        np.testing.assert_array_equal(zk, expectedZk)
        self.assertTrue(zkMeta["fit_success"])
        self.assertEqual(history, {"foo": "bar"})


class TestMultiCoreFailureHandling(unittest.TestCase):
    """A single bad donut must not crash a real multiprocessing run."""

    def testEstimateFromPairsMultiCoreToleratesBadDonut(self) -> None:
        task = _ConcreteTask()
        wfEst = _FakeWfEstimator()

        # Three pairs; the middle extra-focal donut is bad and will raise.
        donutStampsExtra = _FakeDonutStamps([_FakeDonut(0), _FakeDonut(1, bad=True), _FakeDonut(2)])
        donutStampsIntra = _FakeDonutStamps([_FakeDonut(10), _FakeDonut(11), _FakeDonut(12)])

        zkArray, zkMeta = task.estimateFromPairs(donutStampsExtra, donutStampsIntra, wfEst, numCores=2)

        # One row per pair, one column per Noll index. The whole task
        # completed instead of crashing on the bad pair.
        self.assertEqual(zkArray.shape, (3, len(wfEst.nollIndices)))

        # Only the bad pair is flagged and NaN'd; the others succeeded.
        self.assertEqual(zkMeta["fit_success"], [True, False, True])
        self.assertFalse(np.any(np.isnan(zkArray[0])))
        self.assertTrue(np.all(np.isnan(zkArray[1])))
        self.assertFalse(np.any(np.isnan(zkArray[2])))

    def testEstimateFromIndivStampsMultiCoreToleratesBadDonut(self) -> None:
        task = _ConcreteTask()
        wfEst = _FakeWfEstimator()

        # Three extra-focal single donuts (the middle one is bad and will
        # raise) and no intra-focal stamps.
        donutStampsExtra = _FakeDonutStamps([_FakeDonut(0), _FakeDonut(1, bad=True), _FakeDonut(2)])
        donutStampsIntra = _FakeDonutStamps([])

        zkArray, zkMeta = task.estimateFromIndivStamps(donutStampsExtra, donutStampsIntra, wfEst, numCores=2)

        self.assertEqual(zkArray.shape, (3, len(wfEst.nollIndices)))
        self.assertEqual(zkMeta["fit_success"], [True, False, True])
        self.assertTrue(np.all(np.isnan(zkArray[1])))
        self.assertFalse(np.any(np.isnan(zkArray[0])))
        self.assertFalse(np.any(np.isnan(zkArray[2])))


class TestCollateZkMeta(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _ConcreteTask()

    def testUniformKeys(self) -> None:
        metas = [
            {"fit_success": True, "fwhm": 1.0},
            {"fit_success": True, "fwhm": 2.0},
        ]
        collated = self.task._collateZkMeta(metas)
        self.assertEqual(collated["fit_success"], [True, True])
        self.assertEqual(collated["fwhm"], [1.0, 2.0])

    def testMissingKeysFilledAndAligned(self) -> None:
        # Second donut failed and only reported fit_success=False.
        metas = [
            {"fit_success": True, "fwhm": 1.0, "chi_square": 3.0},
            {"fit_success": False},
        ]
        collated = self.task._collateZkMeta(metas)
        # Union of keys, in first-seen order.
        self.assertEqual(list(collated.keys()), ["fit_success", "fwhm", "chi_square"])
        # Every list stays aligned with donut order.
        self.assertEqual(collated["fit_success"], [True, False])
        self.assertTrue(np.isnan(collated["fwhm"][1]))
        self.assertTrue(np.isnan(collated["chi_square"][1]))
        self.assertEqual(collated["fwhm"][0], 1.0)

    def testFitSuccessDefaultsTrueWhenAbsent(self) -> None:
        # A donut that never reported fit_success did not hit the failure path.
        metas = [
            {"fwhm": 1.0},
            {"fit_success": False},
        ]
        collated = self.task._collateZkMeta(metas)
        self.assertEqual(collated["fit_success"], [True, False])


if __name__ == "__main__":
    unittest.main()
