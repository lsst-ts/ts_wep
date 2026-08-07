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

    def _makeInstrument(self, maskParamsFile, maskParams=None) -> MagicMock:
        # Only the attributes touched by _logMaskVersions are needed.
        inst = MagicMock()
        inst.maskParamsFile = maskParamsFile
        inst._maskParams = maskParams
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

    def testLogMaskVersionsExplicitOverride(self) -> None:
        # Explicit maskParams override any danish file.
        inst = self._makeInstrument("RubinObsc.yaml", maskParams={"M1": {}})
        with self.assertLogs(level="INFO") as cm:
            self.task._logMaskVersions(inst)

        self.assertTrue(any("overrides any danish file" in msg for msg in cm.output))
        self.assertFalse(any("resolved to" in msg for msg in cm.output))

    def testLogMaskVersionsNoDanishFile(self) -> None:
        inst = self._makeInstrument(None)
        with self.assertLogs(level="INFO") as cm:
            self.task._logMaskVersions(inst)

        self.assertTrue(any("no danish file" in msg for msg in cm.output))
        self.assertTrue(any("Batoid model: batoid" in msg for msg in cm.output))


class TestEstimateZkFailureHandling(unittest.TestCase):
    """A single bad donut should not abort the whole task."""

    def _makeWfEstimator(self, nollIndices=range(4, 23)) -> MagicMock:
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

        with self.assertLogs(level="ERROR") as cm:
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

        with self.assertLogs(level="ERROR"):
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
