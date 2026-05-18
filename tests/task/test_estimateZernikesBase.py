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


if __name__ == "__main__":
    unittest.main()
