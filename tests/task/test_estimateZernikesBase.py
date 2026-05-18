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


if __name__ == "__main__":
    unittest.main()
