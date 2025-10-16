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

import numbers
import unittest

import lsst.pipe.base as pipeBase
import numpy as np
from lsst.ts.wep.task.combineZernikesSigmaClipTask import (
    CombineZernikesSigmaClipTask,
    CombineZernikesSigmaClipTaskConfig,
)


class TestCombineZernikesSigmaClipTask(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CombineZernikesSigmaClipTaskConfig()
        self.task = CombineZernikesSigmaClipTask()

    def prepareTestData(self) -> tuple[np.ndarray, np.ndarray]:
        inputArray = np.ones((101, 10))
        inputArray[50:100] += 2.0
        inputArray[100] += 100.0
        outputFlags = np.zeros(101, dtype=int)
        outputFlags[100] = 1
        return inputArray, outputFlags

    def testValidateConfigs(self) -> None:
        self.assertEqual(
            {"sigma": 3.0, "stdfunc": "mad_std", "maxiters": 1},
            self.task.sigmaClipKwargs,
        )
        self.assertEqual(3, self.task.maxZernClip)
        self.config.sigmaClipKwargs["sigma"] = 2.0
        self.config.stdMin = 0.005
        self.config.maxZernClip = 5
        self.task = CombineZernikesSigmaClipTask(config=self.config)
        self.assertEqual(2.0, self.task.sigmaClipKwargs["sigma"])
        self.assertEqual(0.005, self.task.stdMin)
        self.assertEqual(5, self.task.maxZernClip)

    def testCombineZernikes(self) -> None:
        zernikeArray, trueFlags = self.prepareTestData()
        combinedZernikes, testFlags = self.task.combineZernikes(zernikeArray)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikes)
        np.testing.assert_array_equal(trueFlags, testFlags)
        self.assertTrue(isinstance(testFlags[0], numbers.Integral))

        # Test that the conditional sigma clipping works
        combinedZernikes, testFlags = self.task.combineZernikes(zernikeArray * 1e-4)
        np.testing.assert_array_almost_equal(np.ones(10) * 2.98e-4, combinedZernikes)
        trueFlags[100] = 0
        np.testing.assert_array_equal(trueFlags, testFlags)
        self.assertTrue(isinstance(testFlags[0], numbers.Integral))

        # Test that zernikes higher than maxZernClip don't remove
        # a row from the final averaging
        zernikeArray[0, 3:] += 100.0
        zernikeArray[49, 3:] -= 100.0
        # Revert the change in the 100th row from previous test
        trueFlags[100] = 1
        combinedZernikes, testFlags = self.task.combineZernikes(zernikeArray)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikes)
        np.testing.assert_array_equal(trueFlags, testFlags)
        self.assertTrue(isinstance(testFlags[0], numbers.Integral))

        # Test that changing the maxZernClip parameter does change
        # whether a row is removed from the final result
        zernikeArray[50, 3:] += 100.0
        zernikeArray[51, 3:] -= 100.0
        self.config.maxZernClip = 5
        self.task = CombineZernikesSigmaClipTask(config=self.config)
        combinedZernikes, testFlags = self.task.combineZernikes(zernikeArray)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikes)
        trueFlags[0] = 1
        trueFlags[49:52] = 1
        np.testing.assert_array_equal(trueFlags, testFlags)
        self.assertTrue(isinstance(testFlags[0], numbers.Integral))

    def testCombineZernikesEffectiveMaxZernClip(self) -> None:
        testWhileZernikeArray = np.ones((3, 10))
        testWhileZernikeArray[0, 4] = 3
        testWhileZernikeArray[1, 3] = 3
        testWhileZernikeArray[2, 2] = 3

        # Test that changing the maxZernClip parameter does change
        # whether a row is removed from the final result
        self.config.maxZernClip = 6
        self.task = CombineZernikesSigmaClipTask(config=self.config)
        combinedZernikes, testFlags = self.task.combineZernikes(testWhileZernikeArray)
        np.testing.assert_array_equal(testWhileZernikeArray[0], combinedZernikes)
        self.assertEqual(self.task.metadata["maxZernClip"], 6)
        self.assertEqual(self.task.metadata["effMaxZernClip"], 4)
        trueFlags = np.array([0, 1, 1])
        np.testing.assert_array_equal(trueFlags, testFlags)
        self.assertTrue(isinstance(testFlags[0], numbers.Integral))

    def testTaskRun(self) -> None:
        zernikeArray, trueFlags = self.prepareTestData()
        combinedZernikesStruct = self.task.run(zernikeArray)
        self.assertEqual(type(combinedZernikesStruct), pipeBase.Struct)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikesStruct.combinedZernikes)
        np.testing.assert_array_equal(trueFlags, combinedZernikesStruct.flags)
        self.assertTrue(isinstance(combinedZernikesStruct.flags[0], numbers.Integral))
