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

import unittest
import numpy as np

import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.CombineZernikesSigmaClipTask import (
    CombineZernikesSigmaClipTaskConfig,
    CombineZernikesSigmaClipTask,
)


class TestCombineZernikesSigmaClipTask(unittest.TestCase):
    def setUp(self):

        self.config = CombineZernikesSigmaClipTaskConfig()
        self.task = CombineZernikesSigmaClipTask()

    def prepareTestData(self):

        inputArray = np.ones((101, 10))
        inputArray[50:100] += 2.0
        inputArray[100] += 100.0
        outputFlags = np.zeros(101)
        outputFlags[100] = 1.0
        return inputArray, outputFlags

    def testValidateConfigs(self):

        self.assertEqual(3.0, self.task.sigma)
        self.config.sigma = 2.0
        self.task = CombineZernikesSigmaClipTask(config=self.config)
        self.assertEqual(2.0, self.task.sigma)

    def testCombineZernikes(self):

        zernikeArray, trueFlags = self.prepareTestData()
        combinedZernikes, testFlags = self.task.combineZernikes(zernikeArray)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikes)
        np.testing.assert_array_equal(trueFlags, testFlags)

    def testTaskRun(self):

        zernikeArray, trueFlags = self.prepareTestData()
        combinedZernikesStruct = self.task.run(zernikeArray)
        self.assertEqual(type(combinedZernikesStruct), pipeBase.Struct)
        np.testing.assert_array_equal(
            np.ones(10) * 2.0, combinedZernikesStruct.combinedZernikes
        )
        np.testing.assert_array_equal(trueFlags, combinedZernikesStruct.flags)
