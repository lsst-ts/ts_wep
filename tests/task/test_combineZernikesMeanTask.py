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
from lsst.ts.wep.task.combineZernikesMeanTask import CombineZernikesMeanTask


class TestCombineZernikesMeanTask(unittest.TestCase):
    def setUp(self) -> None:
        self.task = CombineZernikesMeanTask()

    def prepareTestArray(self) -> np.ndarray:
        inputArray = np.ones((2, 10))
        inputArray[1] += 2.0
        return inputArray

    def testCombineZernikes(self) -> None:
        zernikeArray = self.prepareTestArray()
        combinedZernikes, flags = self.task.combineZernikes(zernikeArray)
        np.testing.assert_array_equal(np.ones(10) * 2.0, combinedZernikes)
        np.testing.assert_array_equal(np.zeros(2), flags)
        self.assertTrue(isinstance(flags[0], numbers.Integral))

    def testTaskRun(self) -> None:
        zernikeArray = self.prepareTestArray()
        combinedZernikesStruct = self.task.run(zernikeArray)
        self.assertEqual(type(combinedZernikesStruct), pipeBase.Struct)
        np.testing.assert_array_equal(
            np.ones(10) * 2.0, combinedZernikesStruct.combinedZernikes
        )
        np.testing.assert_array_equal(np.zeros(2), combinedZernikesStruct.flags)
        self.assertTrue(isinstance(combinedZernikesStruct.flags[0], numbers.Integral))
