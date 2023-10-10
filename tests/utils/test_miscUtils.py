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
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTaskConfig
from lsst.ts.wep.utils import (
    createInstDictFromConfig,
    extractArray,
    getDefocalDisInMm,
    padArray,
    rotMatrix,
)


class TestMiscUtils(unittest.TestCase):
    """Test the miscellaneous utility functions."""

    def testGetDefocalDisInMm(self):
        self.assertEqual(getDefocalDisInMm("lsst"), 1.5)
        self.assertEqual(getDefocalDisInMm("lsstfam"), 1.5)
        self.assertEqual(getDefocalDisInMm("comcam"), 1.5)
        self.assertEqual(getDefocalDisInMm("auxTel"), 0.8)
        instName = "telescope"
        assertMsg = f"Instrument name ({instName}) is not supported."
        with self.assertRaises(ValueError) as context:
            getDefocalDisInMm(instName)
        self.assertTrue(assertMsg in str(context.exception))

    def testCreateInstDictFromConfig(self):
        # Test instDict creation in tasks
        testConfig = CalcZernikesTaskConfig()
        testInstDict = createInstDictFromConfig(testConfig)
        truthInstDict = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": None,
            "pixelSize": 10.0e-6,
        }

        self.assertDictEqual(truthInstDict, testInstDict)

    def testRotMatrix(self):
        # Test rotation with 0 degrees
        testTheta1 = 0
        rotMatrix1 = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(rotMatrix1, rotMatrix(testTheta1))

        # Test rotation with 90 degrees
        testTheta2 = 90
        rotMatrix2 = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(rotMatrix2, rotMatrix(testTheta2))

        # Test rotation with 45 degrees
        testTheta3 = 45
        rotMatrix3 = np.array([[0.707107, -0.707107], [0.707107, 0.707107]])
        np.testing.assert_array_almost_equal(rotMatrix3, rotMatrix(testTheta3))

    def testPadArray(self):
        imgDim = 10
        padPixelSize = 20

        img, imgPadded = self._padRandomImg(imgDim, padPixelSize)

        self.assertEqual(imgPadded.shape[0], imgDim + padPixelSize)

    def _padRandomImg(self, imgDim, padPixelSize):
        img = np.random.rand(imgDim, imgDim)
        imgPadded = padArray(img, imgDim + padPixelSize)

        return img, imgPadded

    def testExtractArray(self):
        imgDim = 10
        padPixelSize = 20
        img, imgPadded = self._padRandomImg(imgDim, padPixelSize)

        imgExtracted = extractArray(imgPadded, imgDim)

        self.assertEqual(imgExtracted.shape[0], imgDim)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
