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

import os
import time
import numpy as np
import unittest

from lsst.ts.wep.cwfs.baseCwfsTestCase import BaseCwfsTestCase
from lsst.ts.wep.utility import getModulePath, CamType, CentroidFindType


class TestWepWithMultiImgs(BaseCwfsTestCase, unittest.TestCase):
    """Test the images to compare with bxin/cwfs package"""

    def setUp(self):
        # Path of test images and validation data
        modulePath = getModulePath()
        self.imageFolderPath = os.path.join(
            modulePath, "tests", "testData", "testImages"
        )
        self.validationDir = os.path.join(
            modulePath, "tests", "testData", "testImages", "validation", "simulation"
        )

        # Get inst information
        self.instParams = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": 1.0,
            "pixelSize": 10.0e-6,
        }

        # Set the tolerance
        self.tolMax = 4
        self.tolRms = 1

        # Record the start time
        self.startTime = time.perf_counter()

    def tearDown(self):
        # Calculate the duration
        t = time.perf_counter() - self.startTime
        print("%s: %.3f s." % (self.id(), t))

    def testCase1(self):
        self._calcWfErrAndCompareWithAns(
            "LSST_NE_SN25", "z11_0.25", (1.185, 1.185), "exp", "offAxis"
        )

    def _calcWfErrAndCompareWithAns(
        self, imageFolderName, baseImageName, fieldXY, algoName, opticalModel
    ):
        imageFileIntra = os.path.join(
            self.imageFolderPath, imageFolderName, f"{baseImageName}_intra.txt"
        )
        imageFileExtra = os.path.join(
            self.imageFolderPath, imageFolderName, f"{baseImageName}_extra.txt"
        )
        wfErr = self.calcWfErr(
            CentroidFindType.RandomWalk,
            fieldXY,
            CamType.LsstCam,
            algoName,
            opticalModel,
            self.instParams,
            imageFileIntra=imageFileIntra,
            imageFileExtra=imageFileExtra,
        )

        # Get the answer of wavefront error
        ansFileName = f"{imageFolderName}_{baseImageName}_{algoName}.txt"
        ansFilePath = os.path.join(self.validationDir, ansFileName)
        wfErrAns = np.loadtxt(ansFilePath)

        self.compareDiffWithTol(wfErr, wfErrAns, self.tolMax, self.tolRms)

    def testCase2(self):
        self._calcWfErrAndCompareWithAns(
            "LSST_NE_SN25", "z11_0.25", (1.185, 1.185), "fft", "offAxis"
        )

    def testCase3(self):
        self._calcWfErrAndCompareWithAns(
            "F1.23_1mm_v61", "z7_0.25", (0, 0), "fft", "paraxial"
        )

    def testCase4(self):
        self._calcWfErrAndCompareWithAns(
            "LSST_C_SN26", "z7_0.25", (0, 0), "fft", "onAxis"
        )

    def testCase5(self):
        self._calcWfErrAndCompareWithAns(
            "LSST_C_SN26", "z7_0.25", (0, 0), "exp", "onAxis"
        )


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
