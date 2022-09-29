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
import yaml
import numpy as np
import unittest

from lsst.ts.wep.cwfs.BaseCwfsTestCase import BaseCwfsTestCase
from lsst.ts.wep.Utility import getModulePath, CamType, CentroidFindType, getConfigDir


class TestImgsLsstFam(BaseCwfsTestCase, unittest.TestCase):
    """Test the images of LSST full-array mode (FAM)"""

    def setUp(self):

        testImageDataDir = os.path.join(
            getModulePath(), "tests", "testData", "testImages"
        )
        self.testImgDir = os.path.join(testImageDataDir, "lsstfam")
        self.validationDir = os.path.join(testImageDataDir, "validation", "lsstfam")

        # Get inst information
        instConfigDir = os.path.join(getConfigDir(), "cwfs", "instData")
        instConfigFile = os.path.join(instConfigDir, "lsstfam", "instParam.yaml")
        with open(instConfigFile, "r") as stream:
            self.instParams = yaml.safe_load(stream)

    def testImages(self):

        sensorNames, fieldXYs = self._getSensorNameAndFieldXY()
        wfErrCwfs = self._getWfErrByCwfs()
        for idx, sensorName in enumerate(sensorNames):
            imgFileIntra, imgFileExtra = self._getTestImgsPath(sensorName)
            wfErr = self.calcWfErr(
                CentroidFindType.Otsu,
                fieldXYs[idx, :],
                CamType.LsstFamCam,
                "exp",
                "offAxis",
                self.instParams,
                imageFileIntra=imgFileIntra,
                imageFileExtra=imgFileExtra,
            )

            self.compareDiffWithTol(wfErr, wfErrCwfs[idx, :], 12, 4)

            wfErrTruth = self._getTruthOfWfErr(sensorName)
            self.compareDiffWithTol(wfErr, wfErrTruth, 60, 17)

    def _getSensorNameAndFieldXY(self):

        fileOfFieldXY = os.path.join(self.testImgDir, "ccdCenterTestImg.txt")
        data = np.loadtxt(fileOfFieldXY)

        sensorNames = []
        for idx in range(data.shape[0]):
            sensorNames.append("R%02d_S%02d" % (data[idx, 2], data[idx, 3]))

        return sensorNames, data[:, 0:2]

    def _getWfErrByCwfs(self):

        fileWfErr = os.path.join(self.validationDir, "famTestResults.txt")

        return np.loadtxt(fileWfErr)

    def _getTestImgsPath(self, sensorName):

        imgFileIntra = os.path.join(self.testImgDir, f"Image_intra_{sensorName}.txt")
        imgFileExtra = os.path.join(self.testImgDir, f"Image_extra_{sensorName}.txt")

        return imgFileIntra, imgFileExtra

    def _getTruthOfWfErr(self, sensorName):

        fileWfErr = os.path.join(self.validationDir, f"zn_{sensorName}.txt")
        wfErr = np.loadtxt(fileWfErr)

        return wfErr[3:22] * 500


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
