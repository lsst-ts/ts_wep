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
import numpy as np
import unittest

from lsst.ts.wep.cwfs.Image import Image
from lsst.ts.wep.cwfs.BaseCwfsTestCase import BaseCwfsTestCase
from lsst.ts.wep.Utility import getModulePath, CamType, CentroidFindType


class TestImgsAuxTelZWO(BaseCwfsTestCase, unittest.TestCase):
    """Test the images of auxiliary telescope ZWO camera"""

    def setUp(self):

        testImageDataDir = os.path.join(
            getModulePath(), "tests", "testData", "testImages"
        )
        self.testImgDir = os.path.join(testImageDataDir, "auxTelZWO")
        self.validationDir = os.path.join(testImageDataDir, "validation", "auxTelZWO")

        self.offset = 80
        self.tolMax = 6
        self.tolRms = 2

    def testCase1paraxial(self):

        imgIntraName, imgExtraName = self._getImgsCase1()
        zer4UpNm = self._calcWfErrAuxTelZWO(
            imgIntraName, imgExtraName, self.offset, "paraxial"
        )

        ans = self._getDataVerify("case1_auxTelZWO_paraxial.txt")
        self.compareDiffWithTol(zer4UpNm, ans, self.tolMax, self.tolRms)

    def _getImgsCase1(self):

        imgIntraName = "1579925613-16Pup_intra-0-1.fits"
        imgExtraName = "1579925662-16Pup_extra-0-1.fits"

        return imgIntraName, imgExtraName

    def _calcWfErrAuxTelZWO(self, imgIntraName, imgExtraName, offset, model):

        # Cut the donut image from input files
        centroidFindType = CentroidFindType.Otsu
        imgIntra = Image(centroidFindType=centroidFindType)
        imgExtra = Image(centroidFindType=centroidFindType)

        imgIntraPath = os.path.join(self.testImgDir, imgIntraName)
        imgExtraPath = os.path.join(self.testImgDir, imgExtraName)

        imgIntra.setImg(imageFile=imgIntraPath)
        imgExtra.setImg(imageFile=imgExtraPath)

        xIntra, yIntra = imgIntra.getCenterAndR()[0:2]
        imgIntraArray = imgIntra.getImg()[
            int(yIntra) - offset : int(yIntra) + offset,
            int(xIntra) - offset : int(xIntra) + offset,
        ]

        xExtra, yExtra = imgExtra.getCenterAndR()[0:2]
        imgExtraArray = imgExtra.getImg()[
            int(yExtra) - offset : int(yExtra) + offset,
            int(xExtra) - offset : int(xExtra) + offset,
        ]

        # Calculate the wavefront error
        fieldXY = (0, 0)
        wfErr = self.calcWfErr(
            centroidFindType,
            fieldXY,
            CamType.AuxTelZWO,
            "exp",
            0.5,
            model,
            imageIntra=imgIntraArray,
            imageExtra=imgExtraArray,
        )

        return wfErr

    def _getDataVerify(self, fileName):

        filePath = os.path.join(self.validationDir, fileName)

        return np.loadtxt(filePath)

    def testCase1onaxis(self):

        imgIntraName, imgExtraName = self._getImgsCase1()
        zer4UpNm = self._calcWfErrAuxTelZWO(
            imgIntraName, imgExtraName, self.offset, "onAxis"
        )

        ans = self._getDataVerify("case1_auxTelZWO_onaxis.txt")
        self.compareDiffWithTol(zer4UpNm, ans, self.tolMax, self.tolRms)

    def testCase2paraxial(self):

        imgIntraName, imgExtraName = self._getImgsCase2()
        zer4UpNm = self._calcWfErrAuxTelZWO(
            imgIntraName, imgExtraName, self.offset, "paraxial"
        )

        ans = self._getDataVerify("case2_auxTelZWO_paraxial.txt")
        self.compareDiffWithTol(zer4UpNm, ans, self.tolMax, self.tolRms)

    def _getImgsCase2(self):

        imgIntraName = "1579925833-16Pup_intra-0-1.fits"
        imgExtraName = "1579925882-16Pup_extra-0-1.fits"

        return imgIntraName, imgExtraName

    def testCase2onaxis(self):

        imgIntraName, imgExtraName = self._getImgsCase2()
        zer4UpNm = self._calcWfErrAuxTelZWO(
            imgIntraName, imgExtraName, self.offset, "onAxis"
        )

        ans = self._getDataVerify("case2_auxTelZWO_onaxis.txt")
        self.compareDiffWithTol(zer4UpNm, ans, self.tolMax, self.tolRms)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
