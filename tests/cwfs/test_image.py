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
from lsst.ts.wep.cwfs.CentroidRandomWalk import CentroidRandomWalk
from lsst.ts.wep.Utility import getModulePath


class TestImage(unittest.TestCase):
    """Test the Image class."""

    def setUp(self):
        self.testDataDir = os.path.join(getModulePath(), "tests", "testData")
        self.imgFile = os.path.join(
            self.testDataDir, "testImages", "LSST_NE_SN25", "z11_0.25_intra.txt"
        )

        self.img = Image()
        self.img.setImg(imageFile=self.imgFile)

    def testGetCentroidFind(self):
        centroidFind = self.img.getCentroidFind()
        self.assertTrue(isinstance(centroidFind, CentroidRandomWalk))

    def testGetImg(self):
        img = self.img.getImg()
        self.assertEqual(img.shape, (120, 120))

    def testGetImgFilePath(self):
        imgFilePath = self.img.getImgFilePath()
        self.assertEqual(imgFilePath, self.imgFile)

    def testSetImgByImageArray(self):
        newImg = np.random.rand(5, 5)
        self.img.setImg(image=newImg)

        self.assertTrue(np.all(self.img.getImg() == newImg))
        self.assertEqual(self.img.getImgFilePath(), "")

    def testSetImgByFitsFile(self):
        opdFitsFile = os.path.join(
            self.testDataDir, "opdOutput", "9006000", "opd_9006000_0.fits.gz"
        )
        self.img.setImg(imageFile=opdFitsFile)

        img = self.img.getImg()
        self.assertEqual(img.shape, (255, 255))

        imgFilePath = self.img.getImgFilePath()
        self.assertEqual(imgFilePath, opdFitsFile)

    def testUpdateImage(self):
        newImg = np.random.rand(5, 5)
        self.img.updateImage(newImg)

        self.assertTrue(np.all(self.img.getImg() == newImg))

    def testUpdateImageWithNoHoldImage(self):
        img = Image()

        newImg = np.random.rand(5, 5)
        self.assertWarns(UserWarning, img.updateImage, newImg)

    def testGetCenterAndR_ef(self):
        realcx, realcy, realR = self.img.getCenterAndR()
        self.assertEqual(int(realcx), 61)
        self.assertEqual(int(realcy), 61)
        self.assertGreater(int(realR), 35)

    def testGetSNR(self):
        # Add the noise to the image
        image = self.img.getImg()
        noisedImg = image + np.random.random(image.shape) * 0.1

        self.img.setImg(image=noisedImg)
        snr = self.img.getSNR()

        self.assertGreater(snr, 15)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
