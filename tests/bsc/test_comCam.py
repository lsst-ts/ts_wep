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

from lsst.ts.wep.bsc.StarData import StarData
from lsst.ts.wep.bsc.ComCam import ComCam
from lsst.ts.wep.Utility import FilterType


class TestComCam(unittest.TestCase):
    """Test the ComCam class and functions of parent class."""

    def setUp(self):

        # Boresight (unit: degree)
        ra = 0.0  # 0 <= RA <= 360
        dec = 30.0  # -90 <= Dec <= 90
        rotSkyPos = 0.0
        self.camera = ComCam()
        self.camera.setObsMetaData(ra, dec, rotSkyPos, mjd=59580.0)

        self.stars = StarData(
            [123, 456, 789],
            [0.1, 0.2, 0.3],
            [2.1, 2.2, 2.3],
            [2.0, 3.0, 4.0],
            [2.1, 2.1, 4.1],
            [2.2, 3.2, 4.2],
            [2.3, 3.3, 4.3],
            [],
            [],
        )

    def testGetWfsCcdList(self):

        wfsCcdList = self.camera.getWfsCcdList()

        self.assertEqual(len(wfsCcdList), 9)
        self.assertTrue("R22_S11" in wfsCcdList)
        self.assertFalse("R21_S11" in wfsCcdList)

    def testSetWfsCcdList(self):

        wfsCcdList = ["a", "b", "c"]
        self.camera.setWfsCcdList(wfsCcdList)

        self.assertEqual(self.camera.getWfsCcdList(), wfsCcdList)

    def testGetWfsCorner(self):

        wfsCorner = self.camera.getWfsCorner("R22_S11")

        self.assertEqual(len(wfsCorner), 2)
        self.assertRaises(KeyError, self.camera.getWfsCorner, "R21_S11")

    def testSetWfsCorners(self):

        wfsCorners = {"a": 1, "b": 2}
        self.camera.setWfsCorners(wfsCorners)

        self.assertEqual(self.camera.getWfsCorner("a"), wfsCorners["a"])

    def testGetCcdDim(self):

        ccdDim = self.camera.getCcdDim("R22_S11")

        self.assertEqual(ccdDim, (4096, 4004))

    def testSetCcdDims(self):

        ccdDims = {"a": (1, 1), "b": (2, 2)}
        self.camera.setCcdDims(ccdDims)

        self.assertEqual(self.camera.getCcdDim("a"), ccdDims["a"])

    def testGetWavefrontSensor(self):

        wfsData = self.camera.getWavefrontSensor()
        self.assertEqual(len(wfsData), 9)

    def testPopulatePixelFromRADecl(self):

        self.assertEqual(len(self.stars.getRaInPixel()), 0)

        populatedStar = self._populatePixelFromRADecl()

        self.assertEqual(len(self.stars.getRaInPixel()), 0)
        self.assertEqual(len(self.stars.getDeclInPixel()), 0)
        self.assertEqual(len(populatedStar.getRaInPixel()), 3)
        self.assertEqual(len(populatedStar.getDeclInPixel()), 3)
        self.assertNotEqual(id(populatedStar), id(self.stars))

    def _populatePixelFromRADecl(self):

        self.stars.setDetector("R22_S11")
        populatedStar = self.camera.populatePixelFromRADecl(self.stars)

        return populatedStar

    def testRemoveStarsNotOnDetectorWithLargeOffset(self):

        stars = self._populatePixelFromRADecl()
        starsOnDet = self.camera.getStarsOnDetector(stars, 1e7)

        self.assertEqual(len(starsOnDet.getId()), 3)
        self.assertEqual(len(starsOnDet.getRA()), 3)
        self.assertEqual(len(starsOnDet.getDecl()), 3)
        self.assertEqual(len(starsOnDet.getRaInPixel()), 3)
        self.assertEqual(len(starsOnDet.getDeclInPixel()), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.U)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.G)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.R)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.I)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Z)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Y)), 0)

        self.assertNotEqual(id(starsOnDet), id(stars))

    def testRemoveStarsNotOnDetectorWithZeroOffset(self):

        stars = self._populatePixelFromRADecl()
        starsOnDet = self.camera.getStarsOnDetector(stars, 0)

        self.assertEqual(len(starsOnDet.getId()), 0)
        self.assertEqual(len(starsOnDet.getRA()), 0)
        self.assertEqual(len(starsOnDet.getDecl()), 0)
        self.assertEqual(len(starsOnDet.getRaInPixel()), 0)
        self.assertEqual(len(starsOnDet.getDeclInPixel()), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.U)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.G)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.R)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.I)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Z)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Y)), 0)

        self.assertEqual(len(stars.getId()), 3)
        self.assertEqual(len(stars.getRA()), 3)
        self.assertEqual(len(stars.getDecl()), 3)
        self.assertEqual(len(stars.getRaInPixel()), 3)
        self.assertEqual(len(stars.getDeclInPixel()), 3)
        self.assertEqual(len(stars.getMag(FilterType.U)), 3)
        self.assertEqual(len(stars.getMag(FilterType.G)), 3)
        self.assertEqual(len(stars.getMag(FilterType.R)), 3)
        self.assertEqual(len(stars.getMag(FilterType.I)), 3)
        self.assertEqual(len(stars.getMag(FilterType.Z)), 0)
        self.assertEqual(len(stars.getMag(FilterType.Y)), 0)

        self.assertNotEqual(id(starsOnDet), id(stars))


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
