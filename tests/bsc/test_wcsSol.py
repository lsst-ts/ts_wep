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

import numpy as np
import unittest

from lsst.ts.wep.bsc.WcsSol import WcsSol


class TestWcsSol(unittest.TestCase):
    """Test the ComCam class."""

    def setUp(self):

        self.wcs = WcsSol()

        self.ra = 1.0
        self.dec = 2.0
        rotSkyPos = 0.0
        self.wcs.setObsMetaData(self.ra, self.dec, rotSkyPos)

    def testSetAndGetCamera(self):

        camera = "FaultCamera"
        self.wcs.setCamera(camera)

        self.assertEqual(self.wcs.getCamera(), camera)

    def testsetObsMetaData(self):

        ra = 10.0
        dec = 20.0
        rotSkyPos = 30.0
        self.wcs.setObsMetaData(ra, dec, rotSkyPos)

        skyOrigin = self.wcs.skyWcs.getSkyOrigin()
        self.assertAlmostEqual(skyOrigin.getRa().asDegrees(), ra)
        self.assertAlmostEqual(skyOrigin.getDec().asDegrees(), dec)

    def testFormatCoordListWiIntFloat(self):

        ra = 10
        dec = 19.0
        raOut, decOut = self.wcs._formatCoordList(ra, dec, "ra", "dec")
        self.assertEqual(raOut, [ra])
        self.assertEqual(decOut, [dec])

    def testFormatCoordListWiLists(self):

        ra = [10.0, 20.0]
        dec = [5.0, 23.0]
        raOut, decOut = self.wcs._formatCoordList(ra, dec, "ra", "dec")
        self.assertEqual(ra, raOut)
        self.assertEqual(dec, decOut)

    def testFormatCoordListAssert(self):

        ra = [10, 20]
        dec = [10.0, 20.0, 30.0]

        with self.assertRaises(AssertionError, msg="Size of ra not same as dec"):
            self.wcs._formatCoordList(ra, dec, "ra", "dec")

    def testRaDecFromPixelCoordsForSingleChip(self):

        xPix = 2032
        yPix = 2000
        chipName = "R22_S11"
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)
        self.assertAlmostEqual(raByWcs, self.ra, places=2)
        self.assertAlmostEqual(decByWcs, self.dec, places=2)

    def testRaDecFromPixelCoordsForChipArray(self):

        xPix = np.array([2032, 2032])
        yPix = np.array([2000, 2000])
        chipName = np.array(["R22_S11", "R22_S11"])
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertEqual(len(raByWcs), 2)
        self.assertEqual(len(decByWcs), 2)

        self.assertAlmostEqual(raByWcs[0], self.ra, places=2)
        self.assertAlmostEqual(raByWcs[1], self.ra, places=2)

    def testPixelCoordsFromRaDecWithoutChipName(self):

        xPix, yPix = self.wcs.pixelCoordsFromRaDec(self.ra, self.dec)

        self.assertAlmostEqual(xPix, 2048, places=-1)
        self.assertAlmostEqual(yPix, 2000, places=-1)

    def testPixelCoordsFromRaDecWithChipName(self):

        chipName = "R22_S11"
        xPix, yPix = self.wcs.pixelCoordsFromRaDec(self.ra, self.dec, chipName=chipName)

        self.assertAlmostEqual(xPix, 2048, places=-1)
        self.assertAlmostEqual(yPix, 2000, places=-1)

    def testPixelCoordsFromRaDecWithWrongType(self):

        with self.assertRaises(
            ValueError,
            msg="chipName is an unallowed type. Can be None, string or array of strings.",
        ):
            self.wcs.pixelCoordsFromRaDec(self.ra, self.dec, chipName=20.0)

    def testFocalPlaneCoordsFromRaDecWithZeroRot(self):

        self.wcs.setObsMetaData(0, 0, 0)
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(0, 0)

        self.assertEqual(xInMm, 0.0)
        self.assertEqual(yInMm, 0.0)

        # 0.2 arcsec = 10 um => 1 um = 0.02 arcsec => 1 mm = 20 arcsec
        # 1 arcsec = 1/3600 degree
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(20.0 / 3600, 0)

        self.assertAlmostEqual(xInMm, -1.0, places=3)
        self.assertAlmostEqual(yInMm, 0.0, places=3)

    def testFocalPlaneCoordsFromRaDecWithNonZeroRot(self):

        self.wcs.setObsMetaData(0, 0, 45)

        # 0.2 arcsec = 10 um => 1 um = 0.02 arcsec => 1 mm = 20 arcsec
        # 1 arcsec = 1/3600 degree
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(20.0 / 3600, 0)

        self.assertAlmostEqual(xInMm, -1 / np.sqrt(2), places=3)
        self.assertAlmostEqual(yInMm, 1 / np.sqrt(2), places=3)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
