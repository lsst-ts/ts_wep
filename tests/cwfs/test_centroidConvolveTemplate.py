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

from lsst.ts.wep.cwfs.CentroidConvolveTemplate import CentroidConvolveTemplate


class TestCentroidConvolveTemplate(unittest.TestCase):
    """Test the CentroidConvolveTemplate class."""

    def setUp(self):

        self.centroidConv = CentroidConvolveTemplate()

    def _createData(self, radiusInner, radiusOuter, imageSize, addNoise=False):

        # Create two images. One with a single donut and one with two donuts.
        singleDonut = np.zeros((imageSize, imageSize))
        doubleDonut = np.zeros((imageSize, imageSize))

        for x in range(160):
            for y in range(160):
                if np.sqrt((80 - x) ** 2 + (80 - y) ** 2) <= radiusOuter:
                    singleDonut[x, y] += 1
                if np.sqrt((80 - x) ** 2 + (80 - y) ** 2) <= radiusInner:
                    singleDonut[x, y] -= 1
                if np.sqrt((50 - x) ** 2 + (80 - y) ** 2) <= radiusOuter:
                    doubleDonut[x, y] += 1
                if np.sqrt((50 - x) ** 2 + (80 - y) ** 2) <= radiusInner:
                    doubleDonut[x, y] -= 1
                if np.sqrt((100 - x) ** 2 + (80 - y) ** 2) <= radiusOuter:
                    doubleDonut[x, y] += 1
                if np.sqrt((100 - x) ** 2 + (80 - y) ** 2) <= radiusInner:
                    doubleDonut[x, y] -= 1
        # Make binary image
        doubleDonut[doubleDonut > 0.5] = 1

        if addNoise is True:
            # Add noise so the images are not binary
            randState = np.random.RandomState(42)
            singleDonut += randState.normal(scale=0.01, size=np.shape(singleDonut))
            doubleDonut += randState.normal(scale=0.01, size=np.shape(doubleDonut))

        eff_radius = np.sqrt(radiusOuter ** 2 - radiusInner ** 2)

        return singleDonut, doubleDonut, eff_radius

    def testGetImgBinary(self):

        singleDonut, doubleDonut, eff_radius = self._createData(
            20, 40, 160, addNoise=False
        )

        noisySingle, noisyDouble, eff_radius = self._createData(
            20, 40, 160, addNoise=True
        )

        binarySingle = self.centroidConv.getImgBinary(noisySingle)

        np.testing.assert_array_equal(singleDonut, binarySingle)

    def testGetCenterAndRWithoutTemplate(self):

        singleDonut, doubleDonut, eff_radius = self._createData(
            20, 40, 160, addNoise=True
        )

        # Test recovery with defaults
        centX, centY, rad = self.centroidConv.getCenterAndR(singleDonut)

        self.assertEqual(centX, 80.0)
        self.assertEqual(centY, 80.0)
        self.assertAlmostEqual(rad, eff_radius, delta=0.1)

    def testGetCenterAndRWithTemplate(self):

        singleDonut, doubleDonut, eff_radius = self._createData(
            20, 40, 160, addNoise=True
        )

        # Test recovery with defaults
        centX, centY, rad = self.centroidConv.getCenterAndR(
            singleDonut, templateDonut=singleDonut
        )

        self.assertEqual(centX, 80.0)
        self.assertEqual(centY, 80.0)
        self.assertAlmostEqual(rad, eff_radius, delta=0.1)

    def testGetCenterAndRFromImgBinary(self):

        singleDonut, doubleDonut, eff_radius = self._createData(20, 40, 160)

        # Test recovery with defaults
        centX, centY, rad = self.centroidConv.getCenterAndRfromImgBinary(singleDonut)

        self.assertEqual(centX, 80.0)
        self.assertEqual(centY, 80.0)
        self.assertAlmostEqual(rad, eff_radius, delta=0.1)

    def testNDonutsAssertion(self):

        singleDonut, doubleDonut, eff_radius = self._createData(20, 40, 160)

        nDonutsAssertMsg = "nDonuts must be an integer >= 1"
        with self.assertRaises(AssertionError, msg=nDonutsAssertMsg):
            cX, cY, rad = self.centroidConv.getCenterAndRfromTemplateConv(
                singleDonut, nDonuts=0
            )

        with self.assertRaises(AssertionError, msg=nDonutsAssertMsg):
            cX, cY, rad = self.centroidConv.getCenterAndRfromTemplateConv(
                singleDonut, nDonuts=-1
            )

        with self.assertRaises(AssertionError, msg=nDonutsAssertMsg):
            cX, cY, rad = self.centroidConv.getCenterAndRfromTemplateConv(
                singleDonut, nDonuts=1.5
            )

    def testGetCenterAndRFromTemplateConv(self):

        singleDonut, doubleDonut, eff_radius = self._createData(20, 40, 160)

        # Test recovery of single donut
        singleCX, singleCY, rad = self.centroidConv.getCenterAndRfromTemplateConv(
            singleDonut
        )
        self.assertEqual(singleCX, [80.0])
        self.assertEqual(singleCY, [80.0])
        self.assertAlmostEqual(rad, eff_radius, delta=0.1)

        # Test recovery of two donuts at once
        doubleCX, doubleCY, rad = self.centroidConv.getCenterAndRfromTemplateConv(
            doubleDonut, templateImgBinary=singleDonut, nDonuts=2
        )
        self.assertCountEqual(doubleCX, [50.0, 100.0])
        self.assertEqual(doubleCY, [80.0, 80.0])
        self.assertAlmostEqual(rad, eff_radius, delta=0.1)


if __name__ == "__main__":

    unittest.main()
