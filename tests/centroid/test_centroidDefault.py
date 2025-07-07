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
from lsst.ts.wep.centroid import CentroidDefault
from scipy.ndimage import generate_binary_structure, iterate_structure


class TestCentroidDefault(unittest.TestCase):
    """Test the CentroidDefault class."""

    def setUp(self) -> None:
        self.centroid = CentroidDefault()

    def testGetImgBinary(self) -> None:
        self.assertRaises(
            NotImplementedError, self.centroid.getImgBinary, np.zeros((2, 2))
        )

    def testGetCenterAndRfromImgBinary(self) -> None:
        structOri = generate_binary_structure(2, 1).astype(int)

        iterations = 7
        donut = iterate_structure(structOri, iterations)
        dY, dX = donut.shape

        cornerX = 10
        cornerY = 20
        imgBinary = np.zeros((120, 120), dtype=int)
        imgBinary[cornerY : cornerY + dY, cornerX : cornerX + dX] = donut

        x, y, r = self.centroid.getCenterAndRfromImgBinary(imgBinary)
        self.assertEqual(x, cornerX + iterations)
        self.assertEqual(y, cornerY + iterations)
        self.assertAlmostEqual(r, 5.9974, places=3)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
