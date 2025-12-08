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
import unittest

import numpy as np

from lsst.ts.wep.centroid import CentroidOtsu
from lsst.ts.wep.utils import getModulePath


class TestCentroidOtsu(unittest.TestCase):
    """Test the CentroidOtsu class."""

    def setUp(self) -> None:
        self.centroid = CentroidOtsu()

    def testGetCenterAndR(self) -> None:
        imgDonut = self._prepareDonutImg(1000)

        realcx, realcy, realR = self.centroid.getCenterAndR(imgDonut)
        self.assertAlmostEqual(realcx, 59.7526, places=3)
        self.assertAlmostEqual(realcy, 59.3366, places=3)
        self.assertAlmostEqual(realR, 47.6698, places=3)

    def _prepareDonutImg(self, seed: int) -> np.ndarray:
        # Read the image file
        imgFile = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "testImages",
            "LSST_NE_SN25",
            "z11_0.25_intra.txt",
        )
        imgDonut = np.loadtxt(imgFile)
        # This assumes this "txt" file is in the format
        # I[0,0]   I[0,1]
        # I[1,0]   I[1,1]
        imgDonut = imgDonut[::-1, :]

        # Add the noise to simulate the amplifier image
        np.random.seed(seed=seed)
        d0, d1 = imgDonut.shape
        noise = np.random.rand(d0, d1) * 10

        return imgDonut + noise


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
