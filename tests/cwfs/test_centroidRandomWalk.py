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

from lsst.ts.wep.cwfs.CentroidRandomWalk import CentroidRandomWalk
from lsst.ts.wep.Utility import getModulePath


class TestCentroidRandomWalk(unittest.TestCase):
    """Test the CentroidRandomWalk class."""

    def setUp(self):

        self.centroid = CentroidRandomWalk()

    def testGetCenterAndR(self):

        imgDonut = self._prepareDonutImg(1000)

        realcx, realcy, realR = self.centroid.getCenterAndR(imgDonut)
        self.assertAlmostEqual(realcx, 59.7495, places=3)
        self.assertAlmostEqual(realcy, 59.3421, places=3)
        self.assertAlmostEqual(realR, 47.3616, places=3)

    def _prepareDonutImg(self, seed):

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
