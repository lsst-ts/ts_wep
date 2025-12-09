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

from lsst.ts.wep.donutImageCheck import DonutImageCheck
from lsst.ts.wep.utils import getModulePath


class TestDonutImageCheck(unittest.TestCase):
    """Test the DonutImageCheck class."""

    def setUp(self) -> None:
        self.donutImgCheck = DonutImageCheck()

    def testIsEffDonutWithEffImg(self) -> None:
        imgFile = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "testImages",
            "LSST_NE_SN25",
            "z11_0.25_intra.txt",
        )
        donutImg = np.loadtxt(imgFile)
        # This assumes this "txt" file is in the format
        # I[0,0]   I[0,1]
        # I[1,0]   I[1,1]
        donutImg = donutImg[::-1, :]

        # test that by default the returnEntro is False
        self.assertFalse(self.donutImgCheck.returnEntro)

        # change that to True
        self.donutImgCheck.returnEntro = True
        # first check that now two outputs are present
        self.assertTrue(len(self.donutImgCheck.isEffDonut(donutImg)) == 2)
        # then test that the values are what is expected
        effective, entro = self.donutImgCheck.isEffDonut(donutImg)

        self.assertTrue(effective)
        np.testing.assert_allclose(
            entro,
            0.027858272652433826,
        )

        # change back to the default
        self.donutImgCheck.returnEntro = False

        # test that now we only get the boolean as before
        self.assertTrue(self.donutImgCheck.isEffDonut(donutImg))

    def testIsEffDonutWithConstImg(self) -> None:
        zeroDonutImg = np.zeros((120, 120))
        self.assertFalse(self.donutImgCheck.isEffDonut(zeroDonutImg))

        onesDonutImg = np.ones((120, 120))
        self.assertFalse(self.donutImgCheck.isEffDonut(onesDonutImg))

    def testIsEffDonutWithRandImg(self) -> None:
        donutImg = np.random.rand(120, 120)
        self.assertFalse(self.donutImgCheck.isEffDonut(donutImg))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
