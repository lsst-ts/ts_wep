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

from lsst.ts.wep.deblend.deblendAdapt import DeblendAdapt
from lsst.ts.wep.utils import getModulePath


class TestDeblendAdapt(unittest.TestCase):
    """Test the DeblendAdapt class."""

    def setUp(self) -> None:
        self.deblend = DeblendAdapt()

    def testDeblendDonutMoreThanOneStarNgr(self) -> None:
        iniGuessXY = [(1, 2), (3, 4)]
        self.assertRaises(ValueError, self.deblend.deblendDonut, [], iniGuessXY)

    def testDeblendDonut(self) -> None:
        template, imgToDeblend, iniGuessXY = self._genBlendedImg()
        imgDeblend, realcx, realcy = self.deblend.deblendDonut(imgToDeblend, iniGuessXY)

        difference = np.sum(np.abs(np.sum(template) - np.sum(imgDeblend)))
        self.assertLess(difference, 20)

        self.assertEqual(np.rint(realcx), 96)
        self.assertEqual(np.rint(realcy), 93)

    def _genBlendedImg(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
        imageFilePath = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "testImages",
            "LSST_NE_SN25",
            "z11_0.25_intra.txt",
        )
        template = np.loadtxt(imageFilePath)

        (
            image,
            imageMain,
            imageNeighbor,
            neighborX,
            neighborY,
        ) = self.deblend.generateMultiDonut(template, 1.3, 0.1, 45.0)

        return template, image, [(neighborX, neighborY)]


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
