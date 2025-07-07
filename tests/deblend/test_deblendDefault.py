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
from lsst.ts.wep.deblend.deblendDefault import DeblendDefault
from lsst.ts.wep.utils import getModulePath


class TestDeblendDefault(unittest.TestCase):
    """Test the DeblendDefault class."""

    def setUp(self) -> None:
        self.deblend = DeblendDefault()

    def testGenerateMultiDonutWithWrongInputs(self) -> None:
        img = np.random.rand(4, 4)
        self.assertRaises(
            ValueError, self.deblend.generateMultiDonut, img, -1.3, 0.3, 45.0
        )
        self.assertRaises(
            ValueError, self.deblend.generateMultiDonut, img, 0, 0.3, 45.0
        )

        self.assertRaises(
            ValueError, self.deblend.generateMultiDonut, img, 1.3, -0.3, 45.0
        )
        self.assertRaises(
            ValueError, self.deblend.generateMultiDonut, img, 1.3, 1.3, 45.0
        )

    def testGenerateMultiDonut(self) -> None:
        template = self._getTemplate()
        (
            image,
            imageMain,
            imageNeighbor,
            neighborX,
            neighborY,
        ) = self.deblend.generateMultiDonut(template, 1.3, 0.3, 45.0)

        self.assertEqual(np.rint(np.sum(image)), 7534)
        self.assertAlmostEqual(neighborX, 130.58742823, places=7)
        self.assertAlmostEqual(neighborY, 130.58742823, places=7)

    def _getTemplate(self) -> np.ndarray:
        imageFilePath = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "testImages",
            "LSST_NE_SN25",
            "z11_0.25_intra.txt",
        )
        return np.loadtxt(imageFilePath)

    def testDeblendDonut(self) -> None:
        self.assertRaises(NotImplementedError, self.deblend.deblendDonut, [], [])


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
