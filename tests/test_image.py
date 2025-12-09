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

from lsst.ts.wep.image import Image
from lsst.ts.wep.utils import BandLabel


class TestImage(unittest.TestCase):
    """Test the Image class."""

    @staticmethod
    def _get_good_inputs() -> dict:
        return {
            "image": np.zeros((160, 160)),
            "fieldAngle": (1, 2),
            "defocalType": "intra",
        }

    def testBadImage(self) -> None:
        with self.assertRaises(TypeError):
            Image(None, (0, 0), "intra")
        with self.assertRaises(ValueError):
            Image(np.zeros((1, 1, 1)), (0, 0), "intra")
        with self.assertRaises(ValueError):
            Image(np.zeros((2, 4)), (0, 0), "intra")

    def testBadFieldAngle(self) -> None:
        with self.assertRaises(ValueError):
            Image(np.zeros((160, 160)), (1, 2, 3), "intra")

    def testBadDefocalType(self) -> None:
        with self.assertRaises(TypeError):
            Image(np.zeros((160, 160)), (0, 0), 1)

    def testBadBandLabel(self) -> None:
        with self.assertRaises(TypeError):
            Image(np.zeros((160, 160)), (0, 0), "intra", bandLabel=1)

    def testBandLabelStringNotInEnum(self) -> None:
        image = Image(np.zeros((160, 160)), (0, 0), "intra", bandLabel="NOT_AN_ENUMERATION")
        assert image.bandLabel == BandLabel.REF

    def testBadPlaneType(self) -> None:
        with self.assertRaises(TypeError):
            Image(np.zeros((160, 160)), (0, 0), "intra", planeType=1)

    def testBadBlendOffsets(self) -> None:
        with self.assertRaises(ValueError):
            Image(np.zeros((160, 160)), (0, 0), "intra", blendOffsets=[1])
        with self.assertRaises(ValueError):
            Image(np.zeros((160, 160)), (0, 0), "intra", blendOffsets=[[1], [1], [1]])

    def testBadMask(self) -> None:
        with self.assertRaises(TypeError):
            Image(np.zeros((160, 160)), (0, 0), "intra").mask = 1
        with self.assertRaises(ValueError):
            Image(np.zeros((160, 160)), (0, 0), "intra").mask = np.zeros((1, 1))

    def testCopy(self) -> None:
        rng = np.random.default_rng(0)
        image1 = Image(rng.normal(size=(160, 160)), (1, 2), "extra")
        image2 = image1.copy()

        # Make sure they contain the same image
        self.assertTrue(np.allclose(image1.image, image2.image))

        # Now change image 2 and make sure image 1 doesn't change with it
        image2.image += 2
        self.assertTrue(np.all(~np.isclose(image1.image, image2.image)))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
