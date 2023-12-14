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
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.estimation import WfAlgorithm


class TestWfAlgorithm(unittest.TestCase):
    """Test the WfAlgorithm base class."""

    def testMustSubclass(self):
        with self.assertRaises(TypeError) as err:
            WfAlgorithm()

        self.assertEquals(
            str(err.exception),
            "Can't instantiate abstract class WfAlgorithm with "
            + "abstract method estimateZk",
        )

    def testValidateInputs(self):
        # Create some dummy inputs
        intra = Image(
            image=np.zeros((180, 180)),
            fieldAngle=(0, 0),
            defocalType="intra",
        )
        extra = Image(
            image=np.zeros((180, 180)),
            fieldAngle=(0, 0),
            defocalType="extra",
        )

        # Test good inputs
        WfAlgorithm._validateInputs(intra, None, 28, Instrument())
        WfAlgorithm._validateInputs(intra, extra, 28, Instrument())

        # Test I1 not an Image
        with self.assertRaises(TypeError):
            WfAlgorithm._validateInputs("fake", None, 28, Instrument())

        # Test bad I1 shape
        rect1 = intra.copy()
        rect1._image = np.zeros((10, 180))
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(rect1, None, 28, Instrument())

        rect2 = intra.copy()
        rect2._image = np.zeros((180, 180, 180))
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(rect2, None, 28, Instrument())

        # Test I2 not an image
        with self.assertRaises(TypeError):
            WfAlgorithm._validateInputs(intra, "fake", 28, Instrument())

        # Test bad I2 shape
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(intra, rect1, 28, Instrument())
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(intra, rect2, 28, Instrument())

        # Test I1 and I2 same side of focus
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(intra, intra, 28, Instrument())
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(extra, extra, 28, Instrument())

        # Test bad jmax
        with self.assertRaises(TypeError):
            WfAlgorithm._validateInputs(intra, extra, "fake", Instrument())
        with self.assertRaises(ValueError):
            WfAlgorithm._validateInputs(intra, extra, 3, Instrument())

        # Test bad instrument
        with self.assertRaises(TypeError):
            WfAlgorithm._validateInputs(intra, extra, 28, "fake")


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
