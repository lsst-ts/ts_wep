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
from typing import Any

import numpy as np
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.estimation import WfAlgorithm


class TestWfAlgorithm(unittest.TestCase):
    """Test the WfAlgorithm base class."""

    def testMustSubclass(self) -> None:
        with self.assertRaises(TypeError) as err:
            WfAlgorithm()

        error_message = str(err.exception)
        # Try to future proof against type error message changes
        # by checking for the class name and the method names
        self.assertIn("Can't instantiate abstract class WfAlgorithm", error_message)
        self.assertIn("_estimateZk", error_message)
        self.assertIn("requiresPairs", error_message)

    def testHistDocstringRequired(self) -> None:
        with self.assertRaises(AttributeError) as err:

            class DummyWfAlg(WfAlgorithm):
                @property
                def requiresPairs(self) -> bool:
                    return False

                def _estimateZk(self, *args: Any, **kwargs: Any) -> np.ndarray:
                    return np.zeros(19)

        self.assertEqual(
            str(err.exception),
            "When subclassing WfAlgorithm you must write a docstring "
            + "for the history property. Please use this to describe "
            + "the contents of the history dictionary.",
        )

    def testValidateInputs(self) -> None:
        # Create a dummy WfAlgorithm class
        class DummyWfAlg(WfAlgorithm):
            @property
            def requiresPairs(self) -> bool:
                return False

            @property
            def history(self) -> dict:
                """Docstring"""
                return super().history

            def _estimateZk(self, *args: Any, **kwargs: Any) -> np.ndarray:
                return np.zeros(19)

        wfAlg = DummyWfAlg()

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
        goodSettings = {
            "I1": intra,
            "I2": extra,
            "nollIndices": np.arange(4, 23),
            "instrument": Instrument(),
            "startWithIntrinsic": True,
            "returnWfDev": False,
            "units": "m",
            "saveHistory": True,
        }
        wfAlg._validateInputs(**goodSettings)

        # Test that I2 can be None
        testSettings = goodSettings.copy()
        testSettings["I2"] = None
        wfAlg._validateInputs(**testSettings)

        # Test I1 not an Image
        testSettings = goodSettings.copy()
        testSettings["I1"] = None
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)

        # Test bad I1 shape
        rect1 = intra.copy()
        rect1._image = np.zeros((10, 180))
        testSettings = goodSettings.copy()
        testSettings["I1"] = rect1
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        rect2 = intra.copy()
        rect2._image = np.zeros((180, 180, 180))
        testSettings = goodSettings.copy()
        testSettings["I1"] = rect2
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        # Test I2 not an image
        testSettings = goodSettings.copy()
        testSettings["I2"] = "fake"
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)

        # Test bad I2 shape
        testSettings = goodSettings.copy()
        testSettings["I2"] = rect1
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)
        testSettings["I2"] = rect2
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        # Test I1 and I2 same side of focus
        testSettings = goodSettings.copy()
        testSettings["I2"] = intra
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)
        testSettings = goodSettings.copy()
        testSettings["I1"] = extra
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        # Test bad Noll indices
        testSettings = goodSettings.copy()
        testSettings["nollIndices"] = np.array([4, 5])
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        # Test bad instrument
        testSettings = goodSettings.copy()
        testSettings["instrument"] = "fake"
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)

        # Test bad startWithIntrinsic
        testSettings = goodSettings.copy()
        testSettings["startWithIntrinsic"] = "fake"
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)

        # Test bad returnWfDev
        testSettings = goodSettings.copy()
        testSettings["returnWfDev"] = "fake"
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)

        # Test bad units
        testSettings = goodSettings.copy()
        testSettings["units"] = -1
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)
        testSettings["units"] = "ergs"
        with self.assertRaises(ValueError):
            wfAlg._validateInputs(**testSettings)

        # Test bad saveHistory
        testSettings = goodSettings.copy()
        testSettings["saveHistory"] = "fake"
        with self.assertRaises(TypeError):
            wfAlg._validateInputs(**testSettings)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
