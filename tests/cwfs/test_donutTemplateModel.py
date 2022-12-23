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

from lsst.ts.wep.cwfs.DonutTemplateModel import DonutTemplateModel
from lsst.ts.wep.Utility import DefocalType, CamType


class TestTemplateModel(unittest.TestCase):
    """Test the TemplateModel class."""

    def setUp(self):

        self.templateMaker = DonutTemplateModel()

        # Set LSST instParams
        self.lsstInstParams = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": 1.0,
            "pixelSize": 10.0e-6,
        }

        # Set AuxTel instParams
        self.auxTelInstParams = {
            "obscuration": 0.3525,
            "focalLength": 21.6,
            "apertureDiameter": 1.2,
            "offset": 32.8,
            "pixelSize": 10.0e-6,
        }

    def testMakeTemplateWithDict(self):

        # Generate a test template on the center chip and specify instParams
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize, instParams=self.lsstInstParams
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)

        # Center of donut should have hole in it
        self.assertEqual(templateArray[int(imageSize / 2), int(imageSize / 2)], 0)

        # Donut at center of focal plane should be symmetrical
        np.testing.assert_array_equal(
            templateArray[: int(imageSize / 2)],
            templateArray[-1 : -1 * (int(imageSize / 2) + 1) : -1],
        )

    def testMakeTemplateWithFile(self):

        # Generate a test template on the center chip w/o specifying instParams
        # This should load default instParams from file.
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize, camType=CamType.LsstFamCam
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)

        # Center of donut should have hole in it
        self.assertEqual(templateArray[int(imageSize / 2), int(imageSize / 2)], 0)

        # Donut at center of focal plane should be symmetrical
        np.testing.assert_array_equal(
            templateArray[: int(imageSize / 2)],
            templateArray[-1 : -1 * (int(imageSize / 2) + 1) : -1],
        )

    def testMakeTemplateCheckCamTypes(self):

        # Generate a test template on the center chip
        imageSize = 160
        for camType in [CamType.LsstCam, CamType.LsstFamCam, CamType.ComCam]:
            print(camType)
            templateArray = self.templateMaker.makeTemplate(
                "R22_S11",
                DefocalType.Extra,
                imageSize,
                camType=camType,
                instParams=self.lsstInstParams,
            )
            self.assertTrue(isinstance(templateArray, np.ndarray))

        for camType in [CamType.AuxTel]:
            templateArray = self.templateMaker.makeTemplate(
                "R22_S11",
                DefocalType.Extra,
                imageSize,
                camType=camType,
                opticalModel="onAxis",
                instParams=self.lsstInstParams,
            )
            self.assertTrue(isinstance(templateArray, np.ndarray))

        with self.assertRaises(ValueError) as context:
            errCamType = 99999
            self.templateMaker.makeTemplate(
                "R22_S11",
                DefocalType.Extra,
                imageSize,
                camType=errCamType,
                instParams=self.lsstInstParams,
            )
        self.assertEqual(
            f"Camera type ({errCamType}) is not supported.", str(context.exception)
        )

    def testTemplateSize(self):

        # Generate a test template on the center chip
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize, instParams=self.lsstInstParams
        )

        smallTemplate = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize - 20, instParams=self.lsstInstParams
        )

        self.assertEqual(np.shape(templateArray), (imageSize, imageSize))
        self.assertEqual(np.shape(smallTemplate), (imageSize - 20, imageSize - 20))
        np.testing.assert_array_equal(templateArray[10:-10, 10:-10], smallTemplate)

    def testMakeTemplateAuxTelOnlyWorksOnAxis(self):

        # Generate a test template for auxTel
        imageSize = 200
        pixelScale = 0.0956949999339899

        opticalModel = "offAxis"
        with self.assertRaises(ValueError) as context:
            self.templateMaker.makeTemplate(
                "RXX_S00",
                DefocalType.Extra,
                imageSize,
                CamType.AuxTel,
                opticalModel,
                pixelScale,
                self.auxTelInstParams,
            )
        self.assertEqual(
            str(
                f"Optical Model {opticalModel} not supported with AuxTel. "
                + "Must use 'onAxis'."
            ),
            str(context.exception),
        )

        opticalModel = "paraxial"
        with self.assertRaises(ValueError) as context:
            self.templateMaker.makeTemplate(
                "RXX_S00",
                DefocalType.Extra,
                imageSize,
                CamType.AuxTel,
                opticalModel,
                pixelScale,
                self.auxTelInstParams,
            )
        self.assertEqual(
            str(
                f"Optical Model {opticalModel} not supported with AuxTel. "
                + "Must use 'onAxis'."
            ),
            str(context.exception),
        )

    def testMakeTemplateAuxTel(self):

        # Generate a test template for auxTel
        imageSize = 200
        pixelScale = 0.0956949999339899
        opticalModel = "onAxis"

        templateArray = self.templateMaker.makeTemplate(
            "RXX_S00",
            DefocalType.Extra,
            imageSize,
            CamType.AuxTel,
            opticalModel,
            pixelScale,
            self.auxTelInstParams,
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)

        # Center of donut should have hole in it
        self.assertEqual(templateArray[int(imageSize / 2), int(imageSize / 2)], 0)

        # Donut at center of focal plane should be symmetrical
        np.testing.assert_array_equal(
            templateArray[: int(imageSize / 2)],
            templateArray[-1 : -1 * (int(imageSize / 2) + 1) : -1],
        )
        # Test that the donut size is correct - same donut
        # for LsstCam would be smaller

        # for auxTel the donut stretches within 20 px from
        # edge of a 200 px template
        self.assertEqual(np.max(templateArray[0:20, :]), 1)

        # but for LsstCam it would not given that image size,
        # as it is smaller
        templateArrayLsst = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize, instParams=self.lsstInstParams
        )
        self.assertEqual(np.max(templateArrayLsst[0:20, :]), 0)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
