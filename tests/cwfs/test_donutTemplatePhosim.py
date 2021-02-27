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
import os
import shutil
import numpy as np

from lsst.ts.wep.cwfs.DonutTemplatePhosim import DonutTemplatePhosim
from lsst.ts.wep.Utility import DefocalType, getModulePath


class TestTemplatePhosim(unittest.TestCase):
    """Test the DonutTemplatePhosim class."""

    def setUp(self):

        modulePath = getModulePath()
        testDataPath = os.path.join(
            modulePath, "tests", "testData", "testDonutTemplates"
        )
        # Specify location of test Phosim donut templates in testData
        extraTemplateName = "extra_template-R22_S11.txt"
        self.templateExtra = np.genfromtxt(
            os.path.join(testDataPath, extraTemplateName)
        )
        intraTemplateName = "intra_template-R22_S11.txt"
        self.templateIntra = np.genfromtxt(
            os.path.join(testDataPath, intraTemplateName)
        )
        self.templateMaker = DonutTemplatePhosim()

        # Specify template location for test
        self.defaultTemplatePath = os.path.join(
            modulePath, "policy", "cwfs", "donutTemplateData", "phosimTemplates"
        )

        # Copy test templates to phosim template folder. R99 is a fake
        # raft so it will not overwrite existing templates in the folder.
        shutil.copyfile(
            os.path.join(testDataPath, extraTemplateName),
            os.path.join(self.defaultTemplatePath, "extra_template-R99_S99.txt")
        )
        shutil.copyfile(
            os.path.join(testDataPath, intraTemplateName),
            os.path.join(self.defaultTemplatePath, "intra_template-R99_S99.txt")
        )

    def tearDown(self):

        # Only remove the files we added to the template folder
        os.remove(os.path.join(self.defaultTemplatePath, "extra_template-R99_S99.txt"))
        os.remove(os.path.join(self.defaultTemplatePath, "intra_template-R99_S99.txt"))

    def testMakeTemplateExtra(self):

        # Generate a test extrafocal template
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R99_S99", DefocalType.Extra, imageSize
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)
        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray, self.templateExtra[40:200, 40:200])

    def testMakeTemplateIntra(self):

        # Generate a test intrafocal template. Test odd imageSize.
        imageSize = 161
        templateArray = self.templateMaker.makeTemplate(
            "R99_S99", DefocalType.Intra, imageSize
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)
        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray, self.templateIntra[39:200, 39:200])

    def testLargerTemplate(self):

        # Request to return a template larger than the phosim template
        imageSize = 280
        templateArray = self.templateMaker.makeTemplate(
            "R99_S99", DefocalType.Extra, imageSize
        )

        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray[20:-20, 20:-20], self.templateExtra)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
