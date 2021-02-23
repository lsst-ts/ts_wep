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
        self.testDataPath = os.path.join(
            modulePath, "tests", "testData", "testDonutTemplates"
        )
        extraTemplateName = "extra_template-R22_S11.txt"
        self.templateExtra = np.genfromtxt(
            os.path.join(self.testDataPath, extraTemplateName)
        )
        intraTemplateName = "intra_template-R22_S11.txt"
        self.templateIntra = np.genfromtxt(
            os.path.join(self.testDataPath, intraTemplateName)
        )
        self.templateMaker = DonutTemplatePhosim()

        # Create default location for test
        self.defaultTemplatePath = os.path.join(
            modulePath, "policy", "cwfs", "donutTemplateData", "phosimTemplates"
        )

        # Keep track of whether directory existed before for tearDown
        if os.path.exists(self.defaultTemplatePath):
            self.defaultPathExists = True
        else:
            os.mkdir(self.defaultTemplatePath)
            self.defaultPathExists = False

        # Copy test templates to default path with non-existent raft name
        shutil.copyfile(
            os.path.join(self.testDataPath, extraTemplateName),
            os.path.join(self.defaultTemplatePath, "extra_template-R99_S99.txt"),
        )

    def tearDown(self):

        # If the folder already existed only remove the files we added
        if self.defaultPathExists:
            os.remove(
                os.path.join(self.defaultTemplatePath, "extra_template-R99_S99.txt")
            )
        else:
            shutil.rmtree(self.defaultTemplatePath)

    def testMakeTemplateExtra(self):

        # Generate a test template on the center chip
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, imageSize, phosimTemplateDir=self.testDataPath
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)
        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray, self.templateExtra[40:200, 40:200])

    def testMakeTemplateIntra(self):

        # Generate a test template on the center chip. Test odd imageSize.
        imageSize = 161
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Intra, imageSize, phosimTemplateDir=self.testDataPath
        )

        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)
        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray, self.templateIntra[39:200, 39:200])

    def testDefaultFilePath(self):

        templateArray = self.templateMaker.makeTemplate(
            "R99_S99", DefocalType.Extra, 240
        )

        np.testing.assert_array_equal(templateArray, self.templateExtra)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
