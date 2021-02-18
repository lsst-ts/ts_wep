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
import numpy as np

from lsst.ts.wep.cwfs.DonutTemplatePhosim import DonutTemplatePhosim
from lsst.ts.wep.Utility import DefocalType, getModulePath


class TestTemplateModel(unittest.TestCase):
    """Test the TemplateModel class."""

    def setUp(self):

        modulePath = getModulePath()
        self.testDataPath = os.path.join(
            modulePath, "tests", "testData", "testDonutTemplates"
        )
        self.templateExtra = np.genfromtxt(
            os.path.join(self.testDataPath, "extra_template-R22_S11.txt")
        )
        self.templateIntra = np.genfromtxt(
            os.path.join(self.testDataPath, "intra_template-R22_S11.txt")
        )
        self.templateMaker = DonutTemplatePhosim()

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
        np.testing.assert_array_equal(templateArray, self.templateExtra[40:-40, 40:-40])

    def testMakeTemplateIntra(self):

        # Generate a test template on the center chip
        imageSize = 160
        templateArray = self.templateMaker.makeTemplate(
            "R22_S11", DefocalType.Intra, imageSize, phosimTemplateDir=self.testDataPath
        )

        print(np.max(templateArray))
        self.assertTrue(isinstance(templateArray, np.ndarray))
        self.assertEqual(templateArray.dtype, int)
        self.assertEqual(np.max(templateArray), 1)
        np.testing.assert_array_equal(np.shape(templateArray), (imageSize, imageSize))
        np.testing.assert_array_equal(templateArray, self.templateIntra[40:-40, 40:-40])


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
