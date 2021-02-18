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

from lsst.ts.wep.CreatePhosimDonutTemplates import CreatePhosimDonutTemplates
from lsst.ts.wep.Utility import getModulePath, DefocalType


class TestCreatePhosimDonutTemplates(unittest.TestCase):
    """Test the CreatePhosimDonutTemplates class."""

    def setUp(self):

        self.createPhosimDonuts = CreatePhosimDonutTemplates()
        self.testDataDir = os.path.join(getModulePath(), "tests", "testData")
        self.tempDir = os.path.join(self.testDataDir, "tmp")
        os.mkdir(self.tempDir)

    def testCreateDetectorLists(self):

        testDetectors = "R22_S00 R22_S11"

        detStrPhosim, detStrFlats = self.createPhosimDonuts.createDetectorLists(
            testDetectors
        )

        self.assertEqual(detStrPhosim, "R22_S00|R22_S11|")
        self.assertEqual(detStrFlats, "R22_S00 R22_S11 ")

    def testCutOutTemplatesAndSave(self):

        testPhosimPath = os.path.join(
            self.testDataDir, "phosimOutput", "donutTemplates"
        )

        self.createPhosimDonuts.cutOutTemplatesAndSave(
            testPhosimPath,
            240,
            DefocalType.Extra,
            9006001,
            phosimTemplateDir=self.tempDir,
            phosimCentroidDir=os.path.join(self.testDataDir, "testDonutTemplates"),
        )

        newTemplate = np.genfromtxt(
            os.path.join(self.tempDir, "extra_template-R22_S11.txt")
        )
        trueTemplate = np.genfromtxt(
            os.path.join(
                self.testDataDir, "testDonutTemplates", "extra_template-R22_S11.txt"
            )
        )
        np.testing.assert_array_equal(newTemplate, trueTemplate)

    def tearDown(self):

        shutil.rmtree(self.tempDir)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
