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
import tempfile
import numpy as np

from lsst.ts.wep.CreatePhosimDonutTemplates import CreatePhosimDonutTemplates
from lsst.ts.wep.Utility import getConfigDir, getModulePath, DefocalType


class TestCreatePhosimDonutTemplates(unittest.TestCase):
    """Test the CreatePhosimDonutTemplates class."""

    def setUp(self):

        self.createPhosimDonuts = CreatePhosimDonutTemplates()
        self.testDataDir = os.path.join(getModulePath(), "tests", "testData")
        # Location where phosim input files exist.
        # Also where temp work directory is created.
        self.templateDataDir = os.path.join(getConfigDir(), "cwfs", "donutTemplateData")
        # Location where templates are created
        self.templateDir = os.path.join(self.templateDataDir, "phosimTemplates")
        # Temporary work directory
        self.tempTestDirectory = tempfile.TemporaryDirectory()
        self.tempWorkDir = self.tempTestDirectory.name
        self.createPhosimDonuts.setTempWorkPaths(self.tempWorkDir)

    def tearDown(self):

        # Remove test template from template directory
        try:
            os.remove(os.path.join(self.templateDir, "extra_template-R99_S11.txt"))
        except FileNotFoundError:
            pass

        # Remove work directory in case of clean up method test failure
        self.tempTestDirectory.cleanup()

    def _copyPhosimFiles(self):

        shutil.copy(
            os.path.join(
                self.testDataDir,
                "repackagedFiles",
                "lsst_a_9005000_f1_R22_S10_E000.fits",
            ),
            os.path.join(
                self.tempWorkDir, "raw", "lsst_a_9006001_f1_R22_S10_E000.fits"
            ),
        )

        shutil.copy(
            os.path.join(
                self.testDataDir,
                "testDonutTemplates",
                "centroid_lsst_e_9006001_f1_R99_S11_E000.txt",
            ),
            os.path.join(self.tempWorkDir, "phosimOutput", "extra"),
        )

    def testCreateAndCleanUpWorkDirectories(self):

        # Test clean up of work directories
        self.createPhosimDonuts.cleanUpWorkDirs()
        self.assertFalse(os.path.exists(self.tempWorkDir))

        # Test creation of work directories
        self.createPhosimDonuts.createWorkDirectories()
        self.assertTrue(os.path.exists(self.tempWorkDir))
        self.assertTrue(
            os.path.exists(os.path.join(self.tempWorkDir, "phosimOutput", "extra"))
        )

    def testCreateDetectorLists(self):

        testDetectors = "R22_S00 R22_S11"

        detStrPhosim, detStrFlats = self.createPhosimDonuts.createDetectorLists(
            testDetectors
        )

        self.assertEqual(detStrPhosim, "R22_S00|R22_S11")
        self.assertEqual(detStrFlats, "R22_S00 R22_S11")

    def testIngestImages(self):

        # Populate the raw phosim output directory
        self.createPhosimDonuts.createWorkDirectories()
        self._copyPhosimFiles()
        # Run the ingest
        self.createPhosimDonuts.ingestImages()

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.tempWorkDir,
                    "input",
                    "raw",
                    "9005000",
                    "R22",
                    "09005000-R22-S10-det093.fits",
                )
            )
        )

    def testMakeFlatsAndIngest(self):

        self.createPhosimDonuts.createWorkDirectories()
        # Populate the repo first
        self._copyPhosimFiles()
        self.createPhosimDonuts.ingestImages()
        # Make Flats and Ingest
        self.createPhosimDonuts.makeFlats("R22_S10")
        self.createPhosimDonuts.ingestCalibs()

        self.assertTrue(
            os.path.exists(os.path.join(self.tempWorkDir, "input", "flat", "g"))
        )

    def testRunISR(self):

        self.createPhosimDonuts.createWorkDirectories()
        # Populate the repo
        self._copyPhosimFiles()
        self.createPhosimDonuts.ingestImages()
        # Add flats
        self.createPhosimDonuts.makeFlats("R22_S10")
        self.createPhosimDonuts.ingestCalibs()

        # Run the ISR
        self.createPhosimDonuts.runISR()
        isrPath = os.path.join(
            "input", "rerun", "run1", "postISRCCD", "09005000-g", "R22"
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.tempWorkDir,
                    isrPath,
                    "postISRCCD_09005000-g-R22-S10-det093.fits",
                )
            )
        )

    def testCutOutTemplatesAndSave(self):

        testPhosimPath = os.path.join(
            self.testDataDir, "phosimOutput", "donutTemplates"
        )
        # Move centroid file into place
        self.createPhosimDonuts.createWorkDirectories()
        self._copyPhosimFiles()

        self.createPhosimDonuts.cutOutTemplatesAndSave(
            testPhosimPath, 240, DefocalType.Extra, 9006001,
        )

        newTemplate = np.genfromtxt(
            os.path.join(self.templateDir, "extra_template-R99_S11.txt")
        )
        trueTemplate = np.genfromtxt(
            os.path.join(
                self.testDataDir, "testDonutTemplates", "extra_template-R22_S11.txt"
            )
        )
        np.testing.assert_array_equal(newTemplate, trueTemplate)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
