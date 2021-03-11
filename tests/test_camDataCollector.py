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
import tempfile
import unittest

from lsst.ts.wep.CamDataCollector import CamDataCollector
from lsst.ts.wep.Utility import getModulePath, runProgram


class TestCamDataCollector(unittest.TestCase):
    """Test the CamIsrWrapper class."""

    def setUp(self):

        testDir = os.path.join(getModulePath(), "tests")
        self.dataDir = tempfile.TemporaryDirectory(dir=testDir)

        self.isrDir = tempfile.TemporaryDirectory(dir=self.dataDir.name)

        self.camDataCollector = CamDataCollector(self.isrDir.name)

    def tearDown(self):

        self.dataDir.cleanup()

    def testGenCamMapper(self):

        self._genMapper()

        mapperFilePath = os.path.join(self.isrDir.name, "_mapper")
        self.assertTrue(os.path.isfile(mapperFilePath))

        numOfLine = self._getNumOfLineInFile(mapperFilePath)
        self.assertEqual(numOfLine, 1)

    def _genMapper(self):

        self.camDataCollector.genLsstCamMapper()

    def _getNumOfLineInFile(self, filePath):

        with open(filePath, "r") as file:
            return sum(1 for line in file.readlines())

    def testIngestCalibs(self):

        # Make fake gain images
        fakeFlatDir = tempfile.TemporaryDirectory(dir=self.dataDir.name)

        detector = "R22_S10"
        self._genFakeFlat(fakeFlatDir.name, detector)

        # Generate the mapper
        self._genMapper()

        # Do the ingestion
        calibFiles = os.path.join(fakeFlatDir.name, "*")
        self.camDataCollector.ingestCalibs(calibFiles)

        # Check the ingested calibration products
        calibRegistryFilePath = os.path.join(self.isrDir.name, "calibRegistry.sqlite3")
        self.assertTrue(os.path.exists(calibRegistryFilePath))

        flatDir = os.path.join(self.isrDir.name, "flat")
        self.assertTrue(os.path.exists(flatDir))

    def _genFakeFlat(self, fakeFlatDir, detector):

        currWorkDir = self._getCurrWorkDir()

        self._changeWorkDir(fakeFlatDir)
        self._makeFakeFlat(detector)
        self._changeWorkDir(currWorkDir)

    def _getCurrWorkDir(self):

        return os.getcwd()

    def _changeWorkDir(self, dirPath):

        os.chdir(dirPath)

    def _makeFakeFlat(self, detector):

        command = "makeGainImages.py"
        argstring = "--detector_list %s" % detector
        runProgram(command, argstring=argstring)

    def testIngestImages(self):

        self._genMapper()

        imgFiles = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "phosimOutput",
            "realComCam",
            "repackagedFiles",
            "intra",
            "MC_H_20211231_006002_R22_S10.fits",
        )
        self.camDataCollector.ingestImages(imgFiles)

        # Check the ingested files
        self._checkIngestion("raw")

    def _checkIngestion(self, imgType):

        registryFilePath = os.path.join(self.isrDir.name, "registry.sqlite3")
        self.assertTrue(os.path.exists(registryFilePath))

        rawDir = os.path.join(self.isrDir.name, imgType)
        self.assertTrue(os.path.exists(rawDir))

    def testIngestEimages(self):

        self._genMapper()

        imgFiles = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "phosimOutput",
            "realComCam",
            "repackagedFiles",
            "intra",
            "MC_H_20211231_006002_R22_S10_e.fits.gz",
        )
        self.camDataCollector.ingestEimages(imgFiles)

        # Check the ingested files
        self._checkIngestion("eimage")


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
