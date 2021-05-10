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
import unittest
import tempfile

from lsst.ts.wep.Utility import (
    getModulePath,
    CamType,
    FilterType,
    runProgram,
    ImageType,
    BscDbType,
)
from lsst.ts.wep.ParamReader import ParamReader

from lsst.ts.wep.ctrlIntf.WEPCalculation import WEPCalculation
from lsst.ts.wep.ctrlIntf.AstWcsSol import AstWcsSol
from lsst.ts.wep.ctrlIntf.RawExpData import RawExpData


class TestWEPCalculation(unittest.TestCase):
    """Test the WEPCalculation class."""

    def setUp(self):

        self.modulePath = getModulePath()

        testDir = os.path.join(self.modulePath, "tests")
        self.testDataDir = os.path.join(testDir, "testData")

        self.dataDir = tempfile.TemporaryDirectory(dir=testDir)
        self.isrDir = tempfile.TemporaryDirectory(dir=self.dataDir.name)

        self.wepCalculation = WEPCalculation(
            AstWcsSol(), CamType.ComCam, self.isrDir.name
        )

    def tearDown(self):

        self.dataDir.cleanup()

    def testGetSettingFile(self):

        settingFile = self.wepCalculation.getSettingFile()
        self.assertTrue(isinstance(settingFile, ParamReader))

    def testGetImageType(self):

        self.assertEqual(self.wepCalculation._getImageType(), ImageType.Amp)

    def testGetBscDbType(self):

        self.assertEqual(
            self.wepCalculation._getBscDbType(), BscDbType.LocalDbForStarFile
        )

    def testGetIsrDir(self):

        isrDir = self.wepCalculation.getIsrDir()
        self.assertEqual(isrDir, self.isrDir.name)

    def testGetSkyFile(self):

        skyFile = self.wepCalculation.getSkyFile()
        self.assertEqual(skyFile, "")

    def testSetSkyFile(self):

        skyFile = "test.txt"
        self.wepCalculation.setSkyFile(skyFile)

        self.assertEqual(self.wepCalculation.getSkyFile(), skyFile)

    def testGetFilter(self):

        filterType = self.wepCalculation.getFilter()
        self.assertEqual(filterType, FilterType.REF)

    def testSetFilter(self):

        filterType = FilterType.R
        self.wepCalculation.setFilter(filterType)

        self.assertEqual(self.wepCalculation.getFilter(), filterType)

    def testGetBoresight(self):

        ra, dec = self.wepCalculation.getBoresight()

        self.assertEqual(ra, 0)
        self.assertEqual(dec, 0)

    def testSetBoresight(self):

        ra = 1.1
        dec = 2.2
        self.wepCalculation.setBoresight(ra, dec)

        self.assertEqual(self.wepCalculation.getBoresight(), (ra, dec))

    def testGetRotAng(self):

        rotAng = self.wepCalculation.getRotAng()
        self.assertEqual(rotAng, 0.0)

    def testSetRotAng(self):

        rotAng = 10.0
        self.wepCalculation.setRotAng(rotAng)

        self.assertEqual(self.wepCalculation.getRotAng(), rotAng)

    def testIngestCalibs(self):

        fakeFlatDir = tempfile.TemporaryDirectory(dir=self.dataDir.name)
        self._genCalibsAndIngest(fakeFlatDir.name)

        calibRegistryPath = os.path.join(self.isrDir.name, "calibRegistry.sqlite3")
        self.assertTrue(os.path.exists(calibRegistryPath))

        ingestFlatDir = os.path.join(self.isrDir.name, "flat")
        self.assertTrue(os.path.exists(ingestFlatDir))

    def _genCalibsAndIngest(self, flatCalibsDir):

        self._genFlatCalibs(flatCalibsDir)
        self.wepCalculation.ingestCalibs(flatCalibsDir)

    def _genFlatCalibs(self, flatCalibsDir):

        detector = "R22_S11 R22_S10"
        self._genFakeFlat(flatCalibsDir, detector)

    def _genFakeFlat(self, fakeFlatDir, detector):

        currWorkDir = os.getcwd()

        os.chdir(fakeFlatDir)

        command = "makeGainImages.py"
        argstring = "--detector_list %s" % detector
        runProgram(command, argstring=argstring)

        os.chdir(currWorkDir)

    def testCalculateWavefrontErrorsWithoutExtraRawExpData(self):

        self.assertRaises(
            ValueError, self.wepCalculation.calculateWavefrontErrors, RawExpData()
        )

    def testCalculateWavefrontErrorsWithMultiVisit(self):

        rawExpData = RawExpData()
        rawExpData.append(1, 0, "")
        rawExpData.append(2, 0, "")

        self.assertRaises(
            ValueError,
            self.wepCalculation.calculateWavefrontErrors,
            rawExpData,
            rawExpData,
        )

    def testCalculateWavefrontErrors(self):

        fakeFlatDir = tempfile.TemporaryDirectory(dir=self.dataDir.name)
        self._genCalibsAndIngest(fakeFlatDir.name)

        self._calculateWavefrontErrorsAndCheck()

    def _calculateWavefrontErrorsAndCheck(self):

        # Update to match rotation angle in headers
        self.wepCalculation.setRotAng(90.0)
        comcamDataDir = os.path.join(self.testDataDir, "phosimOutput", "realComCam")
        rawExpData, extraRawExpData = self._prepareRawExpData(comcamDataDir)

        with self.assertWarns(UserWarning):
            listOfWfErr = self.wepCalculation.calculateWavefrontErrors(
                rawExpData, extraRawExpData=extraRawExpData
            )

        self.assertTrue(len(listOfWfErr), 2)

        self.assertNotEqual(listOfWfErr[0].getSensorId(), listOfWfErr[1].getSensorId())

        for sensorWavefrontData in listOfWfErr:
            self._testSensorWavefrontData(sensorWavefrontData)

    def _prepareRawExpData(self, comcamDataDir):

        intraImgDir = os.path.join(comcamDataDir, "repackagedFiles", "intra")
        extraImgDir = os.path.join(comcamDataDir, "repackagedFiles", "extra")

        rawExpData = RawExpData()
        rawExpData.append(9006002, 0, intraImgDir)

        extraRawExpData = RawExpData()
        extraRawExpData.append(9006001, 0, extraImgDir)

        return rawExpData, extraRawExpData

    def _testSensorWavefrontData(self, sensorWavefrontData):

        sensorId = sensorWavefrontData.getSensorId()
        self.assertTrue(sensorId in (99, 100))

        listOfDonut = sensorWavefrontData.getListOfDonut()
        self.assertTrue(len(listOfDonut), 2)

        avgWfErr = sensorWavefrontData.getAnnularZernikePoly()
        self.assertGreater(avgWfErr.max(), 0.5)

    def testCalculateWavefrontErrorsOfEimg(self):

        settingFile = self.wepCalculation.getSettingFile()
        settingFile.updateSetting("imageType", "eimage")

        self._calculateWavefrontErrorsAndCheck()


if __name__ == "__main__":

    # Run the unit test
    unittest.main()
