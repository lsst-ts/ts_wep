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
import numpy as np
import tempfile
import unittest

from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.Utility import getModulePath


class TestParamReader(unittest.TestCase):
    """Test the ParamReaderYaml class."""

    def setUp(self):
        testDir = os.path.join(getModulePath(), "tests")
        self.configDir = os.path.join(testDir, "testData")
        self.fileName = "testConfigFile.yaml"

        filePath = os.path.join(self.configDir, self.fileName)
        self.paramReader = ParamReader(filePath=filePath)

        self.testTempDir = tempfile.TemporaryDirectory(dir=testDir)

    def tearDown(self):
        self.testTempDir.cleanup()

    def testGetSetting(self):
        znmax = self.paramReader.getSetting("znmax")
        self.assertEqual(znmax, 22)

    def testGetSettingWithWrongParam(self):
        self.assertRaises(ValueError, self.paramReader.getSetting, "wrongParam")

    def testGetFilePath(self):
        ansFilePath = os.path.join(self.configDir, self.fileName)
        self.assertEqual(self.paramReader.getFilePath(), ansFilePath)

    def testSetFilePath(self):
        fileName = "test.yaml"
        filePath = os.path.join(self.configDir, fileName)
        with self.assertWarns(UserWarning):
            self.paramReader.setFilePath(filePath)

        self.assertEqual(self.paramReader.getFilePath(), filePath)

    def testGetContent(self):
        content = self.paramReader.getContent()
        self.assertTrue(isinstance(content, dict))

    def testGetContentWithDefaultSetting(self):
        paramReader = ParamReader()

        content = paramReader.getContent()
        self.assertTrue(isinstance(content, dict))

    def testWriteMatToFile(self):
        self._writeMatToFile()

        numOfFile = self._getNumOfFileInFolder(self.testTempDir.name)
        self.assertEqual(numOfFile, 1)

    def _writeMatToFile(self):
        mat = np.random.rand(3, 4, 5)
        filePath = os.path.join(self.testTempDir.name, "temp.yaml")
        ParamReader.writeMatToFile(mat, filePath)

        return mat, filePath

    def _getNumOfFileInFolder(self, folder):
        return len(
            [
                name
                for name in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, name))
            ]
        )

    def testWriteMatToFileWithWrongFileFormat(self):
        wrongFilePath = os.path.join(self.testTempDir.name, "temp.txt")
        self.assertRaises(
            ValueError, ParamReader.writeMatToFile, np.ones(4), wrongFilePath
        )

    def testGetMatContent(self):
        mat, filePath = self._writeMatToFile()

        self.paramReader.setFilePath(filePath)
        matInYamlFile = self.paramReader.getMatContent()

        delta = np.sum(np.abs(matInYamlFile - mat))
        self.assertLess(delta, 1e-10)

    def testGetMatContentWithDefaultSetting(self):
        paramReader = ParamReader()
        matInYamlFile = paramReader.getMatContent()

        self.assertTrue(isinstance(matInYamlFile, np.ndarray))
        self.assertEqual(len(matInYamlFile), 0)

    def testUpdateSettingSeries(self):
        znmaxValue = 20
        zn3IdxValue = [1, 2, 3]
        settingSeries = {"znmax": znmaxValue, "zn3Idx": zn3IdxValue}
        self.paramReader.updateSettingSeries(settingSeries)

        self.assertEqual(self.paramReader.getSetting("znmax"), znmaxValue)
        self.assertEqual(self.paramReader.getSetting("zn3Idx"), zn3IdxValue)

    def testUpdateSetting(self):
        value = 10
        param = "znmax"
        self.paramReader.updateSetting(param, value)

        self.assertEqual(self.paramReader.getSetting(param), value)

    def testUpdateSettingWithWrongParam(self):
        self.assertRaises(ValueError, self.paramReader.updateSetting, "wrongParam", -1)

    def testSaveSettingWithFilePath(self):
        filePath = self._saveSettingFile()

        self.assertTrue(os.path.exists(filePath))
        self.assertEqual(self.paramReader.getFilePath(), filePath)

    def _saveSettingFile(self):
        filePath = os.path.join(self.testTempDir.name, "newConfigFile.yaml")
        self.paramReader.saveSetting(filePath=filePath)

        return filePath

    def testSaveSettingWithoutFilePath(self):
        filePath = self._saveSettingFile()
        paramReader = ParamReader(filePath=filePath)

        paramReader.saveSetting()
        self.assertEqual(paramReader.getFilePath(), filePath)

        # Check the values are saved actually
        self.assertEqual(paramReader.getSetting("znmax"), 22)

        keysInContent = paramReader.getContent().keys()
        self.assertTrue("dofIdx" in keysInContent)
        self.assertTrue("zn3Idx" in keysInContent)

        self.assertEqual(paramReader.getSetting("zn3Idx"), [1] * 19)

    def testGetAbsPathNotExist(self):
        self.assertRaises(
            ValueError, self.paramReader.getAbsPath, "testFile.txt", getModulePath()
        )

    def testGetAbsPath(self):
        filePath = "README.md"
        self.assertFalse(os.path.isabs(filePath))

        filePathAbs = ParamReader.getAbsPath(filePath, getModulePath())
        self.assertTrue(os.path.isabs(filePathAbs))

    def testNonexistentFile(self):
        with self.assertWarns(UserWarning):
            paramReader = ParamReader(filePath="thisFileDoesntExists")
        self.assertEqual(len(paramReader.getContent().keys()), 0)


if __name__ == "__main__":
    # Run the unit test
    unittest.main()
