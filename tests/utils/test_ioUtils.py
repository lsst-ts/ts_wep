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

from lsst.ts.wep.utils import (
    configClass,
    getAmpImagesFromDir,
    getConfigDir,
    getModulePath,
    getObsLsstCmdTaskConfigDir,
    mergeConfigWithFile,
    readConfigYaml,
    resolveRelativeConfigPath,
)


class TestIoUtils(unittest.TestCase):
    """Test the IO utility functions."""

    def testGetConfigDir(self):
        ansConfigDir = os.path.join(getModulePath(), "policy")
        self.assertEqual(getConfigDir(), ansConfigDir)

    def testResolveRelativeConfigPath(self):
        testPath = "test/path.yaml"
        resolvedPath = resolveRelativeConfigPath(testPath)

        # Test that it adds the correct stem and keeps the rest of the path
        splitPath = resolvedPath.split("/policy/")
        self.assertEqual(splitPath[0], getModulePath())
        self.assertEqual(splitPath[1], testPath)

        # Check that adding "policy:" to the front returns the same result
        self.assertEqual(resolvedPath, resolveRelativeConfigPath(f"policy:{testPath}"))

    def testMergeConfigWithFile(self):
        # Config file used for tests
        configFile = f"{getModulePath()}/tests/testData/testConfigFile.yaml"

        # Load the contents into a dictionary
        config = readConfigYaml(configFile)

        # Test loading without overriding defaults
        mergedConfig = mergeConfigWithFile(configFile, **{key: None for key in config})
        self.assertDictEqual(mergedConfig, config)

        # Test loading while overriding a default
        keys = list(config)
        override = {key: None for key in keys[:-1]}
        override[keys[-1]] = "override"
        mergedConfig = mergeConfigWithFile(configFile, **override)
        for key in keys[:-1]:
            self.assertEqual(config[key], mergedConfig[key])
        self.assertNotEqual(config[keys[-1]], mergedConfig[keys[-1]])
        self.assertEqual(mergedConfig[keys[-1]], "override")

        # Test loading with an extra key
        mergedConfig = mergeConfigWithFile(configFile, **config, extraKey=123)
        self.assertEqual(mergedConfig["extraKey"], 123)

        # Test load fails when there is unrecognized key in config file
        with self.assertRaises(KeyError):
            mergeConfigWithFile(configFile, **{key: None for key in keys[:-1]})

    def testConfigClass(self):
        # Should fail if second argument is not a class
        with self.assertRaises(TypeError):
            configClass(1, 1)

        # If first argument is string, it should pass to configFile argument
        config = configClass("test", dict)
        self.assertDictEqual(config, {"configFile": "test"})

        # If first argument is a dictionary, it should pass keyword arguments
        config = configClass({"test": "config", "with": "dict"}, dict)
        self.assertDictEqual(config, {"test": "config", "with": "dict"})

        # If first argument is None, should call with defaults
        config = configClass(None, dict)
        self.assertDictEqual(config, dict())

        # If first argument is none of these, should raise error
        with self.assertRaises(TypeError):
            configClass(123, dict)

    def testGetObsLsstCmdTaskConfigDir(self):
        obsLsstCmdTaskConfirDir = getObsLsstCmdTaskConfigDir()
        configNormPath = os.path.normpath(obsLsstCmdTaskConfirDir)
        configNormPathList = configNormPath.split(os.sep)

        self.assertEqual(configNormPathList[-1], "config")
        self.assertTrue(("obs_lsst" in configNormPathList))

    def testGetAmpImagesFromDir(self):
        # path to repackaged phosim files
        # with amplifier images and e-images
        defocalImgDir = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "phosimOutput",
            "realComCam",
            "repackagedFiles",
            "extra",
        )
        # test that there are e-images in that dir
        filesInDir = os.listdir(defocalImgDir)
        self.assertTrue("MC_H_20211231_006001_R22_S11_e.fits.gz" in filesInDir)
        self.assertTrue("MC_H_20211231_006001_R22_S10_e.fits.gz" in filesInDir)

        # get names of amp files
        ampFiles = getAmpImagesFromDir(defocalImgDir)

        # assert the returned content
        self.assertIsInstance(ampFiles, list)

        # assert that amp images are on the returned list
        self.assertTrue("MC_H_20211231_006001_R22_S10.fits" in ampFiles)
        self.assertTrue("MC_H_20211231_006001_R22_S11.fits" in ampFiles)

        # assert that no other files are there
        # by checking that the length of list corresponds to
        # two files tested above
        self.assertEqual(len(ampFiles), 2)
