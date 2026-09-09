# This file is part of ts_wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import hashlib
import os
import tempfile
import unittest
from typing import Union

from lsst.ts.wep.utils import (
    computeSha256,
    configClass,
    getConfigDir,
    getModulePath,
    getObsLsstCmdTaskConfigDir,
    mergeConfigWithFile,
    readConfigYaml,
    resolveRelativeConfigPath,
    verifyModelChecksum,
)


class TestIoUtils(unittest.TestCase):
    """Test the IO utility functions."""

    def testGetConfigDir(self) -> None:
        ansConfigDir = os.path.join(getModulePath(), "policy")
        self.assertEqual(getConfigDir(), ansConfigDir)

    def testResolveRelativeConfigPath(self) -> None:
        testPath = "test/path.yaml"
        resolvedPath = resolveRelativeConfigPath(testPath)

        # Test that it adds the correct stem and keeps the rest of the path
        splitPath = resolvedPath.split("/policy/")
        self.assertEqual(splitPath[0], getModulePath())
        self.assertEqual(splitPath[1], testPath)

        # Check that adding "policy:" to the front returns the same result
        self.assertEqual(resolvedPath, resolveRelativeConfigPath(f"policy:{testPath}"))

    def testMergeConfigWithFile(self) -> None:
        # Config file used for tests
        configFile = f"{getModulePath()}/tests/testData/testConfigFile.yaml"

        # Load the contents into a dictionary
        config = readConfigYaml(configFile)

        # Test loading without overriding defaults
        mergedConfig = mergeConfigWithFile(configFile, **{key: None for key in config})
        self.assertDictEqual(mergedConfig, config)

        # Test loading while overriding a default
        keys = list(config)
        override: dict[str, Union[None, str]] = {key: None for key in keys[:-1]}
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

    def testConfigClass(self) -> None:
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

    def testComputeSha256(self) -> None:
        content = b"hello ts_wep model bytes"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            self.assertEqual(computeSha256(path), hashlib.sha256(content).hexdigest())
        finally:
            os.remove(path)

    def testVerifyModelChecksum(self) -> None:
        content = b"some model weights"
        expected = hashlib.sha256(content).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            # Matching checksum: no exception.
            verifyModelChecksum(path, expected)

            # Empty/None expected checksum: check is skipped (no exception,
            # even for a path that does not exist).
            verifyModelChecksum(path, "")
            verifyModelChecksum("/does/not/exist.pt", None)

            # Mismatched checksum: RuntimeError.
            with self.assertRaises(RuntimeError):
                verifyModelChecksum(path, "0" * 64)
        finally:
            os.remove(path)

    def testGetObsLsstCmdTaskConfigDir(self) -> None:
        obsLsstCmdTaskConfirDir = getObsLsstCmdTaskConfigDir()
        configNormPath = os.path.normpath(obsLsstCmdTaskConfirDir)
        configNormPathList = configNormPath.split(os.sep)

        self.assertEqual(configNormPathList[-1], "config")
        self.assertTrue(("obs_lsst" in configNormPathList))
