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

import lsst.obs.lsst as obs_lsst
from lsst.ts.wep.utils import (
    getCameraFromButlerName,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestTaskUtils(unittest.TestCase):
    """Test the task utility functions."""

    def _writePipetaskCmd(
        self,
        repoName,
        instrument,
        collections,
        runName,
        taskName=None,
        pipelineName=None,
    ):
        # Write the part of the command that is always included
        testCmd = f"pipetask run -b {repoName} -i {collections} "
        testCmd += f"--instrument {instrument} "
        testCmd += f"--register-dataset-types --output-run {runName}"

        # Write with taskName
        if taskName is not None:
            testCmd += f" -t {taskName}"

        # Write with pipeline filename
        if pipelineName is not None:
            testCmd += f" -p {pipelineName}"

        return testCmd

    def _writeCleanUpCmd(self, repoName, runName):
        testCmd = f"butler remove-runs {repoName} {runName}"
        testCmd += " --no-confirm"

        return testCmd

    def testWritePipetaskCmd(self):
        repoName = "testRepo"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats"
        runName = "run2"

        # Test writing with task name
        taskName = "lsst.ts.wep.testTask"
        testCmdTask = self._writePipetaskCmd(
            repoName, instrument, collections, runName, taskName=taskName
        )

        pipeOutTask = writePipetaskCmd(
            repoName, runName, instrument, collections, taskName=taskName
        )
        self.assertEqual(testCmdTask, pipeOutTask)

        # Test writing with pipeline
        pipelineYamlFile = "testPipeOut.yaml"
        testCmdYaml = self._writePipetaskCmd(
            repoName, instrument, collections, runName, pipelineName=pipelineYamlFile
        )

        pipeOutYaml = writePipetaskCmd(
            repoName, runName, instrument, collections, pipelineYaml=pipelineYamlFile
        )
        self.assertEqual(testCmdYaml, pipeOutYaml)

        assertMsg = "At least one of taskName or pipelineYaml must not be None"
        with self.assertRaises(ValueError) as context:
            writePipetaskCmd(repoName, runName, instrument, collections)
        self.assertTrue(assertMsg in str(context.exception))

    def testWriteCleanUpRepoCmd(self):
        repoName = "testRepo"
        runName = "run2"

        testCmd = self._writeCleanUpCmd(repoName, runName)
        self.assertEqual(testCmd, writeCleanUpRepoCmd(repoName, runName))

    def testGetCameraFromButlerName(self):
        # Test camera loading
        self.assertEqual(
            obs_lsst.LsstCam().getCamera(), getCameraFromButlerName("LSSTCam")
        )
        self.assertEqual(
            obs_lsst.LsstComCam().getCamera(), getCameraFromButlerName("LSSTComCam")
        )
        self.assertEqual(
            obs_lsst.Latiss().getCamera(), getCameraFromButlerName("LATISS")
        )
        # Test error
        badCamType = "badCam"
        errMsg = f"Camera {badCamType} is not supported."
        with self.assertRaises(ValueError) as context:
            getCameraFromButlerName(badCamType)
        self.assertEqual(str(context.exception), errMsg)
