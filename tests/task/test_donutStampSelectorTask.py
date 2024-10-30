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

import lsst.utils.tests
import numpy as np
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.donutStampSelectorTask import (
    DonutStampSelectorTask,
    DonutStampSelectorTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestDonutStampSelectorTask(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutStamps needed for task.
        """
        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCutoutsFamPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Make sure we are using the right exposure+detector combinations
        pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):
        self.config = DonutStampSelectorTaskConfig()
        self.task = DonutStampSelectorTask(config=self.config, name="Base Task")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }

    def testValidateConfigs(self):

        # Test the default config values
        self.OrigTask = DonutStampSelectorTask(config=self.config, name="Orig Task")
        self.assertEqual(self.OrigTask.config.selectWithEntropy, False)
        self.assertEqual(self.OrigTask.config.selectWithSignalToNoise, True)
        self.assertEqual(self.OrigTask.config.selectWithFracBadPixels, True)
        self.assertEqual(self.OrigTask.config.useCustomSnLimit, False)

        # Test changing configs
        self.config.maxSelect = 10
        self.config.selectWithEntropy = True
        self.config.selectWithSignalToNoise = False
        self.config.selectWithFracBadPixels = False
        self.config.minSignalToNoise = 999
        self.config.maxEntropy = 4
        self.config.maxFracBadPixels = 0.2
        self.ModifiedTask = DonutStampSelectorTask(config=self.config, name="Mod Task")

        self.assertEqual(self.ModifiedTask.config.maxSelect, 10)
        self.assertEqual(self.ModifiedTask.config.selectWithEntropy, True)
        self.assertEqual(self.ModifiedTask.config.selectWithSignalToNoise, False)
        self.assertEqual(self.ModifiedTask.config.selectWithFracBadPixels, False)
        self.assertEqual(self.ModifiedTask.config.minSignalToNoise, 999)
        self.assertEqual(self.ModifiedTask.config.maxEntropy, 4)
        self.assertEqual(self.ModifiedTask.config.maxFracBadPixels, 0.2)

    def testSelectStamps(self):
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

        # test defaults
        selection = self.task.selectStamps(donutStampsIntra)
        donutsQuality = selection.donutsQuality

        # by default, config.selectWithEntropy is False,
        # so we select all donuts
        self.assertEqual(np.sum(donutsQuality["ENTROPY_SELECT"]), 3)

        # by default, SNR selection happens and uses yaml config values
        # so that all donuts here would get selected
        self.assertEqual(np.sum(donutsQuality["SN_SELECT"]), 3)

        # by default, it thresholds on fraction-of-bad-pixels
        # only one of these test donuts is selected
        self.assertEqual(np.sum(donutsQuality["FRAC_BAD_PIX_SELECT"]), 1)

        # Test that overall selection also shows only one donut
        self.assertEqual(np.sum(donutsQuality["FINAL_SELECT"]), 1)
        self.assertEqual(np.sum(selection.selected), 1)

        # switch on selectWithEntropy,
        # set config.maxEntropy so that one donut is selected
        self.config.selectWithFracBadPixels = False
        self.config.selectWithEntropy = True
        entropyThreshold = 2.85
        self.config.maxEntropy = entropyThreshold

        task = DonutStampSelectorTask(config=self.config, name="Entropy Task")
        selection = task.selectStamps(donutStampsIntra)
        donutsQuality = selection.donutsQuality
        self.assertEqual(np.sum(donutsQuality["ENTROPY_SELECT"]), 1)

        # also test that the entropy of the selected donut
        # is indeed below threshold
        self.assertLess(
            donutsQuality["ENTROPY"][donutsQuality["ENTROPY_SELECT"]],
            entropyThreshold,
        )

        # test custom SNR thresholds
        self.config.selectWithEntropy = False
        self.config.useCustomSnLimit = True
        minSignalToNoise = 1658
        self.config.minSignalToNoise = minSignalToNoise
        task = DonutStampSelectorTask(config=self.config, name="SN Task")
        selection = task.selectStamps(donutStampsIntra)
        donutsQuality = selection.donutsQuality
        self.assertEqual(np.sum(donutsQuality["SN_SELECT"]), 2)

        # test that the SN of selected donuts is indeed above the threshold
        for v in donutsQuality["SN"][donutsQuality["SN_SELECT"]]:
            self.assertLess(minSignalToNoise, v)

        # turn all selections off and make sure everything is selected
        self.config.selectWithEntropy = False
        self.config.selectWithSignalToNoise = False
        self.config.selectWithFracBadPixels = False
        task = DonutStampSelectorTask(config=self.config, name="All off")
        selection = task.selectStamps(donutStampsIntra)
        self.assertEqual(np.sum(selection.donutsQuality["ENTROPY_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["SN_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["FRAC_BAD_PIX_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["FINAL_SELECT"]), 3)

        # set maxSelect = 1 and make sure the final selection is only 1
        self.config.maxSelect = 1
        task = DonutStampSelectorTask(config=self.config, name="maxSelect=1")
        selection = task.selectStamps(donutStampsIntra)
        self.assertEqual(np.sum(selection.donutsQuality["ENTROPY_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["SN_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["FRAC_BAD_PIX_SELECT"]), 3)
        self.assertEqual(np.sum(selection.donutsQuality["FINAL_SELECT"]), 1)

    def testTaskRun(self):
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

        # test defaults
        taskOut = self.task.run(donutStampsIntra)
        donutsQuality = taskOut.donutsQuality
        selected = taskOut.selected
        donutStampsSelect = taskOut.donutStampsSelect

        # Test that final selection numbers match
        self.assertEqual(len(donutStampsSelect), selected.sum())
        self.assertEqual(len(donutStampsSelect), donutsQuality["FINAL_SELECT"].sum())
