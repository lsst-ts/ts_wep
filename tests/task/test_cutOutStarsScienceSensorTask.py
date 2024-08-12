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
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.cutOutStarsScienceSensorTask import (
    CutOutStarsScienceSensorTask,
    CutOutStarsScienceSensorTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCutOutStarsScienceSensorTask(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run the pipeline only once since it takes a
        couple minutes with the ISR.
        """

        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        # Check that runs don't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        # Point to the collections for the reference catalogs,
        # the raw images and the camera model in the calib directory
        # that comes from `butler write-curated-calibrations`.
        collections = "LSSTComCam/calib,LSSTComCam/raw/all"
        instrument = "lsst.obs.lsst.LsstComCam"
        cls.cameraName = "LSSTComCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCutoutsStarsPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure=5025082000941'"
        runProgram(pipeCmd)

    def setUp(self):
        self.config = CutOutStarsScienceSensorTaskConfig()
        self.task = CutOutStarsScienceSensorTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdFocus = {
            "instrument": "LSSTComCam",
            "detector": 0,
            "exposure": 5025082000941,
            "visit": 5025082000941,
        }

    def testValidateConfigs(self):
        self.config.donutStampSize = 75
        self.config.initialCutoutPadding = 15
        self.task = CutOutStarsScienceSensorTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 75)
        self.assertEqual(self.task.initialCutoutPadding, 15)

    def testTaskRun(self):
        # Grab two exposures from the same detector at two different visits to
        # get extra and intra
        exposureFocus = self.butler.get(
            "postISRCCD", dataId=self.dataIdFocus, collections=[self.runName]
        )

        sourceCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIdFocus, collections=[self.runName]
        )

        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTComCam"},
            collections="LSSTComCam/calib/unbounded",
        )
        # Test return values when no sources in catalog
        noSrcDonutCatalog = pd.DataFrame(columns=sourceCatalog.columns)
        testOutNoSrc = self.task.run(exposureFocus, noSrcDonutCatalog, camera)

        self.assertEqual(len(testOutNoSrc.starStamps), 0)

        # Test normal behavior
        # set this config to match the pipeline yaml in the example
        self.config.donutStampSize = 100
        self.config.initialCutoutPadding = 10
        self.config.maxRecenterDistance = 40
        self.task = CutOutStarsScienceSensorTask(config=self.config)
        taskOut = self.task.run(exposureFocus, sourceCatalog, camera)

        testStamps = self.task.cutOutStamps(
            exposureFocus, sourceCatalog, DefocalType.Focus, camera.getName()
        )

        for starStamp, cutOutStamp in zip(taskOut.starStamps, testStamps):
            self.assertMaskedImagesAlmostEqual(
                starStamp.stamp_im, cutOutStamp.stamp_im, atol=1e-4
            )

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
