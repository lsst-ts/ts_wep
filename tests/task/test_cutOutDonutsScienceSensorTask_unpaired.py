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
from lsst.daf.butler import Butler, DatasetNotFoundError
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCutOutDonutsScienceSensorTask(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    visitNum: int
    baseRunName: str
    pairTableName: str
    run2Name: str
    run3Name: str
    cameraName: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run the pipeline only once since it takes a
        couple minutes with the ISR.
        """

        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.cameraName = "LSSTCam"

        # Check that runs don't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testCutoutsFamPipeline_unpaired.yaml")
        collectionsList = list(registry.queryCollections())
        cls.runName = "run1"
        if "pretest_run_science" in collectionsList:
            collections += ",pretest_run_science"
            pipelineYaml += "#cutOutDonutsScienceSensorTask"
        elif cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        # Point to the collections for the reference catalogs,
        # the raw images and the camera model in the calib directory
        # that comes from `butler write-curated-calibrations`.

        pipeCmd = writePipetaskCmd(
            cls.repoDir,
            cls.runName,
            instrument,
            collections,
            pipelineYaml=pipelineYaml,
        )
        pipeCmd += " -d 'exposure IN (4021123106001..4021123106007)'"
        runProgram(pipeCmd)

    def setUp(self) -> None:
        self.config = CutOutDonutsScienceSensorTaskConfig(runPaired=False)
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106002,
            "visit": 4021123106002,
        }

    def testValidateConfigs(self) -> None:
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.config.runPaired = False
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)
        self.assertFalse(self.task.runPaired)

    def testPipelineOutput(self) -> None:
        # This mode should not affect anything except the butler output.
        donutStampsExtra = self.butler.get(
            "donutStampsScienceSensor", dataId=self.dataIdExtra, collections=[self.runName]
        )
        self.assertEqual(len(donutStampsExtra), 3)
        donutStampsIntra = self.butler.get(
            "donutStampsScienceSensor", dataId=self.dataIdIntra, collections=[self.runName]
        )
        self.assertEqual(len(donutStampsIntra), 3)

        with self.assertRaises(DatasetNotFoundError):
            self.butler.get("donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName])

    @classmethod
    def tearDownClass(cls) -> None:
        tearDownRunList = list()
        if cls.runName == "run1":
            tearDownRunList.append(cls.runName)
        for runName in tearDownRunList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, runName)
            runProgram(cleanUpCmd)
