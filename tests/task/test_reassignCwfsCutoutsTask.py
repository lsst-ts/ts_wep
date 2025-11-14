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
from lsst.ts.wep.task.cutOutDonutsCwfsTask import (
    CutOutDonutsCwfsTask,
    CutOutDonutsCwfsTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestReassignDonutsCwfsTask(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate donutCatalog needed for task.
        """

        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if "pretest_run_cwfs" in collectionsList:
            cls.runName = "pretest_run_cwfs"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
            instrument = "lsst.obs.lsst.LsstCam"
            pipelineYaml = os.path.join(testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml")

            pipeCmd = writePipetaskCmd(
                cls.repoDir,
                cls.runName,
                instrument,
                collections,
                pipelineYaml=pipelineYaml,
            )
            pipeCmd += ' -d "detector IN (191, 192)"'
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = CutOutDonutsCwfsTaskConfig()
        self.task = CutOutDonutsCwfsTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry
        self.visitNum = 4021123106000

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": self.visitNum,
            "visit": self.visitNum,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 192,
            "exposure": self.visitNum,
            "visit": self.visitNum,
        }

        self.testRunName = "testTaskRun"
        self.collectionsList = list(self.registry.queryCollections())
        if self.testRunName in self.collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, self.testRunName)
            runProgram(cleanUpCmd)

    def tearDown(self) -> None:
        # Get Butler with updated registry
        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.collectionsList = list(self.registry.queryCollections())
        if self.testRunName in self.collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, self.testRunName)
            runProgram(cleanUpCmd)

    def testPipeline(self) -> None:
        # Compare the interactive run to pipetask run results
        donutStampsExtra_extraId = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra_extraId = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsExtraCwfs_extraId = self.butler.get(
            "donutStampsCwfs", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntraCwfs_intraId = self.butler.get(
            "donutStampsCwfs", dataId=self.dataIdIntra, collections=[self.runName]
        )

        # Test that reassigned stamps are correctly assigned
        for reassignStamp, origStamp in zip(donutStampsExtra_extraId, donutStampsExtraCwfs_extraId):
            self.assertMaskedImagesAlmostEqual(reassignStamp.stamp_im, origStamp.stamp_im)  # type: ignore
        for reassignStamp, origStamp in zip(donutStampsIntra_extraId, donutStampsIntraCwfs_intraId):
            self.assertMaskedImagesAlmostEqual(reassignStamp.stamp_im, origStamp.stamp_im)  # type: ignore

        # Test that the intra-focal id does not get reassigned stamps
        with self.assertRaises(DatasetNotFoundError):
            self.butler.get("donutStampsExtra", dataId=self.dataIdIntra, collections=[self.runName])
        with self.assertRaises(DatasetNotFoundError):
            self.butler.get("donutStampsIntra", dataId=self.dataIdIntra, collections=[self.runName])
