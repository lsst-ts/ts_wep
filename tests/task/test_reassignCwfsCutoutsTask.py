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
from lsst.daf import butler as dafButler
from lsst.daf.butler import DatasetNotFoundError
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


class TestCutOutDonutsCwfsTask(lsst.utils.tests.TestCase):
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
        cls.runName = "run2"
        # The visit number for the test data
        cls.visitNum = 4021123106000

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        # Point to the collections for the reference catalogs,
        # the raw images and the camera model in the calib directory
        # that comes from `butler write-curated-calibrations`.
        cls.collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        cls.instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        cls.pipelineYaml = os.path.join(
            testPipelineConfigDir, "testReassignCwfsCutoutsPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir,
            cls.runName,
            cls.instrument,
            cls.collections,
            pipelineYaml=cls.pipelineYaml,
        )
        pipeCmd += f" -d 'exposure IN ({cls.visitNum})'"
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):
        self.config = CutOutDonutsCwfsTaskConfig()
        self.task = CutOutDonutsCwfsTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

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

    def tearDown(self):
        # Get Butler with updated registry
        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.collectionsList = list(self.registry.queryCollections())
        if self.testRunName in self.collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, self.testRunName)
            runProgram(cleanUpCmd)

    def testPipeline(self):

        # Compare the interactive run to pipetask run results
        donutStampsExtra_extraId = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra_extraId = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsExtraCwfs_extraId = self.butler.get(
            "donutStampsExtraCwfs", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntraCwfs_intraId = self.butler.get(
            "donutStampsIntraCwfs", dataId=self.dataIdIntra, collections=[self.runName]
        )

        # Test that reassigned stamps are correctly assigned
        for reassignStamp, origStamp in zip(
            donutStampsExtra_extraId, donutStampsExtraCwfs_extraId
        ):
            self.assertMaskedImagesAlmostEqual(
                reassignStamp.stamp_im, origStamp.stamp_im
            )
        for reassignStamp, origStamp in zip(
            donutStampsIntra_extraId, donutStampsIntraCwfs_intraId
        ):
            self.assertMaskedImagesAlmostEqual(
                reassignStamp.stamp_im, origStamp.stamp_im
            )

        # Test that the intra-focal id does not get reassigned stamps
        with self.assertRaises(DatasetNotFoundError):
            self.butler.get(
                "donutStampsExtra", dataId=self.dataIdIntra, collections=[self.runName]
            )
        with self.assertRaises(DatasetNotFoundError):
            self.butler.get(
                "donutStampsIntra", dataId=self.dataIdIntra, collections=[self.runName]
            )
