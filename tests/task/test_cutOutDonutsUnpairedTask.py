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
from copy import copy

import lsst.afw.image as afwImage
import lsst.utils.tests
from astropy.table import QTable
from lsst.afw.cameraGeom import Camera
from lsst.daf.butler import Butler
from lsst.ts.wep.task.cutOutDonutsUnpairedTask import (
    CutOutDonutsUnpairedTask,
    CutOutDonutsUnpairedTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCutOutDonutsUnpairedTask(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    visitNum: int
    baseRunName: str

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
        # The visit number for the test data
        cls.visitNum = 4021123106000

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if "pretest_run_cwfs" in collectionsList:
            cls.baseRunName = "pretest_run_cwfs"
        else:
            cls.baseRunName = "run1"
            if cls.baseRunName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.baseRunName)
                runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testCutoutsUnpairedPipeline.yaml")

        if cls.baseRunName == "pretest_run_cwfs":
            collections += ",pretest_run_cwfs"
            cls.runName = "run1"
            pipelineYaml += "#cutOutDonutsUnpairedTask"
        else:
            cls.runName = "run1"

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
        self.config = CutOutDonutsUnpairedTaskConfig()
        self.task = CutOutDonutsUnpairedTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
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

    def _getDataFromButler(
        self,
    ) -> tuple[afwImage.Exposure, afwImage.Exposure, QTable, QTable, Camera]:
        # Grab two exposures from the same visits of adjacent detectors
        exposureExtra = self.butler.get(
            "post_isr_image", dataId=self.dataIdExtra, collections=[self.baseRunName]
        )
        exposureIntra = self.butler.get(
            "post_isr_image", dataId=self.dataIdIntra, collections=[self.baseRunName]
        )
        # Get the donut catalogs for each detector
        donutCatalogExtra = self.butler.get(
            "donutTable", dataId=self.dataIdExtra, collections=[self.baseRunName]
        )
        donutCatalogIntra = self.butler.get(
            "donutTable", dataId=self.dataIdIntra, collections=[self.baseRunName]
        )
        # Get the camera from the butler
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections="LSSTCam/calib/unbounded",
        )

        return (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        )

    def testValidateConfigs(self) -> None:
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = CutOutDonutsUnpairedTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testPipeline(self) -> None:
        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test normal behavior
        taskOutExtra = self.task.run(
            copy(exposureExtra),
            donutCatalogExtra,
            camera,
        )
        taskOutIntra = self.task.run(
            copy(exposureIntra),
            donutCatalogIntra,
            camera,
        )

        # Compare the interactive run to pipetask run results
        donutStampsExtra = self.butler.get("donutStamps", dataId=self.dataIdExtra, collections=[self.runName])
        donutStampsIntra = self.butler.get("donutStamps", dataId=self.dataIdIntra, collections=[self.runName])

        for butlerStamp, taskStamp in zip(donutStampsExtra, taskOutExtra.donutStamps):
            self.assertMaskedImagesAlmostEqual(butlerStamp.stamp_im, taskStamp.stamp_im)  # type: ignore
        for butlerStamp, taskStamp in zip(donutStampsIntra, taskOutIntra.donutStamps):
            self.assertMaskedImagesAlmostEqual(butlerStamp.stamp_im, taskStamp.stamp_im)  # type: ignore

    def testEmptyCatalog(self) -> None:
        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test empty catalog behavior
        emptyCatalog = donutCatalogExtra.copy()
        emptyCatalog = emptyCatalog[:0]

        taskOutExtra = self.task.run(
            copy(exposureExtra),
            emptyCatalog,
            camera,
        )
        donutStampsExtra = self.butler.get("donutStamps", dataId=self.dataIdExtra, collections=[self.runName])

        self.assertEqual(len(taskOutExtra.donutStamps), 0)
        metadataKeys = [
            "BANDPASS",
            "DFC_DIST",
            "DFC_TYPE",
            "DET_NAME",
            "CAM_NAME",
            "VISIT",
            "MJD",
            "BORESIGHT_ROT_ANGLE_RAD",
            "BORESIGHT_PAR_ANGLE_RAD",
            "BORESIGHT_RA_RAD",
            "BORESIGHT_DEC_RAD",
        ]
        for metadataKey in metadataKeys:
            self.assertAlmostEqual(
                donutStampsExtra.metadata[metadataKey], taskOutExtra.donutStamps.metadata[metadataKey]
            )
