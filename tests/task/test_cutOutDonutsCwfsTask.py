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

import lsst.utils.tests
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.cutOutDonutsCwfsTask import (
    CutOutDonutsCwfsTask,
    CutOutDonutsCwfsTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
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
            testPipelineConfigDir, "testCutoutsCwfsPipeline.yaml"
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

    def _getDataFromButler(self):
        # Grab two exposures from the same visits of adjacent detectors
        exposureExtra = self.butler.get(
            "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
        )
        exposureIntra = self.butler.get(
            "postISRCCD", dataId=self.dataIdIntra, collections=[self.runName]
        )
        # Get the donut catalogs for each detector
        donutCatalogExtra = self.butler.get(
            "donutCatalogAstropy", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutCatalogIntra = self.butler.get(
            "donutCatalogAstropy", dataId=self.dataIdIntra, collections=[self.runName]
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

    def testValidateConfigs(self):
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = CutOutDonutsCwfsTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testTaskRunNormal(self):
        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test normal behavior: both exposure and donut catalog
        # order is tested for in the task
        taskOut = self.task.run(
            [copy(exposureIntra), copy(exposureExtra)],
            [
                donutCatalogExtra,
                donutCatalogIntra,
            ],
            camera,
        )

        testExtraStamps = self.task.cutOutStamps(
            exposureExtra, donutCatalogExtra, DefocalType.Extra, camera.getName()
        )
        testIntraStamps = self.task.cutOutStamps(
            exposureIntra, donutCatalogIntra, DefocalType.Intra, camera.getName()
        )

        for donutStamp, cutOutStamp in zip(taskOut.donutStampsExtra, testExtraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )
        for donutStamp, cutOutStamp in zip(taskOut.donutStampsIntra, testIntraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )

    def testPipeline(self):
        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test normal behavior
        taskOut = self.task.run(
            [copy(exposureIntra), copy(exposureExtra)],
            [donutCatalogExtra, donutCatalogIntra],
            camera,
        )
        # Test that order of donut catalogs is tested just like exposures
        taskOutReorder = self.task.run(
            [copy(exposureIntra), copy(exposureExtra)],
            [donutCatalogIntra, donutCatalogExtra],
            camera,
        )

        # One way is to ensure that both donut arrays are identical
        for stampNormal, stampReorder in zip(
            taskOut.donutStampsExtra, taskOutReorder.donutStampsExtra
        ):
            self.assertMaskedImagesAlmostEqual(
                stampNormal.stamp_im, stampReorder.stamp_im
            )
        for stampNormal, stampReorder in zip(
            taskOut.donutStampsIntra, taskOutReorder.donutStampsIntra
        ):
            self.assertMaskedImagesAlmostEqual(
                stampNormal.stamp_im, stampReorder.stamp_im
            )

        # Test that legacy dataset works as well (without "detector" column)
        donutCatalogExtra.remove_column("detector")
        donutCatalogIntra.remove_column("detector")

        taskOutLegacy = self.task.run(
            [copy(exposureExtra), copy(exposureIntra)],
            [donutCatalogExtra, donutCatalogIntra],
            camera,
        )
        # The donuts should be identical
        for legacyStamp, normalStamp in zip(
            taskOutLegacy.donutStampsExtra, taskOut.donutStampsExtra
        ):
            self.assertMaskedImagesAlmostEqual(
                legacyStamp.stamp_im, normalStamp.stamp_im
            )
        for legacyStamp, normalStamp in zip(
            taskOutLegacy.donutStampsIntra, taskOut.donutStampsIntra
        ):
            self.assertMaskedImagesAlmostEqual(
                legacyStamp.stamp_im, normalStamp.stamp_im
            )

        # Compare the interactive run to pipetask run results
        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

        for butlerStamp, taskStamp in zip(donutStampsExtra, taskOut.donutStampsExtra):
            self.assertMaskedImagesAlmostEqual(butlerStamp.stamp_im, taskStamp.stamp_im)
        for butlerStamp, taskStamp in zip(donutStampsIntra, taskOut.donutStampsIntra):
            self.assertMaskedImagesAlmostEqual(butlerStamp.stamp_im, taskStamp.stamp_im)
