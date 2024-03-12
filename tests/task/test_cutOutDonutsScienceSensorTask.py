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
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCutOutDonutsScienceSensorTask(lsst.utils.tests.TestCase):
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
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCutoutsFamPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure IN (4021123106001, 4021123106002)'"
        runProgram(pipeCmd)

    def setUp(self):
        self.config = CutOutDonutsScienceSensorTaskConfig()
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
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

    def testValidateConfigs(self):
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testAssignExtraIntraIdxLsstCam(self):
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZNegative, focusZPositive, "LSSTCam"
        )
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZPositive, focusZNegative, "LSSTCam"
        )
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxLsstComCam(self):
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZNegative, focusZPositive, "LSSTComCam"
        )
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZPositive, focusZNegative, "LSSTComCam"
        )
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxLsstComCamSim(self):
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZNegative, focusZPositive, "LSSTComCamSim"
        )
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZPositive, focusZNegative, "LSSTComCamSim"
        )
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxFocusZValueError(self):
        focusZNegative = -1
        focusZPositive = 1
        focusZ0 = 0

        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZPositive, focusZPositive, "LSSTCam")
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZPositive, focusZ0, "LSSTCam")
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZNegative, focusZNegative, "LSSTCam")
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZNegative, focusZ0, "LSSTCam")
        with self.assertRaises(ValueError) as context:
            self.task.assignExtraIntraIdx(focusZ0, focusZPositive, "LSSTCam")
        self.assertEqual(
            "Must have one extra-focal and one intra-focal image.",
            str(context.exception),
        )

    def testAssignExtraIntraIdxInvalidCamera(self):
        cameraName = "WrongCam"
        with self.assertRaises(ValueError) as context:
            self.task.assignExtraIntraIdx(-1, 1, cameraName)
        errorStr = str(
            f"Invalid cameraName parameter: {cameraName}. Camera must  "
            "be one of: 'LSSTCam', 'LSSTComCam', 'LSSTComCamSim' or 'LATISS'",
        )
        self.assertEqual(errorStr, str(context.exception))

    def testTaskRun(self):
        # Grab two exposures from the same detector at two different visits to
        # get extra and intra
        exposureExtra = self.butler.get(
            "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
        )
        exposureIntra = self.butler.get(
            "postISRCCD", dataId=self.dataIdIntra, collections=[self.runName]
        )

        donutCatalogExtra = self.butler.get(
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutCatalogIntra = self.butler.get(
            "donutCatalog", dataId=self.dataIdIntra, collections=[self.runName]
        )
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections="LSSTCam/calib/unbounded",
        )

        # Test return values when no sources in catalog
        noSrcDonutCatalog = pd.DataFrame(columns=donutCatalogExtra.columns)
        testOutNoSrc = self.task.run(
            [exposureExtra, exposureIntra], [noSrcDonutCatalog] * 2, camera
        )

        self.assertEqual(len(testOutNoSrc.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra), 0)

        # Test normal behavior
        taskOut = self.task.run(
            [copy(exposureIntra), copy(exposureExtra)],
            [donutCatalogExtra, donutCatalogIntra],
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

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
