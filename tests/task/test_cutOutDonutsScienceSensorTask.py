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
import numpy as np
import pandas as pd
from scipy.signal import correlate

import lsst.utils.tests
from lsst.afw import image as afwImage
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.CutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    DefocalType,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
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
        collections = "refcats,LSSTCam/calib,LSSTCam/raw/all"
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

    def _generateTestExposures(self):

        # Generate donut template
        template = self.task.getTemplate(
            "R22_S11", DefocalType.Extra, self.task.donutTemplateSize
        )
        correlatedImage = correlate(template, template)
        maxIdx = np.argmax(correlatedImage)
        maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))
        templateCenter = np.array(maxLoc) - self.task.donutTemplateSize / 2

        # Make donut centered in exposure
        initCutoutSize = (
            self.task.donutTemplateSize + self.task.initialCutoutPadding * 2
        )
        centeredArr = np.zeros((initCutoutSize, initCutoutSize), dtype=np.float32)
        centeredArr[
            self.task.initialCutoutPadding : -self.task.initialCutoutPadding,
            self.task.initialCutoutPadding : -self.task.initialCutoutPadding,
        ] += template
        centeredImage = afwImage.ImageF(initCutoutSize, initCutoutSize)
        centeredImage.array = centeredArr
        centeredExp = afwImage.ExposureF(initCutoutSize, initCutoutSize)
        centeredExp.setImage(centeredImage)
        centerCoord = (
            self.task.initialCutoutPadding + templateCenter[1],
            self.task.initialCutoutPadding + templateCenter[0],
        )

        # Make new donut that needs to be shifted by 20 pixels
        # from the edge of the exposure
        offCenterArr = np.zeros((initCutoutSize, initCutoutSize), dtype=np.float32)
        offCenterArr[
            : self.task.donutTemplateSize - 20, : self.task.donutTemplateSize - 20
        ] = template[20:, 20:]
        offCenterImage = afwImage.ImageF(initCutoutSize, initCutoutSize)
        offCenterImage.array = offCenterArr
        offCenterExp = afwImage.ExposureF(initCutoutSize, initCutoutSize)
        offCenterExp.setImage(offCenterImage)
        # Center coord value 20 pixels closer than template center
        # due to stamp overrunning the edge of the exposure.
        offCenterCoord = templateCenter - 20

        return centeredExp, centerCoord, template, offCenterExp, offCenterCoord

    def testValidateConfigs(self):

        self.config.donutTemplateSize = 120
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.assertEqual(self.task.donutTemplateSize, 120)
        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testAssignExtraIntraIdx(self):

        focusZNegative = -1
        focusZPositive = 1
        focusZ0 = 0

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZNegative, focusZPositive
        )
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(
            focusZPositive, focusZNegative
        )
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZPositive, focusZPositive)
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZPositive, focusZ0)
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZNegative, focusZNegative)
        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZNegative, focusZ0)
        with self.assertRaises(ValueError) as context:
            self.task.assignExtraIntraIdx(focusZ0, focusZPositive)
        self.assertEqual(
            "Must have one extra-focal and one intra-focal image.",
            str(context.exception),
        )

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
            [exposureIntra, exposureExtra],
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