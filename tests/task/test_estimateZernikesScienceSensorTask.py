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
from copy import copy
from scipy.signal import correlate

import lsst.utils.tests
from lsst.afw import image as afwImage
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.EstimateZernikesScienceSensorTask import (
    EstimateZernikesScienceSensorTask,
    EstimateZernikesScienceSensorTaskConfig,
)
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    DefocalType,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestEstimateZernikesScienceSensorTask(lsst.utils.tests.TestCase):
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

        collections = "refcats,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testFamPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure IN (4021123106001, 4021123106002)'"
        runProgram(pipeCmd)

    def setUp(self):

        self.config = EstimateZernikesScienceSensorTaskConfig()
        self.task = EstimateZernikesScienceSensorTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106002,
        }

    def _generateTestExposures(self):

        # Generate donut template
        template = self.task.getTemplate("R22_S11", DefocalType.Extra)
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

    def validateConfigs(self):

        self.config.donutTemplateSize = 120
        self.config.donutStampSize = 120
        self.config.initialCutoutSize = 290
        self.task = EstimateZernikesScienceSensorTask(config=self.config)

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

        donutCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )

        # Test return values when no sources in catalog
        noSrcDonutCatalog = copy(donutCatalog)
        noSrcDonutCatalog["detector"] = "R22_S99"
        testOutNoSrc = self.task.run(
            [exposureExtra, exposureIntra], noSrcDonutCatalog, self.cameraName
        )

        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesRaw, [np.ones(19) * np.nan] * 2
        )
        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesAvg, [np.ones(19) * np.nan] * 2
        )
        self.assertEqual(len(testOutNoSrc.donutStampsExtra[0]), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra[1]), 0)

        # Test normal behavior
        taskOut = self.task.run(
            [exposureIntra, exposureExtra], donutCatalog, self.cameraName
        )

        testExtraStamps = self.task.cutOutStamps(
            exposureExtra, donutCatalog, DefocalType.Extra, self.cameraName
        )
        testIntraStamps = self.task.cutOutStamps(
            exposureIntra, donutCatalog, DefocalType.Intra, self.cameraName
        )

        for donutStamp, cutOutStamp in zip(taskOut.donutStampsExtra[0], testExtraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )
        for donutStamp, cutOutStamp in zip(taskOut.donutStampsIntra[1], testIntraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )

        testCoeffsRaw = self.task.estimateZernikes(testExtraStamps, testIntraStamps)
        testCoeffsAvg = self.task.combineZernikes(testCoeffsRaw)
        np.testing.assert_array_equal(taskOut.outputZernikesRaw[0], testCoeffsRaw)
        np.testing.assert_array_equal(taskOut.outputZernikesAvg[0], testCoeffsAvg)

    @classmethod
    def tearDownClass(cls):

        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
