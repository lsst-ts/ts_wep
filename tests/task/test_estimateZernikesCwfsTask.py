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
from copy import copy
from scipy.signal import correlate

import lsst.utils.tests
from lsst.afw import image as afwImage
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.EstimateZernikesCwfsTask import (
    EstimateZernikesCwfsTask,
    EstimateZernikesCwfsTaskConfig,
)
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    DefocalType,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestEstimateZernikesCwfsTask(lsst.utils.tests.TestCase):
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

        collections = "refcats,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testCwfsPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += f" -d 'exposure IN ({cls.visitNum})'"
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):

        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):

        self.config = EstimateZernikesCwfsTaskConfig()
        self.task = EstimateZernikesCwfsTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": self.visitNum,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 192,
            "exposure": self.visitNum,
        }

    def _generateTestExposures(self):

        # Generate donut template
        template = self.task.getTemplate("R00_SW0", DefocalType.Extra)
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

    def _getDataFromButler(self):

        # Grab two exposures from the same visits of adjacent detectors
        exposureExtra = self.butler.get(
            "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
        )
        exposureIntra = self.butler.get(
            "postISRCCD", dataId=self.dataIdIntra, collections=[self.runName]
        )

        donutCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )

        return exposureExtra, exposureIntra, donutCatalog

    def validateConfigs(self):

        self.config.donutTemplateSize = 120
        self.config.donutStampSize = 120
        self.config.initialCutoutSize = 290
        self.task = EstimateZernikesCwfsTask(config=self.config)

        self.assertEqual(self.task.donutTemplateSize, 120)
        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testSelectCwfsSources(self):

        testDf = pd.DataFrame()
        testDf["coord_ra"] = np.zeros(5)
        testDf["coord_dec"] = np.zeros(5)
        testDf["source_flux"] = np.arange(5)
        testDf["centroid_x"] = np.zeros(5)
        testDf["centroid_y"] = np.zeros(5)
        testDf["detector"] = ["R00_SW0", "R44_SW0", "R40_SW1", "R04_SW1", "R22_S11"]
        extraCatalog, intraCatalog = self.task.selectCwfsSources(testDf, (4072, 2000))

        np.testing.assert_array_equal(extraCatalog.values, testDf.iloc[:2].values[::-1])
        np.testing.assert_array_equal(
            intraCatalog.values, testDf.iloc[2:4].values[::-1]
        )

    def testTaskRunNoSources(self):

        exposureExtra, exposureIntra, donutCatalog = self._getDataFromButler()

        # Test return values when no sources in catalog
        noSrcDonutCatalog = copy(donutCatalog)
        noSrcDonutCatalog["detector"] = "R22_S99"
        testOutNoSrc = self.task.run([exposureExtra, exposureIntra], noSrcDonutCatalog)

        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesRaw, np.ones(19) * np.nan
        )
        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesAvg, np.ones(19) * np.nan
        )
        self.assertEqual(len(testOutNoSrc.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra), 0)

        # Test no intra sources in catalog
        extraOnlyDonutCatalog = copy(donutCatalog)
        extraOnlyDonutCatalog["detector"] = "R00_SW0"
        testOutNoIntra = self.task.run(
            [exposureExtra, exposureIntra], extraOnlyDonutCatalog
        )

        np.testing.assert_array_equal(
            testOutNoIntra.outputZernikesRaw, np.ones(19) * np.nan
        )
        np.testing.assert_array_equal(
            testOutNoIntra.outputZernikesAvg, np.ones(19) * np.nan
        )
        self.assertEqual(len(testOutNoIntra.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoIntra.donutStampsIntra), 0)

        # Test no extra sources in catalog
        intraOnlyDonutCatalog = copy(donutCatalog)
        intraOnlyDonutCatalog["detector"] = "R00_SW1"
        testOutNoExtra = self.task.run(
            [exposureExtra, exposureIntra], intraOnlyDonutCatalog
        )

        np.testing.assert_array_equal(
            testOutNoExtra.outputZernikesRaw, np.ones(19) * np.nan
        )
        np.testing.assert_array_equal(
            testOutNoExtra.outputZernikesAvg, np.ones(19) * np.nan
        )
        self.assertEqual(len(testOutNoExtra.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoExtra.donutStampsIntra), 0)

    def testTaskRunNormal(self):

        exposureExtra, exposureIntra, donutCatalog = self._getDataFromButler()

        # Test normal behavior
        taskOut = self.task.run([exposureIntra, exposureExtra], donutCatalog)

        extraCatalog, intraCatalog = self.task.selectCwfsSources(
            donutCatalog, (4072, 2000)
        )
        testExtraStamps = self.task.cutOutStamps(
            exposureExtra, extraCatalog, DefocalType.Extra
        )
        testIntraStamps = self.task.cutOutStamps(
            exposureIntra, intraCatalog, DefocalType.Intra
        )

        for donutStamp, cutOutStamp in zip(taskOut.donutStampsExtra, testExtraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )
        for donutStamp, cutOutStamp in zip(taskOut.donutStampsIntra, testIntraStamps):
            self.assertMaskedImagesAlmostEqual(
                donutStamp.stamp_im, cutOutStamp.stamp_im
            )

        testCoeffsRaw = self.task.estimateZernikes(testExtraStamps, testIntraStamps)
        testCoeffsAvg = self.task.combineZernikes(testCoeffsRaw)
        np.testing.assert_array_equal(taskOut.outputZernikesRaw, testCoeffsRaw)
        np.testing.assert_array_equal(taskOut.outputZernikesAvg, testCoeffsAvg)