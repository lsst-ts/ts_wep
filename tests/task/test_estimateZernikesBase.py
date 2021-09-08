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
from scipy.signal import correlate

import lsst.utils.tests
from lsst.afw import image as afwImage
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.EstimateZernikesBase import (
    EstimateZernikesBaseTask,
    EstimateZernikesBaseConfig,
)
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    DefocalType,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestEstimateZernikesBase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutCatalog needed for task.
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
        pipelineYaml = os.path.join(testPipelineConfigDir, "testBasePipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):

        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):

        self.config = EstimateZernikesBaseConfig()
        self.task = EstimateZernikesBaseTask(config=self.config, name="BaseTask")

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
        self.task = EstimateZernikesBaseTask(config=self.config)

        self.assertEqual(self.task.donutTemplateSize, 120)
        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testGetTemplate(self):

        extra_template = self.task.getTemplate("R22_S11", DefocalType.Extra)
        self.assertEqual(
            np.shape(extra_template),
            (self.config.donutTemplateSize, self.config.donutTemplateSize),
        )

        self.config.donutTemplateSize = 180
        self.task = EstimateZernikesBaseTask(config=self.config, name="BaseTask")
        intra_template = self.task.getTemplate("R22_S11", DefocalType.Intra)
        self.assertEqual(np.shape(intra_template), (180, 180))

    def testShiftCenter(self):

        centerUpperLimit = self.task.shiftCenter(190.0, 200.0, 20.0)
        self.assertEqual(centerUpperLimit, 180.0)
        centerLowerLimit = self.task.shiftCenter(10.0, 0.0, 20.0)
        self.assertEqual(centerLowerLimit, 20.0)
        centerNoChangeUpper = self.task.shiftCenter(100.0, 200.0, 20.0)
        self.assertEqual(centerNoChangeUpper, 100.0)
        centerNoChangeLower = self.task.shiftCenter(100.0, 200.0, 20.0)
        self.assertEqual(centerNoChangeLower, 100.0)

    def testCalculateFinalCentroid(self):

        (
            centeredExp,
            centerCoord,
            template,
            offCenterExp,
            offCenterCoord,
        ) = self._generateTestExposures()
        centerX, centerY, cornerX, cornerY = self.task.calculateFinalCentroid(
            centeredExp, template, centerCoord[0], centerCoord[1]
        )
        # For centered donut final center and final corner should be
        # half stamp width apart
        self.assertEqual(centerX, centerCoord[0])
        self.assertEqual(centerY, centerCoord[1])
        self.assertEqual(cornerX, centerCoord[0] - self.task.donutStampSize / 2)
        self.assertEqual(cornerY, centerCoord[1] - self.task.donutStampSize / 2)

        centerX, centerY, cornerX, cornerY = self.task.calculateFinalCentroid(
            offCenterExp, template, centerCoord[0], centerCoord[1]
        )
        # For donut stamp that would go off the top corner of the exposure
        # then the stamp should start at (0, 0) instead
        self.assertAlmostEqual(centerX, offCenterCoord[0])
        self.assertAlmostEqual(centerY, offCenterCoord[1])
        # Corner of image should be 0, 0
        self.assertEqual(cornerX, 0)
        self.assertEqual(cornerY, 0)

    def testCutOutStamps(self):

        exposure = self.butler.get(
            "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStamps = self.task.cutOutStamps(
            exposure, donutCatalog, DefocalType.Extra, self.cameraName
        )
        self.assertTrue(len(donutStamps), 4)

        stampCentroid = donutStamps[0].centroid_position
        stampBBox = lsst.geom.Box2I(
            lsst.geom.Point2I(stampCentroid.getX() - 80, stampCentroid.getY() - 80),
            lsst.geom.Extent2I(160),
        )
        expCutOut = exposure[stampBBox].image.array
        np.testing.assert_array_equal(donutStamps[0].stamp_im.image.array, expCutOut)

    def testEstimateZernikes(self):

        extraExposure = self.butler.get(
            "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
        )
        intraExposure = self.butler.get(
            "postISRCCD", dataId=self.dataIdIntra, collections=[self.runName]
        )
        donutCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsExtra = self.task.cutOutStamps(
            extraExposure, donutCatalog, DefocalType.Extra, self.cameraName
        )
        donutStampsIntra = self.task.cutOutStamps(
            intraExposure, donutCatalog, DefocalType.Intra, self.cameraName
        )

        zernCoeff = self.task.estimateZernikes(donutStampsExtra, donutStampsIntra)

        self.assertEqual(np.shape(zernCoeff), (len(donutStampsExtra), 19))

    def testCombineZernikes(self):

        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        np.testing.assert_array_equal(self.task.combineZernikes(testArr), np.ones(19))
