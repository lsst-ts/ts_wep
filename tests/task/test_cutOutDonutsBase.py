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
import numpy as np
from lsst.afw import image as afwImage
from lsst.daf import butler as dafButler
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
    createTemplateForDetector,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)
from scipy.signal import correlate


class TestCutOutDonutsBase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutCatalog needed for task.
        """

        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
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
        self.config = CutOutDonutsBaseTaskConfig()
        self.task = CutOutDonutsBaseTask(config=self.config, name="Base Task")

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
        camera = LsstCam.getCamera()
        template = createTemplateForDetector(camera.get("R22_S11"), "extra")
        template = np.pad(template, (self.task.donutStampSize - len(template)) // 2)
        correlatedImage = correlate(template, template)
        maxIdx = np.argmax(correlatedImage)
        maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))
        templateCenter = np.array(maxLoc) - len(template) / 2

        # Make donut centered in exposure
        initCutoutSize = len(template) + self.task.initialCutoutPadding * 2
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
        offCenterArr[: len(template) - 20, : len(template) - 20] = template[20:, 20:]
        offCenterImage = afwImage.ImageF(initCutoutSize, initCutoutSize)
        offCenterImage.array = offCenterArr
        offCenterExp = afwImage.ExposureF(initCutoutSize, initCutoutSize)
        offCenterExp.setImage(offCenterImage)
        # Center coord value 20 pixels closer than template center
        # due to stamp overrunning the edge of the exposure.
        offCenterCoord = templateCenter - 20

        return centeredExp, centerCoord, template, offCenterExp, offCenterCoord

    def testValidateConfigs(self):
        self.assertEqual(self.task.donutStampSize, 160)
        self.assertEqual(self.task.initialCutoutPadding, 5)
        self.assertEqual(self.task.opticalModel, "offAxis")
        self.assertEqual(self.task.instConfigFile, None)

        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.config.opticalModel = "onAxis"
        self.task = CutOutDonutsBaseTask(config=self.config, name="Base Task")

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)
        self.assertEqual(self.task.opticalModel, "onAxis")

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

    def _getExpAndCatalog(self, defocalType):
        """
        Helper function to get exposure and donutCatalog for
        testing cutOutStamps.

        Parameters
        ----------
        defocalType : lsst.ts.wep.utils.DefocalType
            Defocal type of stamp to cut out.

        Returns
        -------
        lsst.afw.image.Exposure
            Exposure related to given defocal type
        pandas.DataFrame
            Donut Catalog for exposure
        """

        if defocalType is DefocalType.Extra:
            dataId = self.dataIdExtra
        else:
            dataId = self.dataIdIntra

        exposure = self.butler.get(
            "postISRCCD", dataId=dataId, collections=[self.runName]
        )
        donutCatalog = self.butler.get(
            "donutCatalog", dataId=dataId, collections=[self.runName]
        )

        return exposure, donutCatalog

    def testBackgroundSubtractionApplied(self):
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)
        with self.assertRaises(KeyError):
            exposure.getMetadata()["BGMEAN"]
        self.task.cutOutStamps(
            exposure, donutCatalog, DefocalType.Extra, self.cameraName
        )
        # cutOutStamps runs background subtraction which is automatically
        # applied to the exposure. Thus, BGMEAN should now exist in the
        # exposure metadata.
        self.assertIsInstance(exposure.getMetadata()["BGMEAN"], float)

    def testCutOutStamps(self):
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)
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
        np.testing.assert_array_almost_equal(
            donutStamps[0].stamp_im.image.array, expCutOut
        )

        # Check that local linear WCS in archive element is consistent with the
        # original exposure WCS.
        exposure_wcs = exposure.wcs
        for stamp in donutStamps:
            stamp_wcs = stamp.getLinearWCS()
            pt = stamp.centroid_position
            for offset in [(0, 0), (10, -20), (-30, 40), (50, 60)]:
                pt_offset = pt + lsst.geom.Extent2D(*offset)
                self.assertFloatsAlmostEqual(
                    exposure_wcs.pixelToSky(pt_offset)
                    .separation(stamp_wcs.pixelToSky(pt_offset))
                    .asArcseconds(),
                    0.0,
                    rtol=0.0,
                    atol=10e-6,  # 10 microarcsecond accurate over stamp region
                )
