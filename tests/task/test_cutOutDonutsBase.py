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
from lsst.daf.base import PropertySet
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
    DonutStamps,
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
        cls.cameraName = "LSSTCam"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if "pretest_run_science" in collectionsList:
            cls.runName = "pretest_run_science"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
            instrument = "lsst.obs.lsst.LsstCam"
            pipelineYaml = os.path.join(
                testPipelineConfigDir, "testBasePipeline.yaml"
            )

            pipeCmd = writePipetaskCmd(
                cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
            )
            # Make sure we are using the right exposure+detector combinations
            pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
            pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        if cls.runName == "run1":
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

        # Make two donuts centered along one axis in exposure
        initCutoutSize = len(template) + self.task.initialCutoutPadding * 2
        centeredArr = np.zeros((initCutoutSize, initCutoutSize), dtype=np.float32)
        centeredArr[
            self.task.initialCutoutPadding : -self.task.initialCutoutPadding,
            self.task.initialCutoutPadding : -self.task.initialCutoutPadding,
        ] += template
        centeredImage = afwImage.ImageF(initCutoutSize * 2, initCutoutSize)
        centeredImage.array = np.concatenate([centeredArr, centeredArr], axis=1)
        centeredExp = afwImage.ExposureF(initCutoutSize * 2, initCutoutSize)
        centeredExp.setImage(centeredImage)
        centerCoord_1 = (
            self.task.initialCutoutPadding + templateCenter[1],
            self.task.initialCutoutPadding + templateCenter[0],
        )
        centerCoord_2 = [centerCoord_1[0] + initCutoutSize, centerCoord_1[1]]
        centerCoordArr = np.array(
            [[centerCoord_1[0], centerCoord_2[0]], [centerCoord_1[1], centerCoord_2[1]]]
        )

        # Make new donut that needs to be shifted by 20 pixels
        # from the edge of the exposure. Then add centered donut to
        # the side
        offCenterArr = np.zeros((initCutoutSize, initCutoutSize), dtype=np.float32)
        offCenterArr[: len(template) - 20, : len(template) - 20] = template[20:, 20:]
        offCenterImage = afwImage.ImageF(initCutoutSize * 2, initCutoutSize)
        offCenterImage.array = np.concatenate([offCenterArr, centeredArr], axis=1)
        offCenterExp = afwImage.ExposureF(initCutoutSize * 2, initCutoutSize)
        offCenterExp.setImage(offCenterImage)
        # Center coord value 20 pixels closer than template center
        # due to stamp overrunning the edge of the exposure.
        offCenterCoord = templateCenter - 20
        offCenterCoordArr = np.array(
            [
                [offCenterCoord[0], centerCoordArr[0, 1]],
                [offCenterCoord[1], centerCoordArr[1, 1]],
            ]
        )

        return centeredExp, centerCoordArr, template, offCenterExp, offCenterCoordArr

    def testValidateConfigs(self):
        self.assertEqual(self.task.donutStampSize, 160)
        self.assertEqual(self.task.initialCutoutPadding, 5)
        self.assertEqual(self.task.opticalModel, "offAxis")
        self.assertEqual(self.task.instConfigFile, None)
        self.assertEqual(self.task.maxRecenterDistance, 20)

        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.config.opticalModel = "onAxis"
        self.config.maxRecenterDistance = 10
        self.task = CutOutDonutsBaseTask(config=self.config, name="Base Task")

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)
        self.assertEqual(self.task.opticalModel, "onAxis")
        self.assertEqual(self.task.maxRecenterDistance, 10)

    def testShiftCenters(self):
        centerArr = np.array([190.0, 10.0, 100.0, 100.0])
        centerUpperLimit = self.task.shiftCenters(centerArr, 200.0, 20.0)
        np.testing.assert_array_equal(centerUpperLimit, [180.0, 10.0, 100.0, 100.0])
        centerLowerLimit = self.task.shiftCenters(centerUpperLimit, 0.0, 20.0)
        np.testing.assert_array_equal(centerLowerLimit, [180.0, 20.0, 100.0, 100.0])
        centerNoChangeUpper = self.task.shiftCenters(centerLowerLimit, 200.0, 20.0)
        np.testing.assert_array_equal(centerNoChangeUpper, [180.0, 20.0, 100.0, 100.0])
        centerNoChangeLower = self.task.shiftCenters(centerNoChangeUpper, 200.0, 20.0)
        np.testing.assert_array_equal(centerNoChangeLower, [180.0, 20.0, 100.0, 100.0])

    def testCalculateFinalCentroids(self):
        (
            centeredExp,
            centerCoord,
            template,
            offCenterExp,
            offCenterCoord,
        ) = self._generateTestExposures()
        centerCoordX = centerCoord[0]
        centerCoordY = centerCoord[1]
        (
            centerX,
            centerY,
            cornerX,
            cornerY,
            initCornerX,
            initCornerY,
            peakHeight,
        ) = self.task.calculateFinalCentroids(
            centeredExp, template, centerCoordX, centerCoordY
        )
        # For centered donut final center and final corner should be
        # half stamp width apart
        np.testing.assert_array_equal(centerX, centerCoordX)
        np.testing.assert_array_equal(centerY, centerCoordY)
        np.testing.assert_array_equal(
            cornerX, centerCoordX - self.task.donutStampSize / 2
        )
        np.testing.assert_array_equal(
            cornerY, centerCoordY - self.task.donutStampSize / 2
        )
        np.testing.assert_array_equal(initCornerX, cornerX)
        np.testing.assert_array_equal(initCornerY, cornerY)

        # Also check the peak height for the centered exposure
        self.assertFloatsAlmostEqual(peakHeight, 9472.0, rtol=0, atol=3e-4)

    def testCalcFinalCentroidsOnEdge(self):
        (
            centeredExp,
            centerCoord,
            template,
            edgeExp,
            edgeCoord,
        ) = self._generateTestExposures()
        centerCoordX = centerCoord[0]
        centerCoordY = centerCoord[1]
        (
            centerX,
            centerY,
            cornerX,
            cornerY,
            initCornerX,
            initCornerY,
            peakHeight,
        ) = self.task.calculateFinalCentroids(
            edgeExp, template, centerCoordX, centerCoordY
        )
        # For donut stamp that would go off the top corner of the exposure
        # then the stamp should start at (0, 0) instead
        np.testing.assert_array_equal(centerX, edgeCoord[0])
        np.testing.assert_array_equal(centerY, edgeCoord[1])
        # Corner of image should be 0, 0
        np.testing.assert_array_equal(cornerX, np.array([0, initCornerX[1]]))
        np.testing.assert_array_equal(cornerY, np.array([0, initCornerY[1]]))
        # Also check the peak height for the off-center exposure
        self.assertFloatsAlmostEqual(peakHeight, [8944.0, 9472.0], rtol=0, atol=3e-4)

    def testMaxRecenter(self):
        maxRecenter = 5
        exp, catalog = self._getExpAndCatalog(DefocalType.Extra)
        # Shift image so that recentering will fail when cutting out
        # donuts at the original positions
        exp.image.array = np.roll(exp.image.array, maxRecenter * 2, axis=0)

        # Set first item in catalog to pass by adjusting catalog entries
        centroid_y_arr = catalog["centroid_y"].value
        centroid_y_arr[1] += maxRecenter * 2
        centroid_y_arr[2] -= maxRecenter * 2
        catalog["centroid_y"] = centroid_y_arr

        # Get original shifts
        donutStampsOrig = self.task.cutOutStamps(
            exp, catalog, DefocalType.Extra, self.cameraName
        )
        xShifts = []
        yShifts = []
        for stamp_cent, catalog_x, catalog_y in zip(
            donutStampsOrig.getCentroidPositions(),
            catalog["centroid_x"].value,
            catalog["centroid_y"].value,
        ):
            xShifts.append(stamp_cent.getX() - int(catalog_x))
            yShifts.append(stamp_cent.getY() - int(catalog_y))

        self.task.maxRecenterDistance = maxRecenter
        # Test that warnings logged due to recentering failures
        with self.assertLogs(logger=self.task.log.logger, level="WARNING") as cm:
            donutStamps = self.task.cutOutStamps(
                exp, catalog, DefocalType.Extra, self.cameraName
            )
        # Test that there are warnings for two objects
        # since first object should pass
        self.assertEqual(len(cm.output), 2)
        # All donuts except the first one should have unchanged values
        recenteringWarnings = [
            warn for warn in cm.output if "Donut Recentering " in warn
        ]
        for stamp, catRow, logMsg, xShift, yShift in zip(
            donutStamps[1:],
            catalog[1:],
            recenteringWarnings,
            xShifts[1:],
            yShifts[1:],
        ):
            self.assertEqual(stamp.centroid_position[0], int(catRow["centroid_x"]))
            self.assertEqual(stamp.centroid_position[1], int(catRow["centroid_y"]))
            errMsg = (
                "WARNING:lsst.Base Task:Donut Recentering Failed. "
                + "Flagging and not shifting center of stamp for extra-focal source"
                + f' at ({catRow["centroid_x"]}, {catRow["centroid_y"]}). '
                + f"Catalog row: {catRow.index+1}. "
                + f"Proposed Shift: ({int(xShift)}, {int(yShift)})."
            )
            self.assertEqual(logMsg, errMsg)
        self.assertEqual(self.task.metadata.arrays["recenterFlagsExtra"], [0, 1, 1])

        # Test that recenterFlags gets Intra focal label correct
        self.task.cutOutStamps(exp, catalog, DefocalType.Intra, self.cameraName)
        self.assertEqual(self.task.metadata.arrays["recenterFlagsIntra"], [0, 1, 1])

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
        astropy.table.QTable
            Donut Catalog for exposure
        """

        if defocalType is DefocalType.Extra:
            dataId = self.dataIdExtra
        else:
            dataId = self.dataIdIntra

        exposure = self.butler.get(
            "post_isr_image", dataId=dataId, collections=[self.runName]
        )
        donutCatalog = self.butler.get(
            "donutTable", dataId=dataId, collections=[self.runName]
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

    def testCutOutStampsTaskRunNormal(self):
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)
        donutStamps = self.task.cutOutStamps(
            exposure, donutCatalog, DefocalType.Extra, self.cameraName
        )
        self.assertTrue(len(donutStamps), 4)
        self.assertTrue(self.task.metadata.arrays["recenterFlagsExtra"], [0, 0, 0, 0])

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

        # Check that all expected metadata is present
        metadata = list(donutStamps.metadata)
        expectedMetadata = [
            "RA_DEG",
            "DEC_DEG",
            "DET_NAME",
            "CAM_NAME",
            "VISIT",
            "DFC_TYPE",
            "DFC_DIST",
            "MAG",
            "CENT_X0",
            "CENT_Y0",
            "CENT_X",
            "CENT_Y",
            "CENT_DX",
            "CENT_DY",
            "CENT_DR",
            "BLEND_CX",
            "BLEND_CY",
            "X0",
            "Y0",
            "SN",
            "SIGNAL_MEAN",
            "SIGNAL_SUM",
            "NPX_MASK",
            "BKGD_STDEV",
            "SQRT_MEAN_VAR",
            "BKGD_VAR",
            "BACKGROUND_IMAGE_MEAN",
            "NOISE_VAR_BKGD",
            "NOISE_VAR_DONUT",
            "EFFECTIVE",
            "ENTROPY",
            "RADIUS",
            "X_PIX_LEFT_EDGE",
            "X_PIX_RIGHT_EDGE",
            "RADIUS_FAIL_FLAG",
            "PEAK_HEIGHT",
            "FRAC_BAD_PIX",
            "MAX_POWER_GRAD",
            "MJD",
            "BORESIGHT_ROT_ANGLE_RAD",
            "BORESIGHT_PAR_ANGLE_RAD",
            "BORESIGHT_ALT_RAD",
            "BORESIGHT_AZ_RAD",
            "BORESIGHT_RA_RAD",
            "BORESIGHT_DEC_RAD",
            "BANDPASS",
        ]
        self.assertCountEqual(metadata, expectedMetadata)

        # test that the visit is properly stored
        self.assertEqual(
            self.dataIdExtra["visit"], donutStamps.metadata.getArray("VISIT")[0]
        )

        # test that each metric has been calculated
        # for all donuts
        for measure in [
            "SIGNAL_SUM",
            "SIGNAL_MEAN",
            "NOISE_VAR_BKGD",
            "NOISE_VAR_DONUT",
            "EFFECTIVE",
            "ENTROPY",
            "RADIUS",
            "X_PIX_LEFT_EDGE",
            "X_PIX_RIGHT_EDGE",
            "RADIUS_FAIL_FLAG",
            "PEAK_HEIGHT",
            "FRAC_BAD_PIX",
        ]:
            self.assertEqual(
                len(donutStamps), len(donutStamps.metadata.getArray(measure))
            )

        # test that the effectiveness contains only binary values
        unique_values = set(np.unique(donutStamps.metadata.getArray("EFFECTIVE")))
        allowed_values = set([0, 1])
        self.assertTrue(allowed_values.issuperset(unique_values))

        # test the calculation of SN
        sn_values = [2149.093503846915, 2160.668317852145, 2041.9917562475855]
        sn_calculated = donutStamps.metadata.getArray("SN")
        self.assertFloatsAlmostEqual(np.sort(np.array(sn_values)),
                                     np.sort(np.array(sn_calculated)), rtol=1e-3)

    def testFilterBadRecentering(self):
        maxRecenter = 25
        self.task.maxRecenterDistance = maxRecenter
        xShift = np.array([10, 10, 0])
        yShift = np.array([10, 10, 50])
        medX = np.median(xShift)
        medY = np.median(yShift)

        shiftFailureIdx = self.task.filterBadRecentering(xShift, yShift)
        np.testing.assert_array_equal(np.array([2]), shiftFailureIdx)

        # Test that median shifts are output to log
        with self.assertLogs(logger=self.task.log.logger, level="INFO") as cm:
            self.task.filterBadRecentering(xShift, yShift)
        infoMsg = f"INFO:lsst.Base Task:Median Recentering Shift: ({medX}, {medY})"
        self.assertEqual(infoMsg, cm.output[0])

        # Test that task metadata stores median shifts
        self.assertEqual(self.task.metadata.scalars["medianXShift"], medX)
        self.assertEqual(self.task.metadata.scalars["medianYShift"], medY)

    def testCalculateSNWithBlends(self):
        """Test that blends work with the masking routines used in calculateSN.
        We previously found that blends can end up in an infinite loop or
        result in nan values for SN."""

        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)
        stamps = self.task.cutOutStamps(
            exposure, donutCatalog, DefocalType.Extra, self.cameraName
        )
        stamp = stamps[0]
        orig_sn_dict = self.task.calculateSN(stamp)

        # Add blend to mask
        stamp.wep_im.blendOffsets = [[-50, -60]]
        stamp.makeMask(self.task.instConfigFile, self.task.opticalModel)
        sn_dict = self.task.calculateSN(stamp)
        for val in sn_dict.values():
            self.assertFalse(np.isnan(val))
        # Blended pixels should be excluded from the original donut mask now
        self.assertTrue(orig_sn_dict["n_px_mask"] > sn_dict["n_px_mask"])

    def testCalculateSNWithLargeMask(self):
        """Test that dilation correction loop runs and logs correctly."""

        self.task.bkgDilationIter = 100
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)
        with self.assertLogs(logger=self.task.log.logger, level="WARNING") as cm:
            self.task.cutOutStamps(
                exposure, donutCatalog, DefocalType.Extra, self.cameraName
            )
        infoMsg = (
            "WARNING:lsst.Base Task:Binary dilation of donut mask reached the edge "
        )
        infoMsg += "of the image; reducing the amount of donut mask dilation to 99"
        self.assertEqual(infoMsg, cm.output[0])

    def testBadPixelMaskDefinitions(self):
        # Load test data
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)

        # Flag donut pixels as bad
        self.config.badPixelMaskDefinitions = ["DONUT"]
        task = CutOutDonutsBaseTask(config=self.config, name="Flag donut pix as bad")
        donutStamps = task.cutOutStamps(
            exposure, donutCatalog, DefocalType.Extra, self.cameraName
        )

        # Check that all the stamps have "bad" pixels
        # (because we flagged donut pixels as bad)
        fracBadPix = np.asarray(donutStamps.metadata.getArray("FRAC_BAD_PIX"))
        self.assertTrue(np.all(fracBadPix > 0))

    def testAddVisitLevelMetadata(self):
        exposure, donutCatalog = self._getExpAndCatalog(DefocalType.Extra)

        donutStamps = DonutStamps([])
        self.assertEqual(donutStamps.metadata, None)

        donutStamps = self.task.addVisitLevelMetadata(
            exposure, donutStamps, donutCatalog, DefocalType.Extra
        )
        # Check that metadata is a PropertySet
        self.assertIsInstance(donutStamps.metadata, PropertySet)

        # Check metadata keys
        key_list = [
            "BANDPASS",
            "DFC_DIST",
            "DFC_TYPE",
            "DET_NAME",
            "CAM_NAME",
            "VISIT",
            "MJD",
            "BORESIGHT_ROT_ANGLE_RAD",
            "BORESIGHT_PAR_ANGLE_RAD",
            "BORESIGHT_ALT_RAD",
            "BORESIGHT_AZ_RAD",
            "BORESIGHT_RA_RAD",
            "BORESIGHT_DEC_RAD",
        ]
        self.assertCountEqual(key_list,list(donutStamps.metadata.keys()))

        # Test a few values
        self.assertEqual(
            donutStamps.metadata.get("BANDPASS"), exposure.filter.bandLabel
        )
        self.assertEqual(
            donutStamps.metadata.get("VISIT"), self.dataIdExtra["visit"]
        )
