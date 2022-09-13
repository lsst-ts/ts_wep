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
import lsst.pipe.base as pipeBase
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

        # Point to the collections for the reference catalogs,
        # the raw images and the camera model in the calib directory
        # that comes from `butler write-curated-calibrations`.
        cls.collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        cls.instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        cls.pipelineYaml = os.path.join(testPipelineConfigDir, "testCwfsPipeline.yaml")

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

        self.config = EstimateZernikesCwfsTaskConfig()
        self.task = EstimateZernikesCwfsTask(config=self.config)

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

    def _generateTestExposures(self):

        # Generate donut template
        template = self.task.getTemplate(
            "R00_SW0", DefocalType.Extra, self.task.donutTemplateSize
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
            "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutCatalogIntra = self.butler.get(
            "donutCatalog", dataId=self.dataIdIntra, collections=[self.runName]
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

        self.config.donutTemplateSize = 120
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = EstimateZernikesCwfsTask(config=self.config)

        self.assertEqual(self.task.donutTemplateSize, 120)
        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testRunQuantum(self):
        # Set up test quantum from butler data
        inputRefs = pipeBase.InputQuantizedConnection()
        badInstrument = "LSSTComCam"
        inputRefs.exposures = [
            dafButler.DatasetRef(
                self.registry.getDatasetType("postISRCCD"),
                {
                    "instrument": badInstrument,
                    "detector": 191,
                    "exposure": 4021123106000,
                },
                id="3104de33-c107-4678-b07e-1fc62407a52e",
                run="run2",
            )
        ]
        inputRefs.camera = self.butler.getDeferred(
            "camera", instrument="LSSTComCam", collections="LSSTComCam/calib/unbounded"
        ).ref
        outputRefs = pipeBase.OutputQuantizedConnection()
        quantum = dafButler.Quantum(
            inputs={
                inputRefs.exposures[0].datasetType: inputRefs.exposures,
                inputRefs.camera.datasetType: [inputRefs.camera],
            }
        )
        butlerQC = pipeBase.ButlerQuantumContext(self.butler, quantum)

        # Test that we will get an error if we try to use an
        # unsupported instrument.
        errMsg = f"{badInstrument} is not a valid camera name."
        with self.assertRaises(ValueError, msg=errMsg) as context:
            self.task.runQuantum(butlerQC, inputRefs, outputRefs)
        self.assertEqual(
            f"{badInstrument} is not a valid camera name.",
            str(context.exception),
        )

        # Test error raised when list sizes are unequal
        inputRefs.exposures = [
            self.butler.getDeferred(
                "postISRCCD", dataId=self.dataIdExtra, collections=[self.runName]
            ).ref,
        ]
        inputRefs.camera = self.butler.getDeferred(
            "camera", instrument="LSSTCam", collections="LSSTCam/calib/unbounded"
        ).ref
        inputRefs.donutCatalog = [
            self.butler.getDeferred(
                "donutCatalog", dataId=self.dataIdExtra, collections=[self.runName]
            ).ref,
            self.butler.getDeferred(
                "donutCatalog", dataId=self.dataIdIntra, collections=[self.runName]
            ).ref,
        ]
        outputRefs = pipeBase.OutputQuantizedConnection()
        quantum = dafButler.Quantum(
            inputs={
                inputRefs.exposures[0].datasetType: inputRefs.exposures,
                inputRefs.donutCatalog[0].datasetType: inputRefs.donutCatalog,
                inputRefs.camera.datasetType: [inputRefs.camera],
            }
        )
        butlerQC = pipeBase.ButlerQuantumContext(self.butler, quantum)
        unequalMsg = "Unequal number of intra and extra focal detectors."
        with self.assertRaises(ValueError) as context:
            self.task.runQuantum(butlerQC, inputRefs, outputRefs)
        self.assertEqual(str(context.exception), unequalMsg)

        # Test error raised when extra and intra focal do not
        # have correct partner
        self.mismatchDataId = copy(self.dataIdIntra)
        self.mismatchDataId["detector"] = 196

        # Test errors raised
        inputRefs.exposures.append(
            self.butler.getDeferred(
                "postISRCCD", dataId=self.mismatchDataId, collections=[self.runName]
            ).ref
        )
        butlerQC = pipeBase.ButlerQuantumContext(self.butler, quantum)
        mismatchMsg = "Intra and extra focal detectors not adjacent."
        with self.assertRaises(ValueError) as context:
            self.task.runQuantum(butlerQC, inputRefs, outputRefs)
        self.assertEqual(str(context.exception), mismatchMsg)

    def testTaskRunNoSources(self):

        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test return values when no sources in catalog
        noSrcDonutCatalog = pd.DataFrame(columns=donutCatalogExtra.columns)
        testOutNoSrc = self.task.run(
            [exposureExtra, exposureIntra], [noSrcDonutCatalog] * 2, camera
        )

        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesRaw, np.ones(19) * np.nan
        )
        np.testing.assert_array_equal(
            testOutNoSrc.outputZernikesAvg, np.ones(19) * np.nan
        )
        self.assertEqual(len(testOutNoSrc.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra), 0)

        # Test no intra sources in catalog
        testOutNoIntra = self.task.run(
            [exposureExtra, exposureIntra],
            [
                donutCatalogExtra,
                pd.DataFrame(columns=donutCatalogExtra.columns),
            ],
            camera,
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
        testOutNoExtra = self.task.run(
            [exposureExtra, exposureIntra],
            [
                pd.DataFrame(columns=donutCatalogIntra.columns),
                donutCatalogIntra,
            ],
            camera,
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

        (
            exposureExtra,
            exposureIntra,
            donutCatalogExtra,
            donutCatalogIntra,
            camera,
        ) = self._getDataFromButler()

        # Test normal behavior
        taskOut = self.task.run(
            [exposureExtra, exposureIntra],
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

        testCoeffsRaw = self.task.estimateZernikes(testExtraStamps, testIntraStamps)
        testCoeffsAvg = self.task.combineZernikes.run(testCoeffsRaw)
        np.testing.assert_array_equal(taskOut.outputZernikesRaw, testCoeffsRaw)
        np.testing.assert_array_equal(
            taskOut.outputZernikesAvg, testCoeffsAvg.combinedZernikes
        )

    def testPipelineOnePairOnly(self):

        pipeCmd = writePipetaskCmd(
            self.repoDir,
            self.testRunName,
            self.instrument,
            self.collections,
            pipelineYaml=self.pipelineYaml,
        )
        pipeCmd += f" -d 'exposure IN ({self.visitNum}) and detector IN (191, 192)'"
        runProgram(pipeCmd)

        # Get Butler with updated registry
        self.butler = dafButler.Butler(self.repoDir)

        donutExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.testRunName]
        )
        donutIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdIntra, collections=[self.testRunName]
        )
        zernAvg = self.butler.get(
            "zernikeEstimateAvg",
            dataId=self.dataIdExtra,
            collections=[self.testRunName],
        )
        zernRaw = self.butler.get(
            "zernikeEstimateRaw",
            dataId=self.dataIdExtra,
            collections=[self.testRunName],
        )

        self.assertEqual(len(donutExtra), 2)
        self.assertEqual(len(donutExtra), len(donutIntra))
        self.assertEqual(np.shape(zernAvg), (19,))
        self.assertEqual(np.shape(zernRaw), (2, 19))

        self.badDataId = copy(self.dataIdExtra)
        self.badDataId["detector"] = 195
        with self.assertRaises(LookupError):
            self.butler.get(
                "donutStampsExtra",
                dataId=self.badDataId,
                collections=[self.testRunName],
            )
