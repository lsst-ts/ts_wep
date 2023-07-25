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

import logging
import os
import unittest
from copy import copy

import lsst.afw.image as afwImage
import numpy as np
import pandas
from lsst.daf import butler as dafButler
from lsst.ts.wep.task import (
    DonutStamps,
    GenerateDonutFromRefitWcsTask,
    GenerateDonutFromRefitWcsTaskConfig,
    RefCatalogInterface,
)
from lsst.ts.wep.utility import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestGenerateDonutFromRefitWcsTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run the pipeline once so we can test outputs in
        multiple tests.
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

        collections = "refcats/largeCatSingleCCD,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testDonutFromRefitWcsPipeline.yaml"
        )
        exposureId = 4021123106008  # Exposure ID for test extra-focal image

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Update task configuration to match pointing information
        pipeCmd += f" -d 'exposure IN ({exposureId}, {exposureId+1})'"
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):
        self.config = GenerateDonutFromRefitWcsTaskConfig()
        self.task = GenerateDonutFromRefitWcsTask(config=self.config)
        self.logger = logging.getLogger("lsst.generateDonutFromRefitWcsTask")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106008,
            "visit": 4021123106008,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106009,
            "visit": 4021123106009,
        }

    def _getInputData(self):
        preFitExp_S11 = self.butler.get(
            "preFitPostISRCCD",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        directDetectCat = self.butler.get(
            "directDetectDonutCatalog",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )

        # Initialize with pointing information
        info = preFitExp_S11.getInfo()
        visitInfo = info.getVisitInfo()
        boresightRa, boresightDec = visitInfo.boresightRaDec
        boresightRotAng = visitInfo.boresightRotAngle

        refCatInterface = RefCatalogInterface(
            boresightRa.asDegrees(),
            boresightDec.asDegrees(),
            boresightRotAng.asDegrees(),
        )
        htmIds = refCatInterface.getHtmIds()
        # Use the shardIds found above to get the locations (`dataRefs`)
        # in the butler repo for the catalog pieces we want
        catalogName = "cal_ref_cat"
        collections = ["refcats/largeCatSingleCCD"]
        dataRefs, dataIds = refCatInterface.getDataRefs(
            htmIds, self.butler, catalogName, collections
        )

        return preFitExp_S11, directDetectCat, dataRefs

    def testValidateConfigs(self):
        # Test some defaults
        self.assertEqual(self.config.astromTask.referenceSelector.magLimit.maximum, 15)
        self.assertEqual(self.config.astromTask.matcher.maxOffsetPix, 3000)

        # Change up config
        self.config.astromTask.referenceSelector.magLimit.maximum = 18
        self.config.astromTask.matcher.maxOffsetPix = 2500

        # Test new settings
        task = GenerateDonutFromRefitWcsTask(config=self.config)
        self.assertEqual(task.config.astromTask.referenceSelector.magLimit.maximum, 18)
        self.assertEqual(task.config.astromTask.matcher.maxOffsetPix, 2500)

    def testTaskFit(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        task = GenerateDonutFromRefitWcsTask(config=self.config)

        # Fit the WCS with the task since Phosim fit will be off
        fitWcsOutput = task.run(
            dataRefs, copy(preFitExp_S11), directDetectCat, dataRefs
        )
        fitWcs = fitWcsOutput.outputExposure.wcs
        fitCatalog = fitWcsOutput.donutCatalog

        # Shift the WCS by a known amount
        truePixelShift = 5
        directDetectCat["centroid_x"] += truePixelShift
        directDetectCat["centroid_y"] += truePixelShift
        shiftedOutput = task.run(
            dataRefs, copy(preFitExp_S11), directDetectCat, dataRefs
        )
        shiftedWcs = shiftedOutput.outputExposure.wcs
        shiftedCatalog = shiftedOutput.donutCatalog

        # Get the shifts out of the new WCS
        fitRa = fitWcs.getSkyOrigin().getRa().asArcseconds()
        shiftedRa = shiftedWcs.getSkyOrigin().getRa().asArcseconds()
        raPixelShift = (fitRa - shiftedRa) / shiftedWcs.getPixelScale().asArcseconds()

        fitDec = fitWcs.getSkyOrigin().getDec().asArcseconds()
        shiftedDec = shiftedWcs.getSkyOrigin().getDec().asArcseconds()
        decPixelShift = (
            fitDec - shiftedDec
        ) / shiftedWcs.getPixelScale().asArcseconds()

        # Test the fit WCS shifts against the input
        self.assertAlmostEqual(
            truePixelShift * np.sqrt(2),
            np.sqrt(raPixelShift**2.0 + decPixelShift**2.0),
            delta=1e-3,
        )

        # Test the catalog
        np.testing.assert_array_almost_equal(
            fitCatalog["centroid_x"] + 5, shiftedCatalog["centroid_x"], decimal=3
        )
        np.testing.assert_array_almost_equal(
            fitCatalog["centroid_y"] + 5, shiftedCatalog["centroid_y"], decimal=3
        )

    def testWcsFailure(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        # Set cutoff so there will be no sources and fit will fail
        self.config.astromTask.referenceSelector.magLimit.maximum = 8.0
        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        task = GenerateDonutFromRefitWcsTask(config=self.config)

        # Fit the WCS with the task since Phosim fit will be off
        with self.assertLogs(self.logger.name, level="WARNING") as context:
            fitWcsOutput = task.run(
                dataRefs, copy(preFitExp_S11), directDetectCat, dataRefs
            )
        self.assertIn("Solving for WCS failed", context[1][1])
        self.assertIn(
            "Returning original exposure and WCS and direct detect catalog as output.",
            context[1][2],
        )
        fitWcs = fitWcsOutput.outputExposure.wcs
        fitCatalog = fitWcsOutput.donutCatalog

        # Test the WCS is the same
        np.testing.assert_array_equal(
            fitWcs.getCdMatrix(), preFitExp_S11.wcs.getCdMatrix()
        )
        self.assertEqual(
            fitWcs.getSkyOrigin().getRa().asDegrees(),
            preFitExp_S11.wcs.getSkyOrigin().getRa().asDegrees(),
        )
        self.assertEqual(
            fitWcs.getSkyOrigin().getDec().asDegrees(),
            preFitExp_S11.wcs.getSkyOrigin().getDec().asDegrees(),
        )
        # Test that catalog is the same
        np.testing.assert_array_equal(
            fitCatalog["centroid_x"], directDetectCat["centroid_x"]
        )
        np.testing.assert_array_equal(
            fitCatalog["centroid_y"], directDetectCat["centroid_y"]
        )
        np.testing.assert_array_equal(
            fitCatalog["coord_ra"], directDetectCat["coord_ra"]
        )
        np.testing.assert_array_equal(
            fitCatalog["coord_dec"], directDetectCat["coord_dec"]
        )
        # Test metadata flag
        self.assertFalse(task.metadata["wcsFitSuccess"])
        self.assertFalse(task.metadata["refCatalogSuccess"])

    def testRefCatalogFailure(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        # Set up task
        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        task = GenerateDonutFromRefitWcsTask(config=self.config)

        # Shift the WCS by a known amount
        truePixelShift = 5
        directDetectCat["centroid_x"] += truePixelShift
        directDetectCat["centroid_y"] += truePixelShift

        # Give incomplete list of reference catalogs
        with self.assertLogs(self.logger.name, level="WARNING") as context:
            fitWcsOutput = task.run(dataRefs, copy(preFitExp_S11), directDetectCat, [])
        self.assertIn(
            "No catalogs cover this detector.",
            context.output[0],
        )
        self.assertIn(
            "Returning new WCS but original direct detect catalog as donutCatalog.",
            context.output[1],
        )
        fitWcs = fitWcsOutput.outputExposure.wcs
        fitCatalog = fitWcsOutput.donutCatalog

        # Test that WCS is different
        self.assertNotAlmostEqual(
            fitWcs.getSkyOrigin().getRa().asDegrees(),
            preFitExp_S11.wcs.getSkyOrigin().getRa().asDegrees(),
        )
        self.assertNotAlmostEqual(
            fitWcs.getSkyOrigin().getDec().asDegrees(),
            preFitExp_S11.wcs.getSkyOrigin().getDec().asDegrees(),
        )
        # But that catalog is the same
        np.testing.assert_array_equal(
            fitCatalog["centroid_x"], directDetectCat["centroid_x"]
        )
        np.testing.assert_array_equal(
            fitCatalog["centroid_y"], directDetectCat["centroid_y"]
        )
        np.testing.assert_array_equal(
            fitCatalog["coord_ra"], directDetectCat["coord_ra"]
        )
        np.testing.assert_array_equal(
            fitCatalog["coord_dec"], directDetectCat["coord_dec"]
        )
        # Test metadata flags
        self.assertTrue(task.metadata["wcsFitSuccess"])
        self.assertFalse(task.metadata["refCatalogSuccess"])

    def testPipelineOutputsInButler(self):
        """Verify that outputs with given names are stored in butler."""

        directDetectCat = self.butler.get(
            "directDetectDonutCatalog",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        preFitExp_S11 = self.butler.get(
            "preFitPostISRCCD",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        finalExp_S11 = self.butler.get(
            "postISRCCD",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        donutStampsExtra_S11 = self.butler.get(
            "donutStampsExtra",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        donutStampsIntra_S11 = self.butler.get(
            "donutStampsIntra",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        zernOutAvg_S11 = self.butler.get(
            "zernikeEstimateAvg",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        zernOutRaw_S11 = self.butler.get(
            "zernikeEstimateRaw",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        self.assertTrue(isinstance(directDetectCat, pandas.DataFrame))
        self.assertTrue(isinstance(preFitExp_S11, afwImage.ExposureF))
        self.assertTrue(isinstance(finalExp_S11, afwImage.ExposureF))
        self.assertTrue(isinstance(donutStampsExtra_S11, DonutStamps))
        self.assertTrue(isinstance(donutStampsIntra_S11, DonutStamps))
        self.assertTrue(isinstance(zernOutAvg_S11, np.ndarray))
        self.assertTrue(isinstance(zernOutRaw_S11, np.ndarray))

    def testMetadataFlags(self):
        """Test that flags for successful WCS fit and successful
        donut catalog generation from reference catalog are recorded."""
        wcsTaskMetadata = self.butler.get(
            "generateDonutFromRefitWcsTask_metadata",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        # This should be true when the pipeline is successful.
        self.assertTrue(
            wcsTaskMetadata["generateDonutFromRefitWcsTask"]["wcsFitSuccess"]
        )

        # This should be true when the pipeline is successful.
        self.assertTrue(
            wcsTaskMetadata["generateDonutFromRefitWcsTask"]["refCatalogSuccess"]
        )
