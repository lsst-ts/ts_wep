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

import astropy.units as u
import lsst.afw.image as afwImage
import numpy as np
from astropy.table import QTable
from lsst.daf import butler as dafButler
from lsst.pipe.base.task import TaskError
from lsst.ts.wep.task import (
    DonutStamps,
    GenerateDonutFromRefitWcsTask,
    GenerateDonutFromRefitWcsTaskConfig,
    RefCatalogInterface,
)
from lsst.ts.wep.utils import (
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
            "directDetectDonutTable",
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
        self.assertEqual(self.config.maxFitScatter, 1.0)
        self.assertEqual(self.config.edgeMargin, 80)
        self.assertEqual(self.config.astromTask.referenceSelector.magLimit.maximum, 15)
        self.assertEqual(self.config.astromRefFilter, "phot_g_mean")

        # Change up config
        self.config.astromTask.referenceSelector.magLimit.maximum = 18
        self.config.astromTask.matcher.maxOffsetPix = 2500
        self.config.astromRefFilter = "g"
        self.config.photoRefFilter = "g"
        self.config.edgeMargin = 400

        # Test new settings
        task = GenerateDonutFromRefitWcsTask(config=self.config)
        self.assertEqual(task.config.astromTask.referenceSelector.magLimit.maximum, 18)
        self.assertEqual(task.config.astromTask.matcher.maxOffsetPix, 2500)
        self.assertEqual(task.config.astromRefFilter, "g")
        self.assertEqual(task.config.photoRefFilter, "g")
        self.assertEqual(task.config.edgeMargin, 400)

    def testTaskFit(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        self.config.astromRefFilter = "g"
        self.config.photoRefFilter = "g"
        self.config.catalogFilterList = []
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
        fitPixelOrigin = np.array(fitWcs.getPixelOrigin())
        shiftedPixelOrigin = np.array(shiftedWcs.getPixelOrigin())
        pixelShift = fitPixelOrigin - shiftedPixelOrigin

        # Test the fit WCS shifts against the input
        self.assertAlmostEqual(
            truePixelShift * np.sqrt(2),
            np.sqrt(np.sum(pixelShift**2.0)),
            delta=1e-3,
        )

        # Test the catalog
        np.testing.assert_array_almost_equal(
            fitCatalog["centroid_x"] + truePixelShift,
            shiftedCatalog["centroid_x"],
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            fitCatalog["centroid_y"] + truePixelShift,
            shiftedCatalog["centroid_y"],
            decimal=3,
        )

        # Test that detector names are present
        self.assertEqual(np.unique(fitCatalog["detector"]), "R22_S11")

    def testConfigCatalogFilterListErr(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        self.config.astromRefFilter = "g"
        self.config.photoRefFilter = "g"
        self.config.catalogFilterList = ["lsst_u"]
        task = GenerateDonutFromRefitWcsTask(config=self.config)

        # Test that requesting a filter in the donut catalog
        # that isn't in the reference catalog raises an error
        errMsg = str(
            "Filter(s) {'lsst_u'} not in available columns in "
            "reference catalog. Check catalogFilterList config "
            "(currently set as ['lsst_u']). "
            "Available ref catalog filters are ['g']."
        )

        with self.assertRaises(TaskError) as context:
            task.run(dataRefs, copy(preFitExp_S11), directDetectCat, dataRefs)
        self.assertEqual(str(context.exception), errMsg)

    def testWcsFailure(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        # Set cutoff so there will be no sources and fit will fail
        self.config.astromTask.referenceSelector.magLimit.maximum = 8.0
        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromRefFilter = "g"
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

    def testRefCatalogFailureWithNoCatalogs(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        # Set up task
        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        self.config.astromRefFilter = "g"
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
        self.assertFalse(
            np.array_equal(
                np.array(fitWcs.getPixelOrigin()),
                np.array(preFitExp_S11.wcs.getPixelOrigin()),
            )
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

    def testRefCatalogFailureWithNonExistentPhotoRefFilter(self):
        preFitExp_S11, directDetectCat, dataRefs = self._getInputData()

        # Set up task
        self.config.astromTask.referenceSelector.magLimit.fluxField = "g_flux"
        self.config.astromTask.referenceSelector.magLimit.maximum = 16.0
        self.config.astromRefFilter = "g"
        self.config.photoRefFilter = "bad_g"
        task = GenerateDonutFromRefitWcsTask(config=self.config)

        # Shift the WCS by a known amount
        truePixelShift = 5
        directDetectCat["centroid_x"] += truePixelShift
        directDetectCat["centroid_y"] += truePixelShift

        # Give incomplete list of reference catalogs
        with self.assertLogs(self.logger.name, level="WARNING") as context:
            fitWcsOutput = task.run(
                dataRefs, copy(preFitExp_S11), directDetectCat, dataRefs
            )
        self.assertIn(
            "Photometric Reference Catalog does not contain photoRefFilter: bad_g",
            context.output[0],
        )
        self.assertIn(
            "Returning new WCS but original direct detect catalog as donutCatalog.",
            context.output[1],
        )
        fitWcs = fitWcsOutput.outputExposure.wcs
        fitCatalog = fitWcsOutput.donutCatalog

        # Test that WCS is different
        self.assertFalse(
            np.array_equal(
                np.array(fitWcs.getPixelOrigin()),
                np.array(preFitExp_S11.wcs.getPixelOrigin()),
            )
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
            "directDetectDonutTable",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        preFitExp_S11 = self.butler.get(
            "preFitPostISRCCD",
            dataId=self.dataIdExtra,
            collections=[f"{self.runName}"],
        )
        finalExp_S11 = self.butler.get(
            "post_isr_image",
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
        self.assertTrue(isinstance(directDetectCat, QTable))
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

    def testFormatDonutCatalog(self):
        """Test the formatDonutCatalog function that creates
        an intermediate afwTable from donut catalog."""

        # create a test donut catalog
        catalogDict = {
            "coord_ra": [6.28, 6.30] * u.rad,
            "coord_dec": [-0.01, 0.01] * u.rad,
            "centroid_x": [500, 1000],
            "centroid_y": [600, 800],
            "detector": ["R22_S11", "R22_S11"],
            "source_flux": [3.44e6, 3.36e5] * u.nJy,
        }
        # one catalog without coordinate error
        catalogNoErr = QTable(catalogDict)

        # one catalog with coordinate error
        catalogDictErr = copy(catalogDict)
        catalogDictErr["coord_raErr"] = [0.01, 0.02] * u.rad
        catalogDictErr["coord_decErr"] = [0.03, 0.04] * u.rad
        catalogWithErr = QTable(catalogDictErr)

        # Create an afwTable
        # from donut catalog without coordinate error present
        aft1 = self.task.formatDonutCatalog(catalogNoErr)
        aft1_table = aft1.asAstropy()

        # from donut catalog with coordinate error present
        aft2 = self.task.formatDonutCatalog(catalogWithErr)
        aft2_table = aft2.asAstropy()

        for c in ["ra", "dec"]:
            # check that before there is no error column
            # in the first catalog
            self.assertNotIn(f"coord_{c}Err", catalogNoErr.columns)

            # check that there is an error column
            # in the second catalog
            self.assertIn(f"coord_{c}Err", catalogWithErr.columns)

            # first, ensure that the coordinate values were
            # inherited from the original columns
            # this is true for both donut catalog
            # with or without coordinate error present
            np.testing.assert_array_almost_equal(
                catalogNoErr[f"coord_{c}"].value, aft1_table[f"coord_{c}"].data
            )
            np.testing.assert_array_almost_equal(
                catalogWithErr[f"coord_{c}"].value, aft2_table[f"coord_{c}"].data
            )

            # second, ensure that the coordinate error
            # values were correctly calculated
            # In the first case, the calculated error
            # should be at least
            # less than 5% of the coordinate value
            f = 0.05  # set percentage of coord_ra, coord_dec
            np.testing.assert_array_less(
                aft1_table[f"coord_{c}Err"].data,
                f * np.abs(catalogNoErr[f"coord_{c}"].value),
            )

            # In the second case, the inherited error should
            # be identical as in the
            # original donut catalog
            np.testing.assert_array_almost_equal(
                catalogWithErr[f"coord_{c}Err"].value, aft2_table[f"coord_{c}Err"].data
            )

        # check the values of source_flux  --> slot_ApFlux_instFlux
        # original donut catalog
        np.testing.assert_array_almost_equal(
            catalogNoErr["source_flux"].value, aft1_table["slot_ApFlux_instFlux"].data
        )
        np.testing.assert_array_almost_equal(
            catalogWithErr["source_flux"].value, aft2_table["slot_ApFlux_instFlux"].data
        )

        # check the calculation of the slot_ApFlux_instFluxErr
        # as 1% of the source_flux
        f = 0.05
        np.testing.assert_array_less(
            aft1_table["slot_ApFlux_instFluxErr"].data,
            f * np.abs(catalogNoErr["source_flux"].value),
        )

        np.testing.assert_array_less(
            aft2_table["slot_ApFlux_instFluxErr"].data,
            f * np.abs(catalogWithErr["source_flux"].value),
        )

        # check that indeed centroid_x, centroid_y translate to
        # slot_Centroid_x , slot_Centroid_y
        for c in ["x", "y"]:
            np.testing.assert_array_almost_equal(
                catalogNoErr[f"centroid_{c}"], aft1_table[f"slot_Centroid_{c}"].data
            )
            np.testing.assert_array_almost_equal(
                catalogWithErr[f"centroid_{c}"], aft2_table[f"slot_Centroid_{c}"].data
            )
