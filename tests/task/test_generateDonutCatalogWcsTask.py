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
import unittest
import numpy as np
import pandas as pd

import lsst.geom
from lsst.daf import butler as dafButler
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.GenerateDonutCatalogWcsTask import (
    GenerateDonutCatalogWcsTask,
    GenerateDonutCatalogWcsTaskConfig,
)
from lsst.ts.wep.Utility import runProgram, writePipetaskCmd, writeCleanUpRepoCmd


class TestGenerateDonutCatalogWcsTask(unittest.TestCase):
    def setUp(self):
        self.config = GenerateDonutCatalogWcsTaskConfig()
        self.config.donutSelector.unblendedSeparation = 1
        self.task = GenerateDonutCatalogWcsTask(config=self.config)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.centerRaft = ["R22_S10", "R22_S11"]

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

    def _getRefCat(self):
        refCatList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            refCatList.append(
                self.butler.getDeferred(ref, collections=["refcats/gen2"])
            )

        return refCatList

    def _createTestDonutCat(self):
        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        # Check that our refObjLoader loads the available objects
        # within a given footprint from a sample exposure
        testDataId = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
        }
        testExposure = self.butler.get(
            "raw", dataId=testDataId, collections="LSSTCam/raw/all"
        )
        # From the test data provided this will create
        # a catalog of 4 objects.
        donutCatSmall = refObjLoader.loadPixelBox(
            testExposure.getBBox(),
            testExposure.getWcs(),
            testExposure.filter.bandLabel,
        )

        return donutCatSmall.refCat

    def testValidateConfigs(self):
        self.config.doDonutSelection = False
        self.task = GenerateDonutCatalogWcsTask(config=self.config)

        self.assertEqual(self.task.config.doDonutSelection, False)

    def testGetRefObjLoader(self):
        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        # Check that our refObjLoader loads the available objects
        # within a given search radius
        donutCatSmall = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(0.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatSmall.refCat), 8)

        donutCatFull = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(2.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatFull.refCat), 24)

    def testRunSelection(self):
        refCatList = self._getRefCat()
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        detector = camera["R22_S11"]

        self.config.donutSelector.useCustomMagLimit = True
        self.config.donutSelector.magMax = 17.0
        self.config.doDonutSelection = False

        self.task = GenerateDonutCatalogWcsTask(config=self.config, name="Base Task")
        refObjLoader = self.task.getRefObjLoader(refCatList)
        wcs = createInitialSkyWcsFromBoresight(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            90.0 * lsst.geom.degrees,
            detector,
            flipX=False,
        )
        # If we have a magLimit at 17 we should cut out
        # the one source at 17.5.
        donutCatBrighterThan17 = self.task.runSelection(
            refObjLoader, detector, wcs, "g"
        )
        self.assertEqual(len(donutCatBrighterThan17), 3)

        # If we increase the mag limit to 18 we should
        # get all the sources in the catalog.
        self.config.donutSelector.magMax = 18.0
        self.task = GenerateDonutCatalogWcsTask(config=self.config, name="Base Task")
        refObjLoader = self.task.getRefObjLoader(refCatList)
        donutCatFull, blendX, blendY = self.task.runSelection(
            refObjLoader, detector, wcs, "g"
        )
        self.assertEqual(len(donutCatFull), 4)

    def testDonutCatalogToDataFrame(self):
        donutCatSmall = self._createTestDonutCat()

        fieldObjects = self.task.donutCatalogToDataFrame(donutCatSmall, "g")
        self.assertEqual(len(fieldObjects), 4)
        self.assertCountEqual(
            fieldObjects.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        # Test that None returns an empty dataframe
        fieldObjectsNone = self.task.donutCatalogToDataFrame()
        self.assertEqual(len(fieldObjectsNone), 0)
        self.assertCountEqual(
            fieldObjects.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        # Test that blendCentersX and blendCentersY
        # get assigned correctly.
        fieldObjectsBlends = self.task.donutCatalogToDataFrame(
            donutCatSmall,
            "g",
        )
        fieldObjectsBlends.at[1, "blend_centroid_x"].append(5)
        fieldObjectsBlends.at[1, "blend_centroid_y"].append(3)
        np.testing.assert_array_equal(
            fieldObjectsBlends["blend_centroid_x"], [[], [5], [], []]
        )
        np.testing.assert_array_equal(
            fieldObjectsBlends["blend_centroid_y"], [[], [3], [], []]
        )

    def testDonutCatalogToDataFrameErrors(self):
        columnList = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "g_flux",
            "blend_centroid_x",
            "blend_centroid_y",
        ]
        donutCatZero = pd.DataFrame([], columns=columnList)

        # Test donutCatalog supplied but no filterName
        filterErrMsg = "If donutCatalog is not None then filterName cannot be None."
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(donutCatZero)
        self.assertTrue(filterErrMsg in str(context.exception))

        # Test blendCenters are both supplied or both left as None
        blendErrMsg = (
            "blendCentersX and blendCentersY must be"
            + " both be None or both be a list."
        )
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(donutCatZero, "g", blendCentersX=[])
        self.assertTrue(blendErrMsg in str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(donutCatZero, "g", blendCentersY=[])
        self.assertTrue(blendErrMsg in str(context.exception))

        # Test blendCenters must be same length as donutCat
        lengthErrMsg = (
            "blendCentersX and blendCentersY must be"
            + " both be None or both be a list."
        )
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(donutCatZero, "g", blendCentersX=[[], []])
        self.assertTrue(lengthErrMsg in str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(donutCatZero, "g", blendCentersY=[[], []])
        self.assertTrue(lengthErrMsg in str(context.exception))

        donutCatSmall = self._createTestDonutCat()

        # Test that each list within blendCentersX
        # has the same length as the list at the same
        # index within blendCentersY.
        xyMismatchErrMsg = (
            "Each list in blendCentersX must have the same "
            + "length as the list in blendCentersY at the "
            + "same index."
        )
        with self.assertRaises(ValueError) as context:
            self.task.donutCatalogToDataFrame(
                donutCatSmall,
                "g",
                blendCentersX=[[1], [0], [], []],
                blendCentersY=[[4], [], [], []],
            )
        self.assertTrue(xyMismatchErrMsg in str(context.exception))

    def testPipeline(self):
        """
        Test that the task runs in a pipeline. Also functions as a test of
        runQuantum function.
        """

        # Run pipeline command
        runName = "run1"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        exposureId = 4021123106001  # Exposure ID for test extra-focal image
        testPipelineConfigDir = os.path.join(self.testDataDir, "pipelineConfigs")
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testDonutCatWcsPipeline.yaml"
        )
        pipetaskCmd = writePipetaskCmd(
            self.repoDir, runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Update task configuration to match pointing information
        pipetaskCmd += f" -d 'exposure IN ({exposureId})'"

        # Check that run doesn't already exist due to previous improper cleanup
        collectionsList = list(self.registry.queryCollections())
        if runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
            runProgram(cleanUpCmd)

        # Run pipeline task
        runProgram(pipetaskCmd)

        # Test instrument matches
        pipelineButler = dafButler.Butler(self.repoDir)
        donutCatDf_S11 = pipelineButler.get(
            "donutCatalog",
            dataId={"instrument": "LSSTCam", "detector": 94, "visit": exposureId},
            collections=[f"{runName}"],
        )
        donutCatDf_S10 = pipelineButler.get(
            "donutCatalog",
            dataId={"instrument": "LSSTCam", "detector": 93, "visit": exposureId},
            collections=[f"{runName}"],
        )

        # Check 4 sources in each detector
        self.assertEqual(len(donutCatDf_S11), 4)
        self.assertEqual(len(donutCatDf_S10), 4)

        # Check outputs are correct
        outputDf = pd.concat([donutCatDf_S11, donutCatDf_S10])
        self.assertEqual(len(outputDf), 8)
        self.assertCountEqual(
            outputDf.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )
        self.assertCountEqual(
            [
                3806.7636478057957,
                2806.982895217227,
                607.3861483168994,
                707.3972344551466,
                614.607342274194,
                714.6336433247832,
                3815.2649173460436,
                2815.0561553920156,
            ],
            outputDf["centroid_x"],
        )
        self.assertCountEqual(
            [
                3196.070534224157,
                2195.666002294077,
                394.8907003737886,
                394.9087004171349,
                396.2407036464963,
                396.22270360324296,
                3196.1965343932648,
                2196.188002312585,
            ],
            outputDf["centroid_y"],
        )
        fluxTruth = np.ones(8)
        fluxTruth[:6] = 3630780.5477010026
        fluxTruth[6:] = 363078.0547701003
        self.assertCountEqual(outputDf["source_flux"], fluxTruth)

        # Clean up
        cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
        runProgram(cleanUpCmd)

    def testDonutCatalogGeneration(self):
        """
        Test that task creates a dataframe with detector information.
        """

        # Create list of deferred loaders for the ref cat
        deferredList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            deferredList.append(
                self.butler.getDeferred(ref, collections=["refcats/gen2"])
            )
        expGenerator = self.registry.queryDatasets(
            datasetType="raw",
            collections=["LSSTCam/raw/all"],
            dimensions=["exposure", "instrument"],
            dataId={"exposure": 4021123106001, "instrument": "LSSTCam"},
        ).expanded()
        expList = []
        for expRef in expGenerator:
            expList.append(
                self.butler.get(
                    "raw", dataId=expRef.dataId, collections=["LSSTCam/raw/all"]
                )
            )

        # run task on all exposures
        donutCatDfList = []
        # Set task to take all donuts regardless of magnitude
        self.task.config.donutSelector.useCustomMagLimit = True
        for exposure in expList:
            taskOutput = self.task.run(deferredList, exposure)
            self.assertEqual(len(taskOutput.donutCatalog), 4)
            donutCatDfList.append(taskOutput.donutCatalog)

        # concatenate catalogs from each exposure into a single catalog
        # to compare against the test input reference catalog
        outputDf = donutCatDfList[0]
        for donutCat in donutCatDfList[1:]:
            outputDf = pd.concat([outputDf, donutCat])

        # Compare ra, dec info to original input catalog
        inputCat = np.genfromtxt(
            os.path.join(
                self.testDataDir, "phosimOutput", "realComCam", "skyComCamInfo.txt"
            ),
            names=["id", "ra", "dec", "mag"],
        )

        self.assertEqual(len(outputDf), 8)
        self.assertCountEqual(np.radians(inputCat["ra"]), outputDf["coord_ra"])
        self.assertCountEqual(np.radians(inputCat["dec"]), outputDf["coord_dec"])
        self.assertCountEqual(
            [
                3806.7636478057957,
                2806.982895217227,
                607.3861483168994,
                707.3972344551466,
                614.607342274194,
                714.6336433247832,
                3815.2649173460436,
                2815.0561553920156,
            ],
            outputDf["centroid_x"],
        )
        self.assertCountEqual(
            [
                3196.070534224157,
                2195.666002294077,
                394.8907003737886,
                394.9087004171349,
                396.2407036464963,
                396.22270360324296,
                3196.1965343932648,
                2196.188002312585,
            ],
            outputDf["centroid_y"],
        )
        fluxTruth = np.ones(8)
        fluxTruth[:6] = 3630780.5477010026
        fluxTruth[6:] = 363078.0547701003
        self.assertCountEqual(outputDf["source_flux"], fluxTruth)
