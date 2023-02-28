import os
import unittest
import numpy as np
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.GenerateDonutDirectDetectTask import (
    GenerateDonutDirectDetectTask,
    GenerateDonutDirectDetectTaskConfig,
)
from lsst.ts.wep.Utility import runProgram, writePipetaskCmd, writeCleanUpRepoCmd


class TestGenerateDonutDirectDetectTask(unittest.TestCase):
    def setUp(self):
        self.config = GenerateDonutDirectDetectTaskConfig()
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.centerRaft = ["R22_S10", "R22_S11"]

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

    def testValidateConfigs(self):
        # Test config in task
        self.config.opticalModel = "another"
        # Test config in measurement sub task
        self.config.measurementTask.nSigmaDetection = 5.0
        # Test config in source selector sub task
        self.config.donutSelector.useCustomMagLimit = True
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        self.assertEqual(self.task.config.opticalModel, "another")
        self.assertEqual(self.task.config.measurementTask.nSigmaDetection, 5.0)
        self.assertEqual(self.task.config.donutSelector.useCustomMagLimit, True)

    def testCreateInstDictFromConfig(self):
        self.config.instObscuration = 0.1
        self.config.instFocalLength = 10.0
        self.config.instApertureDiameter = 10.0
        self.config.instDefocalOffset = 0.01
        self.config.instPixelSize = 0.1
        task = GenerateDonutDirectDetectTask(config=self.config)

        testDict = {
            "obscuration": 0.1,
            "focalLength": 10.0,
            "apertureDiameter": 10.0,
            "offset": 0.01,
            "pixelSize": 0.1,
        }

        self.assertDictEqual(testDict, task.instParams)

    def testUpdateDonutCatalog(self):
        testDataId = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
        }
        testExposure = self.butler.get(
            "raw", dataId=testDataId, collections="LSSTCam/raw/all"
        )

        # setup a test donut catalog DataFrame
        x_center = np.arange(15)
        y_center = 0.5 * np.arange(15)
        flux = 10000.0 * np.ones(15)
        donutCat = pd.DataFrame(
            data=list(zip(x_center, y_center, x_center, y_center, flux)),
            columns=[
                "centroid_x",
                "centroid_y",
                "blend_centroid_x",
                "blend_centroid_y",
                "source_flux",
            ],
        )
        # update the donut catalog
        donutCatUpd = self.task.updateDonutCatalog(donutCat, testExposure)

        # check that all these new columns are present
        newColumns = [
            "centroid_y",
            "centroid_x",
            "detector",
            "coord_ra",
            "coord_dec",
            "source_flux",
            "blend_centroid_x",
            "blend_centroid_y",
        ]
        self.assertEqual(np.sum(np.in1d(newColumns, list(donutCatUpd.columns))), 8)

    def testPipeline(self):
        """
        Test that the task runs in a pipeline.
        """

        # Run pipeline command
        runName = "run1"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        exposureId = 4021123106001  # Exposure ID for test extra-focal image
        testPipelineConfigDir = os.path.join(self.testDataDir, "pipelineConfigs")
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testDonutDirectDetectPipeline.yaml"
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

        # Check 2 unblended sources in each detector
        # and 1 source much brighter than blend so is "isolated"
        self.assertEqual(len(donutCatDf_S11), 3)
        self.assertEqual(len(donutCatDf_S10), 3)

        # Test that brighter sources are at the top of the list
        np.testing.assert_array_equal(
            np.arange(3), np.argsort(donutCatDf_S10["source_flux"].values)[::-1]
        )
        np.testing.assert_array_equal(
            np.arange(3), np.argsort(donutCatDf_S11["source_flux"].values)[::-1]
        )

        # Check outputs are correct
        outputDf = pd.concat([donutCatDf_S11, donutCatDf_S10])
        self.assertEqual(len(outputDf), 6)
        self.assertCountEqual(
            outputDf.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "detector",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        tolerance = 15  # pixels

        result_y = np.sort(outputDf["centroid_y"].values)
        truth_y = np.sort(np.array([3196, 2198, 398, 2196, 3197, 398]))
        diff_y = np.sum(result_y - truth_y)

        result_x = np.sort(outputDf["centroid_x"].values)
        truth_x = np.sort(np.array([3815, 2814, 617, 2811, 3812, 614]))
        diff_x = np.sum(result_x - truth_x)

        self.assertLess(diff_x, tolerance)
        self.assertLess(diff_y, tolerance)

        # Clean up
        cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
        runProgram(cleanUpCmd)
