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

        self.config.donutTemplateSize = 123
        self.config.instName = "telescope"
        self.config.opticalModel = "another"
        self.config.removeBlends = False
        self.config.blendRadius = 456
        self.config.peakThreshold = 0.56
        self.config.binaryChoice = "centroid"
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        self.assertEqual(self.task.config.donutTemplateSize, 123)
        self.assertEqual(self.task.config.instName, "telescope")
        self.assertEqual(self.task.config.opticalModel, "another")
        self.assertEqual(self.task.config.removeBlends, False)
        self.assertEqual(self.task.config.blendRadius, 456)
        self.assertEqual(self.task.config.peakThreshold, 0.56)
        self.assertEqual(self.task.config.binaryChoice, "centroid")

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
        y_center = 0.5*np.arange(15)
        donutCat = pd.DataFrame(data=list(zip(x_center, y_center)),
                                columns=['x_center', 'y_center'])
        # update the donut catalog
        donutCatUpd = self.task.updateDonutCatalog(donutCat, testExposure)

        # check that all these new columns are present
        newColumns = ['centroid_y', 'centroid_x', 'detector', 'coord_ra', 'coord_dec']
        self.assertEqual(np.sum(np.in1d(newColumns, list(donutCatUpd.columns))), 5)

        # check that columns got transposed
        np.testing.assert_array_equal(donutCatUpd['centroid_x'].values, y_center)
        np.testing.assert_array_equal(donutCatUpd['centroid_y'].values, x_center)

    def testPipeline(self):
        """
        Test that the task runs in a pipeline.
        """

        # Run pipeline command
        runName = "run1"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats,LSSTCam/calib,LSSTCam/raw/all"
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
        self.assertEqual(len(donutCatDf_S11), 2)
        self.assertEqual(len(donutCatDf_S10), 2)

        # Check outputs are correct
        outputDf = pd.concat([donutCatDf_S11, donutCatDf_S10])
        self.assertEqual(len(outputDf), 4)
        self.assertCountEqual(
            outputDf.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "blended",
                "blended_with",
                "num_blended_neighbors",
                "detector"
            ],
        )
        self.assertCountEqual(
            [
                3196,
                2198,
                2190,
                3194
            ],
            outputDf["centroid_y"],
        )
        self.assertCountEqual(
            [
                3815,
                2814,
                2813,
                3812
            ],
            outputDf["centroid_x"],
        )

        # Clean up
        cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
        runProgram(cleanUpCmd)
