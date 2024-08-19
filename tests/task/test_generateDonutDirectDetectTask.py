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
from copy import copy

import lsst.afw.image as afwImage
import lsst.utils.tests
import numpy as np
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.generateDonutDirectDetectTask import (
    GenerateDonutDirectDetectTask,
    GenerateDonutDirectDetectTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestGenerateDonutDirectDetectTask(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run the pipeline only once since it takes a
        couple minutes with the ISR.
        """
        # Run pipeline command
        runName = "run1"
        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry

        # Check that run doesn't already exist due to previous improper cleanup
        collectionsList = list(registry.queryCollections())
        if runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, runName)
            runProgram(cleanUpCmd)

        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        cls.cameraName = "LSSTCam"

        exposureId = 4021123106001  # Exposure ID for test extra-focal image
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testDonutDirectDetectPipeline.yaml"
        )
        pipetaskCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Update task configuration to match pointing information
        pipetaskCmd += f" -d 'exposure IN ({exposureId})'"

        # Run pipeline task
        runProgram(pipetaskCmd)

    def setUp(self):
        self.config = GenerateDonutDirectDetectTaskConfig()
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.testDataIdS10 = {
            "instrument": "LSSTCam",
            "detector": 93,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.testDataIdS11 = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }

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

    def testUpdateDonutCatalog(self):

        testExposure = self.butler.get(
            "raw", dataId=self.testDataIdS10, collections=["LSSTCam/raw/all"]
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

    def testTaskRun(self):
        """
        Test that the task runs interactively.
        """

        # Test run with an empty exposure
        # Read a real exposure and substitute the
        # image component with the background noise
        exposure_S11 = self.butler.get(
            "postISRCCD",
            dataId=self.testDataIdS11,
            collections=[self.runName],
        )
        bkgnd = 100 * (
            np.random.random_sample(size=np.shape(exposure_S11.image.array)) - 0.5
        )
        image = afwImage.ImageF(exposure_S11.getBBox())
        image.array[:] = bkgnd
        maskedImage = afwImage.MaskedImageF(image)
        exposure_noSrc = copy(exposure_S11)
        exposure_noSrc.setMaskedImage(maskedImage)

        # Run task on empty exposure
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        taskOutNoSrc = self.task.run(
            exposure_noSrc,
            camera,
        )

        # Test that there are no rows, but all columns are present
        self.assertEqual(len(taskOutNoSrc.donutCatalog), 0)

        expected_columns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
            "blend_centroid_x",
            "blend_centroid_y",
        ]
        self.assertCountEqual(taskOutNoSrc.donutCatalog.columns, expected_columns)

        # Run detection with different sources in each exposure
        exposure_S10 = self.butler.get(
            "postISRCCD",
            dataId=self.testDataIdS10,
            collections=[self.runName],
        )
        taskOut_S11 = self.task.run(
            exposure_S11,
            camera,
        )
        taskOut_S10 = self.task.run(
            exposure_S10,
            camera,
        )

        # Test that the length of catalogs is as expected
        outputDf = pd.concat([taskOut_S11.donutCatalog, taskOut_S10.donutCatalog])
        self.assertEqual(len(outputDf), 6)

        # Test that the interactive output is as expected
        self.assertCountEqual(
            outputDf.columns,
            expected_columns,
        )
        # Test against truth the centroid for all sources
        tolerance = 15  # pixels

        result_y = np.sort(outputDf["centroid_y"].values)
        truth_y = np.sort(np.array([3196, 2198, 398, 2196, 3197, 398]))
        diff_y = np.sum(result_y - truth_y)

        result_x = np.sort(outputDf["centroid_x"].values)
        truth_x = np.sort(np.array([3815, 2814, 617, 2811, 3812, 614]))
        diff_x = np.sum(result_x - truth_x)

        self.assertLess(diff_x, tolerance)
        self.assertLess(diff_y, tolerance)

    def testTaskRunPipeline(self):
        """
        Test that the task runs in a pipeline.
        """
        # Test instrument matches
        donutCatDf_S11 = self.butler.get(
            "donutCatalog",
            dataId=self.testDataIdS11,
            collections=[self.runName],
        )
        donutCatDf_S10 = self.butler.get(
            "donutCatalog",
            dataId=self.testDataIdS10,
            collections=[self.runName],
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

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
