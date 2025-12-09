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

import numpy as np
from astropy.table import QTable, vstack

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.daf.butler import Butler
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
    runName: str
    testDataDir: str
    repoDir: str
    baseRunName: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run the pipeline only once since it takes a
        couple minutes with the ISR.
        """
        # Run pipeline command
        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")

        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry

        # Check that run doesn't already exist due to previous improper cleanup
        collectionsList = list(registry.queryCollections())
        cls.runName = "run1"
        cls.baseRunName = "run1"
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testDonutDirectDetectPipeline.yaml")

        if "pretest_run_science" in collectionsList:
            pipelineYaml += "#generateDonutDirectDetectTask"
            collections += ",pretest_run_science"
            cls.baseRunName = "pretest_run_science"

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Make sure we are using the right exposure+detector combinations
        pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = GenerateDonutDirectDetectTaskConfig()
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
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
        # Run task on empty exposure
        self.camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )

    def testValidateConfigs(self) -> None:
        # Test config in task
        self.config.opticalModel = "another"
        self.config.edgeMargin = 100
        # Test config in measurement sub task
        self.config.measurementTask.nSigmaDetection = 5.0
        # Test config in source selector sub task
        self.config.donutSelector.useCustomMagLimit = True
        self.task = GenerateDonutDirectDetectTask(config=self.config)

        self.assertEqual(self.task.config.opticalModel, "another")
        self.assertEqual(self.task.config.edgeMargin, 100)
        self.assertEqual(self.task.config.measurementTask.nSigmaDetection, 5.0)
        self.assertEqual(self.task.config.donutSelector.useCustomMagLimit, True)

    def testBackgroundSubtractionApplied(self) -> None:
        testExposure = self.butler.get("raw", dataId=self.testDataIdS10, collections=["LSSTCam/raw/all"])
        with self.assertRaises(KeyError):
            testExposure.getMetadata()["BGMEAN"]
        self.task.run(testExposure, self.camera)
        # background subtraction is automatically
        # applied to the exposure. Thus, BGMEAN should now exist in the
        # exposure metadata.
        self.assertIsInstance(testExposure.getMetadata()["BGMEAN"], float)

    def testUpdateDonutCatalog(self) -> None:
        testExposure = self.butler.get("raw", dataId=self.testDataIdS10, collections=["LSSTCam/raw/all"])

        # setup a test donut catalog DataFrame
        x_center = np.arange(15)
        y_center = 0.5 * np.arange(15)
        flux = 10000.0 * np.ones(15)
        data = [x_center, y_center, flux]
        names = [
            "centroid_x",
            "centroid_y",
            "source_flux",
        ]
        donutCat = QTable(data={x: y for x, y in zip(names, data)})
        donutCat.meta["blend_centroid_x"] = x_center
        donutCat.meta["blend_centroid_y"] = y_center
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
        ]
        self.assertCountEqual(newColumns, donutCatUpd.columns)
        metaKeys = ["blend_centroid_x", "blend_centroid_y"]
        self.assertCountEqual(donutCatUpd.meta.keys(), metaKeys)

        # test that the catalog created without
        # source selection will work
        donutCatNoSel = QTable(data={x: y for x, y in zip(names, data)})
        donutCatNoSel.meta["blend_centroid_x"] = ""
        donutCatNoSel.meta["blend_centroid_y"] = ""
        self.task.config.doDonutSelection = False

        # update the donut catalog
        donutCatUpd = self.task.updateDonutCatalog(donutCatNoSel, testExposure)

        # check that the new columns are present
        self.assertCountEqual(newColumns, donutCatUpd.columns)

    def testEmptyTable(self) -> None:
        testTable = self.task.emptyTable()

        # Test that there are no rows, but all columns are present
        self.assertEqual(len(testTable), 0)

        expected_columns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
        ]
        self.assertCountEqual(testTable.columns, expected_columns)

    def testTaskRun(self) -> None:
        """
        Test that the task runs interactively.
        """

        # Test run with an empty exposure
        # Read a real exposure and substitute the
        # image component with the background noise
        exposure_S11 = self.butler.get(
            "post_isr_image",
            dataId=self.testDataIdS11,
            collections=[self.baseRunName],
        )
        bkgnd = 100 * (np.random.random_sample(size=np.shape(exposure_S11.image.array)) - 0.5)
        image = afwImage.ImageF(exposure_S11.getBBox())
        image.array[:] = bkgnd
        maskedImage = afwImage.MaskedImageF(image)
        exposure_noSrc = copy(exposure_S11)
        exposure_noSrc.setMaskedImage(maskedImage)

        taskOutNoSrc = self.task.run(
            exposure_noSrc,
            self.camera,
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
        ]
        self.assertCountEqual(taskOutNoSrc.donutCatalog.columns, expected_columns)

        # Test that all expected metadata keys are present
        expected_metakeys = ["blend_centroid_x", "blend_centroid_y", "visit_info"]
        self.assertCountEqual(
            taskOutNoSrc.donutCatalog.meta.keys(),
            expected_metakeys,
        )

        # Run detection with different sources in each exposure
        exposure_S10 = self.butler.get(
            "post_isr_image",
            dataId=self.testDataIdS10,
            collections=[self.baseRunName],
        )
        taskOut_S11 = self.task.run(
            exposure_S11,
            self.camera,
        )
        taskOut_S10 = self.task.run(
            exposure_S10,
            self.camera,
        )

        # Test that the length of catalogs is as expected
        outputTable = vstack([taskOut_S11.donutCatalog, taskOut_S10.donutCatalog])
        self.assertEqual(len(outputTable), 6)

        # Test that the interactive output is as expected
        self.assertCountEqual(
            outputTable.columns,
            expected_columns,
        )
        # Test against truth the centroid for all sources
        tolerance = 15  # pixels

        result_y = np.sort(outputTable["centroid_y"].value)
        truth_y = np.sort(np.array([3196, 2198, 398, 2196, 3197, 398]))
        diff_y = np.sum(result_y - truth_y)

        result_x = np.sort(outputTable["centroid_x"].value)
        truth_x = np.sort(np.array([3815, 2814, 617, 2811, 3812, 614]))
        diff_x = np.sum(result_x - truth_x)

        self.assertLess(diff_x, tolerance)
        self.assertLess(diff_y, tolerance)

        # Test behavior if no sources get selected
        # This setting will select no donuts
        # on that exposure
        self.task.config.donutSelector.maxFieldDist = 0

        # Run the task
        taskOut_S10_noSources = self.task.run(
            exposure_S10,
            self.camera,
        )

        # Test that there are no rows, but all columns are present
        self.assertCountEqual(taskOut_S10_noSources.donutCatalog.columns, expected_columns)

        # Test that all expected metadata keys are present
        self.assertCountEqual(
            taskOut_S10_noSources.donutCatalog.meta.keys(),
            expected_metakeys,
        )

        # Test the behavior when source selection is turned off
        self.task.config.doDonutSelection = False
        taskOut_S11_noSelection = self.task.run(
            exposure_S11,
            self.camera,
        )
        # Check that the expected columns are present
        self.assertCountEqual(taskOut_S11_noSelection.donutCatalog.columns, expected_columns)
        # Check that the length of catalogs is as expected
        outputTableNoSel = taskOut_S11_noSelection.donutCatalog
        self.assertEqual(len(outputTableNoSel), 3)

        # Test against truth the centroid for all sources
        result_y = np.sort(outputTableNoSel["centroid_y"].value)
        truth_y = np.sort(np.array([3196, 2196, 398]))
        diff_y = np.sum(result_y - truth_y)

        result_x = np.sort(outputTableNoSel["centroid_x"].value)
        truth_x = np.sort(np.array([3812, 2814, 618]))
        diff_x = np.sum(result_x - truth_x)

        self.assertLess(diff_x, tolerance)
        self.assertLess(diff_y, tolerance)

    def testTaskRunPipeline(self) -> None:
        """
        Test that the task runs in a pipeline.
        """
        # Test instrument matches
        donutCatTable_S11 = self.butler.get(
            "donutTable",
            dataId=self.testDataIdS11,
            collections=[self.runName],
        )
        donutCatTable_S10 = self.butler.get(
            "donutTable",
            dataId=self.testDataIdS10,
            collections=[self.runName],
        )

        # Check 2 unblended sources in each detector
        # and 1 source much brighter than blend so is "isolated"
        self.assertEqual(len(donutCatTable_S11), 3)
        self.assertEqual(len(donutCatTable_S10), 3)

        # Test that brighter sources are at the top of the list
        np.testing.assert_array_equal(np.arange(3), np.argsort(donutCatTable_S10["source_flux"].value)[::-1])
        np.testing.assert_array_equal(np.arange(3), np.argsort(donutCatTable_S11["source_flux"].value)[::-1])

        # Check correct detector names
        self.assertEqual(np.unique(donutCatTable_S11["detector"]), "R22_S11")
        self.assertEqual(np.unique(donutCatTable_S10["detector"]), "R22_S10")

        # Check outputs are correct
        outputTable = vstack([donutCatTable_S11, donutCatTable_S10])
        self.assertEqual(len(outputTable), 6)
        self.assertCountEqual(
            outputTable.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "detector",
                "source_flux",
            ],
        )

        tolerance = 15  # pixels

        result_y = np.sort(outputTable["centroid_y"].value)
        truth_y = np.sort(np.array([3196, 2198, 398, 2196, 3197, 398]))
        diff_y = np.sum(result_y - truth_y)

        result_x = np.sort(outputTable["centroid_x"].value)
        truth_x = np.sort(np.array([3815, 2814, 617, 2811, 3812, 614]))
        diff_x = np.sum(result_x - truth_x)

        self.assertLess(diff_x, tolerance)
        self.assertLess(diff_y, tolerance)

        self.assertCountEqual(
            outputTable.meta.keys(),
            ["blend_centroid_x", "blend_centroid_y", "visit_info"],
        )
