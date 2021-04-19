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

from lsst.daf import butler as dafButler
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.GenerateDonutCatalogOnlineTask import (
    GenerateDonutCatalogOnlineTask,
    GenerateDonutCatalogOnlineTaskConfig,
)
from lsst.ts.wep.Utility import runProgram


class TestGenerateDonutCatalogOnlineTask(unittest.TestCase):
    def setUp(self):

        self.config = GenerateDonutCatalogOnlineTaskConfig()
        self.task = GenerateDonutCatalogOnlineTask(config=self.config)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

    def validateConfigs(self):

        self.config.boresightRa = 0.03
        self.config.boresightDec = -0.02
        self.config.boresightRotAng = 90.0
        self.config.filterName = "r"
        self.task = GenerateDonutCatalogOnlineTask(config=self.config)

        self.assertEqual(self.task.boresightRa, 0.03)
        self.assertEqual(self.task.boresightDec, -0.02)
        self.assertEqual(self.task.boresightRotAng, 90.0)
        self.assertEqual(self.task.filterName, "r")

    def testFilterResults(self):

        refCat = self.butler.get(
            "cal_ref_cat", dataId={"htm7": 253952}, collections=["refcats"]
        )
        testDataFrame = refCat.asAstropy().to_pandas()
        filteredDataFrame = self.task.filterResults(testDataFrame)
        np.testing.assert_array_equal(filteredDataFrame, testDataFrame)

    def testPipeline(self):
        """
        Test that the task runs in a pipeline. Also functions as a test of
        runQuantum function.
        """

        # Run pipeline command
        runName = "run1"
        pipetaskCmd = "pipetask run "
        pipetaskCmd += f"-b {self.repoDir} "  # Specify repo
        pipetaskCmd += "-i refcats "  # Specify collections with data to use
        # Specify task
        taskName = "GenerateDonutCatalogOnlineTask.GenerateDonutCatalogOnlineTask"
        pipetaskCmd += f"-t lsst.ts.wep.task.{taskName} "
        pipetaskCmd += "--instrument lsst.obs.lsst.LsstCam "
        pipetaskCmd += f"--register-dataset-types --output-run {runName}"

        runProgram(pipetaskCmd)

        # Test instrument matches
        pipelineButler = dafButler.Butler(self.repoDir)
        outputDf = pipelineButler.get(
            "donutCatalog", dataId={"instrument": "LSSTCam"}, collections=[f"{runName}"]
        )
        self.assertEqual(len(outputDf), 8)
        self.assertEqual(len(outputDf.query('detector == "R22_S11"')), 4)
        self.assertEqual(len(outputDf.query('detector == "R22_S10"')), 4)
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

        # Clean up
        cleanUpCmd = "butler prune-collection "
        cleanUpCmd += f"{self.repoDir} {runName} --purge --unstore"
        runProgram(cleanUpCmd)

    def testDonutCatalogGeneration(self):
        """
        Test that task creates a dataframe with detector information.
        """

        # Create list of deferred loaders for the ref cat
        deferredList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats"]
        ).expanded()
        for ref in datasetGenerator:
            deferredList.append(self.butler.getDeferred(ref, collections=["refcats"]))
        taskOutput = self.task.run("LSSTCam", deferredList)
        outputDf = taskOutput.donutCatalog

        # Compare ra, dec info to original input catalog
        inputCat = np.genfromtxt(
            os.path.join(
                self.testDataDir, "phosimOutput", "realComCam", "skyComCamInfo.txt"
            ),
            names=["id", "ra", "dec", "mag"],
        )

        # Check that all 8 sources are present and 4 assigned to each detector
        self.assertEqual(len(outputDf), 8)
        self.assertCountEqual(np.radians(inputCat["ra"]), outputDf["coord_ra"])
        self.assertCountEqual(np.radians(inputCat["dec"]), outputDf["coord_dec"])
        self.assertEqual(len(outputDf.query('detector == "R22_S11"')), 4)
        self.assertEqual(len(outputDf.query('detector == "R22_S10"')), 4)
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
