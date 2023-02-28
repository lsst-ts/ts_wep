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
import pytest
import tempfile
import numpy as np

import lsst.utils.tests
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.CalcZernikesTask import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
)
from lsst.ts.wep.task.CombineZernikesMeanTask import CombineZernikesMeanTask
from lsst.ts.wep.task.CombineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


@pytest.mark.skipif(
    os.path.exists("/sdf/data/rubin/repo/main") is False,
    reason="requires access to data in /repo/main",
)
class TestCalcZernikesTaskLatiss(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutCatalog needed for task.
        """

        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = "/sdf/data/rubin/repo/main"

        # Create a temporary test directory
        # under /sdf/data/rubin/repo/main/u/$USER
        # to ensure write access is granted
        user = os.getlogin()
        tempDir = os.path.join(cls.repoDir, "u", user)
        cls.testDir = tempfile.TemporaryDirectory(dir=tempDir)
        testDirName = os.path.split(cls.testDir.name)[1]  # temp dir name
        cls.runName = os.path.join("u", user, testDirName)

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        # Point to the collections with
        # the raw images and calibrations
        collections = "LATISS/raw/all,LATISS/calib"
        instrument = "lsst.obs.lsst.Latiss"
        cls.cameraName = "LATISS"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCalcZernikesLatissPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure IN (2021090800487, 2021090800488) AND visit_system=0'"
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):
        self.config = CalcZernikesTaskConfig()
        self.config.opticalModel = "onAxis"
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800487,
            "visit": 2021090800487,
        }
        self.dataIdIntra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800488,
            "visit": 2021090800488,
        }

    def testValidateConfigs(self):
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

    def testEstimateZernikes(self):
        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        # Use dataIdExtra here too because both sets of donutStamps
        # get saved to extraFocal dataId so we can run this task
        # in parallel across detector pairs of the same visit.
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

        zernCoeff = self.task.run(donutStampsExtra, donutStampsIntra)

        self.assertEqual(
            np.shape(zernCoeff.outputZernikesRaw), (len(donutStampsExtra), 19)
        )

        zkList = np.array(
            [
                [
                    8.51874706e-03,
                    1.14889498e-01,
                    -4.59798303e-02,
                    1.17980813e-02,
                    5.47971263e-03,
                    -3.37311212e-02,
                    -1.68802493e-02,
                    -3.78572402e-02,
                    1.05657862e-02,
                    1.10063567e-04,
                    -8.53572243e-03,
                    -1.01936034e-02,
                    1.75246804e-03,
                    -1.78255049e-03,
                    -8.62521565e-04,
                    -5.23579524e-04,
                    4.45220226e-03,
                    2.38692144e-03,
                    9.67399215e-03,
                ],
                [
                    -2.40261395e-02,
                    1.10103556e-01,
                    -1.31705158e-03,
                    8.44028035e-03,
                    3.96194900e-04,
                    -3.09416580e-02,
                    -2.19351288e-02,
                    -3.03180146e-02,
                    6.07601745e-03,
                    1.00489422e-03,
                    -1.02136815e-02,
                    -7.33892033e-03,
                    1.18980188e-03,
                    -2.71901549e-04,
                    3.58675079e-04,
                    3.15012317e-04,
                    5.05772823e-03,
                    1.09876495e-03,
                    7.51209259e-03,
                ],
            ]
        )

        for i in range(2):
            # ensure total rms error is within 0.5 microns from the
            # recorded values with possible changes from ISR pipeline, etc.
            self.assertLess(
                np.sqrt(
                    np.sum(np.square(zernCoeff.outputZernikesRaw[0][i] - zkList[i]))
                ),
                0.5,
            )

    def testGetCombinedZernikes(self):
        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.getCombinedZernikes(testArr)
        np.testing.assert_array_equal(
            combinedZernikesStruct.combinedZernikes, np.ones(19)
        )
        np.testing.assert_array_equal(
            combinedZernikesStruct.flags, np.zeros(len(testArr))
        )
