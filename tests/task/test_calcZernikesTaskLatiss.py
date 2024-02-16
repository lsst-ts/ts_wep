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
import tempfile

import lsst.utils.tests
import numpy as np
import pytest
from lsst.daf import butler as dafButler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CombineZernikesMeanTask,
    CombineZernikesSigmaClipTask,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
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

    def testEstimateZernikesRegression(self):
        """THIS DOES NOT TEST ZERNIKE ACCURACY!!!

        This only tests to see if software changes result in different
        Zernikes. If that is expected and okay, you can change the test
        values below.
        """
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

        # Previous Zernikes for regression test
        zk = np.array(
            [
                -0.00283742,
                0.12024323,
                -0.05781533,
                -0.00683463,
                -0.01198631,
                0.03911933,
                -0.07739852,
                -0.0307383,
                0.0114996,
                -0.00157382,
                -0.00154171,
                -0.00857287,
                0.00426482,
                -0.00160444,
                0.01318693,
                -0.00538555,
                -0.02333674,
                -0.02252251,
                0.00943855,
            ]
        )

        self.assertTrue(np.allclose(zk, zernCoeff.outputZernikesRaw[0]))

    def testGetCombinedZernikes(self):
        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.combineZernikes.run(testArr)
        np.testing.assert_array_equal(
            combinedZernikesStruct.combinedZernikes, np.ones(19)
        )
        np.testing.assert_array_equal(
            combinedZernikesStruct.flags, np.zeros(len(testArr))
        )
