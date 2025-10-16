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

import getpass
import os
import tempfile

import lsst.utils.tests
import numpy as np
import pytest
from lsst.daf.butler import Butler
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
@pytest.mark.skipif(
    not os.getenv("PGPASSFILE"),
    reason="requires access to butler db",
)
class TestCalcZernikesTieTaskLatiss(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    cameraName: str

    @classmethod
    def setUpClass(cls) -> None:
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
        user = getpass.getuser()
        tempDir = os.path.join(cls.repoDir, "u", user)
        testDir = tempfile.TemporaryDirectory(dir=tempDir)
        testDirName = os.path.split(testDir.name)[1]  # temp dir name
        cls.runName = os.path.join("u", user, testDirName)

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
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
        pipelineYaml = os.path.join(testPipelineConfigDir, "testCalcZernikesLatissPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure IN (2021090800487, 2021090800488) AND visit_system=0'"
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = CalcZernikesTaskConfig()
        self.config.doDonutStampSelector = False
        self.config.estimateZernikes.nollIndices = [
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ]
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdIntra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800488,
            "visit": 2021090800488,
        }
        self.dataIdExtra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800487,
            "visit": 2021090800487,
        }

    def testValidateConfigs(self) -> None:
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

    def testEstimateZernikesRegression(self) -> None:
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

        self.assertEqual(np.shape(zernCoeff.outputZernikesRaw), (len(donutStampsExtra), 19))

        # Previous Zernikes for regression test
        zk = np.array(
            [
                0.06301116,
                0.16224293,
                -0.03171561,
                -0.01889007,
                0.00448589,
                0.03581878,
                -0.05704621,
                -0.02594513,
                0.01350136,
                -0.00455589,
                -0.01713792,
                0.00590172,
                0.00505236,
                0.0015623,
                0.01403694,
                -0.00709998,
                -0.03391995,
                -0.02250289,
                0.0147995,
            ]
        )
        np.testing.assert_allclose(zk, zernCoeff.outputZernikesRaw[0], rtol=0, atol=1e-3)

    def testGetCombinedZernikes(self) -> None:
        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.combineZernikes.run(testArr)
        np.testing.assert_array_equal(combinedZernikesStruct.combinedZernikes, np.ones(19))
        np.testing.assert_array_equal(combinedZernikesStruct.flags, np.zeros(len(testArr)))
