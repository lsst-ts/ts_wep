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

# flake8: noqa
import os

from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()

import lsst.utils.tests
import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CombineZernikesMeanTask,
    CombineZernikesSigmaClipTask,
    EstimateZernikesAiDonutTask,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)

TEST_MODEL_PATH = getModulePath() + "/tests/testData/testAiModels/test_aidonut_model_file.pt"


class TestCalcZernikesAiDonutTaskCwfs(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate donutCatalog needed for task.
        """

        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if "pretest_run_cwfs" in collectionsList:
            cls.runName = "pretest_run_cwfs"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
            instrument = "lsst.obs.lsst.LsstCam"
            pipelineYaml = os.path.join(testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml")

            pipeCmd = writePipetaskCmd(
                cls.repoDir,
                cls.runName,
                instrument,
                collections,
                pipelineYaml=pipelineYaml,
            )
            pipeCmd += ' -d "detector IN (191, 192)"'
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = CalcZernikesTaskConfig()
        self.config.estimateZernikes.retarget(EstimateZernikesAiDonutTask)
        self.config.estimateZernikes.modelPath = TEST_MODEL_PATH
        self.config.estimateZernikes.nollIndices = list(range(4, 12))
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": 4021123106000,
            "visit": 4021123106000,
            "physical_filter": "g",
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": 4021123106000,
            "visit": 4021123106000,
            "physical_filter": "g",
        }
        self.donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        self.donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        self.intrinsicTables = [
            self.butler.get(
                "intrinsic_aberrations_temp",
                dataId=self.dataIdExtra,
                collections=["LSSTCam/aos/intrinsic"],
            ),
            self.butler.get(
                "intrinsic_aberrations_temp",
                dataId=self.dataIdIntra | {"detector": 192},
                collections=["LSSTCam/aos/intrinsic"],
            )
        ]

    def testValidateConfigs(self) -> None:
        self.assertEqual(type(self.task.estimateZernikes), EstimateZernikesAiDonutTask)
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

    def testEstimateZernikes(self) -> None:
        zernCoeff = self.task.estimateZernikes.run(self.donutStampsExtra, self.donutStampsIntra).zernikes

        self.assertEqual(np.shape(zernCoeff), (len(self.donutStampsExtra), 8))

    def testTableMetadata(self) -> None:
        # First estimate without pairs
        emptyStamps = DonutStamps([], metadata=self.donutStampsExtra.metadata)
        zkCalcExtra = self.task.run(self.donutStampsExtra, emptyStamps, self.intrinsicTables).zernikes
        zkCalcIntra = self.task.run(emptyStamps, self.donutStampsIntra, self.intrinsicTables).zernikes

        # Check metadata keys exist for extra case
        self.assertIn("cam_name", zkCalcExtra.meta)
        for k in ["intra", "extra"]:
            dict_ = zkCalcExtra.meta[k]
            self.assertIn("det_name", dict_)
            self.assertIn("visit", dict_)
            self.assertIn("dfc_dist", dict_)
            self.assertIn("band", dict_)
            self.assertEqual(dict_["mjd"], self.donutStampsExtra.metadata["MJD"])

        # Check metadata keys exist for intra case
        self.assertIn("cam_name", zkCalcIntra.meta)
        for k in ["intra", "extra"]:
            dict_ = zkCalcIntra.meta[k]
            self.assertIn("det_name", dict_)
            self.assertIn("visit", dict_)
            self.assertIn("dfc_dist", dict_)
            self.assertIn("band", dict_)
            self.assertEqual(dict_["mjd"], self.donutStampsIntra.metadata["MJD"])

        # Now estimate with pairs
        zkCalcPairs = self.task.run(self.donutStampsExtra, self.donutStampsIntra, self.intrinsicTables).zernikes

        # Check metadata keys exist for pairs case
        self.assertIn("cam_name", zkCalcPairs.meta)
        self.assertIn("estimatorInfo", zkCalcPairs.meta)
        for stamps, k in zip([self.donutStampsIntra, self.donutStampsExtra], ["intra", "extra"]):
            dict_ = zkCalcPairs.meta[k]
            if k == stamps.metadata["DFC_TYPE"]:
                self.assertIn("det_name", dict_)
                self.assertIn("visit", dict_)
                self.assertIn("dfc_dist", dict_)
                self.assertIn("band", dict_)
                self.assertEqual(dict_["mjd"], stamps.metadata["MJD"])
