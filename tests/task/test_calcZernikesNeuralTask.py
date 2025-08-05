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
    CalcZernikesNeuralTask, 
    CalcZernikesNeuralTaskConfig
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesNeuralTask(lsst.utils.tests.TestCase):
    """
    Neural test class for CalcZernikesTask with custom estimation algorithm.
    
    This Neural follows the same structure as TestCalcZernikesDanishTaskCwfs
    and can be customized for testing different Zernike estimation algorithms.
    """
    runName: str
    testDataDir: str
    repoDir: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate donutCatalog needed for task.
        
        This method sets up the test data repository and runs the necessary
        pipeline to generate donut stamps for testing.
        """
        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        
        # Use the same run name as the CWFS tests to share data
        if "pretest_run_cwfs" in collectionsList:
            cls.runName = "pretest_run_cwfs"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            # Use the same pipeline configuration as CWFS tests
            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
            instrument = "lsst.obs.lsst.LsstCam"
            
            # Use the CWFS pipeline configuration to generate test data
            pipelineYaml = os.path.join(
                testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml"
            )

            pipeCmd = writePipetaskCmd(
                cls.repoDir,
                cls.runName,
                instrument,
                collections,
                pipelineYaml=pipelineYaml,
            )
            # Use the same detector selection as CWFS tests
            pipeCmd += ' -d "detector IN (191, 192)"'
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up test data after all tests are complete.
        """
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        """
        Set up individual test environment.
        
        This method configures the task with your specific estimation algorithm
        and loads test data for each test method.
        """
        self.repoDir = '/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/butler'
        self.config = CalcZernikesTaskConfig()
        
        # TODO: Replace with your specific estimation task
        # self.config.estimateZernikes.retarget(YourEstimationTask)
        self.config = CalcZernikesNeuralTaskConfig()
        self.config.wavenet_path = '/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/finetune_logs/lightning_logs/version_0/checkpoints/best_finetuned_wavennet.ckpt'
        self.config.alignet_path = '/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/alignnet_logs/lightning_logs/version_0/checkpoints/best_alignnet_120.ckpt'
        self.config.aggregatornet_path = '/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/aggregator_logs/lightning_logs/version_0/checkpoints/best_aggregator.ckpt'
        self.config.dataset_param_path = '/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/dataset_params.yaml'
        self.config.device = 'cpu'

        # Initialize the neural task
        self.task = CalcZernikesNeuralTask(config=self.config, name="Neural Task")
        
        # Initialize Butler and registry for data loading
        # self.butler = Butler.from_config(self.repoDir)
        self.butler = Butler(self.repoDir, collections=["LSSTCam/raw/all", "LSSTCam/calib"], writeable=True)
        self.registry = self.butler.registry

        # Define data IDs for extra and intra focal exposures
        # Using adjacent detectors (191 and 192) from the same visit
        self.visitNum = 2031063000001
        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": self.visitNum,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 192,
            "exposure": self.visitNum,
        }

    def test_load_model(self):
        """
        Test we can load the models
        """
        print(self.task)

    def test_load_intra_extra_exposures(self):
        """
        Test loading a pair of intra and extra focal exposures from the Butler repository.
        
        This test follows the pattern from test_cutOutDonutsCwfsTask.py to load
        exposure data and donut catalogs for testing the neural network algorithm.
        """
        # Butler and registry are already initialized in setUp()
        # Data IDs are also already defined in setUp()

        # Load post-ISR exposures from Butler
      
        exposureExtra = self.butler.get(
            "raw", dataId=self.dataIdExtra, collections=["LSSTCam/raw/all"]
        )
        exposureIntra = self.butler.get(
            "raw", dataId=self.dataIdIntra, collections=["LSSTCam/raw/all"]
        )
        # exposureExtra = self.butler.get(
        #     "post_isr_image", dataId=self.dataIdExtra, collections=[self.runName]
        # )
        # exposureIntra = self.butler.get(
        #     "post_isr_image", dataId=self.dataIdIntra, collections=[self.runName]
        # )
        
        # Load camera information
 
        
        # Verify that we have valid data
        self.assertIsNotNone(exposureExtra, "Extra focal exposure should not be None")
        self.assertIsNotNone(exposureIntra, "Intra focal exposure should not be None")
        
        # Store the loaded data as instance variables for use in other tests
        self.exposureExtra = exposureExtra
        self.exposureIntra = exposureIntra

    def test_run_exposures(self):
        """
        Test running the neural task on the exposures
        """
        exposureExtra = self.butler.get(
            "raw", dataId=self.dataIdExtra, collections=["LSSTCam/raw/all"]
        )
        exposureIntra = self.butler.get(
            "raw", dataId=self.dataIdIntra, collections=["LSSTCam/raw/all"]
        )
        # exposureExtra = self.butler.get(
        #     "post_isr_image", dataId=self.dataIdExtra, collections=[self.runName]
        # )
        # exposureIntra = self.butler.get(
        #     "post_isr_image", dataId=self.dataIdIntra, collections=[self.runName]
        # )
        
        # Load camera information
 
        
        # Verify that we have valid data
        self.assertIsNotNone(exposureExtra, "Extra focal exposure should not be None")
        self.assertIsNotNone(exposureIntra, "Intra focal exposure should not be None")
        
        # Store the loaded data as instance variables for use in other tests

        values =self.task.run(exposureExtra, exposureIntra)