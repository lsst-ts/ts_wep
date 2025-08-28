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

import lsst.utils.tests
import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.wep.task import (
    CalcZernikesNeuralTask,
    CalcZernikesNeuralTaskConfig
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)
from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()


class TestCalcZernikesNeuralTask(lsst.utils.tests.TestCase):
    """
    Test class for CalcZernikesNeuralTask with neural network-based estimation.

    This test class follows the same structure as
    TestCalcZernikesDanishTaskCwfs and tests the neural network-based
    Zernike estimation algorithm.
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

    def tearDown(self) -> None:
        """
        Clean up temporary files after each test.
        """
        if hasattr(self, '_temp_dataset_params'):
            try:
                os.unlink(self._temp_dataset_params)
            except OSError:
                pass  # File might already be deleted

    def setUp(self) -> None:
        """
        Set up individual test environment.

        This method configures the task with your specific estimation algorithm
        and loads test data for each test method.
        """
        self.repoDir = '/home/peterma/research/Rubin_AO_ML/training/butler'
        self.config = CalcZernikesNeuralTaskConfig()

        # Use random weights for testing instead of actual trained models
        # This makes tests portable and doesn't depend on specific file paths
        self.config.wavenetPath = None  # TARTS will use random weights
        self.config.alignetPath = None  # TARTS will use random weights
        self.config.aggregatornetPath = None  # TARTS will use random weights

        # Create a minimal dataset params file for testing
        import tempfile
        import yaml

        # Create temporary dataset params with minimal required parameters
        dataset_params = {
            "version": 0,
            "adjustment_WaveNet": 10,
            "adjustment_AlignNet": 120,
            "refinements": 1,
            "CROP_SIZE": 160,
            "deg_per_pix": 0.00005555555,
            "mm_pix": 0.01,
            "alpha": 1,
            "max_seq_len": 200,
            "noll_zk": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28],
            "aggregator_model": {
                "d_model": 20,
                "nhead": 2,
                "num_layers": 6,
                "dim_feedforward": 128
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(dataset_params, f)
            self.config.datasetParamPath = f.name
            self._temp_dataset_params = f.name  # Store for cleanup

        self.config.device = 'cpu'
        customNollIndices = [4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,27,28]
        self.config.nollIndices = customNollIndices

        # Initialize the neural task
        self.task = CalcZernikesNeuralTask(config=self.config, name="Neural Task")

        # Set the run name for data collection
        self.runName = "pretest_run_cwfs"

        # Initialize Butler and registry for data loading
        self.butler = Butler.from_config(self.repoDir)
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

    def testConfigurableNollIndices(self) -> None:
        """
        Test that Noll indices can be configured via the config.
        """
        # Create a new config with custom Noll indices
        customConfig = CalcZernikesNeuralTaskConfig()
        # Use the same random weights approach for consistency
        customConfig.wavenetPath = None
        customConfig.alignetPath = None
        customConfig.aggregatornetPath = None
        customConfig.datasetParamPath = self.config.datasetParamPath
        customConfig.device = self.config.device

        # Test custom Noll indices (e.g., only Z4-Z9 for basic aberrations)
        customNollIndices = [4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,27,28]
        customConfig.nollIndices = customNollIndices

        # Create task with custom config
        customTask = CalcZernikesNeuralTask(config=customConfig, name="Custom Neural Task")

        # Verify custom Noll indices are used
        self.assertEqual(customTask.nollIndices, customNollIndices,
                        "Task should use custom Noll indices from config")

        # Test that the task works with custom indices
        # Create a simple test exposure (this would need proper mock data
        # in practice)
        # For now, just verify the configuration is applied correctly
        self.assertEqual(len(customTask.nollIndices), 17,
                        "Custom task should have 17 Noll indices")

    def testRunExposures(self) -> None:
        """
        Test running the neural task on the exposures
        """
        exposureExtra = self.butler.get(
            "raw", dataId=self.dataIdExtra, collections=["LSSTCam/raw/all"]
        )
        exposureIntra = self.butler.get(
            "raw", dataId=self.dataIdIntra, collections=["LSSTCam/raw/all"]
        )

        # Load camera information

        # Verify that we have valid data
        self.assertIsNotNone(exposureExtra,
                            "Extra focal exposure should not be None")
        self.assertIsNotNone(exposureIntra,
                            "Intra focal exposure should not be None")

        # Store the loaded data as instance variables for use in other tests

        values = self.task.run(exposureExtra, exposureIntra)
        print(values.outputZernikesRaw.shape)
        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return results")
        self.assertTrue(hasattr(values, 'outputZernikesAvg'),
                       "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, 'outputZernikesRaw'),
                       "Should have outputZernikesRaw")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray,
                            "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray,
                            "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 2,
                       "Should have 2 rows (intra + extra)")
        self.assertEqual(values.outputZernikesRaw.shape[1], len(self.task.nollIndices),
                       f"Should have {len(self.task.nollIndices)} Zernike coefficients")

    def testRunWithOnlyIntraExposure(self) -> None:
        """
        Test running the neural task with only intra-focal exposure available.
        This tests the robustness when extra-focal exposure is missing.
        """
        exposureIntra = self.butler.get(
            "raw", dataId=self.dataIdIntra, collections=["LSSTCam/raw/all"]
        )

        # Verify we have valid intra-focal data
        self.assertIsNotNone(exposureIntra, "Intra focal exposure should not be None")

        # Run with only intra-focal exposure (extra-focal is None)
        values = self.task.run(None, exposureIntra)

        # Verify the output structure
        self.assertIsNotNone(values,
                            "Task run should return results even with missing extra exposure")
        self.assertTrue(hasattr(values, 'outputZernikesAvg'),
                       "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, 'outputZernikesRaw'),
                       "Should have outputZernikesRaw")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray,
                            "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray,
                            "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 2,
                       "Should still have 2 rows")
        self.assertEqual(values.outputZernikesRaw.shape[1], len(self.task.nollIndices),
                       f"Should have {len(self.task.nollIndices)} Zernike coefficients")

        # Verify that both rows contain the same data (intra-focal used for
        # both)
        np.testing.assert_array_equal(
            values.outputZernikesRaw[0],
            values.outputZernikesRaw[1],
            "Both rows should contain identical data when only intra-focal available"
        )

        # Verify that average equals the single exposure result
        np.testing.assert_array_equal(
            values.outputZernikesAvg,
            values.outputZernikesRaw[0],
            "Average should equal single exposure when only intra-focal available"
        )

    def testRunWithOnlyExtraExposure(self) -> None:
        """
        Test running the neural task with only extra-focal exposure available.
        This tests the robustness when intra-focal exposure is missing.
        """
        exposureExtra = self.butler.get(
            "raw", dataId=self.dataIdExtra, collections=["LSSTCam/raw/all"]
        )

        # Verify we have valid extra-focal data
        self.assertIsNotNone(exposureExtra, "Extra focal exposure should not be None")

        # Run with only extra-focal exposure (intra-focal is None)
        values = self.task.run(exposureExtra, None)

        # Verify the output structure
        self.assertIsNotNone(values,
                            "Task run should return results even with missing intra exposure")
        self.assertTrue(hasattr(values, 'outputZernikesAvg'),
                       "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, 'outputZernikesRaw'),
                       "Should have outputZernikesRaw")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray,
                            "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray,
                            "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 2,
                       "Should still have 2 rows")
        self.assertEqual(values.outputZernikesRaw.shape[1], len(self.task.nollIndices),
                       f"Should have {len(self.task.nollIndices)} Zernike coefficients")

        # Verify that both rows contain the same data (extra-focal used for
        # both)
        np.testing.assert_array_equal(
            values.outputZernikesRaw[0],
            values.outputZernikesRaw[1],
            "Both rows should contain identical data when only extra-focal available"
        )

        # Verify that average equals the single exposure result
        np.testing.assert_array_equal(
            values.outputZernikesAvg,
            values.outputZernikesRaw[0],
            "Average should equal single exposure when only extra-focal available"
        )

    def testRunWithNoExposures(self) -> None:
        """
        Test running the neural task with no exposures available.
        This tests the error handling when both exposures are missing.
        """
        # Run with no exposures (both are None)
        values = self.task.run(None, None)

        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return empty results when no exposures available")
        self.assertTrue(hasattr(values, 'outputZernikesAvg'), "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, 'outputZernikesRaw'), "Should have outputZernikesRaw")
        self.assertTrue(hasattr(values, 'zernikes'), "Should have zernikes table")
        self.assertTrue(hasattr(values, 'donutQualityTable'), "Should have donutQualityTable")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray, "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray, "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 1, "Should have 2 rows")
        self.assertEqual(values.outputZernikesRaw.shape[1], len(self.task.nollIndices),
                       f"Should have {len(self.task.nollIndices)} Zernike coefficients")

        # Verify that results contain NaN values (empty results)
        self.assertTrue(np.all(np.isnan(values.outputZernikesAvg)),
                       "outputZernikesAvg should contain NaN values when no exposures available")
        self.assertTrue(np.all(np.isnan(values.outputZernikesRaw)),
                       "outputZernikesRaw should contain NaN values when no exposures available")

        # Verify that zernikes table is empty
        self.assertEqual(
            len(values.zernikes),
            0,
            "Zernikes table should be empty when no exposures available"
        )
