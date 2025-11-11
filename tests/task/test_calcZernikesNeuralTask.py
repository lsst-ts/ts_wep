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
import importlib.util
import pytest
import tempfile
import yaml

import lsst.utils.tests
import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.wep.task import CalcZernikesNeuralTask, CalcZernikesNeuralTaskConfig
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)
from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()

# Skip all tests in this module if the TARTS package is not available
_TARTS_AVAILABLE = importlib.util.find_spec("tarts") is not None
# TODO: Remove this module-wide skip once on-sky testing is complete and
# TARTS becomes available in standard CI/test environments.
pytestmark = pytest.mark.skipif(
    not _TARTS_AVAILABLE,
    reason="requires the TARTS package currently in development",
)


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
            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
            instrument = "lsst.obs.lsst.LsstCam"

            # Use the neural pipeline configuration to generate test data
            pipelineYaml = os.path.join(testPipelineConfigDir, "testCalcZernikesNeural.yaml")

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
        if hasattr(self, "_temp_dataset_params"):
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
        # Use the bundled local test repo, consistent with other task tests
        moduleDir = getModulePath()
        self.repoDir = os.path.join(moduleDir, "tests", "testData", "gen3TestRepo")
        self.config = CalcZernikesNeuralTaskConfig()

        # Use random weights for testing instead of actual trained models
        # This makes tests portable and doesn't depend on specific file paths
        self.config.wavenetPath = None  # TARTS will use random weights
        self.config.alignetPath = None  # TARTS will use random weights
        self.config.aggregatornetPath = None  # TARTS will use random weights

        # Create a minimal dataset params file for testing

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
            "aggregator_model": {"d_model": 20, "nhead": 2, "num_layers": 6, "dim_feedforward": 128},
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(dataset_params, f)
            self.config.datasetParamPath = f.name
            self._temp_dataset_params = f.name  # Store for cleanup

        self.config.device = "cpu"
        customNollIndices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
        self.config.nollIndices = customNollIndices

        # Initialize the neural task
        self.task = CalcZernikesNeuralTask(config=self.config, name="Neural Task")

        # Set the run name for data collection
        self.runName = "pretest_run_cwfs"

        # Initialize Butler and registry for data loading
        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        # Define data ID for single exposure
        # Using detector 191 from the same visit
        self.visitNum = 2031063000001
        self.dataId = {
            "instrument": "LSSTCam",
            "detector": 191,
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
        customNollIndices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
        customConfig.nollIndices = customNollIndices

        # Create task with custom config
        customTask = CalcZernikesNeuralTask(config=customConfig, name="Custom Neural Task")

        # Verify custom Noll indices are used
        self.assertEqual(
            customTask.nollIndices, customNollIndices, "Task should use custom Noll indices from config"
        )

        # Test that the task works with custom indices
        # Create a simple test exposure (this would need proper mock data
        # in practice)
        # For now, just verify the configuration is applied correctly
        self.assertEqual(len(customTask.nollIndices), 17, "Custom task should have 17 Noll indices")

    def testRunExposure(self) -> None:
        """
        Test running the neural task on a single exposure
        """
        exposure = self.butler.get("raw", dataId=self.dataId, collections=["LSSTCam/raw/all"])

        # Verify that we have valid data
        self.assertIsNotNone(exposure, "Exposure should not be None")

        # Run the task with single exposure
        values = self.task.run(exposure)

        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return results")
        self.assertTrue(hasattr(values, "outputZernikesAvg"), "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, "outputZernikesRaw"), "Should have outputZernikesRaw")
        self.assertTrue(hasattr(values, "donutStampsNeural"), "Should have donutStampsNeural")
        self.assertTrue(hasattr(values, "zernikes"), "Should have zernikes table")
        self.assertTrue(hasattr(values, "donutQualityTable"), "Should have donutQualityTable")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray, "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray, "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 1, "Should have 1 row for single exposure")
        self.assertEqual(
            values.outputZernikesRaw.shape[1],
            len(self.task.nollIndices),
            f"Should have {len(self.task.nollIndices)} Zernike coefficients",
        )

    def testRunWithValidExposure(self) -> None:
        """
        Test running the neural task with a valid exposure.
        This tests the normal operation with a single exposure.
        """
        exposure = self.butler.get("raw", dataId=self.dataId, collections=["LSSTCam/raw/all"])

        # Verify we have valid data
        self.assertIsNotNone(exposure, "Exposure should not be None")

        # Run with valid exposure
        values = self.task.run(exposure)

        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return results with valid exposure")
        self.assertTrue(hasattr(values, "outputZernikesAvg"), "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, "outputZernikesRaw"), "Should have outputZernikesRaw")
        self.assertTrue(hasattr(values, "donutStampsNeural"), "Should have donutStampsNeural")
        self.assertTrue(hasattr(values, "zernikes"), "Should have zernikes table")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray, "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray, "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 1, "Should have 1 row for single exposure")
        self.assertEqual(
            values.outputZernikesRaw.shape[1],
            len(self.task.nollIndices),
            f"Should have {len(self.task.nollIndices)} Zernike coefficients",
        )

        # Verify that average equals the raw result for single exposure
        np.testing.assert_array_equal(
            values.outputZernikesAvg,
            values.outputZernikesRaw[0],
            "Average should equal raw result for single exposure",
        )

    def testRunWithDifferentDetector(self) -> None:
        """
        Test running the neural task with a different detector.
        This tests the robustness with different data sources.
        """
        # Use a different detector (192 instead of 191)
        dataId2 = {
            "instrument": "LSSTCam",
            "detector": 192,
            "exposure": self.visitNum,
        }

        exposure = self.butler.get("raw", dataId=dataId2, collections=["LSSTCam/raw/all"])

        # Verify we have valid data
        self.assertIsNotNone(exposure, "Exposure should not be None")

        # Run with different detector exposure
        values = self.task.run(exposure)

        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return results with different detector")
        self.assertTrue(hasattr(values, "outputZernikesAvg"), "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, "outputZernikesRaw"), "Should have outputZernikesRaw")
        self.assertTrue(hasattr(values, "donutStampsNeural"), "Should have donutStampsNeural")
        self.assertTrue(hasattr(values, "zernikes"), "Should have zernikes table")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray, "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray, "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 1, "Should have 1 row for single exposure")
        self.assertEqual(
            values.outputZernikesRaw.shape[1],
            len(self.task.nollIndices),
            f"Should have {len(self.task.nollIndices)} Zernike coefficients",
        )

        # Verify that average equals the raw result for single exposure
        np.testing.assert_array_equal(
            values.outputZernikesAvg,
            values.outputZernikesRaw[0],
            "Average should equal raw result for single exposure",
        )

    def testRunWithNoExposure(self) -> None:
        """
        Test running the neural task with no exposure available.
        This tests the error handling when exposure is missing.
        """
        # Run with no exposure (None)
        values = self.task.run(None)

        # Verify the output structure
        self.assertIsNotNone(values, "Task run should return empty results when no exposure available")
        self.assertTrue(hasattr(values, "outputZernikesAvg"), "Should have outputZernikesAvg")
        self.assertTrue(hasattr(values, "outputZernikesRaw"), "Should have outputZernikesRaw")
        self.assertTrue(hasattr(values, "donutStampsNeural"), "Should have donutStampsNeural")
        self.assertTrue(hasattr(values, "zernikes"), "Should have zernikes table")
        self.assertTrue(hasattr(values, "donutQualityTable"), "Should have donutQualityTable")

        # Verify data types and shapes
        self.assertIsInstance(values.outputZernikesAvg, np.ndarray, "outputZernikesAvg should be numpy array")
        self.assertIsInstance(values.outputZernikesRaw, np.ndarray, "outputZernikesRaw should be numpy array")
        self.assertEqual(values.outputZernikesRaw.shape[0], 1, "Should have 1 row")
        self.assertEqual(
            values.outputZernikesRaw.shape[1],
            len(self.task.nollIndices),
            f"Should have {len(self.task.nollIndices)} Zernike coefficients",
        )

        # Verify that results contain NaN values (empty results)
        self.assertTrue(
            np.all(np.isnan(values.outputZernikesAvg)),
            "outputZernikesAvg should contain NaN values when no exposure available",
        )
        self.assertTrue(
            np.all(np.isnan(values.outputZernikesRaw)),
            "outputZernikesRaw should contain NaN values when no exposure available",
        )

        # Verify that zernikes table is empty
        self.assertEqual(len(values.zernikes), 0, "Zernikes table should be empty when no exposure available")
