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
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task.donutQuickMeasurementTask import (
    DonutQuickMeasurementTask,
    DonutQuickMeasurementTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
    createTemplateForDetector,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestDonutQuickMeasurementTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Produce ISR image for task.
        """

        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        cls.expNum = 4021123106001
        cls.detNum = 94
        if "pretest_run_science" in collectionsList:
            cls.runName = "pretest_run_science"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            collections = "LSSTCam/calib/unbounded,LSSTCam/raw/all"
            instrument = "lsst.obs.lsst.LsstCam"
            pipelineYaml = os.path.join(testPipelineConfigDir, "testIsrPipeline.yaml")

            pipeCmd = writePipetaskCmd(
                cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
            )
            pipeCmd += f" -d 'detector in ({cls.detNum}) and exposure in ({cls.expNum})'"
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self):
        self.config = DonutQuickMeasurementTaskConfig()
        self.task = DonutQuickMeasurementTask(config=self.config)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        # Get image from butler
        testDataId = {
            "instrument": "LSSTCam",
            "detector": self.detNum,
            "exposure": self.expNum,
            "visit": self.expNum,
        }
        self.postIsrExp = self.butler.get(
            "post_isr_image", dataId=testDataId, collections=self.runName
        )

    def _getTemplate(self):
        # Get the detector
        cam = LsstCam().getCamera()
        detector = cam.get("R22_S11")

        # Create the template
        template = createTemplateForDetector(
            detector=detector, defocalType=DefocalType.Extra
        )

        return template

    def testValidateConfigs(self):
        # Check default configuration
        self.origTask = DonutQuickMeasurementTask(config=self.config, name="Orig Task")
        self.assertEqual(self.origTask.config.initialCutoutPadding, 5)
        self.assertTrue(self.origTask.config.doPreConvolution)
        # Test inherited config from QuickFrameMeasurementTask
        self.assertEqual(self.origTask.config.nSigmaDetection, 20)

        # Check configuration changes are passed through
        self.config.initialCutoutPadding = 10
        self.config.doPreConvolution = False
        self.config.nSigmaDetection = 5
        self.modifiedTask = DonutQuickMeasurementTask(
            config=self.config, name="Mod Task"
        )
        self.assertEqual(self.modifiedTask.config.initialCutoutPadding, 10)
        self.assertFalse(self.modifiedTask.config.doPreConvolution)
        self.assertEqual(self.modifiedTask.config.nSigmaDetection, 5)

    def testTaskTemplateError(self):
        with self.assertRaises(ValueError) as context:
            self.task.run(self.postIsrExp)
        self.assertEqual(
            str(
                "Template required if doPreConvolution "
                + "configuration parameter is set to True."
            ),
            str(context.exception),
        )

    def testTaskRunWithPreConvolve(self):
        template = self._getTemplate()

        output = self.task.run(self.postIsrExp, template)

        outputDf = pd.DataFrame.from_dict(output.detectedCatalog, orient="index")

        self.assertEqual(len(outputDf), 3)
        # Check centroids within 10 pixels of expected
        np.testing.assert_allclose(
            np.sort(outputDf["centroid_x"]), [617.0, 2814.0, 3815.0], atol=10
        )
        np.testing.assert_allclose(
            np.sort(outputDf["centroid_y"]), [398.0, 2198.0, 3196.0], atol=10
        )
        # All detected donuts should be same magnitude.
        # Check aperture fluxes are all within 10% of one another.
        relFluxDiff = outputDf["apFlux70"] / np.max(outputDf["apFlux70"])
        np.testing.assert_allclose(relFluxDiff, 1.0, atol=0.1)

    def testTaskRunWithoutPreConvolve(self):
        self.task.config.doPreConvolution = False
        output = self.task.run(self.postIsrExp)

        outputDf = pd.DataFrame.from_dict(output.detectedCatalog, orient="index")

        self.assertEqual(len(outputDf), 3)
        # Check centroids within 10 pixels of expected
        np.testing.assert_allclose(
            np.sort(outputDf["centroid_x"]), [617.0, 2814.0, 3815.0], atol=10
        )
        np.testing.assert_allclose(
            np.sort(outputDf["centroid_y"]), [398.0, 2198.0, 3196.0], atol=10
        )
        # All detected donuts should be same magnitude.
        # Check aperture fluxes are all within 10% of one another.
        relFluxDiff = outputDf["apFlux70"] / np.max(outputDf["apFlux70"])
        np.testing.assert_allclose(relFluxDiff, 1.0, atol=0.1)
