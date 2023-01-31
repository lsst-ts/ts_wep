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
from lsst.ts.wep.Utility import getModulePath
from lsst.ip.isr.isrTask import IsrTask
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from lsst.ts.wep.Utility import DonutTemplateType, DefocalType
from lsst.ts.wep.task.DonutQuickMeasurementTask import (
    DonutQuickMeasurementTaskConfig,
    DonutQuickMeasurementTask,
)


class TestDonutQuickMeasurementTask(unittest.TestCase):
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
            "detector": 94,
            "exposure": 4021123106001,
        }
        self.testExposure = self.butler.get(
            "raw", dataId=testDataId, collections="LSSTCam/raw/all"
        )

    def _getData(self, runIsr=True):

        isrTask = IsrTask()
        isrTask.config.doBias = False
        isrTask.config.doVariance = False
        isrTask.config.doLinearize = False
        isrTask.config.doCrosstalk = False
        isrTask.config.doDefect = False
        isrTask.config.doNanMasking = False
        isrTask.config.doInterpolate = False
        isrTask.config.doBrighterFatter = False
        isrTask.config.doDark = False
        isrTask.config.doFlat = False
        isrTask.config.doApplyGains = True
        isrTask.config.doOverscan = True
        isrTask.config.overscan.fitType = "MEDIAN"
        postIsrExp = isrTask.run(self.testExposure)

        # Create template
        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )

        # Set inst information
        instParams = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": 1.0,
            "pixelSize": 10.0e-6,
        }

        template = templateMaker.makeTemplate(
            "R22_S11", DefocalType.Extra, 160, instParams=instParams
        )

        return postIsrExp.outputExposure, template

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
            self.task.run(self.testExposure)
        self.assertEqual(
            str(
                "Template required if doPreConvolution "
                + "configuration parameter is set to True."
            ),
            str(context.exception),
        )

    def testTaskRunWithPreConvolve(self):

        postIsrExp, template = self._getData()

        output = self.task.run(postIsrExp, template)

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

        postIsrExp, template = self._getData()

        self.task.config.doPreConvolution = False
        output = self.task.run(postIsrExp)

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
