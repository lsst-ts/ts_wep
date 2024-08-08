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
import pandas as pd
from lsst.daf import butler as dafButler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CombineZernikesMeanTask,
    CombineZernikesSigmaClipTask,
    DonutStampSelectorTask,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesTieTaskScienceSensor(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutCatalog needed for task.
        """

        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCalcZernikesScienceSensorSetupPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Make sure we are using the right exposure+detector combinations
        pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
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
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106002,
            "visit": 4021123106002,
        }

    def testValidateConfigs(self):
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)
        self.assertEqual(type(self.task.donutStampSelector), DonutStampSelectorTask)

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

        zernCoeff = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes

        self.assertEqual(np.shape(zernCoeff), (len(donutStampsExtra), 19))

    def testCalcZernikes(self):
        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        structNormal = self.task.run(donutStampsIntra, donutStampsExtra)

        # check that 4 elements are created
        self.assertEqual(len(structNormal), 4)

        # check that donut quality is reported for all donuts
        self.assertEqual(len(structNormal.donutsExtraQuality), len(donutStampsExtra))
        self.assertEqual(len(structNormal.donutsIntraQuality), len(donutStampsIntra))

        # check that all desired quantities are included
        colnames = list(structNormal.donutsIntraQuality.columns)
        desired_colnames = [
            "SN",
            "ENTROPY",
            "ENTROPY_SELECT",
            "SN_SELECT",
            "FINAL_SELECT",
        ]
        np.testing.assert_array_equal(np.sort(colnames), np.sort(desired_colnames))

        # test null run
        structNull = self.task.run([], [])

        for struct in [structNormal, structNull]:
            # test that in accordance with declared connections,
            # donut quality tables are pandas dataFrame,
            # and Zernikes are numpy arrays
            # both for normal run and for null run
            self.assertIsInstance(struct.donutsIntraQuality, pd.DataFrame)
            self.assertIsInstance(struct.donutsExtraQuality, pd.DataFrame)
            self.assertIsInstance(struct.outputZernikesRaw, np.ndarray)
            self.assertIsInstance(struct.outputZernikesAvg, np.ndarray)

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
