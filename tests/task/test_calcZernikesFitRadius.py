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
from copy import copy

import lsst.utils.tests
import numpy as np
from lsst.daf import butler as dafButler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesFitRadius(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate donutCatalog needed for task.
        """
        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(cls.testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
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
            pipeCmd += ' -d "detector IN (191, 192)"'
            runProgram(pipeCmd)


    @classmethod
    def tearDownClass(cls):
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self):
        self.config = CalcZernikesTaskConfig()
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": 4021123106000,
            "visit": 4021123106000,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": 4021123106000,
            "visit": 4021123106000,
        }
        self.donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        self.donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

    def testEstimateZernikesEmpty(self):
        # We are passing empty donutStamps in intra so the zernikes
        # table should only contain the zernikesFromDonutRadius in
        # the average row.
        donutStampsEmpty = DonutStamps([], metadata=copy(self.donutStampsExtra.metadata))
        struct = self.task.run(
            self.donutStampsExtra, donutStampsEmpty
        )
        zkTable = struct.zernikes
        self.assertEqual(len(zkTable), 1.0)
        row = zkTable[zkTable['label'] == 'average'][0]
        for key in row.colnames:
            if key.startswith("Z") and key != "Z4":
                self.assertAlmostEqual(row[key].value, 0.0)
            elif key == 'Z4':
                self.assertAlmostEqual(row[key].value, 2050.4304)
        assert np.all(np.isnan(struct.outputZernikesAvg[0]))
        assert np.all(np.isnan(struct.outputZernikesRaw[0]))

    def testEstimateZernikesBadDonutQuality(self):
        # We are passing non-empty donutStamps in intra and extra
        # but setting the donut quality selection flag to false
        # so the zernikes
        # table should only contain the zernikesFromDonutRadius in
        # the average row.
        self.task.donutStampSelector.config.maxSelect = 0
        struct = self.task.run(
            self.donutStampsExtra, self.donutStampsIntra
        )

        zkTable = struct.zernikes
        self.assertEqual(len(zkTable), 1.0)
        row = zkTable[zkTable['label'] == 'average'][0]
        for key in row.colnames:
            if key.startswith("Z") and key != "Z4":
                self.assertAlmostEqual(row[key].value, 0.0)
            elif key == 'Z4':
                self.assertAlmostEqual(row[key].value, 1905.5042)
        assert np.all(np.isnan(struct.outputZernikesAvg[0]))
        assert np.all(np.isnan(struct.outputZernikesRaw[0]))
