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

import astropy.units as u
import lsst.utils.tests
import numpy as np
from astropy.table import QTable
from lsst.daf import butler as dafButler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CalcZernikesUnpairedTask,
    CalcZernikesUnpairedTaskConfig,
    EstimateZernikesDanishTask,
    EstimateZernikesTieTask,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikeUnpaired(lsst.utils.tests.TestCase):
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
        cls.runName = "run1"
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        cls.cameraName = "LSSTCam"
        pipelineYaml = os.path.join(
            testPipelineConfigDir, "testCutoutsUnpairedPipeline.yaml"
        )
        if "pretest_run_science" in collectionsList:
            pipelineYaml += "#cutOutDonutsUnpairedTask"
            collections += ",pretest_run_science"

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

    def testWithAndWithoutPairs(self):
        # Load data from butler
        donutStampsExtra = self.butler.get(
            "donutStamps", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStamps", dataId=self.dataIdIntra, collections=[self.runName]
        )

        # Loop over EstimateZernikes subtasks
        for subtask in [EstimateZernikesTieTask, EstimateZernikesDanishTask]:
            # Calculate Zernikes with stamps paired
            config = CalcZernikesTaskConfig()
            config.estimateZernikes.retarget(subtask)
            pairedTask = CalcZernikesTask(config=config)

            pairedZk = pairedTask.run(donutStampsExtra, donutStampsIntra)
            pairedZk = pairedZk.outputZernikesAvg

            # Calculate Zernikes with stamps unpaired
            config = CalcZernikesUnpairedTaskConfig()
            config.estimateZernikes.retarget(subtask)
            unpairedTask = CalcZernikesUnpairedTask(config=config)

            extraZk = unpairedTask.run(donutStampsExtra).outputZernikesAvg
            intraZk = unpairedTask.run(donutStampsIntra).outputZernikesAvg
            meanZk = np.mean([extraZk, intraZk], axis=0)

            # Check that results are similar
            diff = np.sqrt(np.sum((meanZk - pairedZk) ** 2))
            self.assertLess(diff, 0.17)

    def testTable(self):
        # Load data from butler
        donutStampsExtra = self.butler.get(
            "donutStamps", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStamps", dataId=self.dataIdIntra, collections=[self.runName]
        )

        # Loop over EstimateZernikes subtasks
        for subtask in [EstimateZernikesTieTask, EstimateZernikesDanishTask]:
            for stamps in [donutStampsExtra, donutStampsIntra]:
                # Calculate Zernikes with stamps unpaired
                config = CalcZernikesUnpairedTaskConfig()
                config.estimateZernikes.retarget(subtask)
                task = CalcZernikesUnpairedTask(config=config)
                structNormal = task.run(stamps)

                # check that 4 elements are created
                self.assertEqual(len(structNormal), 4)

                zkAvg1 = structNormal.outputZernikesAvg[0]
                zkAvgRow = structNormal.zernikes[
                    structNormal.zernikes["label"] == "average"
                ][0]
                zkAvg2 = np.array(
                    [zkAvgRow[f"Z{i}"].to_value(u.micron) for i in range(4, 29)]
                )
                self.assertFloatsAlmostEqual(zkAvg1, zkAvg2, rtol=1e-6, atol=0)

                zkRaw1 = structNormal.outputZernikesRaw
                zkRaw2 = np.full_like(zkRaw1, np.nan)
                i = 0
                for row in structNormal.zernikes:
                    if row["label"] == "average":
                        continue
                    zkRaw2[i] = np.array(
                        [row[f"Z{i}"].to_value(u.micron) for i in range(4, 29)]
                    )
                    i += 1
                self.assertFloatsAlmostEqual(zkRaw1, zkRaw2, rtol=1e-6, atol=0)

                # verify remaining desired columns exist in zernikes table
                desired_colnames = [
                    "used",
                    "intra_field",
                    "extra_field",
                    "intra_centroid",
                    "extra_centroid",
                    "intra_mag",
                    "extra_mag",
                    "intra_sn",
                    "extra_sn",
                    "intra_entropy",
                    "extra_entropy",
                    "intra_frac_bad_pix",
                    "extra_frac_bad_pix",
                    "intra_max_power_grad",
                    "extra_max_power_grad",
                ]
                self.assertLessEqual(
                    set(desired_colnames), set(structNormal.zernikes.colnames)
                )

                # Check metadata keys exist
                self.assertIn("cam_name", structNormal.zernikes.meta)
                for k in ["intra", "extra"]:
                    dict_ = structNormal.zernikes.meta[k]
                    if k == stamps.metadata['DFC_TYPE']:
                        self.assertIn("det_name", dict_)
                        self.assertIn("visit", dict_)
                        self.assertIn("dfc_dist", dict_)
                        self.assertIn("band", dict_)
                    else:
                        self.assertEqual(dict_, {})

                # Turn on the donut stamp selector
                task.doDonutStampSelector = True
                structSelect = task.run(stamps)
                # check that donut quality is reported for all donuts
                self.assertEqual(
                    len(structSelect.donutQualityTable),
                    len(stamps),
                )

                # check that all desired quantities are included
                colnames = list(structSelect.donutQualityTable.columns)
                desired_colnames = [
                    "SN",
                    "ENTROPY",
                    "ENTROPY_SELECT",
                    "SN_SELECT",
                    "FRAC_BAD_PIX",
                    "FRAC_BAD_PIX_SELECT",
                    "MAX_POWER_GRAD",
                    "MAX_POWER_GRAD_SELECT",
                    "FINAL_SELECT",
                    "DEFOCAL_TYPE",
                    "RADIUS",
                    "X_PIX_LEFT_EDGE",
                    "X_PIX_RIGHT_EDGE",
                    "RADIUS_FAIL_FLAG",
                ]
                np.testing.assert_array_equal(
                    np.sort(colnames), np.sort(desired_colnames)
                )

                # test null run
                structNull = task.run([])

                for struct in [structNormal, structNull]:
                    # test that in accordance with declared connections,
                    # donut quality table is an astropy QTable,
                    # and Zernikes are numpy arrays
                    # both for normal run and for null run
                    self.assertIsInstance(struct.donutQualityTable, QTable)
                    self.assertIsInstance(struct.outputZernikesRaw, np.ndarray)
                    self.assertIsInstance(struct.outputZernikesAvg, np.ndarray)
                    self.assertIsInstance(struct.zernikes, QTable)
