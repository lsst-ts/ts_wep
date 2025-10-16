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

import astropy.units as u
import lsst.utils.tests
import numpy as np
from astropy.table import QTable
from lsst.daf.butler import Butler
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CombineZernikesMeanTask,
    CombineZernikesSigmaClipTask,
    DonutStamps,
    DonutStampSelectorTask,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesTieTaskScienceSensor(lsst.utils.tests.TestCase):
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
        if "pretest_run_science" in collectionsList:
            cls.runName = "pretest_run_science"
        else:
            cls.runName = "run1"
            if cls.runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
                runProgram(cleanUpCmd)

            collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
            instrument = "lsst.obs.lsst.LsstCam"
            pipelineYaml = os.path.join(
                testPipelineConfigDir, "testCalcZernikesScienceSensorSetupPipeline.yaml"
            )

            pipeCmd = writePipetaskCmd(
                cls.repoDir,
                cls.runName,
                instrument,
                collections,
                pipelineYaml=pipelineYaml,
            )
            # Make sure we are using the right exposure+detector combinations
            pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
            pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
            runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = CalcZernikesTaskConfig()
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = Butler.from_config(self.repoDir)
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

    def testValidateConfigs(self) -> None:
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)
        self.assertEqual(type(self.task.donutStampSelector), DonutStampSelectorTask)
        self.assertEqual(self.task.doDonutStampSelector, True)

    def testEstimateZernikes(self) -> None:
        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        # Use dataIdExtra here too because both sets of donutStamps
        # get saved to extraFocal dataId so we can run this task
        # in parallel across detector pairs of the same visit.
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )

        zernCoeff = self.task.estimateZernikes.run(donutStampsExtra, donutStampsIntra).zernikes

        self.assertEqual(np.shape(zernCoeff), (len(donutStampsExtra), 25))

    def testCalcZernikes(self) -> None:
        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIdExtra, collections=[self.runName]
        )
        structNormal = self.task.run(donutStampsIntra, donutStampsExtra)

        # check that 4 elements are created
        self.assertEqual(len(structNormal), 4)

        zkAvg1 = structNormal.outputZernikesAvg[0]
        zkAvgRow = structNormal.zernikes[structNormal.zernikes["label"] == "average"][0]
        zkAvg2 = np.array([zkAvgRow[f"Z{i}"].to_value(u.micron) for i in range(4, 29)])
        np.testing.assert_allclose(zkAvg1, zkAvg2, rtol=1e-6, atol=0)

        zkRaw1 = structNormal.outputZernikesRaw
        zkRaw2 = np.full_like(zkRaw1, np.nan)
        i = 0
        for row in structNormal.zernikes:
            if row["label"] == "average":
                continue
            zkRaw2[i] = np.array([row[f"Z{i}"].to_value(u.micron) for i in range(4, 29)])
            i += 1
        np.testing.assert_allclose(zkRaw1, zkRaw2, rtol=1e-6, atol=0)

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
        self.assertLessEqual(set(desired_colnames), set(structNormal.zernikes.colnames))

        # Check metadata keys exist
        self.assertIn("cam_name", structNormal.zernikes.meta)
        for k in ["intra", "extra"]:
            dict_ = structNormal.zernikes.meta[k]
            self.assertIn("det_name", dict_)
            self.assertIn("visit", dict_)
            self.assertIn("dfc_dist", dict_)
            self.assertIn("band", dict_)

        # Turn on the donut stamp selector
        self.task.doDonutStampSelector = True
        structSelect = self.task.run(donutStampsIntra, donutStampsExtra)
        # check that donut quality is reported for all donuts
        self.assertEqual(
            len(structSelect.donutQualityTable),
            len(donutStampsExtra) + len(donutStampsIntra),
        )
        # Check DEFOCAL_TYPE assigned properly
        extra_count = len(np.where(structSelect.donutQualityTable["DEFOCAL_TYPE"] == "extra")[0])
        self.assertEqual(extra_count, len(donutStampsExtra))

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
        ]
        np.testing.assert_array_equal(np.sort(colnames), np.sort(desired_colnames))

        # test null run
        structNull = self.task.run(
            DonutStamps([], metadata=copy(donutStampsExtra.metadata)),
            DonutStamps([], metadata=copy(donutStampsExtra.metadata)),
        )

        for struct in [structNormal, structNull]:
            # test that in accordance with declared connections,
            # donut quality table is an astropy QTable,
            # and Zernikes are numpy arrays
            # both for normal run and for null run
            self.assertIsInstance(struct.donutQualityTable, QTable)
            self.assertIsInstance(struct.outputZernikesRaw, np.ndarray)
            self.assertIsInstance(struct.outputZernikesAvg, np.ndarray)
            self.assertIsInstance(struct.zernikes, QTable)
        self.assertEqual(len(structNormal.donutQualityTable), 6)
        self.assertEqual(len(structNull.donutQualityTable), 0)

        self.config.donutStampSelector.maxSelect = 0
        self.task = CalcZernikesTask(config=self.config)
        structAllDonutsFail = self.task.run(donutStampsIntra, donutStampsExtra)
        self.assertEqual(len(structAllDonutsFail.donutQualityTable), 6)

    def testGetCombinedZernikes(self) -> None:
        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.combineZernikes.run(testArr)
        np.testing.assert_array_equal(combinedZernikesStruct.combinedZernikes, np.ones(19))
        np.testing.assert_array_equal(combinedZernikesStruct.flags, np.zeros(len(testArr)))
