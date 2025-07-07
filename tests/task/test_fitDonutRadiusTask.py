import os

import astropy
import lsst.utils.tests
import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.wep.task.fitDonutRadiusTask import (
    FitDonutRadiusTask,
    FitDonutRadiusTaskConfig,
)
from lsst.ts.wep.utils import getModulePath, runProgram, writeCleanUpRepoCmd


class TestFitDonutRadiusTaskScienceSensor(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    runNameScience: str
    baseRunNameScience: str
    runNameCwfs: str
    baseRunNameCwfs: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate donut stamps needed for task.
        """
        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        cls.repoDir = os.path.join(cls.testDataDir, "gen3TestRepo")
        cls.runNameScience = "run1"
        cls.baseRunNameScience = "pretest_run_science"
        cls.runNameCwfs = "run2"
        cls.baseRunNameCwfs = "pretest_run_cwfs"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        for runName in [cls.runNameScience, cls.runNameCwfs]:
            if runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, runName)
                runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = FitDonutRadiusTaskConfig()
        self.task = FitDonutRadiusTask(config=self.config, name="Base Task")

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtraScience = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.dataIdExtraCwfs = {
            "instrument": "LSSTCam",
            "detector": 191,
            "exposure": 4021123106000,
            "visit": 4021123106000,
        }

    def testValidateConfigs(self) -> None:
        # Test default value
        self.OrigTask = FitDonutRadiusTask(config=self.config, name="Orig Task")
        self.assertEqual(self.OrigTask.widthMultiplier, 0.8)
        self.assertEqual(self.OrigTask.filterSigma, 3)
        self.assertEqual(self.OrigTask.minPeakWidth, 5)
        self.assertEqual(self.OrigTask.minPeakHeight, 0.3)
        self.assertEqual(self.OrigTask.nAngles, 10)
        self.assertEqual(self.OrigTask.binning, 1)

        # Test changing configs
        self.config.widthMultiplier = 2.00
        self.config.filterSigma = 4
        self.config.minPeakWidth = 7
        self.config.minPeakHeight = 0.8
        self.config.nAngles = 20
        self.config.binning = 2
        self.ModifiedTask = FitDonutRadiusTask(config=self.config, name="Mod Task")
        self.assertEqual(self.ModifiedTask.widthMultiplier, 2.00)
        self.assertEqual(self.ModifiedTask.filterSigma, 4)
        self.assertEqual(self.ModifiedTask.minPeakWidth, 7)
        self.assertEqual(self.ModifiedTask.minPeakHeight, 0.8)
        self.assertEqual(self.ModifiedTask.nAngles, 20)
        self.assertEqual(self.ModifiedTask.binning, 2)

    def testTaskRunScienceSensor(self) -> None:
        donutStampsExtra = self.butler.get(
            "donutStampsExtra",
            dataId=self.dataIdExtraScience,
            collections=[self.baseRunNameScience],
        )

        taskOut = self.task.run(donutStampsExtra)
        self.assertEqual(type(taskOut.donutRadiiTable), astropy.table.table.QTable)

        self.assertEqual(len(taskOut.donutRadiiTable), 3)

        # test that the expected columns are present
        expected_columns = [
            "DET_NAME",
            "DFC_DIST",
            "RADIUS",
            "X_PIX_LEFT_EDGE",
            "X_PIX_RIGHT_EDGE",
            "FAIL_FLAG",
        ]
        self.assertLessEqual(
            set(expected_columns), set(taskOut.donutRadiiTable.colnames)
        )
        # test that correct detector names are present
        self.assertEqual(
            set(np.unique(taskOut.donutRadiiTable["DET_NAME"].value)),
            set(["R22_S11"]),
        )
        self.assertEqual(
            set(np.unique(taskOut.donutRadiiTable["FAIL_FLAG"].value)),
            set([0]),
        )
        # test that the mean radius is correct
        np.testing.assert_allclose(
            69.69552203692189,
            np.mean(taskOut.donutRadiiTable["RADIUS"]),
            rtol=1e-1,
            atol=1e-1,
        )

    def testTaskRunCwfs(self) -> None:
        donutStampsIntra = self.butler.get(
            "donutStampsIntra",
            dataId=self.dataIdExtraCwfs,
            collections=[self.baseRunNameCwfs],
        )

        taskOut = self.task.run(donutStampsIntra)
        self.assertEqual(type(taskOut.donutRadiiTable), astropy.table.table.QTable)

        self.assertEqual(len(taskOut.donutRadiiTable), 2)

        expected_columns = [
            "DET_NAME",
            "DFC_DIST",
            "RADIUS",
            "X_PIX_LEFT_EDGE",
            "X_PIX_RIGHT_EDGE",
            "FAIL_FLAG",
        ]
        self.assertLessEqual(
            set(expected_columns), set(taskOut.donutRadiiTable.colnames)
        )
        # test that correct detector names are present
        self.assertEqual(
            set(np.unique(taskOut.donutRadiiTable["DET_NAME"].value)),
            set(["R00_SW1"]),
        )
        self.assertEqual(
            set(np.unique(taskOut.donutRadiiTable["FAIL_FLAG"].value)),
            set([0]),
        )
        np.testing.assert_allclose(
            66.38605539619834,
            np.mean(taskOut.donutRadiiTable["RADIUS"]),
            rtol=1e-1,
            atol=1e-1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for runName in [cls.runNameScience, cls.runNameCwfs]:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, runName)
            runProgram(cleanUpCmd)
