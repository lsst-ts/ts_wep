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
    EstimateZernikesDanishTask,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesDanishTaskCwfs(lsst.utils.tests.TestCase):
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
    def tearDownClass(cls) -> None:
        if cls.runName == "run1":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = CalcZernikesTaskConfig()
        self.config.estimateZernikes.retarget(EstimateZernikesDanishTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.butler = Butler.from_config(self.repoDir)
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

    def testValidateConfigs(self) -> None:
        self.assertEqual(type(self.task.estimateZernikes), EstimateZernikesDanishTask)
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

    def testEstimateZernikes(self) -> None:
        zernCoeff = self.task.estimateZernikes.run(
            self.donutStampsExtra, self.donutStampsIntra
        ).zernikes

        self.assertEqual(np.shape(zernCoeff), (len(self.donutStampsExtra), 25))

    def testEstimateCornerZernikes(self) -> None:
        """
        Test the rotated corner sensors (R04 and R40) and make sure no changes
        upstream in obs_lsst have created issues in Zernike estimation.
        """

        donutStampDir = os.path.join(self.testDataDir, "donutImg", "donutStamps")

        # Test R04
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW1_donutStamps.fits")
        )
        zernCoeffAllR04 = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes
        zernCoeffAvgR04 = self.task.combineZernikes.run(
            zernCoeffAllR04
        ).combinedZernikes
        trueZernCoeffR04 = np.array(
            [
                -0.39401388,
                0.36051539,
                0.60247446,
                0.01628614,
                -0.00294667,
                0.20695479,
                0.15891274,
                0.00473219,
                -0.00297377,
                -0.03348815,
                0.01690553,
                0.10845509,
                0.00102616,
                -0.00204221,
                -0.02738544,
                -0.0324347,
                -0.01002763,
                0.02291608,
                -0.00589446,
                -0.00884343,
                -0.00322051,
                -0.0122419,
                0.00535912,
                0.00531382,
                -0.00154533,
            ]
        )

        # Make sure the total rms error is less than 0.35 microns off
        # from the OPD truth as a sanity check
        self.assertLess(
            np.sqrt(np.sum(np.square(zernCoeffAvgR04 - trueZernCoeffR04))), 0.35
        )

        # Test R40
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R40_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R40_SW1_donutStamps.fits")
        )
        zernCoeffAllR40 = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes
        zernCoeffAvgR40 = self.task.combineZernikes.run(
            zernCoeffAllR40
        ).combinedZernikes
        trueZernCoeffR40 = np.array(
            [
                -0.39401388,
                0.36051539,
                0.60247446,
                0.01628614,
                -0.00294667,
                0.20695479,
                0.15891274,
                0.00473219,
                -0.00297377,
                -0.03348815,
                0.01690553,
                0.10845509,
                0.00102616,
                -0.00204221,
                -0.02738544,
                -0.0324347,
                -0.01002763,
                0.02291608,
                -0.00589446,
                -0.00884343,
                -0.00322051,
                -0.0122419,
                0.00535912,
                0.00531382,
                -0.00154533,
            ]
        )

        # Make sure the total rms error is less than 0.35 microns off
        # from the OPD truth as a sanity check
        self.assertLess(
            np.sqrt(np.sum(np.square(zernCoeffAvgR40 - trueZernCoeffR40))), 0.35
        )

    def testGetCombinedZernikes(self) -> None:
        testArr = np.zeros((2, 25))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.combineZernikes.run(testArr)
        np.testing.assert_array_equal(
            combinedZernikesStruct.combinedZernikes, np.ones(25)
        )
        np.testing.assert_array_equal(
            combinedZernikesStruct.flags, np.zeros(len(testArr))
        )

    def testWithAndWithoutPairs(self) -> None:
        # Load the test data
        donutStampDir = os.path.join(self.testDataDir, "donutImg", "donutStamps")
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW1_donutStamps.fits")
        )

        # First estimate without pairs
        emptyStamps = DonutStamps([])
        zkAllExtra = self.task.estimateZernikes.run(
            donutStampsExtra, emptyStamps
        ).zernikes
        zkAvgExtra = self.task.combineZernikes.run(zkAllExtra).combinedZernikes
        zkAllIntra = self.task.estimateZernikes.run(
            emptyStamps, donutStampsIntra
        ).zernikes
        zkAvgIntra = self.task.combineZernikes.run(zkAllIntra).combinedZernikes

        # Now estimate with pairs
        zkAllPairs = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes
        zkAvgPairs = self.task.combineZernikes.run(zkAllPairs).combinedZernikes

        # Check that all have same number of Zernike coeffs
        self.assertEqual(zkAllExtra.shape[1], zkAllPairs.shape[1])
        self.assertEqual(zkAllIntra.shape[1], zkAllPairs.shape[1])
        self.assertEqual(len(zkAvgExtra), len(zkAvgPairs))
        self.assertEqual(len(zkAvgIntra), len(zkAvgPairs))

        # Check that unpaired is at least as long as paired
        self.assertGreaterEqual(zkAllExtra.shape[0], zkAllPairs.shape[0])
        self.assertGreaterEqual(zkAllIntra.shape[0], zkAllPairs.shape[0])

        # Check that the averages are similar
        zkAvgUnpaired = np.mean([zkAvgExtra, zkAvgIntra], axis=0)
        self.assertLess(np.sqrt(np.sum(np.square(zkAvgPairs - zkAvgUnpaired))), 0.30)

    def testTableMetadata(self) -> None:
        # First estimate without pairs
        emptyStamps = DonutStamps([], metadata=self.donutStampsExtra.metadata)
        zkCalcExtra = self.task.run(self.donutStampsExtra, emptyStamps).zernikes
        zkCalcIntra = self.task.run(emptyStamps, self.donutStampsIntra).zernikes

        # Check metadata keys exist for extra case
        self.assertIn("cam_name", zkCalcExtra.meta)
        for k in ["intra", "extra"]:
            dict_ = zkCalcExtra.meta[k]
            self.assertIn("det_name", dict_)
            self.assertIn("visit", dict_)
            self.assertIn("dfc_dist", dict_)
            self.assertIn("band", dict_)
            self.assertEqual(dict_["mjd"], self.donutStampsExtra.metadata["MJD"])

        # Check metadata keys exist for intra case
        self.assertIn("cam_name", zkCalcIntra.meta)
        for k in ["intra", "extra"]:
            dict_ = zkCalcIntra.meta[k]
            self.assertIn("det_name", dict_)
            self.assertIn("visit", dict_)
            self.assertIn("dfc_dist", dict_)
            self.assertIn("band", dict_)
            self.assertEqual(dict_["mjd"], self.donutStampsIntra.metadata["MJD"])

        # Now estimate with pairs
        zkCalcPairs = self.task.run(
            self.donutStampsExtra, self.donutStampsIntra
        ).zernikes

        # Check metadata keys exist for pairs case
        self.assertIn("cam_name", zkCalcPairs.meta)
        self.assertIn("estimatorInfo", zkCalcPairs.meta)
        self.assertIn("fwhm", zkCalcPairs.meta["estimatorInfo"])
        self.assertEqual(2, len(zkCalcPairs.meta["estimatorInfo"]["fwhm"]))
        for stamps, k in zip(
            [self.donutStampsIntra, self.donutStampsExtra], ["intra", "extra"]
        ):
            dict_ = zkCalcPairs.meta[k]
            if k == stamps.metadata["DFC_TYPE"]:
                self.assertIn("det_name", dict_)
                self.assertIn("visit", dict_)
                self.assertIn("dfc_dist", dict_)
                self.assertIn("band", dict_)
                self.assertEqual(dict_["mjd"], stamps.metadata["MJD"])
