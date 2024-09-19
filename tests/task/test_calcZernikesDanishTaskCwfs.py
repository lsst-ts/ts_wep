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
from lsst.daf import butler as dafButler
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
            testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml"
        )

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += ' -d "detector IN (191, 192, 195, 196, 199, 200, 203, 204)"'
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls):
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)

    def setUp(self):
        self.config = CalcZernikesTaskConfig()
        self.config.estimateZernikes.retarget(EstimateZernikesDanishTask)
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

    def testValidateConfigs(self):
        self.assertEqual(type(self.task.estimateZernikes), EstimateZernikesDanishTask)
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

    def testEstimateZernikes(self):
        zernCoeff = self.task.estimateZernikes.run(
            self.donutStampsExtra, self.donutStampsIntra
        ).zernikes

        self.assertEqual(np.shape(zernCoeff), (len(self.donutStampsExtra), 25))

    def testEstimateCornerZernikes(self):
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

    def testGetCombinedZernikes(self):
        testArr = np.zeros((2, 25))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.combineZernikes.run(testArr)
        np.testing.assert_array_equal(
            combinedZernikesStruct.combinedZernikes, np.ones(25)
        )
        np.testing.assert_array_equal(
            combinedZernikesStruct.flags, np.zeros(len(testArr))
        )

    def testWithAndWithoutPairs(self):
        # Load the test data
        donutStampDir = os.path.join(self.testDataDir, "donutImg", "donutStamps")
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW1_donutStamps.fits")
        )

        # First estimate without pairs
        self.config.estimateZernikes.usePairs = False
        zkAllWithoutPairs = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes
        zkAvgWithoutPairs = self.task.combineZernikes.run(
            zkAllWithoutPairs
        ).combinedZernikes

        # Now estimate with pairs
        self.config.estimateZernikes.usePairs = True
        zkAllWithPairs = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes
        zkAvgWithPairs = self.task.combineZernikes.run(zkAllWithPairs).combinedZernikes

        # Check that without pairs has at least twice the number of zernikes
        self.assertEqual(zkAllWithoutPairs.shape[1], zkAllWithPairs.shape[1])
        self.assertGreaterEqual(zkAllWithoutPairs.shape[0], 2 * zkAllWithPairs.shape[0])

        # Check that the averages are the same
        self.assertTrue(np.allclose(zkAvgWithoutPairs, zkAvgWithPairs))
