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
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCalcZernikesTieTaskCwfs(lsst.utils.tests.TestCase):
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
        self.assertEqual(type(self.task.combineZernikes), CombineZernikesSigmaClipTask)

        self.config.combineZernikes.retarget(CombineZernikesMeanTask)
        self.task = CalcZernikesTask(config=self.config, name="Base Task")

        self.assertEqual(type(self.task.combineZernikes), CombineZernikesMeanTask)

        self.config.estimateZernikes.binning = 2
        self.assertEqual(self.task.estimateZernikes.wfAlgoConfig.binning, 2)

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
                -3.98568849e-01,
                3.29880192e-02,
                6.86914803e-01,
                3.90220393e-02,
                -7.50009174e-02,
                1.81549349e-01,
                1.39330112e-01,
                1.29821719e-05,
                -8.61263006e-03,
                -5.29780246e-03,
                4.29874356e-02,
                9.03785421e-02,
                3.26618090e-03,
                4.18176969e-04,
                -2.90643396e-02,
                -2.52572372e-02,
                -2.51558908e-02,
                4.00444469e-02,
                -6.71305522e-03,
                -1.87932168e-02,
                2.77894381e-04,
                -2.12649168e-02,
                9.96417120e-03,
                -1.45702414e-02,
                -1.05431895e-02,
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
                -4.51261752e-01,
                2.16216207e-01,
                4.66819698e-01,
                -9.23288192e-02,
                4.97984634e-02,
                9.33841073e-02,
                -4.89458314e-02,
                7.17956342e-03,
                1.74395927e-02,
                -2.81949251e-02,
                9.53899106e-03,
                1.29594074e-01,
                -5.42637866e-03,
                1.21212094e-02,
                -1.66611416e-04,
                2.34503797e-02,
                3.56519007e-03,
                3.64561520e-02,
                -8.42437070e-03,
                -1.58178023e-02,
                -8.84574527e-03,
                -1.24650851e-02,
                -2.66815820e-04,
                1.86351265e-04,
                1.70769973e-03,
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
        zkAllExtra = self.task.estimateZernikes.run(donutStampsExtra, []).zernikes
        zkAvgExtra = self.task.combineZernikes.run(zkAllExtra).combinedZernikes
        zkAllIntra = self.task.estimateZernikes.run([], donutStampsIntra).zernikes
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

    def testUnevenPairs(self):
        # Test for when you have more of either extra or intra
        # Load the test data
        stampsExtra = self.donutStampsExtra
        stampsIntra = self.donutStampsIntra

        # Increase length of extra list
        stampsExtra.extend([stampsExtra[0]])

        # Now estimate Zernikes
        self.task.run(stampsExtra, stampsIntra)

    def testRequireConverge(self):
        config = CalcZernikesTaskConfig()
        config.estimateZernikes.requireConverge = True  # Require to converge
        config.estimateZernikes.convergeTol = 0  # But don't allow convergence
        task = CalcZernikesTask(config=config, name="Test requireConverge")

        # Estimate zernikes
        donutStampDir = os.path.join(self.testDataDir, "donutImg", "donutStamps")
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW1_donutStamps.fits")
        )
        output = task.estimateZernikes.run(donutStampsExtra, donutStampsIntra)
        zernikes = output.zernikes

        # Everything should be NaN because we did not converge
        self.assertTrue(np.isnan(zernikes).all())

    def testNollIndices(self):
        # Load the stamps
        donutStampDir = os.path.join(self.testDataDir, "donutImg", "donutStamps")
        donutStampsExtra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW0_donutStamps.fits")
        )
        donutStampsIntra = DonutStamps.readFits(
            os.path.join(donutStampDir, "R04_SW1_donutStamps.fits")
        )

        # Estimate Zernikes 4, 5, 6
        self.task.config.estimateZernikes.nollIndices = [4, 5, 6]
        zk0 = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes[0]

        # Estimate Zernikes 4, 5, 6, 20, 21
        self.task.config.estimateZernikes.nollIndices = [4, 5, 6, 20, 21]
        zk1 = self.task.estimateZernikes.run(
            donutStampsExtra, donutStampsIntra
        ).zernikes[0]

        # Check lengths
        self.assertEqual(len(zk0), 3)
        self.assertEqual(len(zk1), 5)

        # Check that 4, 5, 6 are independent of 20, 21 at less
        self.assertLess(np.sqrt(np.sum(np.square(zk1[:-2] - zk0))), 0.020)
        # self.assertTrue(np.all(np.abs(zk1[:3] - zk0) < 0.035))
