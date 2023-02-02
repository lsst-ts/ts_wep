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
import numpy as np

import lsst.utils.tests
from lsst.daf import butler as dafButler
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.task.CalcZernikesTask import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
)
from lsst.ts.wep.task.CombineZernikesMeanTask import CombineZernikesMeanTask
from lsst.ts.wep.task.CombineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.Utility import (
    getModulePath,
    runProgram,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestCalcZernikesTaskCwfs(lsst.utils.tests.TestCase):
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
            testPipelineConfigDir, "testCalcZernikesCwfsPipeline.yaml"
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

    def testCalcBlendOffsets(self):

        # Test with no blend centroid
        donutStampExtra = self.donutStampsExtra[0]
        eulerAngle = 0
        blendOffsetsNone = self.task.calcBlendOffsets(donutStampExtra, eulerAngle)
        np.testing.assert_array_almost_equal(
            blendOffsetsNone, np.array([["nan"], ["nan"]], dtype=float)
        )
        # Test with blend centroid present
        trueOffset = np.array([[10.0], [10.0]])
        eulerAngle = 0
        centroidPos = donutStampExtra.centroid_position
        centroid10PixOffset = np.array([[centroidPos.x], [centroidPos.y]]) + trueOffset
        donutStampExtra.blend_centroid_positions = centroid10PixOffset.T
        blendOffsetsCentroid = self.task.calcBlendOffsets(donutStampExtra, eulerAngle)
        np.testing.assert_array_almost_equal(blendOffsetsCentroid, trueOffset)
        # Test with euler angle non-zero
        eulerAngle = 180
        blendOffsetsCentroid = self.task.calcBlendOffsets(donutStampExtra, eulerAngle)
        np.testing.assert_array_almost_equal(blendOffsetsCentroid, -1.0 * trueOffset)

    def testEstimateZernikes(self):

        zernCoeff = self.task.estimateZernikes(
            self.donutStampsExtra, self.donutStampsIntra
        )

        self.assertEqual(np.shape(zernCoeff), (len(self.donutStampsExtra), 19))

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
        zernCoeffAllR04 = self.task.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernCoeffAvgR04 = self.task.combineZernikes.run(
            zernCoeffAllR04
        ).combinedZernikes
        trueZernCoeffR04 = np.array(
            [
                -0.71201408,
                1.12248525,
                0.77794367,
                -0.04085477,
                -0.05272933,
                0.16054277,
                0.081405,
                -0.04382461,
                -0.04830676,
                -0.06218882,
                0.10246469,
                0.0197683,
                0.007953,
                0.00668697,
                -0.03570788,
                -0.03020376,
                0.0039522,
                0.04793133,
                -0.00804605,
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
        zernCoeffAllR40 = self.task.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernCoeffAvgR40 = self.task.combineZernikes.run(
            zernCoeffAllR40
        ).combinedZernikes
        trueZernCoeffR40 = np.array(
            [
                -0.6535694,
                1.00838499,
                0.55968811,
                -0.08899825,
                0.00173607,
                0.04133107,
                -0.10913093,
                -0.04363778,
                -0.03149601,
                -0.04941225,
                0.09980538,
                0.03704486,
                -0.00210766,
                0.01737253,
                0.01727539,
                0.01278011,
                0.01212878,
                0.03876888,
                -0.00559142,
            ]
        )
        # Make sure the total rms error is less than 0.35 microns off
        # from the OPD truth as a sanity check
        self.assertLess(
            np.sqrt(np.sum(np.square(zernCoeffAvgR40 - trueZernCoeffR40))), 0.35
        )

    def testGetCombinedZernikes(self):

        testArr = np.zeros((2, 19))
        testArr[1] += 2.0
        combinedZernikesStruct = self.task.getCombinedZernikes(testArr)
        np.testing.assert_array_equal(
            combinedZernikesStruct.combinedZernikes, np.ones(19)
        )
        np.testing.assert_array_equal(
            combinedZernikesStruct.flags, np.zeros(len(testArr))
        )
