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
import unittest

from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.cwfs.compensableImage import CompensableImage
from lsst.ts.wep.cwfs.algorithm import Algorithm
from lsst.ts.wep.utility import getModulePath, getConfigDir, DefocalType, CamType


class TestAlgorithm(unittest.TestCase):
    """Test the Algorithm class."""

    def setUp(self):
        # Get the path of module
        self.modulePath = getModulePath()

        # Define the image folder and image names
        # Image data -- Don't know the final image format.
        # It is noted that image.readFile inuts is based on the txt file
        imageFolderPath = os.path.join(
            self.modulePath, "tests", "testData", "testImages", "LSST_NE_SN25"
        )
        intra_image_name = "z11_0.25_intra.txt"
        extra_image_name = "z11_0.25_extra.txt"

        # Define fieldXY: [1.185, 1.185] or [0, 0]
        # This is the position of donut on the focal plane in degree
        fieldXY = [1.185, 1.185]

        # Define the optical model: "paraxial", "onAxis", "offAxis"
        self.opticalModel = "offAxis"

        # Image files Path
        intra_image_file = os.path.join(imageFolderPath, intra_image_name)
        extra_image_file = os.path.join(imageFolderPath, extra_image_name)

        # Theree is the difference between intra and extra images
        # I1: intra_focal images, I2: extra_focal Images
        self.I1 = CompensableImage()
        self.I2 = CompensableImage()

        self.I1.setImg(fieldXY, DefocalType.Intra, imageFile=intra_image_file)
        self.I2.setImg(fieldXY, DefocalType.Extra, imageFile=extra_image_file)

        # Set up the instrument
        cwfsConfigDir = os.path.join(getConfigDir(), "cwfs")
        instDir = os.path.join(cwfsConfigDir, "instData")
        self.inst = Instrument()
        instConfigDict = {
            # Obscuration (inner_radius / outer_radius of M1M3)
            "obscuration": 0.61,
            # Focal length in m
            "focalLength": 10.312,
            # Aperture diameter in m
            "apertureDiameter": 8.36,
            # Defocal distance offset in mm
            "offset": 1.0,
            # Camera pixel size in m
            "pixelSize": 10.0e-6,
        }
        maskConfigFile = os.path.join(instDir, "lsst", "maskMigrate.yaml")

        self.inst.configFromDict(
            instConfigDict, self.I1.getImgSizeInPix(), CamType.LsstCam, maskConfigFile
        )

        # Set up the algorithm
        algoDir = os.path.join(cwfsConfigDir, "algo")

        self.algoExp = Algorithm(algoDir)
        self.algoExp.config("exp", self.inst)

        self.algoFft = Algorithm(algoDir)
        self.algoFft.config("fft", self.inst)

    def resetAlgorithmsAndImages(self):
        # Reset the algorithms and images to their initial state

        # Reset algorithms
        self.algoExp.reset()
        self.algoFft.reset()

        # Reset images
        fieldXY = [self.I1.fieldX, self.I1.fieldY]
        self.I1.setImg(
            fieldXY,
            self.I1.getDefocalType(),
            image=self.I1.getImgInit(),
        )
        self.I2.setImg(
            fieldXY,
            self.I2.getDefocalType(),
            image=self.I2.getImgInit(),
        )

    def testGetDebugLevel(self):
        self.assertEqual(self.algoExp.getDebugLevel(), 0)

    def testSetDebugLevel(self):
        self.algoExp.config("exp", self.inst, debugLevel=3)
        self.assertEqual(self.algoExp.getDebugLevel(), 3)

        self.algoExp.setDebugLevel(0)
        self.assertEqual(self.algoExp.getDebugLevel(), 0)

    def testGetZer4UpInNm(self):
        zer4UpNm = self.algoExp.getZer4UpInNm()
        self.assertTrue(isinstance(zer4UpNm, np.ndarray))

    def testGetPoissonSolverName(self):
        self.assertEqual(self.algoExp.getPoissonSolverName(), "exp")
        self.assertEqual(self.algoFft.getPoissonSolverName(), "fft")

    def testGetNumOfZernikes(self):
        self.assertEqual(self.algoExp.getNumOfZernikes(), 22)
        self.assertEqual(self.algoFft.getNumOfZernikes(), 22)

    def testGetZernikeTerms(self):
        zTerms = self.algoExp.getZernikeTerms()
        self.assertTrue(type(zTerms[0]), int)
        self.assertEqual(len(zTerms), self.algoExp.getNumOfZernikes())
        self.assertEqual(zTerms[1], 1)
        self.assertEqual(zTerms[-1], self.algoExp.getNumOfZernikes() - 1)

        zTerms = self.algoFft.getZernikeTerms()
        self.assertTrue(type(zTerms[0]), int)
        self.assertEqual(len(zTerms), self.algoExp.getNumOfZernikes())

    def testGetObsOfZernikes(self):
        self.assertEqual(self.algoExp.getObsOfZernikes(), self.inst.obscuration)
        self.assertEqual(self.algoFft.getObsOfZernikes(), self.inst.obscuration)

    def testGetNumOfOuterItr(self):
        self.assertEqual(self.algoExp.getNumOfOuterItr(), 14)
        self.assertEqual(self.algoFft.getNumOfOuterItr(), 14)

    def testGetNumOfInnerItr(self):
        self.assertEqual(self.algoFft.getNumOfInnerItr(), 6)

    def testGetFeedbackGain(self):
        self.assertEqual(self.algoExp.getFeedbackGain(), 0.6)
        self.assertEqual(self.algoFft.getFeedbackGain(), 0.6)

    def testGetOffAxisPolyOrder(self):
        self.assertEqual(self.algoExp.getOffAxisPolyOrder(), 10)
        self.assertEqual(self.algoFft.getOffAxisPolyOrder(), 10)

    def testGetCompensatorMode(self):
        self.assertEqual(self.algoExp.getCompensatorMode(), "zer")
        self.assertEqual(self.algoFft.getCompensatorMode(), "zer")

    def testGetCompSequence(self):
        compSequence = self.algoExp.getCompSequence()
        self.assertTrue(isinstance(compSequence, np.ndarray))
        self.assertEqual(compSequence.dtype, int)
        self.assertEqual(len(compSequence), self.algoExp.getNumOfOuterItr())
        self.assertEqual(compSequence[0], 4)
        self.assertEqual(compSequence[-1], 22)

        compSequence = self.algoFft.getCompSequence()
        self.assertEqual(len(compSequence), self.algoFft.getNumOfOuterItr())

    def testGetBoundaryThickness(self):
        self.assertEqual(self.algoExp.getBoundaryThickness(), 8)
        self.assertEqual(self.algoFft.getBoundaryThickness(), 1)

    def testGetFftDimension(self):
        self.assertEqual(self.algoFft.getFftDimension(), 128)

    def testGetSignalClipSequence(self):
        sumclipSequence = self.algoFft.getSignalClipSequence()
        self.assertTrue(isinstance(sumclipSequence, np.ndarray))
        self.assertEqual(len(sumclipSequence), self.algoExp.getNumOfOuterItr() + 1)
        self.assertEqual(sumclipSequence[0], 0.33)
        self.assertEqual(sumclipSequence[-1], 0.51)

    def testGetMaskScalingFactor(self):
        self.assertAlmostEqual(self.algoExp.getMaskScalingFactor(), 1.0939, places=4)
        self.assertAlmostEqual(self.algoFft.getMaskScalingFactor(), 1.0939, places=4)

    def testGetWavefrontMapEstiInIter0(self):
        self.assertRaises(RuntimeError, self.algoExp.getWavefrontMapEsti)

    def testGetWavefrontMapEstiAndResidual(self):
        self.algoExp.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)

        wavefrontMapEsti = self.algoExp.getWavefrontMapEsti()
        wavefrontMapEsti[np.isnan(wavefrontMapEsti)] = 0
        self.assertGreater(np.sum(np.abs(wavefrontMapEsti)), 4.8e-4)

        wavefrontMapResidual = self.algoExp.getWavefrontMapResidual()
        wavefrontMapResidual[np.isnan(wavefrontMapResidual)] = 0
        self.assertLess(np.sum(np.abs(wavefrontMapResidual)), 2.5e-6)

    def testItr0(self):
        self.algoExp.itr0(self.I1, self.I2, self.opticalModel)

        zer4UpNm = self.algoExp.getZer4UpInNm()
        self.assertEqual(np.sum(np.abs(np.rint(zer4UpNm) - self._getAnsItr0())), 0)

    def _getAnsItr0(self):
        return [
            31,
            -69,
            -21,
            84,
            44,
            -53,
            48,
            -146,
            6,
            10,
            13,
            -5,
            1,
            -12,
            -8,
            7,
            0,
            -6,
            11,
        ]

    def testNextItrWithOneIter(self):
        self.algoExp.nextItr(self.I1, self.I2, self.opticalModel, nItr=1)

        zer4UpNm = self.algoExp.getZer4UpInNm()
        self.assertEqual(np.sum(np.abs(np.rint(zer4UpNm) - self._getAnsItr0())), 0)

    def testNextItrWithTwoIter(self):
        self.algoExp.nextItr(self.I1, self.I2, self.opticalModel, nItr=2)
        zer4UpNm = self.algoExp.getZer4UpInNm()

        ansRint = [
            40,
            -80,
            -18,
            92,
            44.0,
            -52,
            54,
            -146,
            5,
            10,
            15,
            -3,
            -0,
            -12,
            -8,
            7,
            1,
            -3,
            12,
        ]
        self.assertEqual(np.sum(np.abs(np.rint(zer4UpNm) - ansRint)), 0)

    def testIter0AndNextIterToCheckReset(self):
        self.algoExp.itr0(self.I1, self.I2, self.opticalModel)
        tmp1 = self.algoExp.getZer4UpInNm()

        self.algoExp.nextItr(self.I1, self.I2, self.opticalModel, nItr=2)

        # itr0() should reset the images and ignore the effect from nextItr()
        self.algoExp.itr0(self.I1, self.I2, self.opticalModel)
        tmp2 = self.algoExp.getZer4UpInNm()

        difference = np.sum(np.abs(tmp1 - tmp2))
        self.assertEqual(difference, 0)

    def testRunItOfExp(self):
        self.algoExp.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)

        # Check the value
        zk = self.algoExp.getZer4UpInNm()
        self.assertEqual(int(zk[7]), -192)

    def testResetAfterFullCalc(self):
        self.algoExp.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)

        # Reset and check the calculation again
        self.resetAlgorithmsAndImages()

        self.algoExp.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)

        zk = self.algoExp.getZer4UpInNm()
        self.assertEqual(int(zk[7]), -192)

    def testRunItOfFft(self):
        self.algoFft.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)

        zk = self.algoFft.getZer4UpInNm()
        self.assertEqual(int(zk[7]), -192)

    def checkSymmetryI1I2(self, algo):
        # This function checks that the algorithm is symmetric with respect
        # to I1 and I2. It is used by testSymmetryI1I2() below.

        # Calculate Zernikes with I1,I2
        self.resetAlgorithmsAndImages()
        algo.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)
        zkI1I2 = algo.getZer4UpInNm()

        # Calculate Zernikes with I2,I1
        self.resetAlgorithmsAndImages()
        algo.runIt(self.I2, self.I1, self.opticalModel, tol=1e-3)
        zkI2I1 = algo.getZer4UpInNm()

        # Check that the zernikes are the same, regardless of the order
        # of I1 and I2
        self.assertTrue(np.array_equal(zkI1I2, zkI2I1))

    def testSymmetryI1I2(self):
        # Test that the algorithm gives same results when you swap I1 <-> I2.

        # Run test for exp and fft algorithms
        for algo in [self.algoExp, self.algoFft]:
            with self.subTest(algo=algo):
                self.checkSymmetryI1I2(algo)

    def checkAlgHistory(self, algo):
        # This function checks the algorithm history.
        # It is used by testAlgHistory() below.

        # Reset everything
        self.resetAlgorithmsAndImages()

        # Check that history is empty before running the algorithm
        self.assertTrue(len(algo.getHistory()) == 0)

        # Check that history is not created when debugLevel==0
        algo.setDebugLevel(0)
        algo.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)
        self.assertTrue(len(algo.getHistory()) == 0)

        # Reset everything
        self.resetAlgorithmsAndImages()

        # Check that everything is recorded when debugLevel==3
        algo.setDebugLevel(3)
        algo.runIt(self.I1, self.I2, self.opticalModel, tol=1e-3)
        self.assertIsInstance(algo.getHistory(), dict)

        # Check all outer loop items are present
        outerItems = [
            "initI1",
            "initI2",
            "compZk",
            "compI1",
            "compI2",
            "pupilMask",
            "maskedI1",
            "maskedI2",
            "residZk",
            "residWf",
            "totZk",
            "totWf",
            "caustic",
            "converged",
        ]
        if algo.getPoissonSolverName() == "fft":
            outerItems += ["innerLoop"]
        for outerItr in algo.getHistory().values():
            self.assertEqual(set(outerItems), set(outerItr.keys()))

        # If fft, check the inner loops
        innerItems = [
            "initS",
            "FFT",
            "estWf",
            "estS",
        ]
        if algo.getPoissonSolverName() == "fft":
            for outerItr in algo.getHistory().values():
                for innerItr in outerItr["innerLoop"].values():
                    self.assertEqual(set(innerItems), set(innerItr.keys()))

    def testAlgHistory(self):
        # Test that the algorithm history contains all expected entries

        # Run test for exp and fft algorithms
        for algo in [self.algoExp, self.algoFft]:
            with self.subTest(algo=algo):
                self.checkAlgHistory(algo)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
