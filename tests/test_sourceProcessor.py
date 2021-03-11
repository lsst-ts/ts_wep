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

from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.bsc.NbrStar import NbrStar
from lsst.ts.wep.Utility import getModulePath, FilterType


class TestSourceProcessor(unittest.TestCase):
    """Test the source processor class."""

    def setUp(self):

        # Get the path of module
        self.modulePath = getModulePath()

        # Set the source processor
        self.sourProc = SourceProcessor()

        # Set the configuration
        self.sourProc.config(sensorName="R00_SW1")

    def testInit(self):

        self.assertEqual(self.sourProc.sensorName, "R00_SW1")
        self.assertEqual(len(self.sourProc.sensorDimList), 205)
        self.assertEqual(len(self.sourProc.sensorEulerRot), 205)
        self.assertEqual(len(self.sourProc.sensorFocaPlaneInDeg), 205)
        self.assertEqual(len(self.sourProc.sensorFocaPlaneInUm), 205)

        self.assertEqual(self.sourProc.sensorDimList["R00_SW1"], (2000, 4072))
        self.assertEqual(self.sourProc.sensorDimList["R22_S11"], (4004, 4096))
        self.assertEqual(self.sourProc.sensorFocaPlaneInDeg["R22_S11"], (0, 0))
        self.assertNotEqual(
            self.sourProc.sensorFocaPlaneInDeg["R00_SW0"],
            self.sourProc.sensorFocaPlaneInDeg["R00_SW1"],
        )

    def testConfig(self):

        sensorName = "sensorName"
        self.sourProc.config(sensorName=sensorName)

        self.assertEqual(self.sourProc.sensorName, sensorName)

    def testGetEulerZinDeg(self):

        wfsSensorName = "R40_SW0"
        eulerZ = self.sourProc.getEulerZinDeg(wfsSensorName)

        self.assertEqual(eulerZ, 90.004585)

    def testCamXYtoFieldXY(self):

        pixelX = 1000
        pixelY = 2036
        fieldX, fieldY = self.sourProc.camXYtoFieldXY(pixelX, pixelY)

        ansFieldX, ansFieldY = self.sourProc.sensorFocaPlaneInDeg[
            self.sourProc.sensorName
        ]
        self.assertEqual(fieldX, ansFieldX)
        self.assertEqual(fieldY, ansFieldY)

    def testCamXYtoFieldXYforWfs(self):

        oxR00SW1, oyR00SW1 = self._camXYtoFieldXY("R00_SW1", 0, 0)
        oxR00SW0, oyR00SW0 = self._camXYtoFieldXY("R00_SW0", 0, 0)

        oxR40SW1, oyR40SW1 = self._camXYtoFieldXY("R40_SW1", 0, 0)
        oxR40SW0, oyR40SW0 = self._camXYtoFieldXY("R40_SW0", 0, 0)

        oxR44SW1, oyR44SW1 = self._camXYtoFieldXY("R44_SW1", 0, 0)
        oxR44SW0, oyR44SW0 = self._camXYtoFieldXY("R44_SW0", 0, 0)

        oxR04SW1, oyR04SW1 = self._camXYtoFieldXY("R04_SW1", 0, 0)
        oxR04SW0, oyR04SW0 = self._camXYtoFieldXY("R04_SW0", 0, 0)

        # Compare with the same RXX_SYY
        self.assertLess(abs(oyR00SW1 - oyR00SW0), 1e-3)
        self.assertLess(abs(oxR40SW1 - oxR40SW0), 1e-3)
        self.assertLess(abs(oyR44SW1 - oyR44SW0), 1e-3)
        self.assertLess(abs(oxR04SW1 - oxR04SW0), 1e-3)

        # Campare with different RXX_SYY
        self.assertEqual((oxR00SW1 + oxR44SW1, oyR00SW1 + oyR44SW1), (0, 0))
        self.assertEqual((oxR40SW0 + oxR04SW0, oyR40SW0 + oyR04SW0), (0, 0))

    def _camXYtoFieldXY(self, sensorName, pixelX, pixelY):

        self.sourProc.config(sensorName=sensorName)
        fieldX, fieldY = self.sourProc.camXYtoFieldXY(pixelX, pixelY)

        return fieldX, fieldY

    def testDmXY2CamXY(self):

        self.sourProc.config(sensorName="R22_S11")
        self.assertEqual(self.sourProc.dmXY2CamXY(4070, 1000), (1000, 4070))

    def testCamXY2DmXY(self):

        self.sourProc.config(sensorName="R22_S11")
        self.assertEqual(self.sourProc.camXY2DmXY(3000, 4070), (4070, 3000))

    def testIsVignette(self):

        isVignette = self.sourProc.isVignette(1.76, 0)
        self.assertTrue(isVignette)

        noVignette = self.sourProc.isVignette(0.2, 0.2)
        self.assertFalse(noVignette)

    def testSimulateImg(self):

        ccdImgIntra, ccdImgExtra = self._simulateImg()

        self.assertEqual(ccdImgIntra.shape, (4072, 2000))
        self.assertNotEqual(np.sum(np.abs(ccdImgIntra)), 0)

    def _simulateImg(self):

        imageFolderPath = os.path.join(
            self.modulePath, "tests", "testData", "testImages", "LSST_C_SN26"
        )
        defocalDis = 0.25
        nbrStar = self._generateNbrStar()
        ccdImgIntra, ccdImgExtra = self.sourProc.simulateImg(
            imageFolderPath, defocalDis, nbrStar, FilterType.REF, noiseRatio=0
        )

        return ccdImgIntra, ccdImgExtra

    def _generateNbrStar(self):

        nbrStar = NbrStar()
        nbrStar.starId = {523572575: [], 523572679: [523572671]}
        nbrStar.lsstMagG = {
            523572575: 14.66652,
            523572671: 16.00000,
            523572679: 13.25217,
        }
        nbrStar.raDeclInPixel = {
            523572679: (3966.44, 1022.91),
            523572671: (3968.77, 1081.02),
            523572575: (3475.48, 479.33),
        }
        return nbrStar

    def testGetSingleTargetImage(self):

        (
            sglSciNeiImg,
            allStarPosX,
            allStarPosY,
            magRatio,
            offsetX,
            offsetY,
        ) = self._getSingleTargetImage()

        self.assertEqual(sglSciNeiImg.shape, (310, 310))
        self.assertAlmostEqual(allStarPosX[0], 185.02)
        self.assertAlmostEqual(allStarPosX[1], 126.91)
        self.assertAlmostEqual(allStarPosY[0], 206.77)
        self.assertAlmostEqual(allStarPosY[1], 204.44)
        self.assertAlmostEqual(magRatio[0], 0.07959174)
        self.assertEqual(magRatio[1], 1)
        self.assertEqual(offsetX, 896.0)
        self.assertEqual(offsetY, 3762.0)

    def _getSingleTargetImage(self):

        nbrStar = self._generateNbrStar()
        ccdImgIntra, ccdImgExtra = self._simulateImg()
        starIndex = list(nbrStar.getId()).index(523572679)
        (
            sglSciNeiImg,
            allStarPosX,
            allStarPosY,
            magRatio,
            offsetX,
            offsetY,
        ) = self.sourProc.getSingleTargetImage(
            ccdImgIntra, nbrStar, starIndex, FilterType.REF
        )

        return sglSciNeiImg, allStarPosX, allStarPosY, magRatio, offsetX, offsetY

    def testDoDeblending(self):

        (
            sglSciNeiImg,
            allStarPosX,
            allStarPosY,
            magRatio,
            offsetX,
            offsetY,
        ) = self._getSingleTargetImage()

        imgDeblend, realcx, realcy = self.sourProc.doDeblending(
            sglSciNeiImg, allStarPosX, allStarPosY, magRatio
        )

        self.assertEqual(imgDeblend.shape, (310, 310))
        self.assertLess(np.abs(realcx - 126.51), 3)
        self.assertLess(np.abs(realcy - 205.00), 3)

        # Get the real camera position x, y after the deblending
        realCameraX = realcx + offsetX
        realCameraY = realcy + offsetY

        # Compared with DM prediction
        nbrStar = self._generateNbrStar()
        raDeclInPixel = nbrStar.getRaDeclInPixel()
        camX, camY = self.sourProc.dmXY2CamXY(
            raDeclInPixel[523572679][0], raDeclInPixel[523572679][1]
        )
        delta = np.sqrt((realCameraX - camX) ** 2 + (realCameraY - camY) ** 2)
        self.assertLess(delta, 10)

    def testMapSensorAndFieldIdx(self):

        fieldXY = self._getFieldXyOfLsst()
        mapping = self.sourProc.mapSensorAndFieldIdx(fieldXY)

        self.assertEqual(mapping["R22_S11"], 0)
        self.assertEqual(mapping["R21_S10"], 12)

    def _getFieldXyOfLsst(self):

        nArm = 6
        armLen = [0.379, 0.841, 1.237, 1.535, 1.708]
        fieldWFSx = [1.176, -1.176, -1.176, 1.176]
        fieldWFSy = [1.176, 1.176, -1.176, -1.176]
        pointAngle = np.arange(nArm) * (2 * np.pi) / nArm
        fieldX = np.concatenate(
            [np.zeros(1), np.kron(armLen, np.cos(pointAngle)), fieldWFSx]
        )
        fieldY = np.concatenate(
            [np.zeros(1), np.kron(armLen, np.sin(pointAngle)), fieldWFSy]
        )

        return np.array([fieldX, fieldY]).T


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
