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

import unittest
import numpy as np
import os

import lsst.afw.image as afwImage
from lsst.daf.base import PropertyList
from lsst.ts.wep.task.DonutStamp2 import DonutStamp2
from lsst.geom import Point2D

from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.Utility import getModulePath, getConfigDir, DefocalType, CamType

from lsst.ts.wep.cwfs.CentroidRandomWalk import CentroidRandomWalk


class TempAlgo(object):
    """Temporary algorithm class used for the testing."""

    def __init__(self):

        self.numTerms = 22
        self.offAxisPolyOrder = 10
        self.zobsR = 0.61

    def getNumOfZernikes(self):

        return self.numTerms

    def getOffAxisPolyOrder(self):

        return self.offAxisPolyOrder

    def getObsOfZernikes(self):

        return self.zobsR


class TestDonutStamp2(unittest.TestCase):
    def setUp(self):

        self.nStamps = 3
        self.stampSize = 25

        self.testStamps, self.testMetadata = self._makeStamps(
            self.nStamps, self.stampSize
        )
        self.numTerms = 22
        self.offAxisPolyOrder = 10
        self.zobsR = 0.61

        modulePath = getModulePath()
        # Define the instrument folder
        instDir = os.path.join(getConfigDir(), "cwfs", "instData")

        # Define the instrument name
        dimOfDonutOnSensor = 120

        self.inst = Instrument(instDir)
        self.inst.config(
            CamType.LsstCam, dimOfDonutOnSensor, announcedDefocalDisInMm=1.0
        )

        # Define the image folder and image names
        # Image data -- Don't know the final image format.
        # It is noted that image.readFile inuts is based on the txt file
        imageFolderPath = os.path.join(
            modulePath, "tests", "testData", "testImages", "LSST_NE_SN25"
        )
        intra_image_name = "z11_0.25_intra.txt"
        extra_image_name = "z11_0.25_extra.txt"
        self.imgFilePathIntra = os.path.join(imageFolderPath, intra_image_name)
        self.imgFilePathExtra = os.path.join(imageFolderPath, extra_image_name)

        # This is the position of donut on the focal plane in degree
        self.field = Point2D(1.185, 1.185)

        # Define the optical model: "paraxial", "onAxis", "offAxis"
        self.opticalModel = "offAxis"

        # Get the true Zk
        zcAnsFilePath = os.path.join(
            modulePath,
            "tests",
            "testData",
            "testImages",
            "validation",
            "simulation",
            "LSST_NE_SN25_z11_0.25_exp.txt",
        )
        self.zcCol = np.loadtxt(zcAnsFilePath)

        self.stamp = DonutStamp2.factory(self.testStamps[0], self.testMetadata, 1)

    def _makeStamps(self, nStamps, stampSize):

        randState = np.random.RandomState(42)
        stampList = []

        for i in range(nStamps):
            stamp = afwImage.maskedImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
            stampList.append(stamp)

        ras = np.arange(nStamps)
        decs = np.arange(nStamps) + 5
        centX = np.arange(nStamps) + 20
        centY = np.arange(nStamps) + 25
        detectorNames = ["R22_S11"] * nStamps

        metadata = PropertyList()
        metadata["RA_DEG"] = ras
        metadata["DEC_DEG"] = decs
        metadata["CENT_X"] = centX
        metadata["CENT_Y"] = centY
        metadata["DET_NAME"] = detectorNames

        return stampList, metadata

    def testGetOffAxisCoeff(self):

        offAxisCoeff, offAxisOffset = self.stamp.getOffAxisCoeff()
        self.assertTrue(isinstance(offAxisCoeff, np.ndarray))
        self.assertEqual(len(offAxisCoeff), 0)
        self.assertEqual(offAxisOffset, 0.0)

    def testIsCaustic(self):

        self.assertFalse(self.stamp.isCaustic())

    def testGetPaddedMask(self):

        pMask = self.stamp.getPaddedMask()
        self.assertEqual(len(pMask), 0)
        self.assertEqual(pMask.dtype, int)

    def testGetNonPaddedMask(self):

        cMask = self.stamp.getNonPaddedMask()
        self.assertEqual(len(cMask), 0)
        self.assertEqual(cMask.dtype, int)

    def testGetFieldXY(self):

        fieldX, fieldY = self.stamp.getFieldXY()
        self.assertEqual(fieldX, 0)
        self.assertEqual(fieldY, 0)

    def testSetImg(self):

        self._setIntraImg()
        self.assertEqual(self.stamp.getImg().shape, (120, 120))

    def _setIntraImg(self):

        self.stamp.setImg(
            self.fieldXY, DefocalType.Intra, imageFile=self.imgFilePathIntra
        )

    def testUpdateImage(self):

        self._setIntraImg()

        newImg = np.random.rand(5, 5)
        self.stamp.updateImage(newImg)

        self.assertTrue(np.all(self.stamp.getImg() == newImg))

    def testUpdateImgInit(self):

        self._setIntraImg()

        self.stamp.updateImgInit()

        delta = np.sum(np.abs(self.stamp.getImgInit() - self.stamp.getImg()))
        self.assertEqual(delta, 0)

    def testImageCoCenter(self):

        self._setIntraImg()

        self.stamp.imageCoCenter(self.inst)

        xc, yc = self.stamp.getImgObj().getCenterAndR()[0:2]
        self.assertEqual(int(xc), 63)
        self.assertEqual(int(yc), 63)

    def testCompensate(self):

        # Generate a fake algorithm class
        algo = TempAlgo()

        # Test the function of image compensation
        boundaryT = 8
        offAxisCorrOrder = 10
        zcCol = np.zeros(22)
        zcCol[3:] = self.zcCol * 1e-9

        # read the numpy arrays for intra and extra focal images
        intra = np.loadtxt(self.imgFilePathIntra)
        extra = np.loadtxt(self.imgFilePathExtra)

        wfsImgIntra = DonutStamp2.factory(intra, self.testMetadata, 1)
        wfsImgIntra.field = self.field
        wfsImgIntra.defocalType = DefocalType.Intra

        print(type(wfsImgIntra))
        print(type(wfsImgIntra.stamp_im))

        wfsImgExtra = DonutStamp2.factory(extra, self.testMetadata, 1)
        wfsImgExtra.field = self.field
        wfsImgExtra.defocalType = DefocalType.Extra

        for wfsImg in [wfsImgIntra, wfsImgExtra]:
            wfsImg.makeMask(self.inst, self.opticalModel, boundaryT, 1)
            wfsImg.setOffAxisCorr(self.inst, offAxisCorrOrder)
            wfsImg.imageCoCenter(self.inst)
            wfsImg.compensate(self.inst, algo, zcCol, self.opticalModel)

        # Get the common region
        intraImg = wfsImgIntra.getImg()
        extraImg = wfsImgExtra.getImg()

        centroid = CentroidRandomWalk()
        binaryImgIntra = centroid.getImgBinary(intraImg)
        binaryImgExtra = centroid.getImgBinary(extraImg)

        binaryImg = binaryImgIntra + binaryImgExtra
        binaryImg[binaryImg < 2] = 0
        binaryImg = binaryImg / 2

        # Calculate the difference
        res = np.sum(np.abs(intraImg - extraImg) * binaryImg)
        self.assertLess(res, 500)

    def testCenterOnProjection(self):

        template = self._prepareGaussian2D(100, 1)

        dx = 2
        dy = 8
        img = np.roll(np.roll(template, dx, axis=1), dy, axis=0)
        np.roll(np.roll(img, dx, axis=1), dy, axis=0)

        self.assertGreater(np.sum(np.abs(img - template)), 29)
        print(type(self.stamp))

        imgRecenter = self.stamp.centerOnProjection(img, template, window=20)
        self.assertLess(np.sum(np.abs(imgRecenter - template)), 1e-7)

    def _prepareGaussian2D(self, imgSize, sigma):

        x = np.linspace(-10, 10, imgSize)
        y = np.linspace(-10, 10, imgSize)

        xx, yy = np.meshgrid(x, y)

        return (
            1
            / (2 * np.pi * sigma ** 2)
            * np.exp(-(xx ** 2 / (2 * sigma ** 2) + yy ** 2 / (2 * sigma ** 2)))
        )

    def testSetOffAxisCorr(self):

        self._setIntraImg()

        offAxisCorrOrder = 10
        self.stamp.setOffAxisCorr(self.inst, offAxisCorrOrder)

        offAxisCoeff, offAxisOffset = self.stamp.getOffAxisCoeff()
        self.assertEqual(offAxisCoeff.shape, (4, 66))
        self.assertAlmostEqual(offAxisCoeff[0, 0], -2.6362089 * 1e-3)
        self.assertEqual(offAxisOffset, 0.001)

    def testMakeMaskListOfParaxial(self):

        self._setIntraImg()

        model = "paraxial"
        masklist = self.stamp.makeMaskList(self.inst, model)

        masklistAns = np.array([[0, 0, 1, 1], [0, 0, 0.61, 0]])
        self.assertEqual(np.sum(np.abs(masklist - masklistAns)), 0)

    def testMakeMaskListOfOffAxis(self):

        self._setIntraImg()

        model = "offAxis"
        masklist = self.stamp.makeMaskList(self.inst, model)

        masklistAns = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 0.61, 0],
                [-0.21240585, -0.21240585, 1.2300922, 1],
                [-0.08784336, -0.08784336, 0.55802573, 0],
            ]
        )
        self.assertAlmostEqual(np.sum(np.abs(masklist - masklistAns)), 0)

    def testMakeMask(self):

        self._setIntraImg()

        boundaryT = 8
        maskScalingFactorLocal = 1
        model = "offAxis"
        self.stamp.makeMask(self.inst, model, boundaryT, maskScalingFactorLocal)

        image = self.stamp.getImg()
        pMask = self.stamp.getPaddedMask()
        cMask = self.stamp.getNonPaddedMask()
        self.assertEqual(pMask.shape, image.shape)
        self.assertEqual(cMask.shape, image.shape)
        self.assertEqual(np.sum(np.abs(cMask - pMask)), 3001)

    def testFactory(self):

        randState = np.random.RandomState(42)
        for i in range(self.nStamps):
            donutStamp = DonutStamp2.factory(self.testStamps[i], self.testMetadata, i)
            np.testing.assert_array_almost_equal(
                donutStamp.stamp_im.image.array,
                randState.rand(self.stampSize, self.stampSize),
            )
            np.testing.assert_array_equal(
                donutStamp.stamp_im.mask.array,
                np.ones((self.stampSize, self.stampSize)) * 10,
            )
            np.testing.assert_array_equal(
                donutStamp.stamp_im.variance.array,
                np.ones((self.stampSize, self.stampSize)) * 100,
            )
            self.assertEqual(donutStamp.detector_name, "R22_S11")
            skyPos = donutStamp.sky_position
            self.assertEqual(skyPos.getRa().asDegrees(), i)
            self.assertEqual(skyPos.getDec().asDegrees(), i + 5)
            centroidPos = donutStamp.centroid_position
            self.assertEqual(centroidPos.getX(), i + 20)
            self.assertEqual(centroidPos.getY(), i + 25)

            boundaryT = 8
            maskScalingFactorLocal = 1
            model = "offAxis"
            donutStamp.makeMask(self.inst, model, boundaryT, maskScalingFactorLocal)

            self.assertEqual(donutStamp.pMask.shape, donutStamp.cMask.shape)
