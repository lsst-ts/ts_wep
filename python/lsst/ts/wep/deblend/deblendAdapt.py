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

__all__ = ["DeblendAdapt"]

import numpy as np
from lsst.ts.wep.centroid import CentroidFindFactory
from lsst.ts.wep.deblend.deblendDefault import DeblendDefault
from lsst.ts.wep.deblend.nelderMeadModify import nelderMeadModify
from lsst.ts.wep.utils import CentroidFindType
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_opening,
    center_of_mass,
    shift,
)
from scipy.optimize import minimize_scalar
from skimage.filters import threshold_local


class DeblendAdapt(DeblendDefault):
    """DeblendDefault child class to do the deblending by the adaptive
    threshold method."""

    def __init__(self) -> None:
        super(DeblendAdapt, self).__init__()

        # Method to find the centroid of donut
        self._centroidFind = CentroidFindFactory.createCentroidFind(CentroidFindType.RandomWalk)

        # Initial guess of block size used in the adaptive threshold mothod
        self.blockSizeInit = 33

    def deblendDonut(
        self, imgToDeblend: np.ndarray, iniGuessXY: list[tuple]
    ) -> tuple[np.ndarray, float, float]:
        """Deblend the donut image.

        Parameters
        ----------
        imgToDeblend : numpy.ndarray
            Image to deblend.
        iniGuessXY : list[tuple]
            The list contains the initial guess of (x, y) positions of
            neighboring stars as [star 1, star 2, etc.].

        Returns
        -------
        numpy.ndarray
            Deblended donut image.
        float
            Position x of donut in pixel.
        float
            Position y of donut in pixel.

        Raises
        ------
        ValueError
            Only support to deblend single neighboring star.
        """

        # Check the number of neighboring star
        if len(iniGuessXY) != 1:
            raise ValueError("Only support to deblend single neighboring star.")

        # Get the initial guess of the brightest donut
        imgBinary = self._centroidFind.getImgBinary(imgToDeblend)
        realcx, realcy, realR = self._centroidFind.getCenterAndRfromImgBinary(imgBinary)

        # Check the image quality
        if not realcx:
            return np.array([]), realcx, realcy

        # Remove the salt and pepper noise
        imgBinary = binary_opening(imgBinary).astype(float)
        imgBinary = binary_closing(imgBinary).astype(float)

        # Get the binary image by the adaptive threshold method
        imgBinaryAdapt = self._getImgBinaryAdapt(imgToDeblend)

        # Calculate the system error by only taking the background signal
        bg1D = imgToDeblend.flatten()
        bgImgBinary1D = imgBinaryAdapt.flatten()
        background = bg1D[bgImgBinary1D == 0]
        bgPhist, binEdges = np.histogram(background, bins=256)
        sysError = np.mean(binEdges[0:2])

        # Remove the system error
        noSysErrImage = imgToDeblend - sysError
        noSysErrImage[noSysErrImage < 0] = 0

        # Get the residure map
        resImgBinary = imgBinaryAdapt - imgBinary

        # Compensate the zero element for subtraction
        resImgBinary[np.where(resImgBinary < 0)] = 0

        # Remove the salt and pepper noise noise of resImgBinary
        resImgBinary = binary_opening(resImgBinary).astype(float)

        # Calculate the shifts of x and y
        # Only support to deblend single neighboring star at this moment
        starXyNbr = iniGuessXY[0]
        x0 = int(starXyNbr[0] - realcx)
        y0 = int(starXyNbr[1] - realcy)

        xoptNeighbor = nelderMeadModify(
            self._funcResidue,
            np.array([x0, y0]),
            args=(imgBinary, resImgBinary),
            step=15,
        )

        # Shift the main donut image to fitted position of neighboring star
        fitImgBinary = shift(imgBinary, [int(xoptNeighbor[0][1]), int(xoptNeighbor[0][0])])

        # Handle the numerical error of shift. Regenerate a binary image.
        fitImgBinary[fitImgBinary > 0.5] = 1
        fitImgBinary[fitImgBinary < 0.5] = 0

        # Get the overlap region between main donut and neighboring donut
        imgOverlapBinary = imgBinary + fitImgBinary
        imgOverlapBinary[imgOverlapBinary < 1.5] = 0
        imgOverlapBinary[imgOverlapBinary > 1.5] = 1

        # Get the overall binary image
        imgAllBinary = imgBinary + fitImgBinary
        imgAllBinary[imgAllBinary > 1] = 1

        # Get the reference image for the fitting
        imgRef = noSysErrImage * imgAllBinary

        # Calculate the magnitude ratio of image
        imgMainDonut = noSysErrImage * imgBinary
        imgFit = shift(imgMainDonut, [int(xoptNeighbor[0][1]), int(xoptNeighbor[0][0])])

        xoptMagNeighbor = minimize_scalar(
            self._funcMag,
            bounds=(0, 1),
            method="bounded",
            args=(imgMainDonut, imgOverlapBinary, imgFit, imgRef, xoptNeighbor[0]),
        )

        imgDeblend = imgMainDonut - xoptMagNeighbor.x * imgFit * imgOverlapBinary

        # Repair the boundary of image
        imgDeblend = self._repairBoundary(imgOverlapBinary, imgBinary, imgDeblend)

        # Calculate the centroid position of donut
        realcy, realcx = center_of_mass(imgBinary)

        return imgDeblend, realcx, realcy

    def _getImgBinaryAdapt(self, imgInit: np.ndarray) -> np.ndarray:
        """Get the binary image by the adaptive threshold method.

        Parameters
        ----------
        imgInit : numpy.ndarray
            Initial image.

        Returns
        -------
        numpy.ndarray
            Binary image.
        """

        # Adaptive threshold
        delta = 1
        times = 0

        blockSize = self.blockSizeInit
        while (delta > 1e-2) and (times < 10):
            img = imgInit.copy()
            imgBinary = (img > threshold_local(img, blockSize)).astype(float)

            # Calculate the weighting radius
            realR = np.sqrt(np.sum(imgBinary) / np.pi)

            # Calculte the nearest odd number of radius for the blockSize
            if int(realR) % 2 == 0:
                oddRearR = int(realR + 1)
            else:
                oddRearR = int(realR)

            # Critera check of while loop
            delta = abs(blockSize - oddRearR)
            times += 1

            # New value of blockSize
            blockSize = oddRearR

        return imgBinary

    def _funcResidue(self, posShift: tuple | list, imgBinary: np.ndarray, resImgBinary: np.ndarray) -> float:
        """Use the least square method to decide the position of neighboring
        star.

        Parameters
        ----------
        posShift : tuple or list
            (x, y) shift from the main star position to neighboring star.
        imgBinary : numpy.ndarray
            Binary image of the main star.
        resImgBinary : numpy.ndarray[int]
            Binary image of residue of neighboring star.

        Returns
        -------
        float
            The difference between fitted binary image and residue image.
        """

        # Shift the image
        fitImgBinary = shift(imgBinary, [int(posShift[1]), int(posShift[0])])

        # Handle the numerical error of shift. Regenerate a binary image.
        fitImgBinary[fitImgBinary > 0.5] = 1
        fitImgBinary[fitImgBinary < 0.5] = 0

        # Take the least square difference
        delta = np.sum((fitImgBinary - resImgBinary) ** 2)

        return delta

    def _funcMag(
        self,
        magRatio: float,
        imgMainDonut: np.ndarray,
        imgOverlapBinary: np.ndarray,
        imgFit: np.ndarray,
        imgRef: np.ndarray,
        xyShiftNeighbor: tuple | list,
    ) -> float:
        """Use the least square method to decide the magnitude ratio of
        neighboring star.

        Parameters
        ----------
        magRatio : float
            Magnitude ratio between the main star and neighboring star.
        imgMainDonut : numpy.ndarray
            Image of the main star.
        imgOverlapBinary : numpy.ndarray
            Binary image of overlap between the main star and neighboring star.
        imgFit : numpy.ndarray
            Fitted image of neighboring star by the shifting of image of main
            star.
        imgRef : numpy.ndarray
            Reference image for the fitting.
        xyShiftNeighbor : tuple or list
            Fitted image of neighboring star by the shifting of image of main
            star.

        Returns
        -------
        float
            The difference between synthesized image and blended image.
        """

        # Synthesize the image
        imgNew = imgMainDonut - magRatio * imgFit * imgOverlapBinary
        imgNew = imgNew + magRatio * shift(imgNew, [int(xyShiftNeighbor[1]), int(xyShiftNeighbor[0])])

        # Take the least square difference
        delta = np.sum((imgNew - imgRef) ** 2)

        return delta

    def _repairBoundary(
        self,
        imgOverlapBinary: np.ndarray,
        imgBinary: np.ndarray,
        imgDeblend: np.ndarray,
    ) -> np.ndarray:
        """Compensate the values on boundary of overlap region between the
        bright star and neighboring star.

        Parameters
        ----------
        imgOverlapBinary : numpy.ndarray[int]
            Binary impage of overlap between bright star and neighboring star.
        imgBinary : numpy.ndarray[int]
            Binary image of bright star.
        imgDeblend : numpy.ndarray
            Deblended donut image.

        Returns
        -------
        numpy.ndarray
            Repaired deblended donut image.
        """

        # Copy the original data
        repairImgDeblend = imgDeblend.copy()

        # Get the boundary of overlap region
        boundaryOverlap = imgOverlapBinary - binary_erosion(imgOverlapBinary)

        # Find all boundary points
        m, n = np.where(boundaryOverlap == 1)

        for ii in range(len(m)):
            # Correct values that are not on the boundary next to environment
            if imgBinary[m[ii] - 1 : m[ii] + 2, n[ii] - 1 : n[ii] + 2].all():
                # Modify the value in column
                neighborValues = repairImgDeblend[m[ii], n[ii] - 4 : n[ii] + 5]
                temp = neighborValues[neighborValues != 0]
                stdTemp = np.std(temp)
                meanTemp = np.mean(temp)

                for kk in range(9):
                    testValue = repairImgDeblend[m[ii], n[ii] - 4 + kk]
                    if testValue != 0:
                        if (testValue >= meanTemp + 2 * stdTemp) or (testValue <= meanTemp - 2 * stdTemp):
                            repairImgDeblend[m[ii], n[ii] - 4 + kk] = (
                                repairImgDeblend[m[ii], n[ii] - 5 + kk]
                                + repairImgDeblend[m[ii], n[ii] - 3 + kk]
                            ) / 2

                # Modify the value in row
                neighborValues = repairImgDeblend[m[ii] - 4 : m[ii] + 5, n[ii]]
                temp = neighborValues[neighborValues != 0]
                stdTemp = np.std(temp)
                meanTemp = np.mean(temp)

                for kk in range(9):
                    testValue = repairImgDeblend[m[ii] - 4 + kk, n[ii]]
                    if testValue != 0:
                        if (testValue >= meanTemp + 2 * stdTemp) or (testValue <= meanTemp - 2 * stdTemp):
                            repairImgDeblend[m[ii] - 4 + kk, n[ii]] = (
                                repairImgDeblend[m[ii] - 5 + kk, n[ii]]
                                + repairImgDeblend[m[ii] - 3 + kk, n[ii]]
                            ) / 2

        return repairImgDeblend
