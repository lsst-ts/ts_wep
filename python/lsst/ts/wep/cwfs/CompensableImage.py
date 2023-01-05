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

__all__ = ["CompensableImage"]

import os
import re
import warnings
import numpy as np
from copy import copy

from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    shift,
)
from scipy.interpolate import RectBivariateSpline
from scipy.signal import correlate

from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.cwfs.Tool import (
    padArray,
    extractArray,
    ZernikeAnnularGrad,
    ZernikeAnnularJacobian,
)
from lsst.ts.wep.cwfs.Image import Image
from lsst.ts.wep.Utility import DefocalType, CentroidFindType, rotMatrix
from galsim.utilities import horner2d


class CompensableImage(object):
    """Instantiate the class of CompensableImage.

    Parameters
    ----------
    centroidFindType : enum 'CentroidFindType', optional
        Algorithm to find the centroid of donut. (the default is
        CentroidFindType.RandomWalk.)
    """

    def __init__(self, centroidFindType=CentroidFindType.RandomWalk):

        self._image = Image(centroidFindType=centroidFindType)
        self.defocalType = DefocalType.Intra

        # Field coordinate in degree
        self.fieldX = 0
        self.fieldY = 0

        # Blended coordinates in pixels relative
        # to central donut
        self.blendOffsetX = list()
        self.blendOffsetY = list()

        # Initial image before doing the compensation
        self.image0 = None

        # Coefficient to do the off-axis correction
        self.offAxisCoeff = np.array([])

        # Defocused offset in files of off-axis correction
        self.offAxisOffset = 0.0

        # True if the image gets the over-compensation
        self.caustic = False

        # Padded mask for use at the offset planes
        self.mask_comp = np.array([], dtype=int)

        # Non-padded mask corresponding to aperture
        self.mask_pupil = np.array([], dtype=int)

    def getDefocalType(self):
        """Get the defocal type.

        Returns
        -------
        enum 'DefocalType'
            Defocal type.
        """

        return self.defocalType

    def getImgObj(self):
        """Get the image object.

        Returns
        -------
        Image
            Image object.
        """

        return self._image

    def getImg(self):
        """Get the image.

        Returns
        -------
        numpy.ndarray
            Image.
        """

        return self._image.getImg()

    def getImgSizeInPix(self):
        """Get the image size in pixel.

        Returns
        -------
        int
            Image size in pixel.
        """

        return self.getImg().shape[0]

    def getOffAxisCoeff(self):
        """Get the coefficients to do the off-axis correction.

        Returns
        -------
        numpy.ndarray
            Coefficients to do the off-axis correction.
        float
            Defocused offset in files of off-axis correction.
        """

        return self.offAxisCoeff, self.offAxisOffset

    def getImgInit(self):
        """Get the initial image before doing the compensation.

        Returns
        -------
        numpy.ndarray
            Initial image before doing the compensation.
        """

        return self.image0

    def isCaustic(self):
        """The image is caustic or not.

        The image might be caustic from the over-compensation.

        Returns
        -------
        bool
            True if the image is caustic.
        """

        return self.caustic

    def getPaddedMask(self):
        """Get the padded mask use at the offset planes.

        Returns
        -------
        numpy.ndarray[int]
            Padded mask.
        """

        return self.mask_comp

    def getNonPaddedMask(self):
        """Get the non-padded mask corresponding to aperture.

        Returns
        -------
        numpy.ndarray[int]
            Non-padded mask
        """

        return self.mask_pupil

    def getFieldXY(self):
        """Get the field x, y in degree.

        Returns
        -------
        float
            Field x in degree.
        float
            Field y in degree.
        """

        return self.fieldX, self.fieldY

    def setImg(
        self, fieldXY, defocalType, blendOffsets=None, image=None, imageFile=None
    ):
        """Set the wavefront image.

        Parameters
        ----------
        fieldXY : tuple or list
            Position of donut on the focal plane in degree (field x, field y).
        defocalType : enum 'DefocalType'
            Defocal type of image.
        blendOffsets : list, optional
            Positions of blended donuts relative to location of center donut.
            Enter as [xCoordList, yCoordList].
            Length of xCoordList and yCoordList must be same length.
            (the default is None).
        image : numpy.ndarray, optional
            Array of image. (the default is None.)
        imageFile : str, optional
            Path of image file. (the default is None.)

        Raises
        ------
        ValueError
            Input to blendOffsets must have same number of x-coordinates
            and y-coordinates.
        """

        self._image.setImg(image=image, imageFile=imageFile)
        self._checkImgShape()

        if blendOffsets is None:
            blendOffsets = [[], []]

        self.fieldX, self.fieldY = fieldXY
        self.defocalType = defocalType
        self.blendOffsetX = blendOffsets[0]
        self.blendOffsetY = blendOffsets[1]
        if len(self.blendOffsetX) != len(self.blendOffsetY):
            raise ValueError(
                "Length of blend x-coord list must equal length of y-coord list."
            )

        self._resetInternalAttributes()

    def _checkImgShape(self):
        """Check the image shape.

        Raises
        ------
        RuntimeError
            Only square image stamps are accepted.
        RuntimeError
            Number of pixels cannot be odd numbers.
        """

        img = self.getImg()
        if img.shape[0] != img.shape[1]:
            raise RuntimeError("Only square image stamps are accepted.")
        elif img.shape[0] % 2 == 1:
            raise RuntimeError("Number of pixels cannot be odd numbers.")

    def _resetInternalAttributes(self):
        """Reset the internal attributes."""

        self.image0 = None

        self.offAxisCoeff = np.array([])
        self.offAxisOffset = 0.0

        self.caustic = False

        # Reset all mask related parameters
        self.mask_pupil = np.array([], dtype=int)
        self.mask_comp = np.array([], dtype=int)

    def updateImage(self, image):
        """Update the image of donut.

        Parameters
        ----------
        image : numpy.ndarray
            Donut image.
        """

        self._image.updateImage(image)

    def updateImgInit(self):
        """Update the backup of initial image.

        This will be used in the outer loop iteration, which always uses the
        initial image (image0) before each iteration starts.
        """

        # Update the initial image for future use
        self.image0 = self.getImg().copy()

    def imageCoCenter(self, inst, fov=3.5, debugLevel=0):
        """Shift the weighting center of donut to the center of reference
        image with the correction of projection of fieldX and fieldY.
        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        fov : float, optional
            Field of view (FOV) of telescope. (the default is 3.5.)
        debugLevel : int, optional
            Show the information under the running. If the value is higher, the
            information shows more. It can be 0, 1, 2, or 3. (the default is
            0.)
        """

        warnings.warn(
            "This function is deprecated.  Will be removed after January 2023.",
            DeprecationWarning,
        )

        # Calculate the weighting center (x, y) and radius
        x1, y1 = self._image.getCenterAndR()[0:2]

        # Show the co-center information
        if debugLevel >= 3:
            print("imageCoCenter: (x, y) = (%8.2f,%8.2f)\n" % (x1, y1))

        # Calculate the center position on image
        # 0.5 is the half of 1 pixel
        stampCenterx1 = inst.dimOfDonutImg / 2 + 0.5
        stampCentery1 = inst.dimOfDonutImg / 2 + 0.5

        # Shift in the radial direction
        # The field of view (FOV) of LSST camera is 3.5 degree
        radialShift = fov * (inst.defocalDisOffsetInM / 1e-3) * (10e-6 / inst.pixelSize)

        # Calculate the projection of distance of donut to center
        fieldDist = self._getFieldDistFromOrigin()
        radialShift = radialShift * (fieldDist / (fov / 2))

        # Do not consider the condition out of FOV of lsst
        if fieldDist > (fov / 2):
            radialShift = 0

        # Calculate the cos(theta) for projection
        I1c = self.fieldX / fieldDist

        # Calculate the sin(theta) for projection
        I1s = self.fieldY / fieldDist

        # Get the projected x, y-coordinate
        stampCenterx1 = stampCenterx1 + radialShift * I1c
        stampCentery1 = stampCentery1 + radialShift * I1s

        # Shift the image to the projected position
        self.updateImage(
            np.roll(self.getImg(), int(np.round(stampCentery1 - y1)), axis=0)
        )
        self.updateImage(
            np.roll(self.getImg(), int(np.round(stampCenterx1 - x1)), axis=1)
        )

    def _getFieldDistFromOrigin(self, fieldX=None, fieldY=None, minDist=1e-8):
        """Get the field distance from the origin.

        Parameters
        ----------
        fieldX : float, optional
            Field x in degree. If the input is None, the value of self.fieldX
            will be used. (the default is None.)
        fieldY : float, optional
            Field y in degree. If the input is None, the value of self.fieldY
            will be used. (the default is None.)
        minDist : float, optional
            Minimum distace. In some cases, the field distance will be the
            denominator in other functions. (the default is 1e-8.)

        Returns
        -------
        float
            Field distance from the origin.
        """

        if fieldX is None:
            fieldX = self.fieldX

        if fieldY is None:
            fieldY = self.fieldY

        fieldDist = np.hypot(fieldX, fieldY)
        if fieldDist == 0:
            fieldDist = minDist

        return fieldDist

    def compensate(self, inst, algo, zcCol, model):
        """Calculate the image compensated from the affection of wavefront.

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        algo : Algorithm
            Algorithm to solve the Poisson's equation. It can by done by the
            fast Fourier transform or serial expansion.
        zcCol : numpy.ndarray
            Coefficients of wavefront.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".

        Raises
        ------
        RuntimeError
            input:size zcCol in compensate needs to be a numTerms row column
            vector.
        """

        # Check the condition of inputs
        numTerms = algo.getNumOfZernikes()
        if (zcCol.ndim == 1) and (len(zcCol) != numTerms):
            raise RuntimeError(
                "input:size",
                "zcCol in compensate needs to be a %d row column vector. \n" % numTerms,
            )

        # Dimension of image
        sm, sn = self.getImg().shape

        # Dimension of projected image on focal plane
        projSamples = sm

        # Let us create a look-up table for x -> xp first.
        luty, lutx = np.mgrid[
            -(projSamples / 2 - 0.5) : (projSamples / 2 + 0.5),
            -(projSamples / 2 - 0.5) : (projSamples / 2 + 0.5),
        ]

        sensorFactor = inst.getSensorFactor()
        lutx = lutx / (projSamples / 2 / sensorFactor)
        luty = luty / (projSamples / 2 / sensorFactor)

        # Set up the mapping
        lutxp, lutyp, J = self._aperture2image(
            inst, algo, zcCol, lutx, luty, projSamples, model
        )

        show_lutxyp = self._showProjection(
            lutxp, lutyp, sensorFactor, projSamples, raytrace=False
        )
        if np.all(show_lutxyp <= 0):
            self.caustic = True
            return

        # Extend the dimension of image by 20 pixel in x and y direction
        show_lutxyp = padArray(show_lutxyp, projSamples + 20)

        # Get the binary matrix of image on pupil plane if raytrace=False
        struct0 = generate_binary_structure(2, 1)
        struct = iterate_structure(struct0, 4)
        struct = binary_dilation(struct, structure=struct0, iterations=2).astype(int)
        show_lutxyp = binary_dilation(show_lutxyp, structure=struct)
        show_lutxyp = binary_erosion(show_lutxyp, structure=struct)

        # Extract the region from the center of image and get the original one
        show_lutxyp = extractArray(show_lutxyp, projSamples)

        # Recenter the image
        imgRecenter = self.centerOnProjection(
            self.getImg(), show_lutxyp.astype(float), window=20
        )
        self.updateImage(imgRecenter)

        # Construct the interpolant to get the intensity on (x', p') plane
        # that corresponds to the grid points on (x,y)
        yp, xp = np.mgrid[
            -(sm / 2 - 0.5) : (sm / 2 + 0.5), -(sm / 2 - 0.5) : (sm / 2 + 0.5)
        ]

        xp = xp / (sm / 2 / sensorFactor)
        yp = yp / (sm / 2 / sensorFactor)

        # Put the NaN to be 0 for the interpolate to use
        lutxp[np.isnan(lutxp)] = 0
        lutyp[np.isnan(lutyp)] = 0

        # Construct the function for interpolation
        ip = RectBivariateSpline(yp[:, 0], xp[0, :], self.getImg(), kx=1, ky=1)

        # Construct the projected image by the interpolation
        lutIp = ip(lutyp, lutxp, grid=False)

        # Calculate the image on focal plane with compensation based on flux
        # conservation
        # I(x, y)/I'(x', y') = J = (dx'/dx)*(dy'/dy) - (dx'/dy)*(dy'/dx)
        self.updateImage(lutIp * J)

        # Put NaN to be 0
        imgCompensate = self.getImg()
        imgCompensate[np.isnan(imgCompensate)] = 0

        # Check the compensated image has the problem or not.
        # The negative value means the over-compensation from wavefront error
        if np.any(imgCompensate < 0) and np.all(self.image0 >= 0):
            print(
                "WARNING: negative scale parameter, image is within caustic, zcCol (in um)=\n"
            )
            self.caustic = True

        # Put the overcompensated part to be 0
        imgCompensate[imgCompensate < 0] = 0
        self.updateImage(imgCompensate)

    def _aperture2image(self, inst, algo, zcCol, lutx, luty, projSamples, model):
        """Calculate the x, y-coordinate on the focal plane and the related
        Jacobian matrix.

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        algo : Algorithm
            Algorithm to solve the Poisson's equation. It can by done by the
            fast Fourier transform or serial expansion.
        zcCol : numpy.ndarray
            Coefficients of optical basis. It is Zernike polynomials in the
            baseline.
        lutx : numpy.ndarray
            X-coordinate on pupil plane.
        luty : numpy.ndarray
            Y-coordinate on pupil plane.
        projSamples : int
            Dimension of projected image. This value considers the
            magnification ratio of donut image.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".

        Returns
        -------
        numpy.ndarray
            X coordinate on the focal plane.
        numpy.ndarray
            Y coordinate on the focal plane.
        numpy.ndarray
            Jacobian matrix between the pupil and focal plane.
        """

        # Get the radius: R = D/2
        R = inst.apertureDiameter / 2

        # Calculate C = -f(f-l)/l/R^2. This is for the calculation of reduced
        # coordinate.
        if self.defocalType == DefocalType.Intra:
            l = inst.defocalDisOffsetInM
        elif self.defocalType == DefocalType.Extra:
            l = -inst.defocalDisOffsetInM

        myC = -inst.focalLength * (inst.focalLength - l) / l / R**2

        # Get the functions to do the off-axis correction by numerical fitting
        # Order to do the off-axis correction. The order is 10 now.
        offAxisPolyOrder = algo.getOffAxisPolyOrder()
        polyFunc = self._getFunction("poly%d_2D" % offAxisPolyOrder)
        polyGradFunc = self._getFunction("poly%dGrad" % offAxisPolyOrder)

        # Calculate the distance to center
        lutr = np.sqrt(lutx**2 + luty**2)

        # Calculated the extended ring radius (delta r), which is to extended
        # the available pupil area.
        # 1 pixel larger than projected pupil. No need to be EF-like, anything
        # outside of this will be masked off by the computational mask
        sensorFactor = inst.getSensorFactor()
        onepixel = 1 / (projSamples / 2 / sensorFactor)

        # Get the index that the point is out of the range of extended pupil
        idxout = (lutr > 1 + onepixel) | (lutr < inst.obscuration - onepixel)

        # Define the element to be NaN if it is out of range
        lutx[idxout] = np.nan
        luty[idxout] = np.nan

        # Get the index in the extended area of outer boundary with the width
        # of onepixel
        idxbound = (lutr <= 1 + onepixel) & (lutr > 1)

        # Calculate the extended x, y-coordinate (x' = x/r*r', r'=1)
        lutx[idxbound] = lutx[idxbound] / lutr[idxbound]
        luty[idxbound] = luty[idxbound] / lutr[idxbound]

        # Get the index in the extended area of inner boundary with the width
        # of onepixel
        idxinbd = (lutr < inst.obscuration) & (lutr > inst.obscuration - onepixel)

        # Calculate the extended x, y-coordinate (x' = x/r*r', r'=obscuration)
        lutx[idxinbd] = lutx[idxinbd] / lutr[idxinbd] * inst.obscuration
        luty[idxinbd] = luty[idxinbd] / lutr[idxinbd] * inst.obscuration

        # Get the corrected x, y-coordinate on focal plane (lutxp, lutyp)
        if model == "paraxial":
            # No correction is needed in "paraxial" model
            lutxp = lutx
            lutyp = luty

        elif model == "onAxis":

            # Calculate F(x, y) = m * sqrt(f^2-R^2) / sqrt(f^2-(x^2+y^2)*R^2)
            # m is the mask scaling factor
            myA2 = (inst.focalLength**2 - R**2) / (
                inst.focalLength**2 - lutr**2 * R**2
            )

            # Put the unphysical value as NaN
            myA = myA2.copy()
            idx = myA < 0
            myA[idx] = np.nan
            myA[~idx] = np.sqrt(myA2[~idx])

            # Mask scaling factor (for fast beam)
            maskScalingFactor = algo.getMaskScalingFactor()

            # Calculate the x, y-coordinate on focal plane
            # x' = F(x,y)*x + C*(dW/dx), y' = F(x,y)*y + C*(dW/dy)
            lutxp = maskScalingFactor * myA * lutx
            lutyp = maskScalingFactor * myA * luty

        elif model == "offAxis":

            # Get the coefficient of polynomials for off-axis correction
            tt = self.offAxisOffset

            cx = (self.offAxisCoeff[0, :] - self.offAxisCoeff[2, :]) * (tt + l) / (
                2 * tt
            ) + self.offAxisCoeff[2, :]
            cy = (self.offAxisCoeff[1, :] - self.offAxisCoeff[3, :]) * (tt + l) / (
                2 * tt
            ) + self.offAxisCoeff[3, :]

            # This will be inverted back by typesign later on.
            # We do the inversion here to make the (x,y)->(x',y') equations has
            # the same form as the paraxial case.
            cx = np.sign(l) * cx
            cy = np.sign(l) * cy

            # Do the orthogonalization: x'=1/sqrt(2)*(x+y), y'=1/sqrt(2)*(x-y)
            # Calculate the rotation angle for the orthogonalization
            fieldDist = self._getFieldDistFromOrigin()
            costheta = (self.fieldX + self.fieldY) / fieldDist / np.sqrt(2)
            if costheta > 1:
                costheta = 1
            elif costheta < -1:
                costheta = -1

            sintheta = np.sqrt(1 - costheta**2)
            if self.fieldY < self.fieldX:
                sintheta = -sintheta

            # Create the pupil grid in off-axis model. This gives the
            # x,y-coordinate in the extended ring area defined by the parameter
            # of onepixel.

            # Get the mask-related parameters
            maskCa, maskRa, maskCb, maskRb = self._interpMaskParam(
                self.fieldX, self.fieldY, inst.maskOffAxisCorr
            )

            lutx, luty = self._createPupilGrid(
                lutx,
                luty,
                onepixel,
                maskCa,
                maskCb,
                maskRa,
                maskRb,
                self.fieldX,
                self.fieldY,
            )

            # Calculate the x, y-coordinate on focal plane

            # First rotate back to reference orientation
            lutx0 = lutx * costheta + luty * sintheta
            luty0 = -lutx * sintheta + luty * costheta

            # Use the mapping at reference orientation
            lutxp0 = polyFunc(cx, lutx0, y=luty0)
            lutyp0 = polyFunc(cy, lutx0, y=luty0)

            # Rotate back to focal plane
            lutxp = lutxp0 * costheta - lutyp0 * sintheta
            lutyp = lutxp0 * sintheta + lutyp0 * costheta

            # Zemax data are in mm, therefore 1000
            reduced_coordi_factor = 1e-3 / (
                inst.dimOfDonutImg / 2 * inst.pixelSize / sensorFactor
            )

            # Reduced coordinates, so that this can be added with the dW/dz
            lutxp = lutxp * reduced_coordi_factor
            lutyp = lutyp * reduced_coordi_factor

        else:
            print("Wrong optical model type in compensate. \n")
            return

        # Obscuration of annular aperture
        zobsR = algo.getObsOfZernikes()

        # Calculate the x, y-coordinate on focal plane
        # x' = F(x,y)*x + C*(dW/dx), y' = F(x,y)*y + C*(dW/dy)

        # In Model basis (zer: Zernike polynomials)
        if zcCol.ndim == 1:
            lutxp = lutxp + myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dx")
            lutyp = lutyp + myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dy")

        # Make the sign to be consistent
        if self.defocalType == DefocalType.Extra:
            lutxp = -lutxp
            lutyp = -lutyp

        # Calculate the Jacobian matrix
        # In Model basis (zer: Zernike polynomials)
        if zcCol.ndim == 1:
            if model == "paraxial":
                J = (
                    1
                    + myC * ZernikeAnnularJacobian(zcCol, lutx, luty, zobsR, "1st")
                    + myC**2 * ZernikeAnnularJacobian(zcCol, lutx, luty, zobsR, "2nd")
                )

            elif model == "onAxis":
                xpox = maskScalingFactor * myA * (
                    1
                    + lutx**2
                    * R**2.0
                    / (inst.focalLength**2 - R**2 * lutr**2)
                ) + myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dx2")

                ypoy = maskScalingFactor * myA * (
                    1
                    + luty**2
                    * R**2.0
                    / (inst.focalLength**2 - R**2 * lutr**2)
                ) + myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dy2")

                xpoy = maskScalingFactor * myA * lutx * luty * R**2 / (
                    inst.focalLength**2 - R**2 * lutr**2
                ) + myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dxy")

                ypox = xpoy

                J = xpox * ypoy - xpoy * ypox

            elif model == "offAxis":
                xp0ox = (
                    polyGradFunc(cx, lutx0, luty0, "dx") * costheta
                    - polyGradFunc(cx, lutx0, luty0, "dy") * sintheta
                )

                yp0ox = (
                    polyGradFunc(cy, lutx0, luty0, "dx") * costheta
                    - polyGradFunc(cy, lutx0, luty0, "dy") * sintheta
                )

                xp0oy = (
                    polyGradFunc(cx, lutx0, luty0, "dx") * sintheta
                    + polyGradFunc(cx, lutx0, luty0, "dy") * costheta
                )

                yp0oy = (
                    polyGradFunc(cy, lutx0, luty0, "dx") * sintheta
                    + polyGradFunc(cy, lutx0, luty0, "dy") * costheta
                )

                xpox = (
                    xp0ox * costheta - yp0ox * sintheta
                ) * reduced_coordi_factor + myC * ZernikeAnnularGrad(
                    zcCol, lutx, luty, zobsR, "dx2"
                )

                ypoy = (
                    xp0oy * sintheta + yp0oy * costheta
                ) * reduced_coordi_factor + myC * ZernikeAnnularGrad(
                    zcCol, lutx, luty, zobsR, "dy2"
                )

                temp = myC * ZernikeAnnularGrad(zcCol, lutx, luty, zobsR, "dxy")

                # if temp==0,xpoy doesn't need to be symmetric about x=y
                xpoy = (
                    xp0oy * costheta - yp0oy * sintheta
                ) * reduced_coordi_factor + temp

                # xpoy-flipud(rot90(ypox))==0 is true
                ypox = (
                    xp0ox * sintheta + yp0ox * costheta
                ) * reduced_coordi_factor + temp

                J = xpox * ypoy - xpoy * ypox

        return lutxp, lutyp, J

    def _getFunction(self, name):
        """Decide to call the function of _poly10_2D() or _poly10Grad().

        This is to correct the off-axis distortion. A numerical solution with
        2-dimensions 10 order polynomials to map between the telescope
        aperture and defocused image plane is used.

        Parameters
        ----------
        name : str
            Function name to call.

        Returns
        -------
        numpy.ndarray
            Corrected image after the correction.

        Raises
        ------
        RuntimeError
            Raise error if the function name does not exist.
        """

        # Construct the dictionary table for calling function.
        # The reason to use the dictionary is for the future's extension.
        funcTable = dict(poly10_2D=self._poly10_2D, poly10Grad=self._poly10Grad)

        # Look for the function to call
        if name in funcTable:
            return funcTable[name]

        # Error for unknown function name
        raise RuntimeError("Unknown function name: %s" % name)

    def _poly10_2D(self, c, data, y=None):
        """Correct the off-axis distortion by fitting with a 10 order
        polynomial equation.

        Parameters
        ----------
        c : numpy.ndarray
            Parameters of off-axis distrotion.
        data : numpy.ndarray
            X, y-coordinate on aperture. If y is provided this will be just
            the x-coordinate.
        y : numpy.ndarray, optional
            Y-coordinate at aperture. (the default is None.)

        Returns
        -------
        numpy.ndarray
            Corrected parameters for off-axis distortion.
        """

        # Decide the x, y-coordinate data on aperture
        if y is None:
            x = data[0, :]
            y = data[1, :]
        else:
            x = data

        # Correct the off-axis distortion

        # Need to reorder the coefficients for use with GalSim:
        carr = np.zeros((11, 11))
        carr[np.tril_indices(11)] = c
        for i in range(0, 11):
            carr[:, i] = np.roll(carr[:, i], -i)
        return horner2d(x, y, carr)

    def _poly10Grad(self, c, x, y, atype):
        """Correct the off-axis distortion by fitting with a 10 order
        polynomial equation in the gradient part.

        Parameters
        ----------
        c : numpy.ndarray
            Parameters of off-axis distrotion.
        x : numpy.ndarray
            X-coordinate at aperture.
        y : numpy.ndarray
            Y-coordinate at aperture.
        atype : str
            Direction of gradient. It can be "dx" or "dy".

        Returns
        -------
        numpy.ndarray
            Corrected parameters for off-axis distortion.

        Raises
        ------
        ValueError
            If atype is not 'dx' or 'dy'.
        """
        if atype not in ["dx", "dy"]:
            raise ValueError(f"Unknown atype: {atype}")

        carr = np.zeros((11, 11))
        carr[np.tril_indices(11)] = c
        for i in range(0, 11):
            carr[:, i] = np.roll(carr[:, i], -i)
        grad = np.zeros(carr.shape, dtype=np.float64)

        if atype == "dx":
            for (i, j) in zip(*np.nonzero(carr)):
                if i > 0:
                    grad[i - 1, j] = carr[i, j] * i
        elif atype == "dy":
            for (i, j) in zip(*np.nonzero(carr)):
                if j > 0:
                    grad[i, j - 1] = carr[i, j] * j
        return horner2d(x, y, grad)

    def _createPupilGrid(self, lutx, luty, onepixel, ca, cb, ra, rb, fieldX, fieldY):
        """Create the pupil grid in off-axis model.

        This function gives the x,y-coordinate in the extended ring area
        defined by the parameter of onepixel.

        Parameters
        ----------
        lutx : numpy.ndarray
            X-coordinate on pupil plane.
        luty : numpy.ndarray
            Y-coordinate on pupil plane.
        onepixel : float
            Extended delta radius.
        ca : float
            Center of outer ring on the pupil plane.
        cb : float
            Center of inner ring on the pupil plane.
        ra : float
            Radius of outer ring on the pupil plane.
        rb : float
            Radius of inner ring on the pupil plane.
        fieldX : float
            X-coordinate of donut on the focal plane in degree.
        fieldY : float
            Y-coordinate of donut on the focal plane in degree.

        Returns
        -------
        numpy.ndarray
            X-coordinate of extended ring area on pupil plane.
        numpy.ndarray
            Y-coordinate of extended ring area on pupil plane.
        """

        # Rotate the mask center after the off-axis correction based on the
        # position of fieldX and fieldY
        cax, cay, cbx, cby = self._rotateMaskParam(ca, cb, fieldX, fieldY)

        # Get x, y coordinate of extended outer boundary by the linear
        # approximation
        lutx, luty = self._approximateExtendedXY(
            lutx, luty, cax, cay, ra, ra + onepixel, "outer"
        )

        # Get x, y coordinate of extended inner boundary by the linear
        # approximation
        lutx, luty = self._approximateExtendedXY(
            lutx, luty, cbx, cby, rb - onepixel, rb, "inner"
        )

        return lutx, luty

    def _approximateExtendedXY(self, lutx, luty, cenX, cenY, innerR, outerR, config):
        """Calculate the x, y-coordinate on pupil plane in the extended ring
        area by the linear approximation, which is used in the off-axis
        correction.

        Parameters
        ----------
        lutx : numpy.ndarray
            X-coordinate on pupil plane.
        luty : numpy.ndarray
            Y-coordinate on pupil plane.
        cenX : float
            X-coordinate of boundary ring center.
        cenY : float
            Y-coordinate of boundary ring center.
        innerR : float
            Inner radius of extended ring.
        outerR : float
            Outer radius of extended ring.
        config : str
            Configuration to calculate the x,y-coordinate in the extended ring.
            "inner": inner extended ring; "outer": outer extended ring.

        Returns
        -------
        numpy.ndarray
            X-coordinate of extended ring area on pupil plane.
        numpy.ndarray
            Y-coordinate of extended ring area on pupil plane.
        """

        # Calculate the distance to rotated center of boundary ring
        lutr = np.sqrt((lutx - cenX) ** 2 + (luty - cenY) ** 2)

        # Define NaN to be 999 for the comparison in the following step
        tmp = lutr.copy()
        tmp[np.isnan(tmp)] = 999

        # Get the available index that the related distance is between innderR
        # and outerR
        idxbound = (~np.isnan(lutr)) & (tmp >= innerR) & (tmp <= outerR)

        # Deside R based on the configuration
        if config == "outer":
            R = innerR
            # Get the index that the related distance is bigger than outerR
            idxout = tmp > outerR
        elif config == "inner":
            R = outerR
            # Get the index that the related distance is smaller than innerR
            idxout = tmp < innerR

        # Put the x, y-coordinate to be NaN if it is inside/ outside the pupil
        # that is after the off-axis correction.
        lutx[idxout] = np.nan
        luty[idxout] = np.nan

        # Get the x, y-coordinate in this ring area by the linear approximation
        lutx[idxbound] = (lutx[idxbound] - cenX) / lutr[idxbound] * R + cenX
        luty[idxbound] = (luty[idxbound] - cenY) / lutr[idxbound] * R + cenY

        return lutx, luty

    def _rotateMaskParam(self, ca, cb, fieldX, fieldY):
        """Rotate the mask-related parameters of center.

        Parameters
        ----------
        ca : float
            Mask-related parameter of center.
        cb : float
            Mask-related parameter of center.
        fieldX : float
            X-coordinate of donut on the focal plane in degree.
        fieldY : float
            Y-coordinate of donut on the focal plane in degree.

        Returns
        -------
        float
            Projected x element after the rotation.
        float
            Projected y element after the rotation.
        float
            Projected x element after the rotation.
        float
            Projected y element after the rotation.
        """

        # Calculate the sin(theta) and cos(theta) for the rotation
        fieldDist = self._getFieldDistFromOrigin(
            fieldX=fieldX, fieldY=fieldY, minDist=0
        )
        if fieldDist == 0:
            c = 0
            s = 0
        else:
            # Calculate cos(theta)
            c = fieldX / fieldDist

            # Calculate sin(theta)
            s = fieldY / fieldDist

        # Projected x and y coordinate after the rotation
        cax = c * ca
        cay = s * ca

        cbx = c * cb
        cby = s * cb

        return cax, cay, cbx, cby

    def centerOnProjection(self, img, template, window=20):
        """Center the image to the template's center.

        Parameters
        ----------
        img : numpy.array
            Image to be centered with the template. The input image needs to
            be a n-by-n matrix.
        template : numpy.array
            Template image to have the same dimension as the input image
            ('img'). The center of template is the position of input image
            tries to align with.
        window : int, optional
            Size of window in pixel. Assume the difference of centers of input
            image and template is in this range (e.g. [-window/2, window/2] if
            1D). (the default is 20.)

        Returns
        -------
        numpy.array
            Recentered image.
        """

        # Calculate the cross-correlate
        corr = correlate(img, template, mode="same")

        # Calculate the shifts of center

        # Only consider the shifts in a certain window (range)
        # Align the input image to the center of template
        length = template.shape[0]
        center = length // 2

        r = window // 2

        mask = np.zeros(corr.shape)
        mask[center - r : center + r, center - r : center + r] = 1
        idx = np.argmax(corr * mask)

        # The above 'idx' is an interger. Need to rematch it to the
        # two-dimension position (x and y)
        xmatch = idx % length
        ymatch = idx // length

        dx = center - xmatch
        dy = center - ymatch

        # Shift/ recenter the input image
        return np.roll(np.roll(img, dx, axis=1), dy, axis=0)

    def setOffAxisCorr(self, inst, order):
        """Set the coefficients of off-axis correction for x, y-projection of
        intra- and extra-image.

        This is for the mapping of coordinate from the telescope aperture to
        defocal image plane.

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        order : int
            Up to order-th of off-axis correction.
        """

        # List of configuration
        configList = ["cxin", "cyin", "cxex", "cyex"]

        # Get all files in the directory
        instDir = inst.getInstFileDir()
        fileList = [
            f for f in os.listdir(instDir) if os.path.isfile(os.path.join(instDir, f))
        ]

        # Read files
        offAxisCoeff = []
        for config in configList:

            # Construct the configuration file name
            for fileName in fileList:
                m = re.match(r"\S*%s\S*.yaml" % config, fileName)
                if m is not None:
                    matchFileName = m.group()
                    break

            filePath = os.path.join(instDir, matchFileName)
            corrCoeff, offset = self._getOffAxisCorrSingle(filePath)
            offAxisCoeff.append(corrCoeff)

        # Give the values
        self.offAxisCoeff = np.array(offAxisCoeff)
        self.offAxisOffset = offset

    def _getOffAxisCorrSingle(self, confFile):
        """Get the image-related parameters for the off-axis distortion by the
        linear approximation with a series of fitted parameters with LSST
        ZEMAX model.

        Parameters
        ----------
        confFile : str
            Path of configuration file.

        Returns
        -------
        numpy.ndarray
            Coefficients for the off-axis distortion based on the linear
            response.
        float
            Defocal distance in m.
        """

        fieldDist = self._getFieldDistFromOrigin(minDist=0.0)

        # Read the configuration file
        paramReader = ParamReader()
        paramReader.setFilePath(confFile)
        cdata = paramReader.getMatContent()

        # Record the offset (defocal distance)
        offset = cdata[0, 0]

        # Take the reference parameters
        c = cdata[:, 1:]

        # Get the ruler, which is the distance to center
        # ruler is between 1.51 and 1.84 degree here
        ruler = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)

        # Get the fitted parameters for off-axis correction by linear
        # approximation
        corr_coeff = self._linearApprox(fieldDist, ruler, c[:, 2:])

        return corr_coeff, offset

    def _interpMaskParam(self, fieldX, fieldY, maskParam):
        """Get the mask-related parameters for the off-axis distortion and
        vignetting correction by the linear approximation with a series of
        fitted parameters with LSST ZEMAX model.

        Parameters
        ----------
        fieldX : float
            X-coordinate of donut on the focal plane in degree.
        fieldY : float
            Y-coordinate of donut on the focal plane in degree.
        maskParam : numpy.ndarray
            Fitted coefficients for the off-axis distortion and vignetting
            correction.

        Returns
        -------
        float
            'ca' coefficient for the off-axis distortion and vignetting
            correction based on the linear response.
        float
            'ra' coefficient for the off-axis distortion and vignetting
            correction based on the linear response.
        float
            'cb' coefficient for the off-axis distortion and vignetting
            correction based on the linear response.
        float
            'rb' coefficient for the off-axis distortion and vignetting
            correction based on the linear response.
        """

        # Calculate the distance from donut to origin (aperture)
        filedDist = np.sqrt(fieldX**2 + fieldY**2)

        # Get the ruler, which is the distance to center
        # ruler is between 1.51 and 1.84 degree here
        ruler = np.sqrt(2) * maskParam[:, 0]

        # Get the fitted parameters for off-axis correction by linear
        # approximation
        param = self._linearApprox(filedDist, ruler, maskParam[:, 1:])

        # Define related parameters
        ca = param[0]
        ra = param[1]
        cb = param[2]
        rb = param[3]

        return ca, ra, cb, rb

    def _linearApprox(self, fieldDist, ruler, parameters):
        """Get the fitted parameters for off-axis correction by linear
        approximation.

        Parameters
        ----------
        fieldDist : float
            Field distance from donut to origin (aperture).
        ruler : numpy.ndarray
            A series of distance with available parameters for the fitting.
        parameters : numpy.ndarray
            Referenced parameters for the fitting.

        Returns
        -------
        numpy.ndarray
            Fitted parameters based on the linear approximation.
        """

        # Sort the ruler and parameters based on the magnitude of ruler
        sortIndex = np.argsort(ruler)
        ruler = ruler[sortIndex]
        parameters = parameters[sortIndex, :]

        # Compare the distance to center (aperture) between donut and standard
        compDis = ruler >= fieldDist

        # fieldDist is too big and out of range
        if fieldDist > ruler.max():
            # Take the coefficients in the highest boundary
            p2 = parameters.shape[0] - 1
            p1 = 0
            w1 = 0
            w2 = 1

        # fieldDist is too small to be in the range
        elif fieldDist < ruler.min():
            # Take the coefficients in the lowest boundary
            p2 = 0
            p1 = 0
            w1 = 1
            w2 = 0

        # fieldDist is in the range
        else:
            # Find the boundary of fieldDist in the known data
            p2 = compDis.argmax()
            p1 = p2 - 1

            # Calculate the weighting ratio
            w1 = (ruler[p2] - fieldDist) / (ruler[p2] - ruler[p1])
            w2 = 1 - w1

        # Get the fitted parameters for off-axis correction by linear
        # approximation
        param = w1 * parameters[p1, :] + w2 * parameters[p2, :]

        return param

    def makeMaskList(self, inst, model):
        """Calculate the mask list based on the obscuration and optical model.

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".

        Returns
        -------
        numpy.ndarray
            The list of mask.
        """

        # Masklist = [center_x, center_y, radius_of_boundary,
        #             1/ 0 for outer/ inner boundary]
        if model in ("paraxial", "onAxis"):

            if inst.obscuration == 0:
                masklist = np.array([[0, 0, 1, 1]])
            else:
                masklist = np.array([[0, 0, 1, 1], [0, 0, inst.obscuration, 0]])
        else:
            # Get the mask-related parameters
            maskCa, maskRa, maskCb, maskRb = self._interpMaskParam(
                self.fieldX, self.fieldY, inst.maskOffAxisCorr
            )

            # Rotate the mask-related parameters of center
            cax, cay, cbx, cby = self._rotateMaskParam(
                maskCa, maskCb, self.fieldX, self.fieldY
            )
            masklist = np.array(
                [
                    [0, 0, 1, 1],
                    [0, 0, inst.obscuration, 0],
                    [cax, cay, maskRa, 1],
                    [cbx, cby, maskRb, 0],
                ]
            )

        return masklist

    def _showProjection(self, lutxp, lutyp, sensorFactor, projSamples, raytrace=False):
        """Calculate the x, y-projection of image on pupil.

        This can be used to calculate the center of projection in compensate().

        Parameters
        ----------
        lutxp : numpy.ndarray
            X-coordinate on pupil plane. The value of element will be NaN if
            that point is not inside the pupil.
        lutyp : numpy.ndarray
            Y-coordinate on pupil plane. The value of element will be NaN if
            that point is not inside the pupil.
        sensorFactor : float
            Sensor factor.
        projSamples : int
            Dimension of projected image. This value considers the
            magnification ratio of donut image.
        raytrace : bool, optional
            Consider the ray trace or not. If the value is true, the times of
            photon hit will aggregate. (the default is False.)

        Returns
        -------
        numpy.ndarray
            Projection of image. It will be a binary image if raytrace=False.
        """

        # Dimension of pupil image
        n1, n2 = lutxp.shape

        # Construct the binary matrix on pupil. It is noted that if the
        # raytrace is true, the value of element is allowed to be greater
        # than 1.
        show_lutxyp = np.zeros((n1, n2))

        # Get the index in pupil. If a point's value is NaN, this point is
        # outside the pupil.
        idx = ~np.isnan(lutxp)

        # Calculate the projected x, y-coordinate in pixel
        # x=0.5 is center of pixel#1
        xR = np.zeros((n1, n2))
        yR = np.zeros((n1, n2))

        xR[idx] = np.round(
            (lutxp[idx] + sensorFactor) * (projSamples / sensorFactor) / 2 + 0.5
        )
        yR[idx] = np.round(
            (lutyp[idx] + sensorFactor) * (projSamples / sensorFactor) / 2 + 0.5
        )

        # Check the projected coordinate is in the range of image or not.
        # If the check passes, the times will be recorded.
        mask = np.bitwise_and(
            np.bitwise_and(np.bitwise_and(xR > 0, xR < n2), yR > 0), yR < n1
        )

        # Check the projected coordinate is in the range of image or not.
        # If the check passes, the times will be recorded.
        if raytrace:
            for ii, jj in zip(
                np.array(yR - 1, dtype=int)[mask], np.array(xR - 1, dtype=int)[mask]
            ):
                show_lutxyp[ii, jj] += 1
        else:
            show_lutxyp[
                np.array(yR - 1, dtype=int)[mask], np.array(xR - 1, dtype=int)[mask]
            ] = 1

        return show_lutxyp

    def makeMask(
        self, inst, model, boundaryT, maskScalingFactorLocal, compensated=True
    ):
        """Get the binary mask which considers the obscuration and off-axis
        correction.

        There will be two mask parameters to be calculated:
        mask_comp: computation mask, i.e. padded mask,
            for use at the offset planes
        mask_pupil: pupil mask, i.e. non-padded mask,
            corresponding to aperture

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        boundaryT : int
            Extended boundary in pixel. It defines how far the computation mask
            extends beyond the pupil mask. And, in fft, it is also the width of
            Neuman boundary where the derivative of the wavefront is set to
            zero.
        maskScalingFactorLocal : float
            Mask scaling factor (for fast beam) for local correction.
        compensated: bool, optional
            Whether to calculate masks for the compensated or original image.
            If True, mask will be in orientation of the compensated image.
            If False, mask will be in orientation of the original image.
            Note this is only relevant for extrafocal images.
            (the default is True.)
        """

        self.mask_pupil = np.ones(inst.dimOfDonutImg, dtype=int)
        self.mask_comp = self.mask_pupil.copy()

        rMask = (
            inst.apertureDiameter
            / (2 * inst.focalLength / inst.defocalDisOffsetInM)
            * maskScalingFactorLocal
        )

        # Get the mask list
        xSensor, ySensor = inst.getSensorCoor()
        masklist = self.makeMaskList(inst, model)
        for ii in range(masklist.shape[0]):

            # Distance to center on pupil
            r = np.sqrt(
                (xSensor - masklist[ii, 0]) ** 2 + (ySensor - masklist[ii, 1]) ** 2
            )

            # Find the indices that correspond to the mask element, set them to
            # the pass/ block boolean

            # Get the index inside the aperture
            idx = r <= masklist[ii, 2]

            # Get the higher and lower boundary beyond the pupil mask by
            # extension.
            # The extension level is dicided by boundaryT.
            # In fft, this is also the Neuman boundary where the derivative of
            # the wavefront is set to zero.
            if masklist[ii, 3] >= 1:
                aidx = np.nonzero(
                    r <= masklist[ii, 2] * (1 + boundaryT * inst.pixelSize / rMask)
                )
            else:
                aidx = np.nonzero(
                    r <= masklist[ii, 2] * (1 - boundaryT * inst.pixelSize / rMask)
                )

            # Initialize both mask elements to the opposite of the pass/ block
            # boolean
            mask_pupil_ii = (1 - int(masklist[ii, 3])) * np.ones(
                [inst.dimOfDonutImg, inst.dimOfDonutImg], dtype=int
            )
            mask_comp_ii = mask_pupil_ii.copy()

            mask_pupil_ii[idx] = int(masklist[ii, 3])
            mask_comp_ii[aidx] = int(masklist[ii, 3])

            # Multiplicatively add the current mask elements to the model
            # masks.
            # This is try to find the common mask region.

            # non-padded mask corresponding to aperture
            self.mask_pupil = self.mask_pupil * mask_pupil_ii

            # padded mask for use at the offset planes
            self.mask_comp = self.mask_comp * mask_comp_ii

        # If we do not want compensated masks, rotate the extrafocal mask,
        # so that it is in the orientation of the original image
        if not compensated and self.defocalType == DefocalType.Extra:
            self.mask_pupil = np.rot90(self.mask_pupil, 2)
            self.mask_comp = np.rot90(self.mask_comp, 2)

    def makeBlendedMask(
        self,
        inst,
        model,
        boundaryT,
        maskScalingFactorLocal,
        blendPadding=8,
        compensated=True,
    ):
        """Create a binary mask of a central source with the area overlapping
        a blended object cutout.

        Parameters
        ----------
        inst : Instrument
            Instrument to use.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        boundaryT : int
            Extended boundary in pixel. It defines how far the computation mask
            extends beyond the pupil mask. And, in fft, it is also the width of
            Neuman boundary where the derivative of the wavefront is set to
            zero.
        maskScalingFactorLocal : float
            Mask scaling factor (for fast beam) for local correction.
        blendPadding : int, optional
            Number of pixels to increase the radius and expand the
            footprint of the blended source. (the default is 8.)
        compensated: bool, optional
            Whether to calculate masks for the compensated or original image.
            If True, mask will be in orientation of the compensated image.
            If False, mask will be in orientation of the original image.
            Note this is only relevant for extrafocal images.
            (the default is True.)

        Returns
        -------
        numpy.ndarray
            Projection of image with the footprint of any overlapping donuts
            cutout of the mask. It will be a binary image if raytrace=False.
        """

        self.makeMask(
            inst, model, boundaryT, maskScalingFactorLocal, compensated=compensated
        )

        # Add blended donuts if they exist
        if len(self.blendOffsetX) > 0:
            newMaskPupil = self.createBlendedCoadd(self.mask_pupil, blendPadding)
            newMaskComp = self.createBlendedCoadd(self.mask_comp, blendPadding)
            self.mask_pupil = newMaskPupil
            self.mask_comp = newMaskComp

    def createBlendedCoadd(self, maskArray, blendPadding):
        """Cutout regions of the input mask where blended donuts
        overlap with the original donut.

        Parameters
        ----------
        maskArray : numpy.ndarray
            Original input binary mask with just a single unblended donut.
        blendPadding : int
            Number of pixels to increase the radius and expand the
            footprint of the blended source.

        Returns
        -------
        numpy.ndarray
            New maskArray modified to remove the footprint of any
            overlapping donuts from the masked area of the original donut.
        """

        # Switch x,y order because x,y locations in the catalogs and the
        # LSST Science Pipeline objects are consistent with each other but
        # numpy arrays are [row, column] so when using the catalog x,y values
        # with numpy arrays we switch the order to get the right location.
        pixelShift = np.array([self.blendOffsetY, self.blendOffsetX]).T

        # Create binary images of individual blended donuts across mask
        maskBlends = list()
        for xShift, yShift in pixelShift:
            shiftVector = np.array([xShift, yShift], dtype=int)
            # Rotate extrafocal donut position 180 degrees
            if self.defocalType == DefocalType.Extra:
                theta = np.degrees(np.pi)
                shiftVector = np.dot(rotMatrix(theta), shiftVector)
            shiftedMask = shift(copy(maskArray), shiftVector)
            if blendPadding > 0:
                shiftedMask = binary_dilation(shiftedMask, iterations=blendPadding)
            maskBlends.append(shiftedMask)

        # Sum blends with original source mask to find where overlaps exist
        maskCoadd = copy(maskArray)
        for maskBlend in maskBlends:
            maskCoadd += maskBlend
        # Turn into a binary mask of the overlap regions
        maskCoadd[maskCoadd <= 1] = 0
        maskCoadd[maskCoadd > 0] = 1

        # Subtract blended region from single donut image
        # to get mask of only unblended donut area
        maskFinal = maskArray - maskCoadd
        maskFinal[maskFinal < 0] = 0

        return maskFinal
