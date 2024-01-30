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

from typing import Optional, Tuple, Union

import galsim
import numpy as np
from lsst.ts.wep.image import Image
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType, PlaneType
from lsst.ts.wep.utils.ioUtils import configClass
from lsst.ts.wep.utils.miscUtils import centerWithTemplate, polygonContains
from lsst.ts.wep.utils.zernikeUtils import zernikeGradEval
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, shift


class ImageMapper:
    """Class for mapping the pupil to the image plane, and vice versa.

    This class also creates image masks.

    Parameters
    ----------
    instConfig : str or dict or Instrument, optional
        Instrument configuration. If a string, it is assumed this points
        to a config file, which is used to configure the Instrument.
        If the path begins with "policy:", then it is assumed the path is
        relative to the policy directory. If a dictionary, it is assumed to
        hold keywords for configuration. If an Instrument object, that object
        is just used.
        (the default is "policy:instruments/LsstCam.yaml")
    opticalModel : str, optional
        The optical model to use for mapping between the image and pupil
        planes. Can be "onAxis", or "offAxis". onAxis is an analytic model
        appropriate for donuts near the optical axis. It is valid for both
        slow and fast optical systems. The offAxis model is a numerically-fit
        model that is valid for fast optical systems at wide field angles.
        offAxis requires an accurate Batoid model.
        (the default is "offAxis")
    """

    def __init__(
        self,
        instConfig: Union[str, dict, Instrument] = "policy:instruments/LsstCam.yaml",
        opticalModel: str = "offAxis",
    ) -> None:
        self._instrument = configClass(instConfig, Instrument)
        self.opticalModel = opticalModel

    @property
    def instrument(self) -> Instrument:
        """The instrument object that defines the optical geometry."""
        return self._instrument

    @property
    def opticalModel(self) -> str:
        """The name of the optical model to use for image mapping."""
        return self._opticalModel

    @opticalModel.setter
    def opticalModel(self, value: str) -> None:
        """Set the optical model to use for image mapping.

        Parameters
        ----------
        value : str
            The optical model to use for mapping between the image and
            pupil planes. Can be "onAxis", or "offAxis". onAxis is an
            analytic model appropriate for donuts near the optical axis.
            It is valid for both slow and fast optical systems. The offAxis
            model is a numerically-fit model that is valid for fast optical
            systems at wide field angles. offAxis requires an accurate Batoid
            model.

        Raises
        ------
        TypeError
            If the value is not a string
        ValueError
            If the value is not one of the allowed values
        """
        allowedModels = ["onAxis", "offAxis"]
        if not isinstance(value, str):
            raise TypeError("optical model must be a string.")
        elif value not in allowedModels:
            raise ValueError(f"opticalModel must be one of {str(allowedModels)[1:-1]}.")

        self._opticalModel = value

    def _constructForwardMap(
        self,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        zkCoeff: np.ndarray,
        image: Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the forward mapping from the pupil to the image plane.

        Parameters
        ----------
        uPupil : np.ndarray
             Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
             Normalized y coordinates on the image plane
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        image : Image
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the image plane
        np.ndarray
            Normalized y coordinates on the image plane
        np.ndarray
            The Jacobian of the forward map
        np.ndarray
            The determinant of the Jacobian

        Raises
        ------
        RuntimeWarning
            If the optical model is not supported
        """
        # Get the Zernikes for the mapping
        if self.opticalModel == "onAxis":
            zkMap = zkCoeff
        elif self.opticalModel == "offAxis":
            # Get the off-axis coefficients
            offAxisCoeff = self.instrument.getOffAxisCoeff(
                *image.fieldAngle,
                image.defocalType,
                image.bandLabel,
                jmaxIntrinsic=len(zkCoeff) + 3,
            )

            # Add these coefficients to the input coefficients
            size = max(zkCoeff.size, offAxisCoeff.size)
            zkMap = np.zeros(size)
            zkMap[: zkCoeff.size] = zkCoeff
            zkMap[: offAxisCoeff.size] += offAxisCoeff
        else:
            raise RuntimeError(f"Optical model {self.opticalModel} not supported")

        # Calculate all 1st- and 2nd-order Zernike derivatives
        d1Wdu = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=1,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d1Wdv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=0,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdudu = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=2,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdvdv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=0,
            vOrder=2,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdudv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=1,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdvdu = d2Wdudv

        # Plus the first order derivatives at the center of the pupil
        d1Wdu0 = zernikeGradEval(
            np.zeros(1),
            np.zeros(1),
            uOrder=1,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d1Wdv0 = zernikeGradEval(
            np.zeros(1),
            np.zeros(1),
            uOrder=0,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )

        # Get the required info about the telescope geometry
        N = self.instrument.focalRatio
        l = self.instrument.defocalOffset  # noqa: E741

        # Calculate the mapping determined by the optical model
        if self.opticalModel == "onAxis":
            # The onAxis model is analytic and intended for fast optical
            # systems, near the optical axis

            # Determine defocal sign from the image plane at z = f +/- l
            # I.e., the extrafocal image at z = f + l is associated with +1,
            # and the intrafocal image at z = f - l is associated with -1.
            defocalSign = +1 if image.defocalType == DefocalType.Extra else -1

            # Calculate the prefactor
            prefactor = np.sqrt(4 * N**2 - 1)

            # Calculate the factors F and C
            rPupil = np.sqrt(uPupil**2 + vPupil**2)
            with np.errstate(invalid="ignore"):
                F = -defocalSign / np.sqrt(4 * N**2 - rPupil**2)
            C = -2 * N / l

            # Map the pupil points onto the image plane
            uImage = prefactor * (F * uPupil + C * (d1Wdu - d1Wdu0))
            vImage = prefactor * (F * vPupil + C * (d1Wdv - d1Wdv0))

            # Calculate the elements of the Jacobian
            J00 = prefactor * (F + F**3 * uPupil**2 + C * d2Wdudu)
            J01 = prefactor * (F**3 * uPupil * vPupil + C * d2Wdvdu)
            J10 = prefactor * (F**3 * vPupil * uPupil + C * d2Wdudv)
            J11 = prefactor * (F + F**3 * vPupil**2 + C * d2Wdvdv)

        else:
            # The offAxis model uses a numerically-fit model from Batoid
            # This model is able to account for wide field distortion effects
            # in fast optical systems, however it is generally applicable to
            # all optical systems for which you have a good Batoid model

            # Calculate the prefactor
            prefactor = -2 * N * np.sqrt(4 * N**2 - 1) / l

            # Map the pupil points onto the image plane
            uImage = prefactor * (d1Wdu - d1Wdu0)
            vImage = prefactor * (d1Wdv - d1Wdv0)

            # Calculate the elements of the Jacobian
            J00 = prefactor * d2Wdudu
            J01 = prefactor * d2Wdvdu
            J10 = prefactor * d2Wdudv
            J11 = prefactor * d2Wdvdv

        # Assemble the Jacobian
        jac = np.array(
            [
                [J00, J01],
                [J10, J11],
            ]
        )

        # Calculate the determinant
        jacDet = J00 * J11 - J01 * J10

        return uImage, vImage, jac, jacDet

    def _constructInverseMap(
        self,
        uImage: np.ndarray,
        vImage: np.ndarray,
        zkCoeff: np.ndarray,
        image: Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the inverse mapping from the image plane to the pupil.

        Parameters
        ----------
        uImage : np.ndarray
            Normalized x coordinates on the image plane
        vImage : np.ndarray
            Normalized y coordinates on the image plane
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        image : Image
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the pupil plane
        np.ndarray
            Normalized y coordinates on the pupil plane
        np.ndarray
            The Jacobian of the inverse mapping
        np.ndarray
            The determinant of the Jacobian
        """
        # Create a test grid on the pupil to pre-fit the image -> pupil mapping
        uPupilTest = np.linspace(-1, 1, 10)
        uPupilTest, vPupilTest = np.meshgrid(uPupilTest, uPupilTest)

        # Mask outside the pupil
        rPupilTest = np.sqrt(uPupilTest**2 + vPupilTest**2)
        pupilMask = rPupilTest <= 1
        pupilMask &= rPupilTest >= self.instrument.obscuration
        uPupilTest = uPupilTest[pupilMask]
        vPupilTest = vPupilTest[pupilMask]

        # Project the test pupil grid onto the image plane
        uImageTest, vImageTest, jac, jacDet = self._constructForwardMap(
            uPupilTest,
            vPupilTest,
            zkCoeff,
            image,
        )

        # Use test points to fit Zernike coeff for image -> pupil mapping
        rImageMax = np.sqrt(uImageTest**2 + vImageTest**2).max()
        invCoeff, *_ = np.linalg.lstsq(
            galsim.zernike.zernikeBasis(
                6,
                uImageTest,
                vImageTest,
                R_outer=rImageMax,
            ).T,
            np.array([uPupilTest, vPupilTest]).T,
            rcond=None,
        )

        # Now we will map our image points to the pupil using the coefficients
        # we just fit, and then map them back to the image plane using the
        # analytic forward mapping
        # Ideally, this round-trip mapping will return the same image points
        # we started with, however our initial image -> pupil mapping will not
        # be perfect, so this will not be the case. We will iteratively apply
        # Newton's method to reduce the residuals, and thereby improve the
        # mapping

        # Map the image points to the pupil
        uPupil = galsim.zernike.Zernike(
            invCoeff[:, 0],
            R_outer=rImageMax,
        )(uImage, vImage)
        vPupil = galsim.zernike.Zernike(
            invCoeff[:, 1],
            R_outer=rImageMax,
        )(uImage, vImage)

        # Map these pupil points back to the image (RT = round-trip)
        uImageRT, vImageRT, jac, jacDet = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            image,
        )

        # Calculate the residuals of the round-trip mapping
        duImage = uImageRT - uImage
        dvImage = vImageRT - vImage

        # Now iterate Newton's method to improve the mapping
        # (i.e. minimize the residuals)
        for _ in range(10):
            # Add corrections to the pupil coordinates using Newton's method
            uPupil -= (+jac[1, 1] * duImage - jac[0, 1] * dvImage) / jacDet
            vPupil -= (-jac[1, 0] * duImage + jac[0, 0] * dvImage) / jacDet

            # Map these new pupil points to the image plane
            uImageRT, vImageRT, jac, jacDet = self._constructForwardMap(
                uPupil,
                vPupil,
                zkCoeff,
                image,
            )

            # Calculate the new residuals
            duImage = uImageRT - uImage
            dvImage = vImageRT - vImage

            # If the residuals are small enough, stop iterating
            maxResiduals = np.max([np.abs(duImage), np.abs(dvImage)], axis=0)
            if np.all(maxResiduals <= 1e-5):
                break

        # Set not-converged points to NaN
        notConverged = maxResiduals > 1e-5
        uPupil[notConverged] = np.nan
        vPupil[notConverged] = np.nan
        jac[..., notConverged] = np.nan
        jacDet[notConverged] = np.nan

        # Invert the Jacobian
        jac = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / jacDet
        jacDet = 1 / jacDet

        return uPupil, vPupil, jac, jacDet

    def _getImageGridInsidePupil(
        self,
        zkCoeff: np.ndarray,
        image: Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return image grid and mask for which pixels are inside the pupil.

        Note the pupil considered is the pupil mapped to the image plane.

        Parameters
        ----------
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        image : Image
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the image plane
        np.ndarray
            Normalized y coordinates on the image plane
        np.ndarray
            Binary mask array indicating whether each pixel is inside the pupil
        """
        # Map pupil edge to the image to determine edge of pupil on the image
        theta = np.linspace(0, 2 * np.pi, 100)
        uPupil, vPupil = np.cos(theta), np.sin(theta)
        uImageEdge, vImageEdge, *_ = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            image,
        )
        imageEdge = np.array([uImageEdge, vImageEdge]).T

        # Create an image grid
        nPixels = image.image.shape[0]
        uImage, vImage = self.instrument.createImageGrid(nPixels)

        # Determine which image pixels have corners inside the pupil
        dPixel = uImage[0, 1] - uImage[0, 0]
        corners = np.append(uImage[0] - dPixel / 2, uImage[0, -1] + dPixel / 2)
        inside = polygonContains(*np.meshgrid(corners, corners), imageEdge)

        # Select pixels that have at least one corner inside
        inside = inside[:-1, :-1] | inside[1:, :-1] | inside[:-1, 1:] | inside[1:, 1:]

        return uImage, vImage, inside

    def _maskWithCircle(
        self,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        uPupilCirc: float,
        vPupilCirc: float,
        rPupilCirc: float,
        fwdMap: Optional[tuple] = None,
    ) -> np.ndarray:
        """Return a fractional mask for a single circle.

        Parameters
        ----------
        uPupil : np.ndarray
            Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
            Normalized y coordinates on the pupil plane
        uPupilCirc : float
            The u coordinate of the circle center
        vPupilCirc : float
            The v coordinate of the circle center
        rPupilCirc : float
            The normalized radius of the circle
        fwdMap : tuple
            A tuple containing (uImage, vImage, jac, jacDet), i.e.
            the output of self._constructForwardMap(uPupil, vPupil, ...)
            If not None, the mask is mapped to the image plane.
            (the default is None)

        Returns
        -------
        np.ndarray
            Fractional mask with the same shape as uPupil
        """
        # Center the pupil coordinates on the circle's center
        uPupilCen = uPupil - uPupilCirc
        vPupilCen = vPupil - vPupilCirc

        # Pixel scale in normalized coordinates is inverse of the donut radius
        pixelScale = 1 / self.instrument.donutRadius

        # If a forward map is provided, begin preparing for mapping the mask
        # to the image plane
        if fwdMap is not None:
            uImage, vImage, jac, jacDet = fwdMap

            # Calculate quantities for the forward map
            invJac = np.array(
                [
                    [+jac[1, 1], -jac[0, 1]],  # type: ignore
                    [-jac[1, 0], +jac[0, 0]],  # type: ignore
                ]
            )
            invJac /= jacDet

            # Use a local linear approximation to center the image coordinates
            uImageCen = uImage - jac[0, 0] * uPupilCirc - jac[0, 1] * vPupilCirc
            vImageCen = vImage - jac[1, 0] * uPupilCirc - jac[1, 1] * vPupilCirc

            # Calculate the diagonal distance across each pixel on the pupil
            diagL = np.sqrt(
                (invJac[0, 0] + invJac[0, 1]) ** 2  # type: ignore
                + (invJac[1, 0] + invJac[1, 1]) ** 2  # type: ignore
            )
            diagL *= pixelScale

        else:
            # Use the pupil coordinates as the image coordinates
            uImageCen = uPupilCen
            vImageCen = vPupilCen

            # Diagonal distance across a regular pixel
            diagL = np.sqrt(2) * pixelScale

        # Assign pixels to groups based on whether they're definitely
        # inside/outside the circle, or on the border
        rPupilCen = np.sqrt(uPupilCen**2 + vPupilCen**2)
        inside = rPupilCen < (rPupilCirc - diagL / 2)
        outside = rPupilCen > (rPupilCirc + diagL / 2)
        border = ~(inside | outside)

        # We can go ahead and assign fractional mask 1 (0) to pixels
        # totally inside (outside) the circle
        out = np.zeros_like(uPupil)
        out[inside] = 1
        out[outside] = 0

        # If nothing is on the border, go ahead and return the mask
        if not border.any():
            return out

        # Calculate coefficients for the line (y = m*x + b) that is tangent to
        # the circle where the ray that passes through each point intersects
        # the circle (in pupil coordinates)
        uPupilCen, vPupilCen = uPupilCen[border], vPupilCen[border]
        m = -uPupilCen / vPupilCen  # slope
        b = (
            np.sqrt(uPupilCen**2 + vPupilCen**2) * rPupilCirc / vPupilCen
        )  # intercept

        # Select the border image coordinates
        uImageCen, vImageCen = uImageCen[border], vImageCen[border]

        if fwdMap is not None:
            # Transform the slope and intercept to image coordinates
            invJac = invJac[..., border]  # type: ignore
            a1 = m * invJac[0, 0] - invJac[1, 0]
            a2 = m * uPupilCen + b - vPupilCen
            a3 = -m * invJac[0, 1] + invJac[1, 1]
            m = a1 / a3
            b = (a2 - a1 * uImageCen) / a3 + vImageCen

        # Use symmetry to map everything onto situation where -1 <= mImage <= 0
        mask = m > 0
        uImageCen[mask] = -uImageCen[mask]
        m[mask] = -m[mask]

        mask = m < -1
        uImageCen[mask], vImageCen[mask] = vImageCen[mask], uImageCen[mask]
        m[mask], b[mask] = 1 / m[mask], -b[mask] / m[mask]

        # Calculate the v intercept on the right side of the pixel
        vStar = m * (uImageCen + pixelScale / 2) + b

        # Calculate fractional distance of intercept from top of pixel
        gamma = (vImageCen + pixelScale / 2 - vStar) / pixelScale

        # Now determine illumination for border pixels
        borderOut = np.zeros_like(uPupilCen)

        # Pixels that are totally inside the circle
        mask = gamma < 0
        borderOut[mask] = 1

        # Pixels that are totally outside the circle
        mask = gamma > (1 - m)
        borderOut[mask] = 0

        # Pixels for which the circle crosses the left and bottom sides
        mask = (1 < gamma) & (gamma < (1 - m))
        borderOut[mask] = -0.5 / m[mask] * (1 - (gamma[mask] + m[mask])) ** 2

        # Pixels for which the circle crosses the left and right sides
        mask = (-m < gamma) & (gamma < 1)
        borderOut[mask] = 1 - gamma[mask] - m[mask] / 2

        # Pixels for which the circle crosses the top and right
        mask = (0 < gamma) & (gamma < -m)
        borderOut[mask] = 1 + 0.5 * gamma[mask] ** 2 / m[mask]

        # Values below the (centered) u axis need to be flipped
        mask = vImageCen < 0
        borderOut[mask] = 1 - borderOut[mask]

        # Put the border values into the global output array
        out[border] = borderOut

        return out

    def _maskLoop(
        self,
        image: Image,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        fwdMap: Optional[tuple] = None,
    ) -> np.ndarray:
        """Loop through mask elements to create the mask.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        uPupil : np.ndarray
            Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
            Normalized y coordinates on the pupil plane
        fwdMap : tuple
            A tuple containing (uImage, vImage, jac, jacDet), i.e.
            the output of self._constructForwardMap(uPupil, vPupil, ...)
            If not None, the mask is mapped to the image plane.
            (the default is None)

        Returns
        -------
        np.ndarray
            A flattened mask array
        """
        # Get the field angle
        angle = image.fieldAngle

        # Get the angle radius
        rTheta = np.sqrt(np.sum(np.square(angle)))

        # Flatten the pupil arrays
        uPupil, vPupil = uPupil.ravel(), vPupil.ravel()

        # If a forward map is provided, flatten those arrays too
        if fwdMap is not None:
            uImage, vImage, jac, jacDet = fwdMap
            uImage, vImage = uImage.ravel(), vImage.ravel()
            jac = jac.reshape(2, 2, -1)
            jacDet = jacDet.ravel()

        # Get the mask parameters from the instrument
        maskParams = self.instrument.maskParams

        # Start with a full mask
        mask = np.ones_like(uPupil)

        # Loop over each mask element
        for key, val in maskParams.items():
            # Get the indices of non-zero pixels
            idx = np.nonzero(mask)[0]

            # If all the pixels are zero, stop here
            if not idx.any():
                break

            # Only apply this mask if we're past thetaMin
            if rTheta < val["thetaMin"]:
                continue

            # Calculate the radius and center of the mask in meters
            radius = np.polyval(val["radius"], rTheta)
            rCenter = np.polyval(val["center"], rTheta)

            # Convert to normalized pupil coordinates
            radius /= self.instrument.radius
            rCenter /= self.instrument.radius

            # Use angle to convert radius to u and v components
            uCenter = 0 if rTheta == 0 else rCenter * angle[0] / rTheta
            vCenter = 0 if rTheta == 0 else rCenter * angle[1] / rTheta

            # Calculate the mask values
            maskVals = self._maskWithCircle(
                uPupil=uPupil[idx],
                vPupil=vPupil[idx],
                uPupilCirc=uCenter,
                vPupilCirc=vCenter,
                rPupilCirc=radius,  # type: ignore
                fwdMap=None
                if fwdMap is None
                else (uImage[idx], vImage[idx], jac[..., idx], jacDet[idx]),
            )

            # Assign the mask values
            if key.endswith("Inner"):
                mask[idx] = np.minimum(mask[idx], 1 - maskVals)
            else:
                mask[idx] = np.minimum(mask[idx], maskVals)

        return mask

    def _maskBlends(
        self,
        centralMask: np.ndarray,
        blendMask: np.ndarray,
        blendOffsets: np.ndarray,
        binary: bool,
    ) -> np.ndarray:
        """Shift the central mask to mask the blends.

        Parameters
        ----------
        mask : np.ndarray
            The central mask
        blendOffsets : np.ndarray
            The blend offsets
        binary : bool
            Whether to return a binary mask

        Returns
        -------
        np.ndarray
            The mask with the blends masked
        """
        # If no blends, just return the original mask
        if blendOffsets.size == 0:
            return centralMask

        # Shift blend mask to each offset and subtract from the central mask
        for offset in blendOffsets:
            centralMask -= shift(blendMask, offset)

        # Clip negative pixels
        centralMask = np.clip(centralMask, 0, 1)

        if binary:
            centralMask = (centralMask > 0.5).astype(int)

        return centralMask

    def createPupilMask(
        self,
        image: Image,
        *,
        binary: bool = True,
        dilate: int = 0,
        dilateBlends: int = 0,
        maskBlends: bool = False,
    ) -> np.ndarray:
        """Create the pupil mask for the stamp.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        binary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is
            returned instead. (the default is True)
        dilate : int, optional
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            binary==False. (the default is 0)
        dilateBlends : int, optional
            How many times to dilate the blended masks. Note this only matters
            if maskBlends==True, and is not an option if binary==False.
            (the default is 0)
        maskBlends : bool, optional
            Whether to mask the blends (i.e. the blended regions are masked
            out). (the default is False)

        Returns
        -------
        np.ndarray
            The pupil mask

        Raises
        ------
        ValueError
            Invalid dilate value.
        """
        # Check the dilate values
        dilate, dilateBlends = int(dilate), int(dilateBlends)
        if dilate < 0:
            raise ValueError("dilate must be a non-negative integer.")
        elif dilateBlends < 0:
            raise ValueError("dilateBlends must be a non-negative integer.")
        elif dilate > 0 and not binary:
            raise ValueError("If dilate is greater than zero, binary must be True.")

        # Get the pupil grid
        uPupil, vPupil = self.instrument.createPupilGrid()

        # Get the mask by looping over the mask elements
        mask = self._maskLoop(
            image=image,
            uPupil=uPupil,
            vPupil=vPupil,
            fwdMap=None,
        )

        # Restore the mask shape
        mask = mask.reshape(uPupil.shape)

        # Set the mask to binary?
        if binary:
            mask = (mask > 0.5).astype(int)

        # Dilate the mask?
        centralMask = mask.copy()
        blendMask = mask.copy()
        if dilate > 0:
            centralMask = binary_dilation(centralMask, iterations=dilate).astype(int)
        if dilateBlends > 0:
            blendMask = binary_dilation(blendMask, iterations=dilateBlends).astype(int)

        # Mask the blends?
        if maskBlends and image.blendOffsets.size > 0:
            totalMask = self._maskBlends(
                centralMask,
                blendMask,
                image.blendOffsets,
                binary,
            )
        else:
            totalMask = centralMask

        return totalMask

    def createImageMask(
        self,
        image: Image,
        zkCoeff: Optional[np.ndarray] = None,
        *,
        binary: bool = True,
        dilate: int = 0,
        dilateBlends: int = 0,
        maskBlends: bool = False,
        _invMap: Optional[tuple] = None,
    ) -> np.ndarray:
        """Create the image mask for the stamp.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        binary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is
            returned instead. (the default is True)
        dilate : int, optional
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            binary==False. (the default is 0)
        dilateBlends : int, optional
            How many times to dilate the blended masks. Note this only matters
            if maskBlends==True, and is not an option if binary==False.
            (the default is 0)
        maskBlends : bool, optional
            Whether to mask the blends (i.e. the blended regions are masked
            out). (the default is False)

        Returns
        -------
        np.ndarray
        """
        # Check the dilate values
        dilate, dilateBlends = int(dilate), int(dilateBlends)
        if dilate < 0:
            raise ValueError("dilate must be a non-negative integer.")
        elif dilateBlends < 0:
            raise ValueError("dilateBlends must be a non-negative integer.")
        elif dilate > 0 and not binary:
            raise ValueError("If dilate is greater than zero, binary must be True.")

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *image.fieldAngle,
                image.bandLabel,
            )

        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(zkCoeff, image)

        # Get the inverse mapping from image plane to pupil plane
        if _invMap is None:
            # Construct the inverse mapping
            uPupil, vPupil, invJac, invJacDet = self._constructInverseMap(
                uImage[inside],
                vImage[inside],
                zkCoeff,
                image,
            )
        else:
            uPupil, vPupil, invJac, invJacDet = _invMap

        # Rearrange into the forward map
        jac = np.array(
            [
                [+invJac[1, 1], -invJac[0, 1]],  # type: ignore
                [-invJac[1, 0], +invJac[0, 0]],  # type: ignore
            ]
        )
        jac /= invJacDet
        jacDet = 1 / invJacDet

        # Package the forward mapping
        fwdMap = (uImage[inside], vImage[inside], jac, jacDet)

        # Get the mask by looping over the mask elements
        mask = np.zeros_like(inside, dtype=float)
        mask[inside] = self._maskLoop(
            image=image,
            uPupil=uPupil,
            vPupil=vPupil,
            fwdMap=fwdMap,
        )

        # Set the mask to binary?
        if binary:
            mask = (mask > 0.5).astype(int)

        # Dilate the mask?
        centralMask = mask.copy()
        blendMask = mask.copy()
        if dilate > 0:
            centralMask = binary_dilation(centralMask, iterations=dilate).astype(int)
        if dilateBlends > 0:
            blendMask = binary_dilation(blendMask, iterations=dilateBlends).astype(int)

        # Mask the blends?
        if maskBlends and image.blendOffsets.size > 0:
            totalMask = self._maskBlends(
                centralMask,
                blendMask,
                image.blendOffsets,
                binary,
            )
        else:
            totalMask = centralMask

        return totalMask

    def getProjectionSize(
        self,
        fieldAngle: Union[np.ndarray, tuple, list],
        defocalType: Union[DefocalType, str],
        bandLabel: Union[BandLabel, str] = BandLabel.REF,
        zkCoeff: Optional[np.ndarray] = None,
    ) -> int:
        """Return size of the pupil projected onto the image plane (in pixels).

        The returned number is the number of pixels per side needed to contain
        the image template in a square array.

        Note this function returns a conservative estimate, as it does
        not account for vignetting.

        Parameters
        ----------
        fieldAngle : np.ndarray or tuple or list
            The field angle in degrees.
        defocalType : DefocalType or str
            Whether the image is intra- or extra-focal. Can be specified
            using a DefocalType Enum or the corresponding string.
        bandLabel : BandLabel or str
            Photometric band for the exposure. Can be specified using a
            BandLabel Enum or the corresponding string. If None, BandLabel.REF
            is used. The empty string "" also maps to BandLabel.REF.
            (the default is BandLabel.REF)
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)

        Returns
        -------
        int
            Number of pixels on a side needed to contain the pupil projection.
        """
        # Create a dummy Image
        dummyImage = Image(
            image=np.zeros((0, 0)),
            fieldAngle=fieldAngle,
            defocalType=defocalType,
            bandLabel=bandLabel,
        )

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *dummyImage.fieldAngle,
                dummyImage.bandLabel,
            )

        # Project the pupil onto the image plane
        theta = np.linspace(0, 2 * np.pi, 100)
        uPupil, vPupil = np.cos(theta), np.sin(theta)
        uImageEdge, vImageEdge, *_ = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            dummyImage,
        )

        # What is the max u or v coordinate
        maxCoord = np.max(np.abs(np.concatenate((uImageEdge, vImageEdge))))

        # Convert this to a pixel number
        width = 2 * maxCoord * self.instrument.donutRadius + 2
        nPixels = np.ceil(width).astype(int)

        return nPixels

    def centerOnProjection(
        self,
        image: Image,
        zkCoeff: Optional[np.ndarray] = None,
        binary: bool = True,
        rMax: float = 10,
        **maskKwargs,
    ) -> Image:
        """Center the stamp on a projection of the pupil.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed for
        creating the masks for the projection.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata needed for the mapping.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        binary : bool, optional
            If True, a binary mask is used to estimate the center of the image,
            otherwise a forward model of the image is used. The latter will
            likely result in a more accurate center, but takes longer to
            calculate. (the default is True)
        rMax : float, optional
            The maximum pixel distance the image can be shifted.
            (the default is 10)
        """
        # Make a copy of the stamp
        stamp = image.copy()

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *image.fieldAngle,
                image.bandLabel,
            )

        # Create the image template
        if binary:
            template = self.createImageMask(stamp, zkCoeff, binary=True, **maskKwargs)
        else:
            template = self.mapPupilToImage(stamp, zkCoeff, **maskKwargs).image

        # Center the image
        stamp.image = centerWithTemplate(stamp.image, template, rMax)

        return stamp

    def mapPupilToImage(
        self,
        image: Image,
        zkCoeff: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        **maskKwargs,
    ) -> Image:
        """Map the pupil to the image plane.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed to
        self.createPupilMask() when the image is masked. Note this only
        happens if mask=None.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata needed for the mapping.
            It is assumed that mapping the pupil to the image plane is meant
            to model the image contained in this stamp.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        mask : np.ndarray, optional
            You can provide an image mask if you have already computed one.
            This is just to speed up computation. If not provided, the image
            mask will be computed from scratch.

        Returns
        -------
        Image
            The stamp object mapped to the image plane.
        """
        # Make a copy of the stamp
        stamp = image.copy()

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *image.fieldAngle,
                image.bandLabel,
            )

        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(zkCoeff, stamp)

        # Construct the inverse mapping
        uPupil, vPupil, jac, jacDet = self._constructInverseMap(
            uImage[inside],
            vImage[inside],
            zkCoeff,
            stamp,
        )

        # Create the image mask
        if mask is None:
            if "binary" not in maskKwargs:
                maskKwargs["binary"] = False
            mask = self.createImageMask(
                stamp,
                zkCoeff,
                **maskKwargs,
                _invMap=(uPupil, vPupil, jac, jacDet),
            )

        # Fill the image (this assumes that, except for vignetting,
        # the pupil is uniformly illuminated)
        stamp.image = np.zeros_like(stamp.image)
        stamp.image[inside] = mask[inside] * jacDet

        # Also save the mask
        stamp.mask = mask

        # And set the plane type
        stamp.planeType = PlaneType.Image

        return stamp

    def mapImageToPupil(
        self,
        image: Image,
        zkCoeff: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        **maskKwargs,
    ) -> Image:
        """Map a stamp from the image to the pupil plane.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed to
        self.createPupilMask() when the image is masked. Note this only
        happens if mask=None.

        Parameters
        ----------
        image : Image
            A stamp object containing the array to be mapped from the image
            to the pupil plane, plus the required metadata.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        mask : np.ndarray, optional
            You can provide a pupil mask if you have already computed one.
            This is just to speed up computation. If not provided, the pupil
            mask will be computed from scratch.

        Returns
        -------
        Image
            The stamp object mapped to the image plane.
        """
        # Make a copy of the stamp
        stamp = image.copy()

        # Create regular pupil and image grids
        uPupil, vPupil = self.instrument.createPupilGrid()
        uImage, vImage = self.instrument.createImageGrid(stamp.image.shape[0])

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *image.fieldAngle,
                image.bandLabel,
            )

        # Construct the forward mapping
        uImageMap, vImageMap, jac, jacDet = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            stamp,
        )

        # Interpolate the array onto the pupil plane
        pupil = interpn(
            (vImage[:, 0], uImage[0, :]),
            stamp.image,
            (vImageMap, uImageMap),
            method="linear",
            bounds_error=False,
        )
        pupil *= jacDet

        # Set NaNs to zero
        pupil = np.nan_to_num(pupil)

        # Mask the pupil
        if mask is None:
            if "binary" not in maskKwargs:
                maskKwargs["binary"] = True
            mask = self.createPupilMask(stamp, **maskKwargs)
        pupil *= mask

        # Update the stamp with the new pupil image
        stamp.image = pupil

        # Also save the mask
        stamp.mask = mask

        # And set the plane type
        stamp.planeType = PlaneType.Pupil

        return stamp
