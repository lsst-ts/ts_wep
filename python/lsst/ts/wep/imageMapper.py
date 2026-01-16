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

from typing import Any

import galsim
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, shift

from lsst.ts.wep.image import Image
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType, PlaneType
from lsst.ts.wep.utils.ioUtils import configClass
from lsst.ts.wep.utils.miscUtils import centerWithTemplate, polygonContains
from lsst.ts.wep.utils.zernikeUtils import makeDense, zernikeGradEval


class ImageMapper:
    """Class for mapping the pupil to the image plane, and vice versa.

    This class also creates image masks.

    Details mapping between the pupil and image planes are derived
    and discussed in https://sitcomtn-111.lsst.io

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
        planes. Can be "offAxis", "onAxis", or "paraxial". offAxis is a
        numerical model that is valid for all optical systems, but requires
        an accurate Batoid model. onAxis is an analytic model that is valid
        for all optical systems near the optical axis. paraxial is an
        analytic model that is valid for slow optical systems near the
        optical axis. offAxis is recommended when you have a Batoid model
        and onAxis is recommended when you do not. paraxial is primarily
        meant for testing (the default is "offAxis")
    """

    def __init__(
        self,
        instConfig: str | dict | Instrument = "policy:instruments/LsstCam.yaml",
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
            The optical model to use for mapping between the image and pupil
            planes. Can be "offAxis", "onAxis", or "paraxial". offAxis is a
            numerical model that is valid for all optical systems, but requires
            an accurate Batoid model. onAxis is an analytic model that is valid
            for all optical systems near the optical axis. paraxial is an
            analytic model that is valid for slow optical systems near the
            optical axis. offAxis is recommended when you have a Batoid model
            and onAxis is recommended when you do not. paraxial is primarily
            meant for testing (the default is "offAxis")

        Raises
        ------
        TypeError
            If the value is not a string
        ValueError
            If the value is not one of the allowed values
        """
        allowedModels = ["offAxis", "onAxis", "paraxial"]
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
        nollIndices: np.ndarray | None,
        image: Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the forward mapping from the pupil to the image plane.

        Parameters
        ----------
        uPupil : np.ndarray
             Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
             Normalized y coordinates on the pupil plane
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
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
        # Resolve the Noll indices
        if nollIndices is None:
            nollIndices = np.arange(4, len(zkCoeff) + 4)

        # Turn zkCoeff into dense array, starting at Noll index 4
        zkCoeff = makeDense(zkCoeff, nollIndices)

        # Get the Zernikes for the mapping
        if self.opticalModel == "onAxis" or self.opticalModel == "paraxial":
            zkMap = zkCoeff
        elif self.opticalModel == "offAxis":
            # Get the off-axis coefficients
            offAxisCoeff = self.instrument.getOffAxisCoeff(
                *image.fieldAngle,
                image.defocalType,
                image.bandLabel,
                nollIndicesIntr=nollIndices,
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
        if self.opticalModel == "paraxial":
            # The paraxial model is analytic and intended for slow optical
            # systems near the optical axis. This model is never recommended,
            # but is here for testing purposes

            # See Equations 13 and 18 of https://sitcomtn-111.lsst.io

            # Determine defocal sign from the image plane at z = f +/- l
            # I.e., the extrafocal image at z = f + l is associated with +1,
            # and the intrafocal image at z = f - l is associated with -1.
            ds = +1 if image.defocalType == DefocalType.Extra else -1

            # Calculate the factor C
            C = -4 * N**2 / l

            # Map pupil points onto the image plane
            uImage = -ds * uPupil + C * (d1Wdu - d1Wdu0)
            vImage = -ds * vPupil + C * (d1Wdv - d1Wdv0)

            # Calculate the elements of the Jacobian
            J00 = -ds + C * d2Wdudu
            J01 = C * d2Wdudv
            J10 = C * d2Wdvdu
            J11 = -ds + C * d2Wdvdv

        elif self.opticalModel == "onAxis":
            # The onAxis model is analytic and intended for fast optical
            # systems near the optical axis

            # See Equations 12 and 16 of https://sitcomtn-111.lsst.io

            # Determine defocal sign from the image plane at z = f +/- l
            # I.e., the extrafocal image at z = f + l is associated with +1,
            # and the intrafocal image at z = f - l is associated with -1.
            ds = +1 if image.defocalType == DefocalType.Extra else -1

            # Calculate the pieces we will use below
            rPupil = np.sqrt(uPupil**2 + vPupil**2)
            A = np.sqrt(4 * N**2 - 1)
            B = np.sqrt(4 * N**2 - rPupil**2)
            C = -2 * N / l

            # Map pupil points onto the image plane
            uImage = -ds * A / B * uPupil + A * C * (d1Wdu - d1Wdu0)
            vImage = -ds * A / B * vPupil + A * C * (d1Wdv - d1Wdv0)

            # Calculate the elements of the Jacobian
            J00 = A / B * (-ds + uPupil**2 / B**2) + A * C * d2Wdudu
            J01 = A / B * (uPupil * vPupil / B**2) + A * C * d2Wdudv
            J10 = A / B * (vPupil * uPupil / B**2) + A * C * d2Wdvdu
            J11 = A / B * (-ds + vPupil**2 / B**2) + A * C * d2Wdvdv

        else:
            # The offAxis model uses a numerically-fit model from Batoid
            # This model is able to account for wide field distortion effects
            # in fast optical systems, however it is generally applicable to
            # all optical systems for which you have a good Batoid model

            # See Equation 21 of https://sitcomtn-111.lsst.io

            # Calculate the pieces we will use below
            A = np.sqrt(4 * N**2 - 1)
            C = -2 * N / l

            # Map the pupil points onto the image plane
            uImage = A * C * (d1Wdu - d1Wdu0)
            vImage = A * C * (d1Wdv - d1Wdv0)

            # Calculate the elements of the Jacobian
            J00 = A * C * d2Wdudu
            J01 = A * C * d2Wdvdu
            J10 = A * C * d2Wdudv
            J11 = A * C * d2Wdvdv

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
        nollIndices: np.ndarray | None,
        image: Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
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
            nollIndices,
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
            nollIndices,
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
                nollIndices,
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
        nollIndices: np.ndarray | None,
        image: Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return image grid and mask for which pixels are inside the pupil.

        Note the pupil considered is the pupil mapped to the image plane.

        Parameters
        ----------
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
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
            nollIndices,
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
        fwdMap: tuple | None = None,
    ) -> np.ndarray:
        """Return a fractional mask for a single circle.

        This is based on the masking model in Danish:
        https://github.com/jmeyers314/danish
        which is an improvement on the masking model from Janish (2012):
        http://hdl.handle.net/1721.1/78543
        These improvements are discussed in Section 2.1 of
        https://sitcomtn-111.lsst.io

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
        m = -uPupilCen / vPupilCen
        b = np.sqrt(uPupilCen**2 + vPupilCen**2) * rPupilCirc / vPupilCen

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
        fwdMap: tuple | None = None,
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
        rTheta = np.hypot(*angle)

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
        for item in maskParams:
            if item == "Spider_3D":
                continue
            for edge in maskParams[item]:
                # Get the params for this object
                params = maskParams[item][edge]

                # Get the indices of non-zero pixels
                idx = np.nonzero(mask)[0]

                # If all the pixels are zero, stop here
                if not idx.any():
                    break

                # Only apply if we're in the theta range for this element
                if rTheta < params["thetaMin"] or rTheta > params["thetaMax"]:
                    continue

                # Calculate the radius and center of the mask in meters
                radius = np.polyval(params["radius"], rTheta)
                rCenter = np.polyval(params["center"], rTheta)

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
                    fwdMap=(
                        None if fwdMap is None else (uImage[idx], vImage[idx], jac[..., idx], jacDet[idx])
                    ),
                )

                # Assign the mask values
                if params["clear"]:
                    mask[idx] = np.minimum(mask[idx], maskVals)
                else:
                    mask[idx] = np.minimum(mask[idx], 1 - maskVals)

        return mask

    def _createBlendMask(
        self,
        image: Image,
        mask: np.ndarray[float],
        dilateBlends: int,
        isPupilMask: bool,
    ) -> np.ndarray:
        """Create a blend mask with the specified dilation.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        mask : np.ndarray[float]
            The un-dilated source mask that is used to construct blend masks.
        dilateBlends : int
            How many times to dilate the blended masks.
        isPupilMask : bool
            Whether this function is called during creation of pupil masks.
            If False, it is assumed this function is called during creation
            of image masks.

        Returns
        -------
        np.ndarray[float]
            Blend mask array
        """
        # Get the blend offsets. Note for extrafocal images, the orientation
        # is flipped with respect to the pupil
        blendOffsets = image.blendOffsets.copy()
        if isPupilMask and (image.defocalType == DefocalType.Extra):
            blendOffsets *= -1

        # Make the central blend mask and pad it so dilation doesn't
        # bleed off the edge of the stamp
        blendMask0 = mask.copy()
        if dilateBlends > 0:
            blendMask0 = np.pad(blendMask0, dilateBlends)
            blendMask0 = binary_dilation(blendMask0, iterations=dilateBlends)
            blendMask0 = blendMask0.astype(int)

        # Shift the central blend mask to the position of the blends
        blendMask = np.zeros_like(mask)
        for offset in blendOffsets:
            blendMask += shift(blendMask0, offset[::-1])[
                dilateBlends : len(blendMask0) - dilateBlends,
                dilateBlends : len(blendMask0) - dilateBlends,
            ]

        # Clip the values
        blendMask = np.clip(blendMask, 0, 1)

        return blendMask

    def _autoDilateBlendMask(
        self,
        image: Image,
        mask: np.ndarray[float],
        sourceMask: np.ndarray[float],
        autoDilateMaxIter: int,
        autoDilateFracChange: float,
        isPupilMask: bool,
    ) -> np.ndarray:
        """Create blend mask while automatically determining the dilation.

        This works by looking at the median of the top 5% of pixels ranked
        by brightness and iteratively dilating the mask until this median
        value stops changing.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        mask : np.ndarray[float]
            The un-dilated source mask that is used to construct blend masks.
        sourceMask : np.ndarray[float]
            The dilated source mask.
        autoDilateMaxIter : int, optional
            Maximum number of iterations in the dilateBlends auto-algorithm.
            (the default is 8)
        autoDilateFracChange : float, optional
            In the dilateBlends auto-algorithm, this is the max fractional
            amount the median of the top 5% of brightest pixels can change
            between dilation iterations of the blend mask before auto dilation
            stops. (the default is 0.005.)
        isPupilMask : bool
            Whether this function is called during creation of pupil masks.
            If False, it is assumed this function is called during creation
            of image masks.

        Returns
        -------
        np.ndarray[float]
            Blend mask array
        """
        # Create a dummy values to start
        blendMaskOld = np.zeros_like(mask)
        top5medianOld = 0

        # Increase amount of blend dilation until criterion is met
        # or max number of iterations reached
        for iterNum in range(autoDilateMaxIter + 1):
            # Create blend mask with zero dilation
            blendMask = self._createBlendMask(
                image=image,
                mask=mask,
                dilateBlends=iterNum,
                isPupilMask=isPupilMask,
            )

            # Mask the blends
            sourceMaskTemp = np.clip(sourceMask - blendMask, 0, 1)

            # Get median value of top 5% of unmasked pixels
            unmaskedPix = sourceMaskTemp * image.image
            unmaskedPixPos = unmaskedPix[unmaskedPix > 0]
            top5median = np.percentile(unmaskedPixPos, 97.5)

            # Check how much the median changed
            medianChange = np.abs(top5median - top5medianOld)
            if medianChange < autoDilateFracChange * top5median:
                break

            # Update old parameters
            blendMaskOld = blendMask.copy()
            top5medianOld = top5median

        return blendMaskOld

    def _createMaskTriplet(
        self,
        image: Image,
        mask: np.ndarray[float],
        isBinary: bool,
        dilate: int,
        dilateBlends: int | str,
        autoDilateMaxIter: int,
        autoDilateFracChange: float,
        doMaskBlends: bool,
        isPupilMask: bool,
    ) -> None:
        """Create the triple of source, blend, and background masks.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the masks.
        mask : np.ndarray[float]
            Fiducial source mask.
        isBinary : bool
            Whether to return a binary mask. If False, a fractional mask is
            returned instead.
        dilate : int
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            isBinary is False.
        dilateBlends : int or str
            How many times to dilate the blended masks. Can also be set to
            "auto" in which case an algorithm automatically attempts to infer
            the correct amount of dilation.
        autoDilateMaxIter : int
            Maximum number of iterations in the dilateBlends auto-algorithm.
        autoDilateFracChange : float
            In the dilateBlends auto-algorithm, this is the max fractional
            amount the median of the top 5% of brightest pixels can change
            between dilation iterations of the blend mask before auto dilation
            stops.
        doMaskBlends : bool
            Whether to subtract the blend mask from the source mask.
        isPupilMask : bool
            Whether this function is called during creation of pupil masks.
            If False, it is assumed this function is called during creation
            of image masks.
        """
        # Set the mask to binary?
        if isBinary:
            mask = (mask > 0.5).astype(int)

        # Dilate the mask?
        sourceMask = mask.copy()
        if dilate > 0:
            sourceMask = binary_dilation(sourceMask, iterations=dilate)
            sourceMask = sourceMask.astype(int)

        # Create the blend mask
        if len(image.blendOffsets) == 0:
            blendMask = np.zeros_like(mask)
        elif dilateBlends == "auto":
            # Use algorithm to automatically dilate the blend mask
            blendMask = self._autoDilateBlendMask(
                image=image,
                mask=mask,
                sourceMask=sourceMask,
                autoDilateMaxIter=autoDilateMaxIter,
                autoDilateFracChange=autoDilateFracChange,
                isPupilMask=isPupilMask,
            )
        else:
            # Create the blend mask
            blendMask = self._createBlendMask(
                image=image,
                mask=mask,
                dilateBlends=dilateBlends,  # type: ignore
                isPupilMask=isPupilMask,
            )

        # If masking the blend in the source mask then we subtract
        # the overlapping area out of the source mask.
        if doMaskBlends:
            sourceMask -= blendMask
            sourceMask = np.clip(sourceMask, 0, 1)

        # Create the background mask
        backgroundMask = np.ones_like(sourceMask)
        backgroundMask -= sourceMask
        backgroundMask -= blendMask
        backgroundMask = np.clip(backgroundMask, 0, 1)

        # Set the masks
        image.mask = sourceMask
        image.maskBlends = blendMask
        image.maskBackground = backgroundMask

    def _maskArgCheck(
        self,
        image: Image,
        isBinary: bool,
        dilate: int,
        dilateBlends: int | str,
        ignorePlane: bool,
        isPupilMask: bool,
    ) -> None:
        """Check arguments for mask creation.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        isBinary : bool
            Whether to return a binary mask. If False, a fractional mask is
            returned instead.
        dilate : int
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            isBinary is False.
        dilateBlends : int or str
            How many times to dilate the blended masks. Can also be set to
            "auto" in which case an algorithm automatically attempts to infer
            the correct amount of dilation.
        ignorePlane : bool
            If False, check whether image.planeType is PlaneType.Pupil or not.
        isPupilMask : bool
            Whether this function is called during creation of pupil masks.
            If False, it is assumed this function is called during creation
            of image masks.

        Raises
        ------
        ValueError
            The image is not on the pupil plane or the dilate values are
            invalid.
        ValueError
            Value of dilate must be a non-negative integer.
        ValueError
            If dilate is greater than zero then isBinary must be True.
        ValueError
            Value of dilateBlends must be non-negative integer or 'auto'.
        """
        # Check the image plane
        requiredPlane = PlaneType.Pupil if isPupilMask else PlaneType.Image
        if (not ignorePlane) and (image.planeType != requiredPlane):
            raise ValueError(
                f"The image is not on the {'pupil' if isPupilMask else 'image'} plane. "
                "If you want to ignore this check, set ignorePlane=True."
            )

        # Check the dilate values
        dilate = int(dilate)
        if dilate < 0:
            raise ValueError("dilate must be a non-negative integer.")
        elif dilate > 0 and not isBinary:
            raise ValueError("If dilate is greater than zero, isBinary must be True.")

        if dilateBlends != "auto":
            dilateBlends = int(dilateBlends)
            if dilateBlends < 0:
                raise ValueError("dilateBlends must be a non-negative integer, or 'auto'.")

    def createPupilMasks(
        self,
        image: Image,
        *,
        isBinary: bool = True,
        dilate: int = 0,
        dilateBlends: int | str = 0,
        autoDilateMaxIter: int = 8,
        autoDilateFracChange: float = 5e-3,
        doMaskBlends: bool = False,
        ignorePlane: bool = False,
    ) -> None:
        """Create source mask, blend mask, and background mask on pupil plane.

        Note the masks are stored in image.mask, image.maskBlends, and
        image.maskBackground. The mask is 1 for source pixels and 0 for
        other pixels. The blend mask has 1 for blend pixels and 0 for
        other pixels. The background mas has 1 for background pixels
        and 0 for other pixels.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        isBinary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is
            returned instead. (the default is True)
        dilate : int, optional
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            isBinary is False. (the default is 0)
        dilateBlends : int or str, optional
            How many times to dilate the blended masks. Can also be set to
            "auto" in which case an algorithm automatically attempts to infer
            the correct amount of dilation. (the default is 0)
        autoDilateMaxIter : int, optional
            Maximum number of iterations in the dilateBlends auto-algorithm.
            (the default is 8)
        autoDilateFracChange : float, optional
            In the dilateBlends auto-algorithm, this is the max fractional
            amount the median of the top 5% of brightest pixels can change
            between dilation iterations of the blend mask before auto dilation
            stops. (the default is 0.005.)
        doMaskBlends : bool, optional
            Whether to subtract the blend mask from the source mask.
            (the default is False)
        ignorePlane : bool, optional
            If False, check that image.planeType == PlaneType.Pupil.
            (the default is False)

        Raises
        ------
        ValueError
            The image is not on the pupil plane or the dilate values are
            invalid.
        """
        # Check arguments
        self._maskArgCheck(
            image=image,
            isBinary=isBinary,
            dilate=dilate,
            dilateBlends=dilateBlends,
            ignorePlane=ignorePlane,
            isPupilMask=True,
        )

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

        # Create the mask triplet
        self._createMaskTriplet(
            image=image,
            mask=mask,
            isBinary=isBinary,
            dilate=dilate,
            dilateBlends=dilateBlends,
            autoDilateMaxIter=autoDilateMaxIter,
            autoDilateFracChange=autoDilateFracChange,
            doMaskBlends=doMaskBlends,
            isPupilMask=True,
        )

    def createImageMasks(
        self,
        image: Image,
        zkCoeff: np.ndarray | None = None,
        nollIndices: np.ndarray | None = None,
        *,
        isBinary: bool = True,
        dilate: int = 0,
        dilateBlends: int = 0,
        autoDilateMaxIter: int = 8,
        autoDilateFracChange: float = 5e-3,
        doMaskBlends: bool = False,
        ignorePlane: bool = False,
        _invMap: tuple | None = None,
    ) -> None:
        """Create source mask, blend mask, and background mask on image plane.

        Note the masks are stored in image.mask, image.maskBlends, and
        image.maskBackground. The mask is 1 for source pixels and 0 for
        other pixels. The blend mask has 1 for blend pixels and 0 for
        other pixels. The background mas has 1 for background pixels
        and 0 for other pixels.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata required for constructing
            the mask.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
        isBinary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is
            returned instead. (the default is True)
        dilate : int, optional
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. Note this is not an option if
            isBinary is False. (the default is 0)
        dilateBlends : int or str, optional
            How many times to dilate the blended masks. Can also be set to
            "auto" in which case an algorithm automatically attempts to infer
            the correct amount of dilation. (the default is 0)
        autoDilateMaxIter : int, optional
            Maximum number of iterations in the dilateBlends auto-algorithm.
            (the default is 8)
        autoDilateFracChange : float, optional
            In the dilateBlends auto-algorithm, this is the max fractional
            amount the median of the top 5% of brightest pixels can change
            between dilation iterations of the blend mask before auto dilation
            stops. (the default is 0.005.)
        doMaskBlends : bool, optional
            Whether to subtract the blend mask from the source mask.
            (the default is False)
        ignorePlane : bool, optional
            If False, check that image.planeType == PlaneType.Pupil.
            (the default is False)

        Raises
        ------
        ValueError
            The image is not on the image plane or the dilate values are
            invalid.
        """
        self._maskArgCheck(
            image=image,
            isBinary=isBinary,
            dilate=dilate,
            dilateBlends=dilateBlends,
            ignorePlane=ignorePlane,
            isPupilMask=False,
        )

        if zkCoeff is None:
            # Get the intrinsic Zernikes
            zkCoeff = self.instrument.getIntrinsicZernikes(
                *image.fieldAngle,
                band=image.bandLabel,
            )

        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(zkCoeff, nollIndices, image)

        # Get the inverse mapping from image plane to pupil plane
        if _invMap is None:
            # Construct the inverse mapping
            uPupil, vPupil, invJac, invJacDet = self._constructInverseMap(
                uImage[inside],
                vImage[inside],
                zkCoeff,
                nollIndices,
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

        # Create the mask triplet
        self._createMaskTriplet(
            image=image,
            mask=mask,
            isBinary=isBinary,
            dilate=dilate,
            dilateBlends=dilateBlends,
            autoDilateMaxIter=autoDilateMaxIter,
            autoDilateFracChange=autoDilateFracChange,
            doMaskBlends=doMaskBlends,
            isPupilMask=False,
        )

    def getProjectionSize(
        self,
        fieldAngle: np.ndarray | tuple | list,
        defocalType: DefocalType | str,
        bandLabel: BandLabel | str = BandLabel.REF,
        zkCoeff: np.ndarray | None = None,
        nollIndices: np.ndarray | None = None,
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
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)

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
                band=dummyImage.bandLabel,
            )

        # Project the pupil onto the image plane
        theta = np.linspace(0, 2 * np.pi, 100)
        uPupil, vPupil = np.cos(theta), np.sin(theta)
        uImageEdge, vImageEdge, *_ = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            nollIndices,
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
        zkCoeff: np.ndarray | None = None,
        nollIndices: np.ndarray | None = None,
        isBinary: bool = True,
        rMax: float = 10,
        **maskKwargs: Any,
    ) -> Image:
        """Center the stamp on a projection of the pupil.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed for
        creating the masks for the projection.

        Note this function also sets the masks for the image.

        Parameters
        ----------
        image : Image
            A stamp object containing the metadata needed for the mapping.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
        isBinary : bool, optional
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
                band=image.bandLabel,
            )

        # Create the image template
        if isBinary:
            self.createImageMasks(
                stamp,
                zkCoeff,
                nollIndices,
                isBinary=True,
                **maskKwargs,
            )
            template = stamp.mask.copy()
        else:
            template = self.mapPupilToImage(stamp, zkCoeff, nollIndices, **maskKwargs).image

        # Center the image
        stamp.image = centerWithTemplate(stamp.image, template, rMax)

        return stamp

    def mapPupilToImage(
        self,
        image: Image,
        zkCoeff: np.ndarray | None = None,
        nollIndices: np.ndarray | None = None,
        masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        **maskKwargs: Any,
    ) -> Image:
        """Map the pupil to the image plane.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed to
        self.createPupilMasks() when the image is masked. Note this only
        happens if masks=None.

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
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
        masks : np.ndarray, optional
            You can provide the image masks if they have already been computed.
            This is just to speed up computation. If not provided, the masks
            are created after the mapping.

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
                band=image.bandLabel,
            )

        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(
            zkCoeff,
            nollIndices,
            stamp,
        )

        # Construct the inverse mapping
        uPupil, vPupil, jac, jacDet = self._constructInverseMap(
            uImage[inside],
            vImage[inside],
            zkCoeff,
            nollIndices,
            stamp,
        )

        # Set the plane type
        stamp.planeType = PlaneType.Image

        # Create the image mask
        if masks is None or all(item is None for item in masks):
            self.createImageMasks(
                stamp,
                zkCoeff,
                nollIndices,
                **maskKwargs,
                _invMap=(uPupil, vPupil, jac, jacDet),
            )
        else:
            stamp.mask = masks[0]
            stamp.maskBlends = masks[1]
            stamp.maskBackground = masks[2]

        # Fill the image (this assumes that, except for vignetting,
        # the pupil is uniformly illuminated)
        stamp.image = np.zeros_like(stamp.image)
        stamp.image[inside] = stamp.mask[inside] * jacDet

        return stamp

    def mapImageToPupil(
        self,
        image: Image,
        zkCoeff: np.ndarray | None = None,
        nollIndices: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        **maskKwargs: Any,
    ) -> Image:
        """Map a stamp from the image to the pupil plane.

        In addition to the parameters listed below, you can provide any
        keyword argument for mask creation, and these will be passed to
        self.createPupilMasks() when the image is masked. Note this only
        happens if masks=None.

        Parameters
        ----------
        image : Image
            A stamp object containing the array to be mapped from the image
            to the pupil plane, plus the required metadata.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default are the intrinsic Zernikes at the donut position)
        nollIndices : np.ndarray, optional
            These are the Noll indices corresponding to the coefficients in
            zkCoeff. If None, it is assumed zkCoeff contains coefficients
            for consecutive indices starting with Noll index 4.
            (the default is None)
        masks : np.ndarray, optional
            You can provide the image masks if they have already been computed.
            This is just to speed up computation. If not provided, the masks
            are created after the mapping.

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
                band=image.bandLabel,
            )

        # Construct the forward mapping
        uImageMap, vImageMap, jac, jacDet = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            nollIndices,
            stamp,
        )

        # Interpolate the array onto the pupil plane
        stamp.image = interpn(
            (vImage[:, 0], uImage[0, :]),
            stamp.image,
            (vImageMap, uImageMap),
            method="linear",
            bounds_error=False,
        )
        stamp.image *= jacDet

        # Set NaNs to zero
        stamp.image = np.nan_to_num(stamp.image)

        # Set the plane type
        stamp.planeType = PlaneType.Pupil

        # Mask the pupil
        if masks is None or all(item is None for item in masks):
            self.createPupilMasks(stamp, **maskKwargs)
        else:
            stamp.mask = masks[0]
            stamp.maskBlends = masks[1]
            stamp.maskBackground = masks[2]

        stamp.image *= stamp.mask

        return stamp
