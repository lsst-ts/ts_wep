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

__all__ = ["DanishAlgorithm"]

import warnings
from typing import Optional, Tuple, Union

import danish
import numpy as np
from galsim import GalSimFFTSizeError
from lsst.ts.wep import Image, ImageMapper, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.utils import binArray
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares


class DanishAlgorithm(WfAlgorithm):
    """Wavefront estimation algorithm class for Danish.

    The Danish algorithm is based on Janish 2012:
    http://hdl.handle.net/1721.1/78543
    Implemented and improved by Josh Meyers:
    https://github.com/jmeyers314/danish

    Parameters
    ----------
    lstsqKwargs : dict, optional
        A dictionary containing any of the keyword arguments for
        scipy.optimize.least_squares, except `fun`, `x0`, `jac`, or `args`.
    binning : int, optional
        Binning factor to apply to the donut stamps before estimating
        Zernike coefficients. The default value of 1 means no binning.
    """

    def __init__(self, lstsqKwargs: Optional[dict] = None, binning: int = 1) -> None:
        self.binning = binning
        self.lstsqKwargs = lstsqKwargs

    @property
    def requiresPairs(self) -> bool:
        """Whether the algorithm requires pairs to estimate Zernikes."""
        return False

    @property
    def binning(self) -> int:
        """Binning factor to apply to donut stamps."""
        return self._binning

    @binning.setter
    def binning(self, value: int) -> None:
        """Set the binning factor to apply to the donut stamps.

        Parameters
        ----------
        value : int
            The binning factor. A value of 1 means no binning.
        """
        if not isinstance(value, int):
            raise TypeError("binning must be an integer.")
        if value < 1:
            raise ValueError("binning must be greater than or equal to 1.")
        self._binning = value

    @property
    def lstsqKwargs(self) -> dict:
        """Keyword arguments for scipy.optimize.least_squares"""
        return self._lstsqKwargs

    @lstsqKwargs.setter
    def lstsqKwargs(self, value: Union[dict, None]) -> None:
        """Set the keyword arguments for scipy.optimize.least_squares.

        Parameters
        ----------
        value : dict, optional
            A dictionary containing any of the keyword arguments for
            scipy.optimize.least_squares, except `fun`, `x0`, `jac`, or `args`.
            (the default is an empty dictionary)
        """
        # If None, get empty dictionary
        if value is None:
            value = dict()

        # Cast to a dict
        value = dict(value)

        # Make sure these keys are not provided
        notAllowed = ["fun", "x0", "jac", "args"]
        for key in notAllowed:
            if key in value:
                raise KeyError(f"Please do not provide '{key}' in lstsqKwargs.")

        self._lstsqKwargs = value

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary that contains intermediate products
        from the Zernike fitting. The dict contains entries for "intra"
        and/or "extra", plus the final zernike estimate under "zk".

        The "intra" and "extra" entries are dictionaries that contain the
        following entries
            - "image" - the image that is being fit
            - "variance" - the background variance that was used for fitting
            - "zkStart" - the starting Zernike coefficients
            - "lstsqResult" - dictionary of results returned by least_squares
            - "zkFit" - the Zernike coefficients fit to the donut
            - "zkSum" - zkFit + the intrinsic Zernikes
            - "model" - the final forward-modeled donut image
            - "GalSimFFTSizeError" - whether this was hit during least_squares

        Note the units for all Zernikes are in meters, and all Zernikes start
        with Noll index 4.
        """
        return super().history

    def _estimateSingleZk(
        self,
        image: Image,
        zkStart: np.ndarray,
        instrument: Instrument,
        factory: danish.DonutFactory,
        saveHistory: bool,
    ) -> Tuple[np.ndarray, dict]:
        """Estimate Zernikes (in meters) for a single donut stamp.

        Parameters
        ----------
        image : Image
            The ts_wep image of the donut stamp
        zkStart : np.ndarray
            The starting point for the Zernikes
        instrument : Instrument
            The ts_wep Instrument
        factory : danish.DonutFactory
            The Danish donut factory
        saveHistory : bool
            Whether to create a history to be saved

        Returns
        -------
        np.ndarray
            The Zernike coefficients (in meters) for Noll indices >= 4
        dict
            The single-stamp history. This is empty if saveHistory is False.
        """
        # Warn about using Danish for blended donuts
        if np.isfinite(image.blendOffsets).sum() > 0:
            warnings.warn("Danish is currently only setup for non-blended donuts.")

        # Determine the maximum Noll index
        jmax = len(zkStart) + 3

        # Create reference Zernikes by adding off-axis coefficients to zkStart
        zkStart = np.pad(zkStart, (4, 0))
        offAxisCoeff = instrument.getOffAxisCoeff(
            *image.fieldAngle,
            image.defocalType,
            image.bandLabel,
            jmaxIntrinsic=jmax,
            return4Up=False,
        )
        size = max(zkStart.size, offAxisCoeff.size)
        zkRef = np.zeros(size)
        zkRef[: zkStart.size] = zkStart
        zkRef[: offAxisCoeff.size] += offAxisCoeff

        # Create the background mask if it does not exist
        if image.maskBackground is None:
            mapper = ImageMapper(instrument, "offAxis")
            mapper.createImageMasks(image, zkStart)

        # Get robust estimate of background noise
        maskBackground = binary_erosion(image.maskBackground, iterations=10)
        background = image.image[maskBackground]
        q75, q25 = np.percentile(background, [75, 25])
        backgroundStd = (q75 - q25) / 1.349

        # Get the image array
        img = image.image

        if self.binning > 1:
            img = binArray(img, self.binning)
            backgroundStd /= self.binning

        # If size of image is even, cut off final row/column
        if img.shape[0] % 2 == 0:
            img = img[:-1, :-1]

        # Create the Danish donut model
        model = danish.SingleDonutModel(
            factory,
            z_ref=zkRef,
            z_terms=np.arange(4, jmax + 1),
            thx=np.deg2rad(image.fieldAngle[0]),
            thy=np.deg2rad(image.fieldAngle[1]),
            npix=img.shape[0],
        )

        # Create the initial guess for the model parameters
        x0 = [0.0, 0.0, 1.0] + [0.0] * (jmax - 3)

        # Use scipy to optimize the parameters
        try:
            result = least_squares(
                model.chi,
                jac=model.jac,
                x0=x0,
                args=(img, backgroundStd**2),
                **self.lstsqKwargs,
            )
            result = dict(result)

            # Unpack the parameters
            dx, dy, fwhm, *zkFit = result["x"]
            zkFit = np.array(zkFit)

            # Add the starting zernikes back into the result
            zkSum = zkFit + zkStart[4:]

            # Flag that we didn't hit GalSimFFTSizeError
            galSimFFTSizeError = False

            # If we're saving the history, compute the model image
            if saveHistory:
                modelImage = model.model(
                    dx,
                    dy,
                    fwhm,
                    zkFit,
                    sky_level=backgroundStd,
                    flux=img.sum(),
                )

        # Sometimes this happens with Danish :(
        except GalSimFFTSizeError:
            # Fill dummy objects
            result = None
            zkFit = np.full(jmax - 3, np.nan)
            zkSum = np.full(jmax - 3, np.nan)
            if saveHistory:
                modelImage = np.full_like(img, np.nan)

            # Flag the error
            galSimFFTSizeError = True

        if saveHistory:
            # Save the history
            hist = {
                "image": img.copy(),
                "variance": backgroundStd**2,
                "zkStart": zkStart.copy(),
                "lstsqResult": result,
                "zkFit": zkFit.copy(),
                "zkSum": zkSum.copy(),
                "model": modelImage,
                "GalSimFFTSizeError": galSimFFTSizeError,
            }

        else:
            hist = {}

        return zkSum, hist

    def _estimateZk(
        self,
        I1: Image,
        I2: Optional[Image],
        zkStartI1: np.ndarray,
        zkStartI2: Optional[np.ndarray],
        instrument: Instrument,
        saveHistory: bool,
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image or None
            A second image, on the opposite side of focus from I1. Can be None.
        zkStartI1 : np.ndarray
            The starting Zernikes for I1 (in meters, for Noll indices >= 4)
        zkStartI2 : np.ndarray or None
            The starting Zernikes for I2 (in meters, for Noll indices >= 4)
        instrument : Instrument
            The Instrument object associated with the DonutStamps.
        saveHistory : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from
            the images, in meters.
        """
        # Create the Danish donut factory
        factory = danish.DonutFactory(
            R_outer=instrument.radius,
            R_inner=instrument.radius * instrument.obscuration,
            mask_params=instrument.maskParams,
            focal_length=instrument.focalLength,
            pixel_scale=instrument.pixelSize * self.binning,
        )

        # Create an empty history
        hist = {}

        # Estimate for I1
        zk1, hist[I1.defocalType.value] = self._estimateSingleZk(
            I1,
            zkStartI1,
            instrument,
            factory,
            saveHistory,
        )

        if I2 is not None:
            # If I2 provided, estimate for that donut as well
            zk2, hist[I2.defocalType.value] = self._estimateSingleZk(
                I2,
                zkStartI2,
                instrument,
                factory,
                saveHistory,
            )

            # Average the Zernikes
            zk = np.mean([zk1, zk2], axis=0)
        else:
            zk = zk1

        if saveHistory:
            hist["zk"] = zk
            self._history = hist

        return zk
