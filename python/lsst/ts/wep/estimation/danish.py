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
    jointFitPair : bool, optional
        Whether to jointly fit intra/extra pairs, when a pair is provided.
        If False, Zernikes are estimated for each individually, then
        averaged. (the default is True)
    """

    def __init__(
        self,
        lstsqKwargs: Optional[dict] = None,
        binning: int = 1,
        jointFitPair: bool = True,
    ) -> None:
        self.binning = binning
        self.lstsqKwargs = lstsqKwargs
        self.jointFitPair = jointFitPair

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
    def jointFitPair(self) -> bool:
        """Whether to jointly fit intra/extra pairs."""
        return self._jointFitPair

    @jointFitPair.setter
    def jointFitPair(self, value: bool) -> None:
        """Whether to jointly fit intra/extra pairs, when a pair is provided.

        Parameters
        ----------
        value : bool
            Whether to jointly fit intra/extra pairs, when a pair is provided.
            If False, Zernikes are estimated for each individually, then
            averaged.
        """
        if not isinstance(value, bool):
            raise TypeError("jointFitPair must be a bool.")
        self._jointFitPair = value

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
            - "nollIndices" - Noll indices for which Zernikes are estimated
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

    def _prepDanish(
        self,
        image: Image,
        zkStart: np.ndarray,
        nollIndices: np.ndarray,
        instrument: Instrument,
    ):
        # Warn about using Danish for blended donuts
        if np.isfinite(image.blendOffsets).sum() > 0:
            warnings.warn("Danish is currently only setup for non-blended donuts.")

        # Create reference Zernikes by adding off-axis coefficients to zkStart
        offAxisCoeff = instrument.getOffAxisCoeff(
            *image.fieldAngle,
            image.defocalType,
            image.bandLabel,
            nollIndicesModel=np.arange(0, 79),
            nollIndicesIntr=nollIndices,
        )
        zkRef = offAxisCoeff.copy()
        zkRef[nollIndices] += zkStart

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

        # Convert field angle from degrees to radians
        angle = np.deg2rad(image.fieldAngle)

        return img, angle, zkRef, backgroundStd

    def _estimateSingleZk(
        self,
        image: Image,
        zkStart: np.ndarray,
        nollIndices: np.ndarray,
        instrument: Instrument,
        factory: danish.DonutFactory,
        saveHistory: bool,
    ) -> Tuple[np.ndarray, dict, dict]:
        """Estimate Zernikes (in meters) for a single donut stamp.

        Parameters
        ----------
        image : Image
            The ts_wep image of the donut stamp
        zkStart : np.ndarray
            The starting point for the Zernikes
        nollIndices : np.ndarray
            Noll indices for which to estimate Zernikes
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
        dict
            Output from the danish algorithm to pass on as metadata.
        """
        img, angle, zkRef, backgroundStd = self._prepDanish(
            image=image,
            zkStart=zkStart,
            nollIndices=nollIndices,
            instrument=instrument,
        )

        # Create the Danish donut model
        model = danish.SingleDonutModel(
            factory,
            z_ref=zkRef,
            z_terms=nollIndices,
            thx=angle[0],
            thy=angle[1],
            npix=img.shape[0],
        )

        # Create the initial guess for the model parameters
        x0 = [0.0, 0.0, 1.0] + [0.0] * len(nollIndices)

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
            zkSum = zkFit + zkStart

            # Flag that we didn't hit GalSimFFTSizeError
            galSimFFTSizeError = False

            # If we're saving the history, compute the model image
            if saveHistory:
                modelImage = model.model(
                    dx,
                    dy,
                    fwhm,
                    zkFit,
                    sky_level=backgroundStd**2,
                    flux=img.sum(),
                )

        # Sometimes this happens with Danish :(
        except GalSimFFTSizeError:
            # Fill dummy objects
            result = None
            zkFit = np.full_like(zkStart, np.nan)
            zkSum = np.full_like(zkStart, np.nan)
            if saveHistory:
                modelImage = np.full_like(img, np.nan)

            # Flag the error
            galSimFFTSizeError = True

        if saveHistory:
            # Save the history
            hist = {
                "image": img.copy(),
                "variance": backgroundStd**2,
                "nollIndices": nollIndices.copy(),
                "zkStart": zkStart.copy(),
                "lstsqResult": result,
                "zkFit": zkFit.copy(),
                "zkSum": zkSum.copy(),
                "model": modelImage,
                "GalSimFFTSizeError": galSimFFTSizeError,
            }

        else:
            hist = {}

        # Save final fwhm value in metadata
        zkMeta = {"fwhm": fwhm}

        return zkSum, hist, zkMeta

    def _estimatePairZk(
        self,
        I1: Image,
        I2: Optional[Image],
        zkStartI1: np.ndarray,
        zkStartI2: Optional[np.ndarray],
        nollIndices: np.ndarray,
        instrument: Instrument,
        factory: danish.DonutFactory,
        saveHistory: bool,
    ) -> Tuple[np.ndarray, dict, dict]:
        """Estimate Zernikes (in meters) for pairs of donut stamps.

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
        nollIndices : np.ndarray
            Noll indices for which you wish to estimate Zernike coefficients.
        instrument : Instrument
            The Instrument object associated with the DonutStamps.
        factory : danish.DonutFactory
            The Danish donut factory
        saveHistory : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        np.ndarray
            The Zernike coefficients (in meters) for Noll indices >= 4
        dict
            The single-stamp history. This is empty if saveHistory is False.
        dict
            Output from the danish algorithm to pass on as metadata.
        """
        # Prep quantities for both images
        img1, angle1, zkRef1, backgroundStd1 = self._prepDanish(
            image=I1,
            zkStart=zkStartI1,
            nollIndices=nollIndices,
            instrument=instrument,
        )
        img2, angle2, zkRef2, backgroundStd2 = self._prepDanish(
            image=I2,
            zkStart=zkStartI2,
            nollIndices=nollIndices,
            instrument=instrument,
        )

        # Package these into lists for Danish
        imgs = [img1, img2]
        thxs = [angle1[0], angle2[0]]
        thys = [angle1[1], angle2[1]]
        zkRefs = [zkRef1, zkRef2]
        skyLevels = [backgroundStd1**2, backgroundStd2**2]

        # Create Double Zernike tuples
        dzTerms = [(1, j) for j in nollIndices]

        # Set field radius to max value from mask params
        fieldRadius = np.deg2rad(
            np.max(
                [
                    edge["thetaMax"]
                    for item in instrument.maskParams.values()
                    for edge in item.values()
                ]
            )
        )

        # Create model
        model = danish.MultiDonutModel(
            factory,
            z_refs=zkRefs,
            dz_terms=dzTerms,
            field_radius=fieldRadius,
            thxs=thxs,
            thys=thys,
            npix=imgs[0].shape[0],
        )

        # Initial guess
        x0 = [0.0] * 2 + [0.0] * 2 + [0.7] + [0.0] * len(dzTerms)

        # Use scipy to optimize the parameters
        try:
            result = least_squares(
                model.chi,
                jac=model.jac,
                x0=x0,
                args=(imgs, skyLevels),
                **self.lstsqKwargs,
            )
            result = dict(result)

            # Unpack the parameters
            dxs, dys, fwhm, zkFit = model.unpack_params(result["x"])

            # Add the starting zernikes back into the result
            zkSum = zkFit + np.nanmean([zkStartI1, zkStartI2], axis=0)

            # Flag that we didn't hit GalSimFFTSizeError
            galSimFFTSizeError = False

            # If we're saving the history, compute the model image
            if saveHistory:
                modelImages = model.model(
                    dxs,
                    dys,
                    fwhm,
                    zkFit,
                    sky_levels=skyLevels,
                    fluxes=np.sum(imgs, axis=(1, 2)),
                )

        # Sometimes this happens with Danish :(
        except GalSimFFTSizeError:
            # Fill dummy objects
            result = None
            zkFit = np.full_like(zkStartI1, np.nan)
            zkSum = np.full_like(zkStartI1, np.nan)
            if saveHistory:
                modelImages = np.full_like(imgs, np.nan)

            # Flag the error
            galSimFFTSizeError = True

        if saveHistory:
            # Save the history
            hist = {}
            hist[I1.defocalType.value] = {
                "image": imgs[0].copy(),
                "variance": skyLevels[0],
                "nollIndices": nollIndices.copy(),
                "zkStart": zkStartI1.copy(),
                "lstsqResult": result,
                "zkFit": zkFit.copy(),
                "zkSum": zkSum.copy(),
                "model": modelImages[0],
                "GalSimFFTSizeError": galSimFFTSizeError,
            }
            hist[I2.defocalType.value] = {
                "image": imgs[1].copy(),
                "variance": skyLevels[1],
                "nollIndices": nollIndices.copy(),
                "zkStart": zkStartI2.copy(),
                "lstsqResult": result,
                "zkFit": zkFit.copy(),
                "zkSum": zkSum.copy(),
                "model": modelImages[1],
                "GalSimFFTSizeError": galSimFFTSizeError,
            }

        else:
            hist = {}

        # Save final fwhm value in metadata
        zkMeta = {"fwhm": fwhm}

        return zkSum, hist, zkMeta

    def _estimateZk(
        self,
        I1: Image,
        I2: Optional[Image],
        zkStartI1: np.ndarray,
        zkStartI2: Optional[np.ndarray],
        nollIndices: np.ndarray,
        instrument: Instrument,
        saveHistory: bool,
    ) -> Tuple[np.ndarray, dict]:
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
        nollIndices : np.ndarray
            Noll indices for which you wish to estimate Zernike coefficients.
        instrument : Instrument
            The Instrument object associated with the DonutStamps.
        saveHistory : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        np.ndarray
            Zernike coefficients for the provided Noll indices, estimated from
            the images, in meters.
        dict
            Output from the danish algorithm to pass on as metadata.
        """
        # Create the Danish donut factory
        factory = danish.DonutFactory(
            R_outer=instrument.radius,
            R_inner=instrument.radius * instrument.obscuration,
            mask_params=instrument.maskParams,
            focal_length=instrument.focalLength,
            pixel_scale=instrument.pixelSize * self.binning,
        )

        if I2 is None or not self.jointFitPair:
            # Create an empty history
            hist = {}

            # Estimate for I1
            zk1, hist[I1.defocalType.value], zk1Meta = self._estimateSingleZk(
                I1,
                zkStartI1,
                nollIndices,
                instrument,
                factory,
                saveHistory,
            )

            if I2 is not None:
                # If I2 provided, estimate for that donut as well
                zk2, hist[I2.defocalType.value], zk2Meta = self._estimateSingleZk(
                    I2,
                    zkStartI2,
                    nollIndices,
                    instrument,
                    factory,
                    saveHistory,
                )

                # Average the Zernikes
                zk = np.mean([zk1, zk2], axis=0)
                zkMeta = {"fwhm": np.mean([zk1Meta["fwhm"], zk2Meta["fwhm"]])}
            else:
                zk = zk1
                zkMeta = zk1Meta

            hist["zk"] = zk

        else:
            zk, hist, zkMeta = self._estimatePairZk(
                I1,
                I2,
                zkStartI1,
                zkStartI2,
                nollIndices,
                instrument,
                factory,
                saveHistory,
            )

        if saveHistory:
            self._history = hist

        return zk, zkMeta
