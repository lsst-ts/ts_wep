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

__all__ = ["Instrument"]

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import batoid
import numpy as np
from lsst.ts.wep.utils import BandLabel, DefocalType, EnumDict, mergeConfigWithFile


class Instrument:
    """Object with relevant geometry of the primary mirror and focal plane.

    Parameters
    ----------
    configFile: Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/instruments/LsstCam.yaml)
    name: str, optional
        The name of the instrument.
    diameter : float, optional
        The diameter of the primary mirror in meters.
    obscuration : float, optional
        The fractional obscuration of the primary mirror.
    focalLength : float, optional
        The effective focal length in meters.
    defocalOffset : float, optional
        The defocal offset of the images in meters.
    pixelSize : float, optional
        The pixel size in meters.
    refBand : BandLabel or str, optional
        When getting the wavelength or loading the Batoid model, use this
        value in place of BandLabel.REF. It should be a BandLabel Enum, or
        one of the corresponding strings. If set to None, this value defaults
        to BandLabel.REF.
    wavelength : float or dict, optional
        The effective wavelength of the instrument in meters. Can be a float,
        or a dictionary that corresponds to different bands. The keys in this
        dictionary are expected to correspond to the strings specified in the
        BandLabel enum in jf_wep.utils.enums.
    batoidModelName : str, optional
        Name of Batoid model (used via batoid.Optic.fromYaml(batoidModelName))
        If the string contains "{band}", it is assumed there are different
        Batoid models for different photometric bands, and the names of these
        bands will be filled in at runtime using the strings specified in the
        BandLabel enum in jf_wep.utils.enums.
    batoidOffsetOptic : str, optional
        The optic to offset in the Batoid model in order to simulate
        defocal images. When this is None, it defaults to "Detector".
    batoidOffsetValue : float, optional
        The value, in meters, to offset the optic in the Batoid model to
        simulate defocal images. When this is None, it defaults to the
        defocalOffset.
    maskParams : dict, optional
        Dictionary of mask parameters. Each key in this dictionary corresponds
        to a different mask element. The corresponding values are dictionaries
        that define circles with different centers and radii. The key, value
        pairs are
            - thetaMin: the minimum field angle in degrees for which this mask
            element is relevant
            - center: list of polynomial coeffs (in meters) for np.polyval()
            to determine the center of the circle
            - radius: list of polynomial coeffs (in meters) for np.polyval()
            to determine the radius of the circle
        None defaults to an empty dictionary.
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/instruments/LsstCam.yaml",
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        obscuration: Optional[float] = None,
        focalLength: Optional[float] = None,
        defocalOffset: Optional[float] = None,
        pixelSize: Optional[float] = None,
        refBand: Union[BandLabel, str, None] = None,
        wavelength: Union[float, dict, None] = None,
        batoidModelName: Optional[str] = None,
        batoidOffsetOptic: Optional[str] = None,
        batoidOffsetValue: Optional[float] = None,
        maskParams: Optional[dict] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeConfigWithFile(
            configFile,
            name=name,
            diameter=diameter,
            obscuration=obscuration,
            focalLength=focalLength,
            defocalOffset=defocalOffset,
            pixelSize=pixelSize,
            refBand=refBand,
            wavelength=wavelength,
            batoidModelName=batoidModelName,
            batoidOffsetOptic=batoidOffsetOptic,
            batoidOffsetValue=batoidOffsetValue,
            maskParams=maskParams,
        )

        # Set each parameter
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        """The name of the instrument."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the instrument.

        Parameters
        ----------
        value : str
            The name of the instrument.
        """
        self._name = str(value)

    @property
    def diameter(self) -> float:
        """The primary mirror diameter in meters."""
        return self._diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        """Set the mirror diameter.

        Parameters
        ----------
        value : float
            The mirror diameter in meters.

        Raises
        ------
        ValueError
            If the diameter is not positive
        """
        value = float(value)
        if value <= 0:
            raise ValueError("diameter must be positive.")
        self._diameter = value

    @property
    def radius(self) -> float:
        """The primary mirror radius in meters."""
        return self.diameter / 2

    @property
    def area(self) -> float:
        """The primary mirror area in square meters."""
        return np.pi * self.radius**2 * (1 - self.obscuration**2)

    @property
    def obscuration(self) -> float:
        """The fractional obscuration."""
        return self._obscuration

    @obscuration.setter
    def obscuration(self, value: float) -> None:
        """Set the fractional obscuration.

        Parameters
        ----------
        value : float
            The fractional obscuration

        Raises
        ------
        ValueError
            If the fractional obscuration is not between 0 and 1 (inclusive)
        """
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError("The obscuration must be between 0 and 1 (inclusive).")
        self._obscuration = value

    @property
    def focalLength(self) -> float:
        """The focal length in meters."""
        return self._focalLength

    @focalLength.setter
    def focalLength(self, value: float) -> None:
        """Set the focal length.

        Parameters
        ----------
        value : float
            The focal length in meters

        Raises
        ------
        ValueError
            If the focal length is not positive
        """
        value = float(value)
        if value <= 0:
            raise ValueError("focalLength must be positive.")
        self._focalLength = value

    @property
    def focalRatio(self) -> float:
        """The f-number."""
        return self.focalLength / self.diameter

    @property
    def defocalOffset(self) -> float:
        """The defocal offset in meters."""
        return self._defocalOffset  # type: ignore

    @defocalOffset.setter
    def defocalOffset(self, value: float) -> None:
        """Set the defocal offset.

        Parameters
        ----------
        value : float
            The defocal offset in meters.
        """
        value = np.abs(float(value))
        self._defocalOffset = value

    @property
    def pupilOffset(self) -> float:
        """The pupil offset in meters."""
        return self.focalLength**2 / self.defocalOffset

    @property
    def pixelSize(self) -> float:
        """The pixel size in meters."""
        return self._pixelSize

    @pixelSize.setter
    def pixelSize(self, value: float) -> None:
        """Set the pixel size.

        Parameters
        ----------
        value : float
            The pixel size in meters.

        Raises
        ------
        ValueError
            If the pixel size is not positive
        """
        value = float(value)
        if value <= 0:
            raise ValueError("pixelSize must be positive.")
        self._pixelSize = value

    @property
    def pixelScale(self) -> float:
        """The pixel scale in arcseconds per pixel."""
        return np.rad2deg(self.pixelSize / self.focalLength) * 3600

    @property
    def donutRadius(self) -> float:
        """The expected donut radius in pixels."""
        rMeters = self.defocalOffset / np.sqrt(4 * self.focalRatio**2 - 1)
        rPixels = rMeters / self.pixelSize
        return rPixels

    @property
    def donutDiameter(self) -> float:
        """The expected donut diameter in pixels."""
        return 2 * self.donutRadius

    @property
    def refBand(self) -> BandLabel:
        """Band to use with Batoid and wavelength when band == BandLabel.REF"""
        return self._refBand

    @refBand.setter
    def refBand(self, value: Union[BandLabel, str]) -> None:
        """Set reference band for loading Batoid model and getting wavelength.

        Parameters
        ----------
        value : BandLabel or str
            The reference band. Should be a BandLabel Enum, or one of the
            corresponding strings. If set to None, this value defaults to
            BandLabel.REF.
        """
        if value is None:
            self._refBand = BandLabel.REF
        else:
            self._refBand = BandLabel(value)

    @property
    def wavelength(self) -> EnumDict:
        """Return the effective wavelength(s) in meters."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Union[float, dict]) -> None:
        """Set the effective wavelength(s).

        Parameters
        ----------
        value : float or dict
            The effective wavelength(s) in meters. Can either be a single
            float, or a dictionary mapping BandLabels to floats.

        Raises
        ------
        TypeError
            If the provided value is not a float or dictionary
        ValueError
            If the provided value is a dictionary, and the dictionary does not
            contain a wavelength for the reference band,
        """
        # Make sure the value is a float or dictionary
        if (
            not isinstance(value, float)
            and not isinstance(value, dict)
            and not isinstance(value, EnumDict)
        ):
            raise TypeError("wavelength must be a float or a dictionary.")

        # Save wavelength info in a BandLabel EnumDict
        if isinstance(value, dict) or isinstance(value, EnumDict):
            value = EnumDict(BandLabel, value)
            try:
                value[BandLabel.REF] = value[self.refBand]
            except KeyError:
                raise ValueError(
                    "The wavelength dictionary must contain a wavelength "
                    "for the reference band."
                )
        else:
            value = EnumDict(BandLabel, {BandLabel.REF: value, self.refBand: value})

        # Set the new value
        self._wavelength = value

        # Also clear the caches for the functions that use this value
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()

    @property
    def batoidModelName(self) -> Union[str, None]:
        """The Batoid model name."""
        return self._batoidModelName

    @batoidModelName.setter
    def batoidModelName(self, value: Optional[str]) -> None:
        """Set the Batoid model name.

        The Batoid model name is used to load the Batoid model via
        batoid.Optic.fromYaml(batoidModelName)

        The name must match one of the yaml files in the batoid/data directory:
        https://github.com/jmeyers314/batoid/tree/main/batoid/data
        You can use "{band}" in the name, and this will be replaced with a band
        name when loading the batoid model.

        E.g. Setting the name to "LSST_{band}" allows one to load the Batoid
        models corresponding to "LSST_u.yaml", "LSST_g.yaml", etc. using the
        getBatoidModel() method below.

        Parameters
        ----------
        value : str or None
            The name of the Batoid model.

        Raises
        ------
        TypeError
            If value is not a string or None
        """
        # Make sure the value is a string or None
        if not isinstance(value, str) and value is not None:
            raise TypeError("batoidModelName must be a string, or None.")

        # Set the new value
        self._batoidModelName = value

        # Also clear the caches for the functions that use this value
        self.getBatoidModel.cache_clear()
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()

    @property
    def batoidOffsetOptic(self) -> Union[str, None]:
        """The optic that is offset in the Batoid model."""
        if self.batoidModelName is None:
            return None
        elif self._batoidOffsetOptic is None:
            return "Detector"
        else:
            return self._batoidOffsetOptic

    @batoidOffsetOptic.setter
    def batoidOffsetOptic(self, value: Union[str, None]) -> None:
        """Set the optic that is offset in the Batoid model.

        This is the optic to offset in the Batoid model in order to simulate
        defocal images. When this is None, it defaults to "Detector".

        Parameters
        ----------
        value : str or None
            The name of the optic to be offset in the Batoid model.

        Raises
        ------
        RuntimeError
            If no Batoid model is set
        TypeError
            If value is not a string or None
        ValueError
            If the optic is not found in the Batoid model
        """
        if value is not None and self.batoidModelName is None:
            raise RuntimeError("There is no Batoid model set.")
        elif value is not None and not isinstance(value, str):
            raise TypeError("batoidOffsetOptic must be a string or None.")
        elif value is not None and value not in self.getBatoidModel()._names:
            raise ValueError(f"Optic {value} not found in the Batoid model.")

        self._batoidOffsetOptic = value

    @property
    def batoidOffsetValue(self) -> Union[float, None]:
        """Amount in meters the optic is offset in the Batoid model."""
        if self.batoidModelName is None:
            return None
        elif self._batoidOffsetValue is None:
            return self.defocalOffset
        else:
            return self._batoidOffsetValue

    @batoidOffsetValue.setter
    def batoidOffsetValue(self, value: Union[str, None]) -> None:
        """Set amount in meters the optic is offset in the batoid model.

        Parameters
        ----------
        value : float or None
            The offset value. If None, this value is defaulted to
            self.defocalOffset.

        Raises
        ------
        RuntimeError
            If no Batoid model is set
        """
        if value is not None and self.batoidModelName is None:
            raise RuntimeError("There is no Batoid model set.")
        elif value is None:
            self._batoidOffsetValue = value
        else:
            self._batoidOffsetValue = np.abs(float(value))

    @lru_cache(10)
    def getBatoidModel(
        self, band: Union[BandLabel, str] = BandLabel.REF
    ) -> batoid.Optic:
        """Return the Batoid model for the instrument and the requested band.

        Parameters
        ----------
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which Batoid
            model to load. Only relevant if self.batoidModelName contains
            "{band}". (the default is BandLabel.REF)
        """
        # If batoidModelName is None, we can't load a model, so return None
        if self.batoidModelName is None:
            return None

        # Get the band enum
        band = BandLabel(band)

        # Replace the reference band
        band = self.refBand if band == BandLabel.REF else band

        # Fill any occurrence of "{band}" with the band string
        batoidModelName = self.batoidModelName.format(band=band.value)

        # Load the Batoid model
        return batoid.Optic.fromYaml(batoidModelName)

    @lru_cache(100)
    def _getIntrinsicZernikesCached(
        self,
        xAngle: float,
        yAngle: float,
        band: Union[BandLabel, str],
        jmax: int,
    ) -> np.ndarray:
        """Cached interior function for the getIntrinsicZernikes method.

        We need to do this because numpy arrays are mutable.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains
            "{band}".
        jmax : int, optional
            The maximum Noll index of the intrinsic Zernikes.

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax + 1)

        # Get the intrinsic Zernikes in wavelengths
        zkIntrinsic = batoid.zernike(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            self.wavelength[band],
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= self.wavelength[band]

        return zkIntrinsic

    def getIntrinsicZernikes(
        self,
        xAngle: float,
        yAngle: float,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
        return4Up: bool = True,
    ) -> np.ndarray:
        """Return the intrinsic Zernikes associated with the optical design.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains
            "{band}". (the default is BandLabel.REF)
        jmax : int, optional
            The maximum Noll index of the intrinsic Zernikes.
            (the default is 66)
        return4Up : bool, optional
            Whether to only return the coefficients for Noll indices >= 4.
            (the default is True)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters
        """
        zk = self._getIntrinsicZernikesCached(xAngle, yAngle, band, jmax).copy()

        if return4Up:
            # Keep only Noll indices >= 4
            zk = zk[4:]

        return zk

    @lru_cache(100)
    def _getIntrinsicZernikesTACached(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: Union[BandLabel, str],
        jmax: int,
    ) -> np.ndarray:
        """Cached function for batoid.zernikeTA.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        defocalType : DefocalType or str
            The DefocalType Enum or corresponding string, specifying which side
            of focus to model.
        band : BandLabel or str
            The BandLabel Enum or corresponding string, specifying which
            batoid model to load. Only relevant if self.batoidModelName
            contains "{band}".
        jmax : int
            The maximum Noll index of the off-axis model Zernikes.

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax + 1)

        # Offset the focal plane
        defocalType = DefocalType(defocalType)
        defocalSign = +1 if defocalType == DefocalType.Extra else -1
        optic = self.batoidOffsetOptic
        offset = defocalSign * self.batoidOffsetValue
        batoidModel = batoidModel.withGloballyShiftedOptic(optic, [0, 0, offset])

        # Get the off-axis model Zernikes in wavelengths
        zkIntrinsic = batoid.zernikeTA(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            self.wavelength[band],
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
            nrad=200,
            naz=20,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= self.wavelength[band]

        return zkIntrinsic

    def getOffAxisCoeff(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
        jmaxIntrinsic: int = 66,
        return4Up: bool = True,
    ) -> np.ndarray:
        """Return the Zernike coefficients associated with the off-axis model.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        defocalType : DefocalType or str
            The DefocalType Enum or corresponding string, specifying which side
            of focus to model.
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which
            batoid model to load. Only relevant if self.batoidModelName
            contains "{band}". (the default is BandLabel.REF)
        jmax : int, optional
            The maximum Noll index of the off-axis model Zernikes.
            (the default is 66)
        jmaxIntrinsic : int, optional
            The off-axis coefficients are calculated by subtracting the
            intrinsic Zernikes from batoid.zernikeTA. This value sets the
            maximum Noll index of the intrinsic Zernikes that are subtracted
            from batoid.zernikeTA. It is usually the jmax of the Zernikes
            being estimated by the wavefront estimators.
        return4Up : bool, optional
            Whether to only return the coefficients for Noll indices >= 4.
            (the default is True)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Get zernikeTA
        zkTA = self._getIntrinsicZernikesTACached(
            xAngle,
            yAngle,
            defocalType,
            band,
            jmax,
        )

        # Get regular intrinsic zernikes
        zk = self._getIntrinsicZernikesCached(
            xAngle,
            yAngle,
            band,
            min(jmax, jmaxIntrinsic),
        )

        # Subtract the intrinsics from zernikeTA
        offAxisCoeff = zkTA.copy()
        offAxisCoeff[: zk.size] -= zk

        if return4Up:
            # Keep only Noll indices >= 4
            offAxisCoeff = offAxisCoeff[4:]

        return offAxisCoeff

    @property
    def maskParams(self) -> dict:
        """The mask parameter dictionary."""
        # Get the parameters if they exist
        params = getattr(self, "_maskParams", None)

        # If they don't exist, use the primary inner and outer radii
        if params is None:
            params = {
                "pupilOuter": {
                    "thetaMin": 0,
                    "center": [0],
                    "radius": [self.radius],
                },
                "pupilInner": {
                    "thetaMin": 0,
                    "center": [0],
                    "radius": [self.obscuration],
                },
            }

        return params

    @maskParams.setter
    def maskParams(self, value: Optional[dict]) -> None:
        """Set the mask parameters.

        Parameters
        ----------
        value : dict or None
            Dictionary of mask parameters. Each key in this dictionary
            corresponds to a different mask element. The corresponding
            values are dictionaries that define circles with different
            centers and radii. The key, value pairs are
                - thetaMin: the minimum field angle in degrees for which
                this mask element is relevant
                - center: list of polynomial coefficients (in meters) for
                np.polyval() to determine the center of the circle
                - radius: list of polynomial coefficients (in meters) for
                np.polyval() to determine the radius of the circle
            None defaults to an empty dictionary.

        Raises
        ------
        TypeError
            If value is not a dictionary or None

        """
        if isinstance(value, dict):
            self._maskParams = value
        elif value is None:
            self._maskParams = dict()
        else:
            raise TypeError("maskParams must be a dictionary or None.")

    def createPupilGrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a grid for the pupil.

        The coordinates of the grid are in normalized pupil coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the outer
        edge of the pupil, and u^2 + v^2 = obscuration^2 is the inner edge.

        The number of pixels is chosen to match the resolution of the image.

        Returns
        -------
        np.ndarray
            The 2D u-grid on the pupil plane
        np.ndarray
            The 2D v-grid on the pupil plane
        """
        # Set the resolution equal to the resolution of the image
        nPixels = np.ceil(self.donutDiameter).astype(int)

        # Create a 1D array with the correct number of pixels
        grid = np.linspace(-1.01, 1.01, nPixels)

        # Create u and v grids
        uPupil, vPupil = np.meshgrid(grid, grid)

        return uPupil, vPupil

    def createImageGrid(self, nPixels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create an (nPixel x nPixel) grid for the image.

        The coordinates of the grid are in normalized image coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the outer
        edge of the unaberrated donut, and u^2 + v^2 = obscuration^2 is the
        inner edge.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            The 2D u-grid on the image plane
        np.ndarray
            The 2D v-grid on the image plane
        """
        # Create a 1D array with the correct number of pixels
        grid = np.arange(nPixels, dtype=float)

        # Center the grid
        grid -= grid.mean()

        # Convert to pupil normalized coordinates
        grid /= self.donutRadius

        # Create u and v grids
        uImage, vImage = np.meshgrid(grid, grid)

        return uImage, vImage
