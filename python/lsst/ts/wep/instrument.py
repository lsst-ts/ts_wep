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
from lsst.ts.wep.utils import BandLabel, DefocalType, mergeConfigWithFile


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
    wavelength : float or dict, optional
        The effective wavelength of the instrument in meters. Can be a float, or
        a dictionary that corresponds to different bands. The keys in this
        dictionary are expected to correspond to the strings specified in the
        BandLabel enum in jf_wep.utils.enums.
    batoidModelName : str, optional
        The name used to load the Batoid model, via
        batoid.Optic.fromYaml(batoidModelName). If the string contains "{band}",
        then it is assumed there are different Batoid models for different
        photometric bands, and the names of these bands will be filled in at
        runtime using the strings specified in the BandLabel enum in
        jf_wep.utils.enums.
    batoidRefBand : BandLabel or str, optional
        When loading the Batoid model, use this value in place of BandLabel.REF.
        It should be a BandLabel Enum, or one of the corresponding strings.
        If set to None, this value defaults to BandLabel.REF. Note this only
        matters if batoidModelName contains "{band}".
    maskParams : dict, optional
        Dictionary of mask parameters. Each key in this dictionary corresponds
        to a different mask element. The corresponding values are dictionaries
        that define circles with different centers and radii. The key, value
        pairs are
            - thetaMin: the minimum field angle in degrees for which this mask
            element is relevant
            - center: list of polynomial coefficients (in meters) for np.polyval
            to determine the center of the circle
            - radius: list of polynomial coefficients (in meters) for np.polyval
            to determine the radius of the circle
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
        wavelength: Union[float, dict, None] = None,
        batoidModelName: Optional[str] = None,
        batoidRefBand: Union[BandLabel, str, None] = None,
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
            wavelength=wavelength,
            batoidModelName=batoidModelName,
            batoidRefBand=batoidRefBand,
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
    def wavelength(self) -> Union[float, dict]:
        """Return the effective wavelength(s) in meters."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Union[float, dict]) -> None:
        """Set the effective wavelength(s).

        Parameters
        ----------
        value : float or dict
            The effective wavelength(s) in meters. Can either be a single float,
            or a dictionary mapping BandLabels to floats.

        Raises
        ------
        TypeError
            If the provided value is not a float or dictionary
        """
        # Make sure the value is a float or dictionary
        if not isinstance(value, float) and not isinstance(value, dict):
            raise TypeError("wavelength must be a float or a dictionary.")

        # Convert the wavelength dictionary to use BandLabels and floats
        if isinstance(value, dict):
            value = {BandLabel(key): float(val) for key, val in value.items()}

        # Set the new value
        self._wavelength = value

        # Also clear the caches for the functions that use this value
        self._getIntrinsicZernikesCached.cache_clear()
        self._getOffAxisCoeffCached.cache_clear()

    @property
    def batoidModelName(self) -> Union[str, None]:
        """The Batoid model name."""
        return self._batoidModelName

    @batoidModelName.setter
    def batoidModelName(self, value: Optional[str]) -> None:
        """Set the Batoid model name.

        The name should match one of the yaml files in the batoid/data directory:
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
        ValueError
            If value is not a string or None
        """
        # Make sure the value is a string or None
        if not isinstance(value, str) and value is not None:
            raise ValueError("batoidModelName must be a string, or None.")

        # Set the new value
        self._batoidModelName = value

        # Also clear the caches for the functions that use this value
        self.getBatoidModel.cache_clear()
        self._getIntrinsicZernikesCached.cache_clear()
        self._getOffAxisCoeffCached.cache_clear()

    @property
    def batoidRefBand(self) -> BandLabel:
        """The band to use with Batoid when band == BandLabel.REF"""
        return self._batoidRefBand

    @batoidRefBand.setter
    def batoidRefBand(self, value: Union[BandLabel, str]) -> None:
        """Set the band to use with Batoid when band == BandLabel.REF.

        Parameters
        ----------
        value : BandLabel or str
            When loading the Batoid model, use this value in place of BandLabel.REF.
            It should be a BandLabel Enum, or one of the corresponding strings.
            If set to None, this value defaults to BandLabel.REF. Note this only
            matters if batoidModelName contains "{band}".
        """
        if value is None:
            self._batoidRefBand = BandLabel.REF
        else:
            self._batoidRefBand = BandLabel(value)

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

        # Make sure the band is a BandLabel
        band = BandLabel(band)

        # Replace the reference band
        band = self.batoidRefBand if band == BandLabel.REF else band

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
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax - 3)

        # Get the wavelength
        if isinstance(self.wavelength, dict):
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength

        # Get the intrinsic Zernikes in wavelengths
        zkIntrinsic = batoid.zernike(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        # Keep only Noll indices >= 4
        zkIntrinsic = zkIntrinsic[4:]

        return zkIntrinsic.copy()

    def getIntrinsicZernikes(
        self,
        xAngle: float,
        yAngle: float,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
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

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        return self._getIntrinsicZernikesCached(xAngle, yAngle, band, jmax).copy()

    @lru_cache(100)
    def _getOffAxisCoeffCached(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: Union[BandLabel, str],
        jmax: int,
    ) -> np.ndarray:
        """Cached interior function for the getOffAxisCoeff method.

        We need to do this because numpy arrays are mutable.

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
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains "{band}".
        jmax : int, optional
            The maximum Noll index of the off-axis model Zernikes.

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax - 3)

        # Offset the focal plane
        defocalType = DefocalType(defocalType)
        defocalSign = +1 if defocalType == DefocalType.Extra else -1
        offset = defocalSign * self.defocalOffset
        batoidModel = batoidModel.withGloballyShiftedOptic("Detector", [0, 0, offset])

        # Get the wavelength
        if isinstance(self.wavelength, dict):
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength

        # Get the off-axis model Zernikes in wavelengths
        zkIntrinsic = batoid.zernikeTA(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
            nrad=200,
            naz=20,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        # Keep only Noll indices >= 4
        zkIntrinsic = zkIntrinsic[4:]

        return zkIntrinsic

    def getOffAxisCoeff(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
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
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains "{band}".
            (the default is BandLabel.REF)
        jmax : int, optional
            The maximum Noll index of the off-axis model Zernikes.
            (the default is 66)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        return self._getOffAxisCoeffCached(
            xAngle, yAngle, defocalType, band, jmax
        ).copy()

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
            Dictionary of mask parameters. Each key in this dictionary corresponds
            to a different mask element. The corresponding values are dictionaries
            that define circles with different centers and radii. The key, value
            pairs are
                - thetaMin: the minimum field angle in degrees for which this mask
                element is relevant
                - center: list of polynomial coefficients (in meters) for np.polyval
                to determine the center of the circle
                - radius: list of polynomial coefficients (in meters) for np.polyval
                to determine the radius of the circle
        Raises
        ------
        TypeError
            If value is not a dictionary or None

        """
        if isinstance(value, dict) or value is None:
            self._maskParams = value
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
