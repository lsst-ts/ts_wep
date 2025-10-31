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

from __future__ import annotations

__all__ = ["Instrument"]

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import batoid
import numpy as np
from astropy.table import Table
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType, EnumDict
from lsst.ts.wep.utils.ioUtils import mergeConfigWithFile, resolveRelativeConfigPath
from scipy.optimize import minimize_scalar

from .utils.zernikeUtils import createZernikeBasis, createZernikeGradBasis


@lru_cache(10)
def _getBatoidModelFromFileName(filename: str) -> batoid.Optic:
    return batoid.Optic.fromYaml(filename)


class Instrument:
    """Object with relevant geometry of the primary mirror and focal plane.

    The value of every parameter is first pulled from the configFile, and
    then overridden by any parameters explicitly passed as keyword arguments
    to the class constructor.

    Parameters
    ----------
    configFile: Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy:", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy:instruments/LsstCam.yaml)
    name: str, optional
        The name of the instrument.
        (the default is None)
    diameter : float, optional
        The diameter of the primary mirror in meters. If None, but
        batoidModelName is set, this value will be pulled from the Batoid
        model. (the default is None)
    obscuration : float, optional
        The fractional obscuration of the primary mirror. If None, but
        batoidModelName is set, this value will be pulled from the Batoid
        model. (the default is None)
    focalLength : float, optional
        The effective focal length in meters. If None, but batoidModelName
        is set, this value will be pulled from the Batoid model.
        (the default is None)
    defocalOffset : float, optional
        The defocal offset of the images in meters. If None, but
        batoidModelName, batoidOffsetOptic, and batoidOffsetValue are set,
        this value will be calculated using the Batoid model.
        (the default is None)
    pixelSize : float, optional
        The pixel size in meters. (the default is None)
    refBand : BandLabel or str, optional
        When getting the wavelength or loading the Batoid model, use this
        value in place of BandLabel.REF. It should be a BandLabel Enum, or
        one of the corresponding strings. If set to None, this value defaults
        to BandLabel.REF. (the default is None)
    wavelength : float or dict, optional
        The effective wavelength of the instrument in meters. Can be a float,
        or a dictionary that corresponds to different bands. The keys in this
        dictionary are expected to correspond to the strings specified in the
        BandLabel enum in ts_wep.utils.enums. If set to None, this defaults
        to {BandLabel.REF: 500e-9}. (the default is None)
    batoidModelName : str, optional
        Name of Batoid model. If the string contains "{band}", it is assumed
        there are different Batoid models for different photometric bands,
        and the names of these bands will be filled in at runtime using the
        strings specified in the BandLabel enum in jf_wep.utils.enums.
        (the default is None)
    batoidOffsetOptic : str or None, optional
        The optic to offset in the Batoid model in order to calculate
        the equivalent detector offset for the model.
        (the default is None)
    batoidOffsetValue : float or None, optional
        The value in meters to offset the optic in the Batoid model to
        calculate the equivalent detector offset for the model. The
        detector offset is then used for everything else. Note that
        depending on the model, the sign of this value might matter.
        (the default is None)
    heightMap : str or astropy.table.Table or None, optional
        Astropy table that holds the focal plane height map, or the path to
        where it is saved. The table columns should have names "x", "y", and
        "z" with astropy units of length. The x and y coordinates must be
        aligned with the x,y coordinates of the Detector in the Batoid model.
        The +z direction is in the extrafocal direction. If None, no height
        map is used. (the default is None)
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

    Notes
    -----
    The following parameters are required to instantiate the Instrument:
        - diameter
        - obscuration
        - focalLength
        - defocalOffset
        - pixelSize
    With the exception of pixelSize, if not explicitly set, these parameters
    can be pulled from the Batoid model specified by batoidModelName.
    Note that the calculation of defocalOffset also requires that
    batoidOffsetOptic and batoidOffsetValue are set.
    """

    def __init__(
        self,
        configFile: Path | str | None = "policy:instruments/LsstCam.yaml",
        *,
        name: str | None = None,
        diameter: float | None = None,
        obscuration: float | None = None,
        focalLength: float | None = None,
        defocalOffset: float | None = None,
        pixelSize: float | None = None,
        refBand: BandLabel | str | None = None,
        wavelength: float | dict | None = None,
        batoidModelName: str | None = None,
        batoidOffsetOptic: str | None = None,
        batoidOffsetValue: float | None = None,
        heightMap: str | Table | None = None,
        maskParams: dict | None = None,
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
            heightMap=heightMap,
            maskParams=maskParams,
        )

        # Set each parameter
        for key, value in params.items():
            setattr(self, key, value)

        # Check the config
        self.checkConfig()

    def copy(self) -> Instrument:
        """Return a deep copy of the instrument.

        Notes
        -----
        Any cached data from the original instrument will need to be
        repopulated in the copied instrument.
        """
        return Instrument(
            name=self.name,
            diameter=self.diameter,
            obscuration=self.obscuration,
            focalLength=self.focalLength,
            defocalOffset=self.defocalOffset,
            pixelSize=self.pixelSize,
            refBand=self.refBand,
            wavelength=self.wavelength,
            batoidModelName=self.batoidModelName,
            batoidOffsetOptic=self.batoidOffsetOptic,
            batoidOffsetValue=self.batoidOffsetValue,
            maskParams=self.maskParams,
        )

    def checkConfig(self) -> None:
        """Access every attribute to make sure no errors are thrown."""
        for item in dir(self):
            if item[0] != "_":
                getattr(self, item)

    def clearCaches(self) -> None:
        """Clear the Batoid caches."""
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()
        self._focalLengthBatoid = None
        self._defocalOffsetBatoid = None

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
        if self._diameter is not None:
            return self._diameter
        elif self.batoidModelName is not None:
            return self.getBatoidModel().pupilSize
        else:
            raise ValueError(
                "There is currently no diameter set. Please set either the diameter, or the batoidModelName."
            )

    @diameter.setter
    def diameter(self, value: float | None) -> None:
        """Set the mirror diameter.

        Parameters
        ----------
        value : float or None
            The mirror diameter in meters.

        Raises
        ------
        ValueError
            If the value is negative or zero
        """
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("diameter must be positive.")
        self._diameter = value
        self.createZernikeBasis.cache_clear()
        self.createZernikeGradBasis.cache_clear()

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
        if self._obscuration is not None:
            return self._obscuration
        elif self.batoidModelName is not None:
            return self.getBatoidModel().pupilObscuration
        else:
            raise ValueError(
                "There is currently no obscuration set. "
                "Please set either the obscuration, or the batoidModelName."
            )

    @obscuration.setter
    def obscuration(self, value: float | None) -> None:
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
        if value is not None:
            value = float(value)
            if value < 0 or value > 1:
                raise ValueError("obscuration must be between 0 and 1 (inclusive).")
        self._obscuration = value

    @property
    def focalLength(self) -> float:
        """The focal length in meters."""
        if self._focalLength is not None:
            return self._focalLength
        if self._focalLengthBatoid is not None:
            return self._focalLengthBatoid
        elif self.batoidModelName is not None:
            focalLengthBatoid = batoid.analysis.focalLength(
                self.getBatoidModel(),
                0,
                0,
                self.wavelength[self.refBand],
            )
            self._focalLengthBatoid = focalLengthBatoid
            return focalLengthBatoid
        else:
            raise ValueError(
                "There is currently no focalLength set. "
                "Please set either the focalLength, or the batoidModelName."
            )

    @focalLength.setter
    def focalLength(self, value: float | None) -> None:
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
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("focalLength must be positive.")
        self._focalLength = value
        self.createZernikeBasis.cache_clear()
        self.createZernikeGradBasis.cache_clear()

    @property
    def focalRatio(self) -> float:
        """The f-number."""
        return self.focalLength / self.diameter

    @property
    def defocalOffset(self) -> float:
        """The defocal offset in meters."""
        if self._defocalOffset is not None:
            return self._defocalOffset
        elif self._defocalOffsetBatoid is not None:
            return self._defocalOffsetBatoid
        elif self.batoidModelName is not None and self._batoidOffsetValue is not None:
            # Load the model and wavelength info
            batoidModel = self.getBatoidModel()
            offsetOptic = self.batoidOffsetOptic
            eps = batoidModel.pupilObscuration
            wavelength = self.wavelength[BandLabel.REF]
            batoidOffsetValue = self.batoidOffsetValue

            # Calculate dZ4 for the optic
            shift = np.array([0, 0, batoidOffsetValue])
            dZ4optic = batoid.zernike(
                batoidModel.withLocallyShiftedOptic(offsetOptic, +shift),
                *np.zeros(2),
                wavelength,
                eps=eps,
                jmax=4,
                nx=128,
            )[4]

            # Calculate the equivalent detector offset
            result = minimize_scalar(
                lambda offset: np.abs(
                    (self.offsetToZ4Defocus(offset) - dZ4optic) / dZ4optic
                ),
                bounds=(-0.1, 0.1),
            )
            if not result.success or result.fun > 1e-3:
                raise RuntimeError(
                    "Calculating defocalOffset from batoidOffsetValue failed."
                )

            # Save the calculated offset
            defocalOffsetBatoid = np.abs(result.x)
            self._defocalOffsetBatoid = defocalOffsetBatoid
            return defocalOffsetBatoid
        else:
            raise ValueError(
                "There is currently no defocalOffset set. "
                "Please set either the defocalOffset, OR the batoidModelName, "
                "the batoidOffsetOptic, and the batoidOffsetValue."
            )

    @defocalOffset.setter
    def defocalOffset(self, value: float | None) -> None:
        """Set the defocal offset.

        Parameters
        ----------
        value : float
            The defocal offset in meters.
        """
        if value is not None:
            value = np.abs(float(value))
        self._defocalOffset = value

        # Clear relevant caches
        self._getIntrinsicZernikesTACached.cache_clear()
        self.createZernikeBasis.cache_clear()
        self.createZernikeGradBasis.cache_clear()

    def offsetToZ4Defocus(self, offset: float) -> float:
        """Convert the defocus offset to Z4.

        Parameters
        ----------
        offset : float
            The defocus offset in meters.

        Returns
        -------
        float
            The Z4 value in meters.
        """
        batoidModel = self.getBatoidModel()
        return batoid.zernike(
            batoidModel.withLocallyShiftedOptic("Detector", [0, 0, offset]),
            *np.zeros(2),
            self.wavelength[BandLabel.REF],
            eps=batoidModel.pupilObscuration,
            jmax=4,
            nx=128,
        )[4]

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
        try:
            value = float(value)
        except TypeError:
            raise TypeError("pixelSize must be a number.")
        if value <= 0:
            raise ValueError("pixelSize must be positive.")
        self._pixelSize = value
        self.createZernikeBasis.cache_clear()
        self.createZernikeGradBasis.cache_clear()

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
    def refBand(self, value: BandLabel | str | None) -> None:
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

        # Clear relevant caches
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()
        self._focalLengthBatoid = None
        self._defocalOffsetBatoid = None

    @property
    def wavelength(self) -> EnumDict:
        """Return the effective wavelength(s) in meters."""
        if self._wavelength is None:
            return EnumDict(BandLabel, {BandLabel.REF: 500e-9, self.refBand: 500e-9})
        else:
            return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float | dict | None) -> None:
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
            and value is not None
        ):
            raise TypeError("wavelength must be a float, dictionary, or None.")

        # Save wavelength info in a BandLabel EnumDict
        if isinstance(value, dict) or isinstance(value, EnumDict):
            enumDictValue = EnumDict(BandLabel, value)
            try:
                enumDictValue[BandLabel.REF] = enumDictValue[self.refBand]
            except KeyError:
                raise ValueError(
                    "The wavelength dictionary must contain a wavelength for the reference band."
                )
            value = enumDictValue
        elif value is not None:
            value = EnumDict(BandLabel, {BandLabel.REF: value, self.refBand: value})

        # Set the new value
        self._wavelength = value

        # Clear relevant caches
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()
        self._focalLengthBatoid = None
        self._defocalOffsetBatoid = None

    @property
    def batoidModelName(self) -> str | None:
        """The Batoid model name."""
        return self._batoidModelName

    @batoidModelName.setter
    def batoidModelName(self, value: str | None) -> None:
        """Set the Batoid model name.

        The Batoid model name is used to load the Batoid model via
        batoid.Optic.fromYaml(batoidModelName + ".yaml")

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
        oldValue = getattr(self, "_batoidModelName", None)
        self._batoidModelName = value

        # Make sure the Batoid model can be found
        try:
            self.getBatoidModel()
        except FileNotFoundError:
            # Undo the change
            self._batoidModelName = oldValue

            # Raise the error
            raise ValueError(
                f"batoidModelName {value} does not match any of the models "
                f"in Batoid version {batoid.__version__}."
            )

        # Clear relevant caches
        self._getIntrinsicZernikesCached.cache_clear()
        self._getIntrinsicZernikesTACached.cache_clear()
        self._focalLengthBatoid = None
        self._defocalOffsetBatoid = None

    @property
    def batoidOffsetOptic(self) -> str | None:
        """The optic that is offset in the Batoid model."""
        return self._batoidOffsetOptic

    @batoidOffsetOptic.setter
    def batoidOffsetOptic(self, value: str | None) -> None:
        """Set the optic that is offset in the Batoid model.

        This optic is offset in order to calculate the equivalent
        detector offset for the model.

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
        if value is not None:
            if self.batoidModelName is None:
                raise RuntimeError("There is no Batoid model set.")
            elif not isinstance(value, str):
                raise TypeError("batoidOffsetOptic must be a string or None.")
            elif value not in self.getBatoidModel()._names:
                raise ValueError(f"Optic {value} not found in the Batoid model.")

        self._batoidOffsetOptic = value

        # Clear relevant caches
        self._getIntrinsicZernikesTACached.cache_clear()
        self._defocalOffsetBatoid = None

    @property
    def batoidOffsetValue(self) -> float | None:
        """Amount in meters the optic is offset in the Batoid model."""
        return self._batoidOffsetValue

    @batoidOffsetValue.setter
    def batoidOffsetValue(self, value: float | None) -> None:
        """Set amount in meters the optic is offset in the batoid model.

        This is the amount that batoidOffsetOptic is offset in the Batoid
        model to calculate the equivalent detector offset for the model.
        Note depending on the model, the sign of this value might matter.

        Parameters
        ----------
        value : float or None
            The offset value

        Raises
        ------
        RuntimeError
            If no Batoid model is set
        """
        if value is not None:
            if self.batoidModelName is None:
                raise RuntimeError("There is no Batoid model set.")
            value = float(value)
        self._batoidOffsetValue = value

        # Clear relevant caches
        self._getIntrinsicZernikesTACached.cache_clear()
        self._defocalOffsetBatoid = None

    @property
    def heightMap(self) -> batoid.surface.Bicubic | None:
        """Focal plane height map."""
        return self._heightMap

    @heightMap.setter
    def heightMap(self, value: str | Table | None) -> None:
        """Set the focal plane height map.

        Parameters
        ----------
        value : str or None
            Astropy table that holds the focal plane height map, or the path to
            where it is saved. The table columns should have names "x", "y",
            and "z" with astropy units of length. The x and y coordinates must
            be aligned with the x,y coordinates of the Detector in the Batoid
            model. The +z direction is in the extrafocal direction. If None,
            no height map is used.
        """
        if value is None:
            self._heightMap = None
            return

        if isinstance(value, str) or isinstance(value, Path):
            # Check the file exists
            path = Path(resolveRelativeConfigPath(value))
            if not path.exists():
                raise FileNotFoundError(f"Height map file {value} does not exist.")

            # Load the height map table
            value = Table.read(path)

        # Extract grid values from table
        x = np.unique(value["x"].to("m").value)  # type: ignore
        y = np.unique(value["y"].to("m").value)  # type: ignore
        z = value["z"].to("m").value.reshape(x.size, y.size)  # type: ignore

        # Create the surface
        self._heightMap = batoid.Bicubic(x, y, z)

        # Clear relevant caches
        self._getIntrinsicZernikesTACached.cache_clear()

    def getBatoidModel(
        self,
        band: BandLabel | str = BandLabel.REF,
        includeHeights: bool = True,
    ) -> batoid.Optic:
        """Return the Batoid model for the instrument and the requested band.

        Parameters
        ----------
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which Batoid
            model to load. Only relevant if self.batoidModelName contains
            "{band}". (the default is BandLabel.REF)
        includeHeights : bool, optional
            If True, and a height map is provided, perturb the focal plane
            surface using the height map. If False, do not apply the height
            map. (the default is True)
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
        model = _getBatoidModelFromFileName(batoidModelName + ".yaml")

        # If a height map is provided, perturb the focal plane
        if includeHeights and getattr(self, "_heightMap", None) is not None:
            model = model.withPerturbedSurface("Detector", self.heightMap)

        return model

    @lru_cache(100)
    def _getIntrinsicZernikesCached(
        self,
        xAngle: float,
        yAngle: float,
        band: BandLabel | str,
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
            The Zernike coefficients in meters, starting with index 0
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band, includeHeights=False)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax + 1)

        # Get the wavelength
        if len(self.wavelength) > 1:
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength[BandLabel.REF]

        # Get the intrinsic Zernikes in wavelengths
        zkIntrinsic = batoid.zernikeGQ(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
            rings=12,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        return zkIntrinsic

    def getIntrinsicZernikes(
        self,
        xAngle: float,
        yAngle: float,
        band: BandLabel | str = BandLabel.REF,
        nollIndices: Sequence[int] = tuple(np.arange(4, 79)),
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
        nollIndices : np.ndarray, optional
            Noll indices for which to return Zernikes.
            (the default is indices 4-78)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters
        """
        # Make sure this is an array
        nollIndices = np.array(nollIndices)

        # Retrieve cached Zernikes
        zk = self._getIntrinsicZernikesCached(xAngle, yAngle, band, max(nollIndices))

        return zk[nollIndices]

    @lru_cache(100)
    def _getIntrinsicZernikesTACached(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: BandLabel | str,
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
            The Zernike coefficients in meters, starting with index 0

        Notes
        -----
        In the ZernikeTA calculation below, we use nrad=10 and choose naz so
        the pupil is approximately uniformly sampled. Not all the Zernike
        coefficients have converged with nrad=10, but we chose this number so
        the image positions have converged. In particular, for nrad=10, the
        residuals with Batoid are less than 0.5 microns.
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
        offset = [0, 0, defocalSign * self.defocalOffset]
        batoidModel = batoidModel.withLocallyShiftedOptic("Detector", offset)

        # Get the wavelength
        if len(self.wavelength) > 1:
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength[BandLabel.REF]

        # Get the off-axis model Zernikes in wavelengths
        zkIntrinsic = batoid.zernikeTA(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
            focal_length=self.focalLength,
            nrad=10,
            naz=int(2 * np.pi * 10),
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        return zkIntrinsic

    def getOffAxisCoeff(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: BandLabel | str = BandLabel.REF,
        nollIndicesModel: Sequence = tuple(np.arange(4, 79)),
        nollIndicesIntr: Sequence = tuple(np.arange(4, 79)),
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
        nollIndicesModel : np.ndarray, optional
            Noll indices of Zernikes retrieved for the off-axis model.
            (the default is indices 4-78)
        nollIndicesIntr : np.ndarray, optional
            Noll indices of Zernikes you are estimating in the TIE or
            Danish. The off-axis coefficients are calculated by retrieving
            coefficients from batoid.zernikeTA, and then subtracting off the
            intrinsic Zernikes for Noll indices you are estimating. This is
            allows you to determine whether intrinsic Zernikes are included
            in wavefront estimates when using WfEstimator.
            (the default is indices 4-22).

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Make sure these are arrays
        nollIndicesModel = np.array(nollIndicesModel)
        nollIndicesIntr = np.array(nollIndicesIntr)

        # Get zernikeTA
        zkTA = self._getIntrinsicZernikesTACached(
            xAngle,
            yAngle,
            defocalType,
            band,
            max(nollIndicesModel),
        )

        # Get regular intrinsic zernikes
        zk = self._getIntrinsicZernikesCached(
            xAngle,
            yAngle,
            band,
            max(nollIndicesIntr),
        )

        # Subtract intrinsics from zernikeTA
        offAxisCoeff = np.zeros(max(zkTA.size, zk.size), dtype=float)
        offAxisCoeff[nollIndicesModel] = zkTA[nollIndicesModel]
        offAxisCoeff[nollIndicesIntr] -= zk[nollIndicesIntr]

        return offAxisCoeff[nollIndicesModel]

    @property
    def maskParams(self) -> dict:
        """The mask parameter dictionary."""
        # Get the parameters if they exist
        params = getattr(self, "_maskParams", None)

        # If they don't exist, use the primary inner and outer radii
        if params is None:
            params = {
                "Pupil": {
                    "outer": {
                        "clear": True,
                        "thetaMin": 0,
                        "thetaMax": np.inf,
                        "center": [0],
                        "radius": [self.radius],
                    },
                    "inner": {
                        "clear": False,
                        "thetaMin": 0,
                        "thetaMax": np.inf,
                        "center": [0],
                        "radius": [self.obscuration * self.radius],
                    },
                }
            }

        return params

    @maskParams.setter
    def maskParams(self, value: dict | None) -> None:
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

    @property
    def nPupilPixels(self) -> int:
        """The number of pupil pixels (on a side).

        This number is set so that the resolution of the pupil roughly
        matches the resolution of the image.
        """
        return np.ceil(self.donutDiameter).astype(int)

    def createPupilGrid(self) -> tuple[np.ndarray, np.ndarray]:
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
        # Create a 1D array with the correct number of pixels
        grid = np.linspace(-1.01, 1.01, self.nPupilPixels)

        # Create u and v grids
        uPupil, vPupil = np.meshgrid(grid, grid)

        return uPupil, vPupil

    @lru_cache(100)
    def createZernikeBasis(self, jmax: int) -> np.ndarray:
        uPupil, vPupil = self.createPupilGrid()
        return createZernikeBasis(
            uPupil, vPupil, jmax=jmax, obscuration=self.obscuration
        )

    @lru_cache(100)
    def createZernikeGradBasis(self, jmax: int) -> np.ndarray:
        uPupil, vPupil = self.createPupilGrid()
        return createZernikeGradBasis(
            uPupil, vPupil, jmax=jmax, obscuration=self.obscuration
        )

    def createImageGrid(self, nPixels: int) -> tuple[np.ndarray, np.ndarray]:
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
