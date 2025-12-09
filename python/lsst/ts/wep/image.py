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

__all__ = ["Image"]

from copy import deepcopy
from typing import Optional, Union

import numpy as np
from typing_extensions import Self

from lsst.ts.wep.utils import BandLabel, DefocalType, PlaneType


class Image:
    """Class to hold a donut image along with metadata.

    All quantities in this object are assumed to be in the global
    camera coordinate system (CCS). See https://sitcomtn-003.lsst.io
    for details.

    I specify the "global" coordinate system because corner wavefront
    sensor images from the butler are rotated by some multiple of 90
    degrees with respect to the global coordinate system. These images
    should be de-rotated before being stored in this object.

    Parameters
    ----------
    image : np.ndarray
        The square numpy array containing the donut image.
    fieldAngle : np.ndarray or tuple or list
        The field angle of the donut, in degrees. The field angle
        is the angle to the source, measured from the optical axis.
        See the note on coordinate systems above.
    defocalType : DefocalType or str
        Whether the image is intra- or extra-focal.
        Can be specified using a DefocalType Enum or the corresponding string.
    bandLabel : BandLabel or str, optional
        Photometric band for the exposure. Can be specified using a
        BandLabel Enum or the corresponding string. If None, BandLabel.REF
        is used. The empty string "" also maps to BandLabel.REF.
        (the default is BandLabel.REF)
    planeType : PlaneType or str, optional
        Whether the image is on the image plane or the pupil plane.
        Can be specified using a PlaneType Enum, or the corresponding string.
        (the default is PlaneType.Image)
    blendOffsets : np.ndarray or tuple or list, optional
        Positions of blended donuts relative to central donut, in pixels.
        Must be provided in the format [[dx1, dy1], [dx2, dy2], ...].
        Note these shifts must be in the global CCS (see the note on
        coordinate systems above).
        (the default is an empty array, i.e. no blends)
    """

    def __init__(
        self,
        image: np.ndarray,
        fieldAngle: Union[np.ndarray, tuple, list],
        defocalType: Union[DefocalType, str],
        bandLabel: Union[BandLabel, str] = BandLabel.REF,
        planeType: Union[PlaneType, str] = PlaneType.Image,
        blendOffsets: Union[np.ndarray, tuple, list, None] = None,
    ) -> None:
        self.image = image
        self.fieldAngle = fieldAngle  # type: ignore
        self.defocalType = defocalType  # type: ignore
        self.bandLabel = bandLabel  # type: ignore
        self.planeType = planeType  # type: ignore
        self.blendOffsets = blendOffsets  # type: ignore

        # Set all mask variables
        self._mask = None
        self._maskBlends = None
        self._maskBackground = None

    @property
    def image(self) -> np.ndarray:
        """The donut image array."""
        return self._image

    @image.setter
    def image(self, value: np.ndarray) -> None:
        """Set the donut image array.

        Parameters
        ----------
        value : np.ndarray
            The square numpy array containing the donut image.

        Raises
        ------
        TypeError
            If the image is not a numpy array
        ValueError
            If the array is not square
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("image must be a numpy array.")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("The image array must be square.")
        self._image = value.copy()

    @property
    def fieldAngle(self) -> np.ndarray:
        """The field angle in degrees."""
        return self._fieldAngle

    @fieldAngle.setter
    def fieldAngle(self, value: Union[np.ndarray, tuple, list]) -> None:
        """Set the field angle.

        Parameters
        ----------
        value : np.ndarray, tuple, or list
            The field angle of the donut, in degrees. The field angle
            is the angle to the source, measured from the optical axis.

        Raises
        ------
        ValueError
            The provided value is not an iterable with 2 values
        """
        value = np.array(value, dtype=float).squeeze()
        if value.shape != (2,):
            raise ValueError("Field angle must have shape (2,).")
        self._fieldAngle = value

    @property
    def defocalType(self) -> DefocalType:
        """Return the DefocalType Enum of the image."""
        return self._defocalType

    @defocalType.setter
    def defocalType(self, value: Union[DefocalType, str]) -> None:
        """Set the defocal type.

        Parameters
        ----------
        value : DefocalType or str
            Whether the image is intra- or extra-focal.
            Can be specified using DefocalType Enum or corresponding string.

        Raises
        ------
        TypeError
            The provided value is not a DefocalType Enum or string.
        """
        if isinstance(value, str) or isinstance(value, DefocalType):
            self._defocalType = DefocalType(value)
        else:
            raise TypeError("defocalType must be a DefocalType Enum, or one of the corresponding strings.")

    @property
    def bandLabel(self) -> BandLabel:
        """The BandLabel Enum of the image."""
        return self._bandLabel

    @bandLabel.setter
    def bandLabel(self, value: Union[BandLabel, str, None]) -> None:
        """Set the band label.

        Parameters
        ----------
        value : BandLabel or str or None
            Photometric band for the exposure. Can be specified using a
            BandLabel Enum or the corresponding string. If None, BandLabel.REF
            is used. The empty string "" also maps to BandLabel.REF.

        Raises
        ------
        TypeError
            The provided value is not a BandLabel Enum or string.
        """
        if value is None or value == "":
            self._bandLabel = BandLabel.REF
        elif isinstance(value, BandLabel):
            self._bandLabel = BandLabel(value)
        elif isinstance(value, str):
            self._bandLabel = (
                BandLabel(value) if value in {band_label.value for band_label in BandLabel} else BandLabel.REF
            )
        else:
            raise TypeError("bandLabel must be a BandLabel Enum, or one of the corresponding strings.")

    @property
    def planeType(self) -> PlaneType:
        """The PlaneType Enum of the image."""
        return self._planeType

    @planeType.setter
    def planeType(self, value: Union[PlaneType, str]) -> None:
        """Set the plane type.

        Parameters
        ----------
        value : PlaneType or str
            A PlaneType Enum or one of the corresponding strings.

        Raises
        ------
        TypeError
            The provided value is not a PlaneType Enum or string.
        """
        if isinstance(value, str) or isinstance(value, PlaneType):
            self._planeType = PlaneType(value)
        else:
            raise TypeError("planeType must be a PlaneType Enum, or one of the corresponding strings.")

    @property
    def blendOffsets(self) -> Union[np.ndarray, None]:
        """The blend offsets array for the image."""
        return self._blendOffsets

    @blendOffsets.setter
    def blendOffsets(self, value: Union[np.ndarray, tuple, list, None]) -> None:
        """Set the blend offsets array for the image.

        Parameters
        ----------
        value : np.ndarray or tuple or list or None
            Positions of blended donuts relative to central donut, in pixels.
            Must be provided in the format [[dx1, dy1], [dx2, dy2], ...].
            If None, an empty array is populated for you.

        Raises
        ------
        ValueError
            If the provided value does not have the correct shape
        """
        # If None, populate an empty array with the correct shape
        if value is None:
            value = np.zeros((0, 2))

        # Convert to float array
        value = np.atleast_2d(np.array(value, dtype=float))
        assert isinstance(value, np.ndarray)  # for mypy

        # Check shape
        if value.shape[1] != 2 or value.ndim != 2:
            raise ValueError(
                "blendOffsets must have shape (N, 2), where N is the number of blends you wish to mask."
            )

        self._blendOffsets = value

    @property
    def mask(self) -> Union[np.ndarray, None]:
        """The image source mask.

        This mask has value 1 for source pixels and value 0 for other pixels.
        """
        return self._mask

    @mask.setter
    def mask(self, value: Optional[np.ndarray]) -> None:
        """Set the image source mask.

        Note mask creation is meant to be handled by the ImageMapper class.
        This mask has value 1 for source pixels and value 0 for other pixels.

        Parameters
        ----------
        mask : np.ndarray
            The mask for the image.

        Raises
        ------
        TypeError
            If the mask is not an array, or None
        ValueError
            If the mask is an array and does not match the shape of the image
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("mask must be an array or None.")
        elif isinstance(value, np.ndarray) and value.shape != self.image.shape:
            raise ValueError("mask must have the same shape as self.image.")
        elif isinstance(value, np.ndarray):
            value = value.copy()
        self._mask = value

    @property
    def maskBlends(self) -> Union[np.ndarray, None]:
        """The image source mask.

        This mask has value 1 for blend pixels and value 0 for other pixels.
        """
        return self._maskBlends

    @maskBlends.setter
    def maskBlends(self, value: Optional[np.ndarray]) -> None:
        """Set the image blend mask.

        Note mask creation is meant to be handled by the ImageMapper class.
        This mask has value 1 for blend pixels and value 0 for other pixels.

        Parameters
        ----------
        mask : np.ndarray
            The mask for the image.

        Raises
        ------
        TypeError
            If the mask is not an array, or None
        ValueError
            If the mask is an array and does not match the shape of the image
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("mask must be an array or None.")
        elif isinstance(value, np.ndarray) and value.shape != self.image.shape:
            raise ValueError("mask must have the same shape as self.image.")
        elif isinstance(value, np.ndarray):
            value = value.copy()
        self._maskBlends = value

    @property
    def maskBackground(self) -> Union[np.ndarray, None]:
        """The image background mask.

        This mask has value 1 for background pixels and value 0
        for other pixels.
        """
        return self._maskBackground

    @maskBackground.setter
    def maskBackground(self, value: Optional[np.ndarray]) -> None:
        """Set the image background mask.

        Note mask creation is meant to be handled by the ImageMapper class.
        This mask has value 1 for background pixels and value 0
        for other pixels.

        Parameters
        ----------
        mask : np.ndarray
            The mask for the image.

        Raises
        ------
        TypeError
            If the mask is not an array, or None
        ValueError
            If the mask is an array and does not match the shape of the image
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("mask must be an array or None.")
        elif isinstance(value, np.ndarray) and value.shape != self.image.shape:
            raise ValueError("mask must have the same shape as self.image.")
        elif isinstance(value, np.ndarray):
            value = value.copy()
        self._maskBackground = value

    @property
    def masks(self) -> tuple:
        """Return (self.mask, self.maskBlends, self.maskBackground)."""
        return (self.mask, self.maskBlends, self.maskBackground)

    def copy(self) -> Self:
        """Return a copy of the DonutImage object.

        Returns
        -------
        DonutImage
            A deep copy of self.
        """
        return deepcopy(self)
