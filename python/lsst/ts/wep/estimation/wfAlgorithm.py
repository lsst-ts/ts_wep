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

__all__ = ["WfAlgorithm"]

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.utils import mergeConfigWithFile


class WfAlgorithm(ABC):
    """Base class for wavefront estimation algorithms

    Parameters
    ----------
    configFile : str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file

    ...

    """

    def __init__(
        self,
        configFile: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeConfigWithFile(
            configFile,
            **kwargs,
        )

        # Configure parameters
        for key, value in params.items():
            setattr(self, key, value)

    @staticmethod
    def _validateInputs(
        I1: Image,
        I2: Optional[Image],
        jmax: int = 28,
        instrument: Instrument = Instrument(),
    ) -> None:
        """Validate the inputs to estimateWf.

        Parameters
        ----------
        I1 : DonutStamp
            An Image object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second image, on the opposite side of focus from I1.
        jmax : int, optional
            The maximum Zernike Noll index to estimate.
            (the default is 28)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Raises
        ------
        TypeError
            If any input is the wrong type
        ValueError
            If I1 or I2 are not square arrays, or if jmax < 4
        """
        # Validate I1
        if not isinstance(I1, Image):
            raise TypeError("I1 must be an Image object.")
        if len(I1.image.shape) != 2 or not np.allclose(*I1.image.shape):  # type: ignore
            raise ValueError("I1.image must be square.")

        # Validate I2 if provided
        if I2 is not None:
            if not isinstance(I2, Image):
                raise TypeError("I2 must be an Image object.")
            if len(I2.image.shape) != 2 or not np.allclose(
                *I2.image.shape  # type: ignore
            ):
                raise ValueError("I2.image must be square.")
            if I2.defocalType == I1.defocalType:
                raise ValueError("I1 and I2 must be on opposite sides of focus.")

        # Validate jmax
        if not isinstance(jmax, int):
            raise TypeError("jmax must be an integer.")
        if jmax < 4:
            raise ValueError("jmax must be greater than or equal to 4.")

        # Validate the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("instrument must be an Instrument.")

    @abstractmethod
    def estimateZk(
        self,
        I1: Image,
        I2: Optional[Image],
        jmax: int = 28,
        instrument: Instrument = Instrument(),
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : DonutStamp
            An Image object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second image, on the opposite side of focus from I1.
        jmax : int, optional
            The maximum Zernike Noll index to estimate.
            (the default is 28)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from
            the image (or pair of images), in meters.
        """
        ...
