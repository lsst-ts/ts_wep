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

import warnings
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
    saveHistory : bool, optional
        Whether to save the algorithm history in the self.history attribute.
        If True, then self.history contains information about the most recent
        time the algorithm was run.

    ...

    """

    def __init__(
        self,
        configFile: Optional[str] = None,
        saveHistory: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeConfigWithFile(
            configFile,
            saveHistory=saveHistory,
            **kwargs,
        )

        # Configure parameters
        for key, value in params.items():
            setattr(self, key, value)

        # Instantiate an empty history
        self._history = {}  # type: ignore

    def __init_subclass__(self) -> None:
        """This is called when you subclass.

        I am using this to force you to write a docstring
        for the history property.
        """
        if self.history.__doc__ is None:
            raise AttributeError(
                "You must write a docstring for the history property. "
                "Please use this to describe the contents of the history dict."
            )

    @property
    def saveHistory(self) -> bool:
        """Whether the algorithm history is saved."""
        return self._saveHistory

    @saveHistory.setter
    def saveHistory(self, value: bool) -> None:
        """Set boolean that determines whether algorithm history is saved.

        Parameters
        ----------
        value : bool
            Boolean that determines whether the algorithm history is saved.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("saveHistory must be a boolean.")

        self._saveHistory = value

        # If we are turning history-saving off, delete any old history
        # This is to avoid confusion
        if value is False:
            self._history = {}

    @property
    def history(self) -> dict:
        # Return the algorithm history
        # Note I have not written a real docstring here, so that I can force
        # subclasses to write a new docstring for this method
        if not self._saveHistory:
            warnings.warn(
                "saveHistory is False. If you want the history to be saved, "
                "run self.config(saveHistory=True)."
            )

        return self._history

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
