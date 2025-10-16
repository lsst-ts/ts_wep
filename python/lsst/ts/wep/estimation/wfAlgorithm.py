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
from typing import Optional, Sequence, Tuple

import numpy as np
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.utils import (
    checkNollIndices,
    convertZernikesToPsfWidth,
    makeDense,
    makeSparse,
)


class WfAlgorithm(ABC):
    """Base class for wavefront estimation algorithms

    Parameters
    ----------
    ...

    """

    def __init_subclass__(cls) -> None:
        """This is called when you subclass."""
        if cls.history.__doc__ is None:
            raise AttributeError(
                "When subclassing WfAlgorithm you must write a docstring for "
                "the history property. Please use this to describe the contents "
                "of the history dictionary."
            )

    @property
    @abstractmethod
    def requiresPairs(self) -> bool:
        """Whether the algorithm requires pairs to estimate Zernikes."""
        ...

    @property
    def history(self) -> dict:
        # Return the algorithm history
        # Note I have not written a real docstring here, so that I can force
        # subclasses to write a new docstring for this method
        if getattr(self, "_history", None) is None:
            self._history: dict = dict()

        return self._history

    def _validateInputs(
        self,
        I1: Image,
        I2: Image | None,
        nollIndices: np.ndarray,
        instrument: Instrument,
        startWithIntrinsic: bool,
        returnWfDev: bool,
        units: str,
        saveHistory: bool,
    ) -> None:
        """Validate the inputs to estimateWf.

        Parameters
        ----------
        I1 : DonutStamp
            An Image object containing an intra- or extra-focal donut image.
        I2 : DonutStamp or None
            A second image, on the opposite side of focus from I1.
        nollIndices : np.ndarray
            List, tuple, or array of Noll indices for which you wish to
            estimate Zernike coefficients. Note these values must be unique,
            ascending, and >= 4.
        instrument : Instrument
            The Instrument object associated with the DonutStamps.
        startWithIntrinsic : bool
            Whether to start the Zernike estimation process from the intrinsic
            Zernikes rather than zero.
        returnWfDev : bool
            If False, the full OPD is returned. If True, the wavefront
            deviation is returned. The wavefront deviation is defined as
            the OPD - intrinsic Zernikes.
        units : str
            Units in which the Zernike amplitudes are returned.
            Options are "m", "nm", "um", or "arcsecs".

        Raises
        ------
        TypeError
            If any input is the wrong type
        ValueError
            If I1 or I2 are not square arrays, or if jmax < 4,
            or unsupported unit
        """
        # Check if we require both images
        if (I1 is None or I2 is None) and self.requiresPairs:
            raise ValueError(
                f"{self.__class__.__name__} requires a pair of "
                "intrafocal and extrafocal donuts to estimate Zernikes. "
                "Please provide both I1 and I2."
            )

        # Validate I1
        if not isinstance(I1, Image):
            raise TypeError("I1 must be an Image object.")
        if I1.image.ndim != 2 or not I1.image.shape[0] == I1.image.shape[1]:
            raise ValueError("I1.image must be square.")

        # Validate I2 if provided
        if I2 is not None:
            if not isinstance(I2, Image):
                raise TypeError("I2 must be an Image object.")
            if I2.image.ndim != 2 or not I2.image.shape[0] == I2.image.shape[1]:
                raise ValueError("I2.image must be square.")
            if I2.defocalType == I1.defocalType:
                raise ValueError("I1 and I2 must be on opposite sides of focus.")

        # Validate nollIndices
        checkNollIndices(nollIndices)

        # Validate the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("instrument must be an Instrument.")

        # Validate startWithIntrinsic
        if not isinstance(startWithIntrinsic, bool):
            raise TypeError("startWithIntrinsic must be a bool.")

        # Validate returnWfDev
        if not isinstance(returnWfDev, bool):
            raise TypeError("returnWfDev must be a bool.")

        # Validate units
        if not isinstance(units, str):
            raise TypeError("units must be a str.")
        allowedUnits = ["m", "nm", "um", "arcsec"]
        if units not in allowedUnits:
            raise ValueError(f"units must be one of {allowedUnits}")

        # Validate saveHistory
        if not isinstance(saveHistory, bool):
            raise TypeError("saveHistory must be a bool.")

    @abstractmethod
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
        """Private Zernike estimation method that should be subclassed.

        Note that unlike the public method, this method MUST return the
        coefficients for Noll indices >= 4 (in meters) corresponding to
        the full OPD.

        Parameters
        ----------
        I1 : DonutStamp
            An Image object containing an intra- or extra-focal donut image.
        I2 : DonutStamp or None
            A second image, on the opposite side of focus from I1.
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
            Zernike coefficients representing the OPD in meters,
            for Noll indices >= 4.
        dict
            Metadata containing extra output from wavefront
            estimation algorithm.
        """
        ...

    def estimateZk(
        self,
        I1: Image,
        I2: Optional[Image] = None,
        nollIndices: Sequence = tuple(np.arange(4, 23)),
        instrument: Optional[Instrument] = None,
        startWithIntrinsic: bool = True,
        returnWfDev: bool = False,
        units: str = "m",
        saveHistory: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : DonutStamp
            An Image object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second image, on the opposite side of focus from I1.
            (the default is None)
        nollIndices : Sequence, optional
            List, tuple, or array of Noll indices for which you wish to
            estimate Zernike coefficients. Note these values must be unique,
            ascending, and >= 4. (the default is indices 4-22)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)
        startWithIntrinsic : bool, optional
            Whether to start the Zernike estimation process from the intrinsic
            Zernikes rather than zero. (the default is True)
        returnWfDev : bool, optional
            If False, the full OPD is returned. If True, the wavefront
            deviation is returned. The wavefront deviation is defined as
            the OPD - intrinsic Zernikes. (the default is False)
        units : str, optional
            Units in which the Zernike amplitudes are returned.
            Options are "m", "nm", "um", or "arcsecs".
            (the default is "m")
        saveHistory : bool, optional
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.
            (the default is False)

        Returns
        -------
        np.ndarray
            Zernike coefficients estimated from the image (or pair of images)
        dict
            Metadata containing extra output from wavefront estimation
            algorithm.
        """
        if instrument is None:
            instrument = Instrument()

        # Convert nollIndices to an array
        nollIndices = np.array(nollIndices)

        # Validate the inputs
        self._validateInputs(
            I1,
            I2,
            nollIndices,
            instrument,
            startWithIntrinsic,
            returnWfDev,
            units,
            saveHistory,
        )

        # Get the intrinsic Zernikes?
        if startWithIntrinsic or returnWfDev:
            zkIntrinsicI1 = instrument.getIntrinsicZernikes(
                *I1.fieldAngle,
                I1.bandLabel,
                nollIndices,
            )
            zkIntrinsicI2 = (
                None
                if I2 is None
                else instrument.getIntrinsicZernikes(
                    *I2.fieldAngle,
                    I2.bandLabel,
                    nollIndices,
                )
            )

        # Determine the Zernikes to start with
        if startWithIntrinsic:
            zkStartI1 = zkIntrinsicI1
            zkStartI2 = zkIntrinsicI2
        else:
            zkStartI1 = np.zeros_like(nollIndices, dtype=float)
            zkStartI2 = None if I2 is None else np.zeros_like(nollIndices, dtype=float)

        # Clear the algorithm history
        self._history = {}

        # Estimate the Zernikes
        zk, zkMeta = self._estimateZk(
            I1=I1,
            I2=I2,
            zkStartI1=zkStartI1,
            zkStartI2=zkStartI2,
            nollIndices=nollIndices,
            instrument=instrument,
            saveHistory=saveHistory,
        )

        # Calculate the wavefront deviation?
        if returnWfDev:
            zkIntrinsic = zkIntrinsicI1 if I2 is None else np.mean([zkIntrinsicI1, zkIntrinsicI2], axis=0)
            zk -= zkIntrinsic

        # Convert to desired units
        if units == "m":
            pass
        elif units == "um":
            zk *= 1e6
        elif units == "nm":
            zk *= 1e9
        elif units == "arcsec":
            zk = makeDense(zk, nollIndices)
            zk = convertZernikesToPsfWidth(
                zernikes=zk * 1e6,
                diameter=instrument.diameter,
                obscuration=instrument.obscuration,
            )
            zk = makeSparse(zk, nollIndices)
        else:  # pragma: no cover
            raise RuntimeError(f"Conversion to unit '{units}' not supported.")

        return zk, zkMeta
