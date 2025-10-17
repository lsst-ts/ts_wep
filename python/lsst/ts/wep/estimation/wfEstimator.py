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

__all__ = ["WfEstimator"]

from typing import Optional, Sequence, Union

import numpy as np
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.estimation.wfAlgorithmFactory import WfAlgorithmFactory
from lsst.ts.wep.utils import checkNollIndices, configClass


class WfEstimator:
    """Class providing a high-level interface for wavefront estimation.

    Parameters
    ----------
    algoName : WfAlgorithmName or str, optional
        Name of the algorithm to use. Can be specified using a WfAlgorithmName
        Enum or the corresponding string.
        (the default is "tie")
    algoConfig : dict or WfAlgorithm, optional
        Algorithm configuration. If a string, it is assumed this points to a
        config file, which is used to configure the algorithm. If the path
        begins with "policy:", then it is assumed the path is relative to the
        policy directory. If a dictionary, it is assumed to hold keywords for
        configuration. If a WfAlgorithm object, that object is just used.
        If None, the algorithm defaults are used.
        (The default is None)
    instConfig : str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
        (the default is "policy:instrument/LsstCam.yaml")
    nollIndices : Sequence, optional
        List, tuple, or array of Noll indices for which you wish to
        estimate Zernike coefficients. Note these values must be unique,
        ascending, and >= 4. (the default is indices 4-11)
    startWithIntrinsic : bool, optional
        Whether to start the Zernike estimation process from the intrinsic
        Zernikes rather than zero.
        (the default is True)
    returnWfDev : bool, optional
        If False, the full OPD is returned. If True, the wavefront
        deviation is returned. The wavefront deviation is defined as
        the OPD - intrinsic Zernikes.
        (the default is False)
    units : str, optional
        Units in which the Zernike coefficients are returned.
        Options are "m", "um", "nm", or "arcsec".
        (the default is "um")
    saveHistory : bool, optional
        Whether to save the algorithm history in the self.history
        attribute. If True, then self.history contains information
        about the most recent time the algorithm was run.
        (the default is False)
    """

    def __init__(
        self,
        algoName: str = "tie",
        algoConfig: Union[dict, WfAlgorithm, None] = None,
        instConfig: Union[str, dict, Instrument] = "policy:instruments/LsstCam.yaml",
        nollIndices: Sequence = tuple(np.arange(4, 12)),
        startWithIntrinsic: bool = True,
        returnWfDev: bool = False,
        units: str = "um",
        saveHistory: bool = False,
    ) -> None:
        # Set the algorithm and instrument
        self.algo = WfAlgorithmFactory.createWfAlgorithm(algoName, algoConfig)
        self.instrument = configClass(instConfig, Instrument)

        # Set the other parameters
        self.nollIndices = nollIndices
        self.startWithIntrinsic = startWithIntrinsic
        self.returnWfDev = returnWfDev
        self.units = units
        self.saveHistory = saveHistory

        # Copy the history docstring
        self.__class__.history.__doc__ = self.algo.__class__.history.__doc__

    @property
    def algo(self) -> WfAlgorithm:
        """Return the WfAlgorithm object."""
        return self._algo

    @algo.setter
    def algo(self, value: WfAlgorithm) -> None:
        """Set the wavefront algorithm.

        Parameters
        ----------
        value : WfAlgorithm
            A WfAlgorithm object.

        Raises
        ------
        TypeError
            If the value is not a WfAlgorithm
        """
        if not isinstance(value, WfAlgorithm):
            raise TypeError("algo must be an WfAlgorithm.")
        self._algo = value

    @property
    def instrument(self) -> Instrument:
        """Return the Instrument object."""
        return self._instrument

    @instrument.setter
    def instrument(self, value: Instrument) -> None:
        """Set the Instrument.

        Parameters
        ----------
        value : Instrument
            An Instument object.

        Raises
        ------
        TypeError
            If the value is not an Instrument object
        """
        if not isinstance(value, Instrument):
            raise TypeError("instrument must be an Instrument object.")
        self._instrument = value

    @property
    def nollIndices(self) -> np.ndarray:
        """Return the Noll indices for which Zernikes are estimated."""
        return self._nollIndices

    @nollIndices.setter
    def nollIndices(self, value: Sequence) -> None:
        """Set the Noll indices for which the Zernikes are estimated.

        Parameters
        ----------
        nollIndices : Sequence, optional
            List, tuple, or array of Noll indices for which you wish to
            estimate Zernike coefficients. Note these values must be unique,
            ascending, and >= 4. (the default is indices 4-22)
        """
        value = np.array(value)
        checkNollIndices(value)
        self._nollIndices = value

    @property
    def startWithIntrinsic(self) -> bool:
        """Whether to start Zernike estimation with the intrinsics."""
        return self._startWithIntrinsic

    @startWithIntrinsic.setter
    def startWithIntrinsic(self, value: bool) -> None:
        """Set startWithIntrinsic.

        Parameters
        ----------
        value : bool
            Whether to start the Zernike estimation process from the intrinsic
            Zernikes rather than zero.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("startWithIntrinsic must be a bool.")
        self._startWithIntrinsic = value

    @property
    def returnWfDev(self) -> bool:
        """Whether to return the wavefront deviation instead of the OPD."""
        return self._returnWfDev

    @returnWfDev.setter
    def returnWfDev(self, value: bool) -> None:
        """Set returnWfDev.

        Parameters
        ----------
        value : bool
            If False, the full OPD is returned. If True, the wavefront
            deviation is returned. The wavefront deviation is defined as
            the OPD - intrinsic Zernikes.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("returnWfDev must be a bool.")
        self._returnWfDev = value

    @property
    def units(self) -> str:
        """Return the wavefront units.

        For details about this parameter, see the class docstring.
        """
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """Set the units of the Zernike coefficients."""
        allowed_units = ["m", "um", "nm", "arcsec"]
        if value not in allowed_units:
            raise ValueError(
                f"Unit '{value}' not supported. Please choose one of {str(allowed_units)[1:-1]}."
            )
        self._units = value

    @property
    def saveHistory(self) -> bool:
        """Whether to save the algorithm history."""
        return self._saveHistory

    @saveHistory.setter
    def saveHistory(self, value: bool) -> None:
        """Set saveHistory.

        Parameters
        ----------
        value : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("saveHistory must be a bool.")
        self._saveHistory = value

    @property
    def history(self) -> dict:
        # Return the algorithm history
        # This does not have a real docstring, because the dosctring from
        # the algorithm history is added during the class __init__ above
        return self.algo.history

    def estimateZk(
        self,
        I1: Image,
        I2: Optional[Image] = None,
    ) -> np.ndarray:
        """Estimate Zernike coefficients of the wavefront from the stamp(s).

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image, optional
            A second image, on the opposite side of focus from I1.
            (the default is None)

        Returns
        -------
        np.ndarray
            Zernike coefficients estimated from the stamp(s)

        Raises
        ------
        ValueError
            If I1 and I2 are on the same side of focus.
        """
        return self.algo.estimateZk(
            I1=I1,
            I2=I2,
            nollIndices=self.nollIndices,
            instrument=self.instrument,
            startWithIntrinsic=self.startWithIntrinsic,
            returnWfDev=self.returnWfDev,
            units=self.units,
            saveHistory=self.saveHistory,
        )
