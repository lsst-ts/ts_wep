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

from typing import Optional, Union

import numpy as np
from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.estimation.wfAlgorithmFactory import WfAlgorithmFactory
from lsst.ts.wep.utils import configClass, mergeConfigWithFile


class WfEstimator:
    """Class providing a high-level interface for wavefront estimation.

    Any explicitly passed parameters override the values found in configFile.

    Parameters
    ----------
    configFile : str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/estimation/wfEstimator.yaml)
    algoName : WfAlgorithmName or str, optional
        Name of the algorithm to use. Can be specified using a WfAlgorithmName
        Enum or the corresponding string.
    algoConfig : str or dict or WfAlgorithm, optional
        Algorithm configuration. If a string, it is assumed this points to a
        config file, which is used to configure the algorithm. If the path
        begins with "policy/", then it is assumed the path is relative to the
        policy directory. If a dictionary, it is assumed to hold keywords for
        configuration. If a WfAlgorithm object, that object is just used.
        If None, the algorithm defaults are used.
    instConfig : str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
    jmax : int, optional
        The maximum Zernike Noll index to estimate.
    startWithIntrinsic : bool, optional
        Whether to start the Zernike estimation process from the intrinsic
        Zernikes rather than zero.
    returnWfDev : bool, optional
        If False, the full OPD is returned. If True, the wavefront
        deviation is returned. The wavefront deviation is defined as
        the OPD - intrinsic Zernikes. (the default is False)
    return4Up : bool, optional
        If True, the returned Zernike coefficients start with
        Noll index 4. If False, they follow the Galsim convention
        of starting with index 0 (which is meaningless), so the
        array index of the output corresponds to the Noll index.
        In this case, indices 0-3 are always set to zero, because
        they are not estimated by our pipeline.
    units : str, optional
        Units in which the Zernike amplitudes are returned.
        Options are "m", "nm", "um", or "arcsecs".
    saveHistory : bool, optional
        Whether to save the algorithm history in the self.history
        attribute. If True, then self.history contains information
        about the most recent time the algorithm was run.
        (the default is False)
    """

    def __init__(
        self,
        configFile: Optional[str] = "policy/estimation/wfEstimator.yaml",
        algoName: Optional[str] = None,
        algoConfig: Union[str, dict, WfAlgorithm, None] = None,
        instConfig: Union[str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        startWithIntrinsic: Optional[bool] = None,
        returnWfDev: Optional[bool] = None,
        return4Up: Optional[bool] = None,
        units: Optional[str] = None,
        saveHistory: Optional[bool] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeConfigWithFile(
            configFile,
            algoName=algoName,
            algoConfig=algoConfig,
            instConfig=instConfig,
            jmax=jmax,
            startWithIntrinsic=startWithIntrinsic,
            returnWfDev=returnWfDev,
            return4Up=return4Up,
            units=units,
            saveHistory=saveHistory,
        )

        # Set the algorithm
        self._algo = WfAlgorithmFactory.createWfAlgorithm(
            params["algoName"], params["algoConfig"]
        )

        # Set the instrument
        self._instrument = configClass(instConfig, Instrument)

        # Set the other parameters
        self.jmax = params["jmax"]
        self.startWithIntrinsic = params["startWithIntrinsic"]
        self.returnWfDev = params["returnWfDev"]
        self.return4Up = params["return4Up"]
        self.units = params["units"]
        self.saveHistory = params["saveHistory"]

        # Copy the history docstring
        self.__class__.history.__doc__ = self.algo.__class__.history.__doc__

    @property
    def algo(self) -> WfAlgorithm:
        """Return the WfAlgorithm object."""
        return self._algo

    @property
    def instrument(self) -> Instrument:
        """Return the Instrument object."""
        return self._instrument

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index that will be estimated."""
        return self._jmax

    @jmax.setter
    def jmax(self, value: int) -> None:
        """Set jmax

        Parameters
        ----------
        value : int
            The maximum Zernike Noll index to estimate.

        Raises
        ------
        ValueError
            If jmax is < 4
        """
        value = int(value)
        if value < 4:
            raise ValueError("jmax must be greater than or equal to 4.")
        self._jmax = value

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
    def return4Up(self) -> bool:
        """Whether to return Zernikes starting with Noll index 4."""
        return self._return4Up

    @return4Up.setter
    def return4Up(self, value: bool) -> None:
        """Set return4Up.

        Parameters
        ----------
        value : bool
            If True, the returned Zernike coefficients start with
            Noll index 4. If False, they follow the Galsim convention
            of starting with index 0 (which is meaningless), so the
            array index of the output corresponds to the Noll index.
            In this case, indices 0-3 are always set to zero, because
            they are not estimated by our pipeline.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("return4Up must be a bool.")
        self._return4Up = value

    @property
    def units(self) -> str:
        """Return the wavefront units.

        For details about this parameter, see the class docstring.
        """
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """Set the units of the Zernike coefficients."""
        allowed_units = ["m", "um", "nm", "arcsecs"]
        if value not in allowed_units:
            raise ValueError(
                f"Unit '{value}' not supported. "
                f"Please choose one of {str(allowed_units)[1:-1]}."
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
            Numpy array of the Zernike coefficients estimated from the stamp
            or pair of stamp. The array contains Noll coefficients from
            4 - self.jmax, inclusive. The unit is determined by self.units.

        Raises
        ------
        ValueError
            If I1 and I2 are on the same side of focus.
        """
        return self.algo.estimateZk(
            I1=I1,
            I2=I2,
            jmax=self.jmax,
            instrument=self.instrument,
            startWithIntrinsic=self.startWithIntrinsic,
            returnWfDev=self.returnWfDev,
            return4Up=self.return4Up,
            units=self.units,
            saveHistory=self.saveHistory,
        )
