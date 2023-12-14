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
from lsst.ts.wep.utils import (
    configClass,
    convertZernikesToPsfWidth,
    mergeConfigWithFile,
)


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
    units : str, optional
        Units in which the Zernike amplitudes are returned.
        Options are "m", "nm", "um", or "arcsecs".
    """

    def __init__(
        self,
        configFile: Optional[str] = "policy/estimation/wfEstimator.yaml",
        algoName: Optional[str] = None,
        algoConfig: Union[str, dict, WfAlgorithm, None] = None,
        instConfig: Union[str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        units: Optional[str] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeConfigWithFile(
            configFile,
            algoName=algoName,
            algoConfig=algoConfig,
            instConfig=instConfig,
            jmax=jmax,
            units=units,
        )

        # Set the algorithm
        self._algo = WfAlgorithmFactory.createWfAlgorithm(
            params["algoName"], params["algoConfig"]
        )

        # Set the instrument
        self._instrument = configClass(instConfig, Instrument)

        # Set the other parameters
        self.jmax = params["jmax"]
        self.units = params["units"]

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
        """Set jmax"""
        value = int(value)
        if value < 4:
            raise ValueError("jmax must be greater than or equal to 4.")
        self._jmax = value

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
        # Estimate wavefront Zernike coefficients (in meters)
        zk = self.algo.estimateZk(I1, I2, self.jmax, self.instrument)

        # Convert to desired units
        if self.units == "m":
            pass
        elif self.units == "um":
            zk *= 1e6
        elif self.units == "nm":
            zk *= 1e9
        elif self.units == "arcsecs":
            zk = convertZernikesToPsfWidth(
                zernikes=zk,
                diameter=self.instrument.diameter,
                obscuration=self.instrument.obscuration,
            )
        else:
            raise RuntimeError(f"Conversion to unit '{self.units}' not supported.")

        return zk
