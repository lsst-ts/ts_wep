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

__all__ = ["CombineZernikesBaseConfig", "CombineZernikesBaseTask"]

import abc
import logging
from typing import Any

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np


class CombineZernikesBaseConfig(pexConfig.Config):
    pass


class CombineZernikesBaseTask(pipeBase.Task, metaclass=abc.ABCMeta):
    """
    Base class for algorithms that combine Zernikes from the individual
    pairs of donuts on a detector into a single array of Zernike values
    for that detector.
    """

    ConfigClass = CombineZernikesBaseConfig
    _DefaultName = "combineZernikes"

    def __init__(self, **kwargs: Any) -> None:
        pipeBase.Task.__init__(self, **kwargs)
        self.log = logging.getLogger(type(self).__name__)  # type: ignore

    def run(self, zernikeArray: np.ndarray) -> pipeBase.Struct:
        """
        Combine the zernikes from the input array of Zernike
        coefficients from each individual donut pair.

        Parameters
        ----------
        zernikeArray : numpy.ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:

            - combinedZernikes : numpy.ndarray
                The final combined Zernike coefficients from the CCD.
            - combineFlags : numpy.ndarray
                Flag indicating a particular set of Zernike
                coefficients was not used in the final estimate.
                If the values in a row in the `zernikeArray`
                were used then its index is 0.
                A value of 1 means the coefficients from that row
                in the input `zernikeArray` were not used.
        """

        combinedZernikes, flags = self.combineZernikes(zernikeArray)

        # Make sure that flags contains only integers
        flags = np.array(flags, dtype=int)
        self.log.info(
            f"Using {len(flags) - np.sum(flags)} pairs out of {len(zernikeArray)} in final Zernike estimate."
        )

        # Save flags and summary values in task metadata
        self.metadata["numDonutsTotal"] = len(flags)
        self.metadata["numDonutsUsed"] = len(flags) - np.sum(flags)
        self.metadata["numDonutsRejected"] = np.sum(flags)
        self.metadata["combineZernikesFlags"] = list(flags)
        return pipeBase.Struct(combinedZernikes=combinedZernikes, flags=flags)

    @abc.abstractmethod
    def combineZernikes(self, zernikeArray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Class specific algorithm to combine the Zernike
        coefficients from each individual donut pair into
        a single set of coefficients for the detector.

        Parameters
        ----------
        zernikeArray : numpy.ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        numpy.ndarray
            The final combined Zernike coefficients from the CCD.
        numpy.ndarray
            A binary array where a value of 0 in any index indicates
            that the row in the `zernikeArray` was used
            in the final combination and a value of 1 indicates it
            was not included in the final combination.
        """
        raise NotImplementedError("CombineZernikesBaseTask is abstract.")
