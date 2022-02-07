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

import abc
import logging

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


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

    def __init__(self, **kwargs):

        pipeBase.Task.__init__(self, **kwargs)
        self.log = logging.getLogger(type(self).__name__)

    def run(self, zernikeArray):
        """
        Combine the zernikes from the input array of Zernike
        coefficients from each individual donut pair.

        Parameters
        ----------
        zernikeArray: numpy ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        struct: `lsst.pipe.base.Struct`
            The struct contains the following data:

            - combinedZernikes: numpy ndarray
                The final combined Zernike coefficients from the CCD.
            - combineFlags: numpy ndarray
                Flag indicating a particular set of Zernike
                coefficients was not used in the final estimate.
                A value of 1 means the coefficients from that row
                in the input `zernikeArray` were not used.
        """

        combinedZernikes, flags = self.combineZernikes(zernikeArray)
        return pipeBase.Struct(combinedZernikes=combinedZernikes, flags=flags)

    @abc.abstractmethod
    def combineZernikes(self, zernikeArray):
        """
        Class specific algorithm to combine the Zernike
        coefficients from each individual donut pair into
        a single set of coefficients for the detector.

        Parameters
        ----------
        zernikeArray: numpy ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        numpy ndarray
            The final combined Zernike coefficients from the CCD.
        numpy ndarray
            A binary array where a value of 1 in any index indicates
            that the row in the zernikeArray was not used
            in the final combination.
        """
        raise NotImplementedError("CombineZernikesBaseTask is abstract.")
