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

import numpy as np

from lsst.ts.wep.task.CombineZernikesBase import CombineZernikesBaseTask


class CombineZernikesMeanTask(CombineZernikesBaseTask):
    """
    Combine the raw Zernike measurements into an average
    measurement with an unweighted mean.
    """

    def combineZernikes(self, zernikeArray):
        """
        Combine the Zernike coefficients with an
        unweighted mean.

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

        return np.mean(zernikeArray, axis=0), np.zeros(len(zernikeArray))
