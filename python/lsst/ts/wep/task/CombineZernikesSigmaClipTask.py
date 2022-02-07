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
from astropy.stats import sigma_clip

import lsst.pex.config as pexConfig
from lsst.ts.wep.task.CombineZernikesBase import (
    CombineZernikesBaseTask,
    CombineZernikesBaseConfig,
)


class CombineZernikesSigmaClipTaskConfig(CombineZernikesBaseConfig):
    """
    Configuration for combining Zernike coefficients with sigma clipping.
    """

    sigma = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Number of standard deviations for the clipping limit.",
    )


class CombineZernikesSigmaClipTask(CombineZernikesBaseTask):
    """
    Combine the raw Zernike measurements into an average
    measurement by rejecting outliers with sigma clipping.
    """

    ConfigClass = CombineZernikesSigmaClipTaskConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the sigma from the config
        self.sigma = self.config.sigma

    def combineZernikes(self, zernikeArray):
        """
        Combine the Zernike coefficients with a sigma
        clipping algorithm. The sigma used for clipping
        is set in the configuration class for this task.

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
            The indices of the rows in the zernikeArray that were used
            in the final combination.
        """

        sigArray = sigma_clip(zernikeArray, sigma=self.sigma, axis=0)
        # Find which donuts have outlier values from the mask
        flagArray = np.sum(sigArray.mask, axis=1)
        # Create a binary flag array that only has a max of 1 instead
        # of the total number of masked values in a row
        binaryFlagArray = np.zeros(len(zernikeArray))
        binaryFlagArray[np.where(flagArray > 0.5)] = 1.0
        # Identify which rows to use when calculating final mean
        keepIdx = np.where(binaryFlagArray == 0)
        self.log.info(
            f"Using {len(keepIdx)} pairs out of {len(zernikeArray)} "
            "in final Zernike estimate."
        )

        return np.mean(zernikeArray[keepIdx], axis=0), binaryFlagArray
