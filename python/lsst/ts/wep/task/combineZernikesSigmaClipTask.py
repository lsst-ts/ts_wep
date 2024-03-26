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

__all__ = ["CombineZernikesSigmaClipTaskConfig", "CombineZernikesSigmaClipTask"]

import lsst.pex.config as pexConfig
import numpy as np
from lsst.ts.wep.task.combineZernikesBase import (
    CombineZernikesBaseConfig,
    CombineZernikesBaseTask,
)
from lsst.ts.wep.utils import conditionalSigmaClip


class CombineZernikesSigmaClipTaskConfig(CombineZernikesBaseConfig):
    """
    Configuration for combining Zernike coefficients with sigma clipping.
    """

    sigma = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Number of standard deviations for the clipping limit.",
    )
    stdMin = pexConfig.Field(
        dtype=float,
        default=0.005,
        doc="Minimum threshold for clipping the standard deviation in um.",
    )
    maxZernClip = pexConfig.Field(
        dtype=int,
        default=3,
        doc="""When looking for outliers to clip. Check the first
        maxZernClip Zernike coefficients.
        This is to prevent clipping based upon the highest Zernikes which
        have smaller values and more variation from noise.""",
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
        self.stdMin = self.config.stdMin
        self.maxZernClip = self.config.maxZernClip

    def combineZernikes(self, zernikeArray):
        sigArray = conditionalSigmaClip(
            zernikeArray, sigma=self.sigma, stdMin=self.stdMin, stdFunc="mad_std"
        )
        # Create a binary flag array that indicates
        # donuts have outlier values. This array is 1 if
        # it has any outlier values.
        binaryFlagArray = np.any(
            np.isnan(sigArray[:, : self.maxZernClip]), axis=1
        ).astype(int)
        # Identify which rows to use when calculating final mean
        keepIdx = ~np.any(np.isnan(sigArray), axis=1)

        return np.mean(zernikeArray[keepIdx], axis=0), binaryFlagArray
