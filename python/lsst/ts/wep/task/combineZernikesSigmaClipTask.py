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

from typing import Any

import lsst.pex.config as pexConfig
import numpy as np
from lsst.ts.wep.task.combineZernikesBase import (
    CombineZernikesBaseConfig,
    CombineZernikesBaseTask,
)
from lsst.ts.wep.utils import conditionalSigmaClip
from astropy.table import Table


class CombineZernikesSigmaClipTaskConfig(CombineZernikesBaseConfig):
    """
    Configuration for combining Zernike coefficients with sigma clipping.
    """

    sigmaClipKwargs: pexConfig.DictField = pexConfig.DictField(
        keytype=str, default=dict(), doc="Arguments for astropy.stats.sigma_clip."
    )
    stdMin: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.005,
        doc="Minimum threshold for clipping the standard deviation in um.",
    )
    maxZernClip: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=3,
        doc="""When looking for outliers to clip. Check the first
        maxZernClip Zernike coefficients.
        This is to prevent clipping based upon the highest Zernikes which
        have smaller values and more variation from noise.""",
    )
    zkClipType: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="deviation",
        doc="""Which type of Zernikes to perform clipping on. Options are:
        'opd', 'intrinsic', 'deviation'.""",
    )


class CombineZernikesSigmaClipTask(CombineZernikesBaseTask):
    """
    Combine the raw Zernike measurements into an average
    measurement by rejecting outliers with sigma clipping.
    """

    ConfigClass = CombineZernikesSigmaClipTaskConfig
    config: CombineZernikesSigmaClipTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set the sigma_clip settings from the config
        self.sigmaClipKwargs = {"sigma": 3.0, "maxiters": 1, "stdfunc": "mad_std"}
        for key, val in self.config.sigmaClipKwargs.items():
            self.sigmaClipKwargs[key] = val
        self.maxZernClip = self.config.maxZernClip
        self.stdMin = self.config.stdMin
        self.zkClipType = self.config.zkClipType

    def _combineZernikes(self, zkTable: Table) -> Table:
        # Extract the array of Zernikes to perform clipping on
        if self.zkClipType == "opd":
            columns = zkTable.meta["opd_columns"]
        elif self.zkClipType == "intrinsic":
            columns = zkTable.meta["intrinsic_columns"]
        elif self.zkClipType == "deviation":
            columns = zkTable.meta["deviation_columns"]
        else:
            raise ValueError(f"Unknown zkClipType: {self.zkClipType}")
        subTable = zkTable[zkTable["label"] != "average"][columns]
        zernikeArray = np.array([subTable[col] for col in columns]).T

        # Perform conditional sigma clipping
        sigArray = conditionalSigmaClip(
            zernikeArray, sigmaClipKwargs=self.sigmaClipKwargs, stdMin=self.stdMin
        )
        # Create a binary flag array that indicates donuts have outlier values.
        # This array is 1 if it has any outlier values. If all available donuts
        # have a clipped value in the first maxZernClip coefficients then
        # reduce maxZernClip by 1 until we get one that passes.
        numRejected = len(sigArray)
        effMaxZernClip = self.maxZernClip + 1

        while numRejected == len(sigArray):
            effMaxZernClip -= 1
            binaryFlagArray = np.any(np.isnan(sigArray[:, :effMaxZernClip]), axis=1).astype(int)
            numRejected = np.sum(binaryFlagArray)

        # Identify which rows to use when calculating final mean
        keepIdx = ~np.array(binaryFlagArray, dtype=bool)
        used = zkTable["used"][zkTable["label"] != "average"]
        used[keepIdx] = True
        zkTable["used"][zkTable["label"] != "average"] = used

        self.log.info(f"MaxZernClip config: {self.maxZernClip}. MaxZernClip used: {effMaxZernClip}.")
        if effMaxZernClip < self.maxZernClip:
            self.log.warning(
                f"EffMaxZernClip ({effMaxZernClip}) was less than MaxZernClip config ({self.maxZernClip})."
            )
        self.metadata["maxZernClip"] = self.maxZernClip
        self.metadata["effMaxZernClip"] = effMaxZernClip

        # Calculate means
        for j in zkTable.meta["noll_indices"]:
            self._setAvg(zkTable, f"Z{j}", np.nanmean, useIdx=keepIdx)
            self._setAvg(zkTable, f"Z{j}_intrinsic", np.nanmean, useIdx=keepIdx)
            self._setAvg(zkTable, f"Z{j}_deviation", np.nanmean, useIdx=keepIdx)

        return zkTable
