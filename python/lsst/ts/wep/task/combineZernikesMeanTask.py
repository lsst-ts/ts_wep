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

__all__ = ["CombineZernikesMeanTask"]

import numpy as np
from lsst.ts.wep.task.combineZernikesBase import CombineZernikesBaseTask
from astropy.table import Table


class CombineZernikesMeanTask(CombineZernikesBaseTask):
    """
    Combine the raw Zernike measurements into an average
    measurement with an unweighted mean.
    """

    def combineZernikes(self, zkTable: Table) -> Table:
        # Set all donuts to used (no rejection in this simple mean)
        zkTable["used"] = True

        # Calculate mean of every Zernike column
        for colName in zkTable.colnames:
            if colName.startswith("Z"):
                zkTable[colName][0] = np.mean(zkTable[colName][1:])

        return zkTable
