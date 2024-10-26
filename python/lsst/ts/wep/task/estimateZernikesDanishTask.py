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

__all__ = ["EstimateZernikesDanishConfig", "EstimateZernikesDanishTask"]

import lsst.pex.config as pexConfig
from lsst.ts.wep.task.estimateZernikesBase import (
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)
from lsst.ts.wep.utils import WfAlgorithmName


class EstimateZernikesDanishConfig(EstimateZernikesBaseConfig):
    """Danish-specific configuration parameters for Zernike estimation."""

    lstsqKwargs = pexConfig.DictField(
        keytype=str,
        default=dict(),
        doc="A dictionary containing any of the keyword arguments for "
        + "scipy.optimize.least_squares, except `fun`, `x0`, `jac`, or `args`.",
    )


class EstimateZernikesDanishTask(EstimateZernikesBaseTask):
    """Estimate Zernike coefficients using the TIE algorithm."""

    ConfigClass = EstimateZernikesDanishConfig

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum."""
        return WfAlgorithmName.Danish
