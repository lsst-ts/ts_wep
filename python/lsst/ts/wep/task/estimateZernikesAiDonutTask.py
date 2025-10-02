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

__all__ = ["EstimateZernikesAiDonutConfig", "EstimateZernikesAiDonutTask"]

from typing import Callable, Iterable

import lsst.pex.config as pexConfig
from lsst.ts.wep.task.estimateZernikesBase import (
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)
from lsst.ts.wep.utils import WfAlgorithmName


class EstimateZernikesAiDonutConfig(EstimateZernikesBaseConfig):
    """AiDonut-specific configuration parameters for Zernike estimation."""

    modelPath: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="",
        doc="Path to the AiDonut model file.",
    )
    device: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="cpu",
        doc="Device to run the model on ('cpu' or 'cuda').",
    )


class EstimateZernikesAiDonutTask(EstimateZernikesBaseTask):
    """Estimate Zernike coefficients using the AiDonut algorithm."""

    ConfigClass = EstimateZernikesAiDonutConfig

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum."""
        return WfAlgorithmName.AiDonut

    def _applyToList(self, fun: Callable, args: Iterable, numCores: int) -> list:
        """Apply a function to a list of arguments, optionally in parallel.

        Overriding this method from the base class to avoid multiprocessing
        issues with AiDonut.

        Parameters
        ----------
        fun : Callable
            The function to apply. Must take a single argument.
        args : Iterable
            An iterable of arguments to apply the function to.
        numCores : int
            The number of cores to use. Must be 1 for AiDonut. If not 1,
            we will still run, but will raise a warning.

        Returns
        -------
        list
            A list of results from applying the function to the arguments.
        """
        if numCores != 1:
            self.log.warn(
                "AiDonut does not support multiprocessing. Running on a single core."
            )

        return [fun(arg) for arg in args]
