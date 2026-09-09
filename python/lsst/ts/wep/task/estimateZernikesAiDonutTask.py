# This file is part of ts_wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

__all__ = ["EstimateZernikesAiDonutConfig", "EstimateZernikesAiDonutTask"]

import os
from typing import Any, Callable, Iterable

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.estimateZernikesBase import (
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)
from lsst.ts.wep.utils import WfAlgorithmName, computeSha256


class EstimateZernikesAiDonutConfig(EstimateZernikesBaseConfig):
    """AiDonut-specific configuration parameters for Zernike estimation."""

    modelPath: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="",
        doc="Path to the AiDonut model file.",
    )
    modelSha256: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="",
        doc="Expected SHA-256 hex digest of the AiDonut model file. If set, "
        "the model file is verified against this digest before loading and a "
        "RuntimeError is raised on mismatch. If empty (default) the check is "
        "skipped. Pins an exact model version and catches unfetched git-lfs "
        "pointer stubs.",
    )
    device: pexConfig.Field = pexConfig.Field(
        dtype=str,
        default="cpu",
        doc="Device to run the model on ('cpu' or 'cuda').",
    )
    temperature: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.005,
        doc="Temperature for softmax weighting of predictions based on model "
        "uncertainty. Lower values put greater weight on lower-uncertainty "
        "predictions.",
    )


class EstimateZernikesAiDonutTask(EstimateZernikesBaseTask):
    """Estimate Zernike coefficients using the AiDonut algorithm."""

    ConfigClass = EstimateZernikesAiDonutConfig

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum."""
        return WfAlgorithmName.AiDonut

    def _recordModelChecksum(self) -> None:
        """Record the loaded model checksum in the task metadata.

        Writes ``modelChecksums`` as a ``"<basename>=<sha256>"`` string so an
        on-sky run can be traced back to the exact model version used, via the
        ts_aos_ai ``model_history.yaml`` ledger. This mirrors the provenance
        recorded by ``CalcZernikesNeuralTask`` for TARTS. No-op if no model
        file is configured (the algorithm raises its own error if the path is
        set but missing).
        """
        if not self.config.modelPath:
            return
        modelPath = os.path.expandvars(self.config.modelPath)
        if not os.path.isfile(modelPath):
            return
        self.metadata["modelChecksums"] = f"{os.path.basename(modelPath)}={computeSha256(modelPath)}"

    def run(
        self,
        donutStampsExtra: Any,
        donutStampsIntra: Any,
        numCores: int = 1,
    ) -> pipeBase.Struct:
        """Record model provenance, then estimate Zernikes via base task."""
        self._recordModelChecksum()
        return super().run(donutStampsExtra, donutStampsIntra, numCores=numCores)

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
            self.log.warn("AiDonut does not support multiprocessing. Running on a single core.")

        return [fun(arg) for arg in args]
