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

__all__ = ["WfAlgorithmFactory"]

from typing import Union

from lsst.ts.wep.estimation.aiDonut import AiDonutAlgorithm
from lsst.ts.wep.estimation.danish import DanishAlgorithm
from lsst.ts.wep.estimation.tie import TieAlgorithm
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.utils import WfAlgorithmName, configClass


class WfAlgorithmFactory:
    """Factory for loading different wavefront estimation algorithms."""

    @staticmethod
    def createWfAlgorithm(
        algoName: Union[WfAlgorithmName, str],
        algoConfig: Union[dict, WfAlgorithm, None] = None,
    ) -> WfAlgorithm:
        """Return a configured WfAlgorithm.

        Parameters
        ----------
        algoName : WfAlgorithmName or str
            A WfAlgorithmName enum or the corresponding string, indicating
            which WfAlgorithm to use.
        algoConfig : str or dict or WfAlgorithm, optional
            Algorithm configuration. If a string, it is assumed this points
            to a config file, which is used to configure the algorithm. If the
            path begins with "policy:", then it is assumed the path is relative
            to the policy directory. If a dictionary, it is assumed to hold
            keywords for configuration. If a WfAlgorithm object, that object is
            just used. If None, the algorithm defaults are used.
            (the default is None)
        """
        # Convert to enum
        algoName = WfAlgorithmName(algoName)

        # Return the configured algorithm
        if algoName == WfAlgorithmName.TIE:
            return configClass(algoConfig, TieAlgorithm)
        elif algoName == WfAlgorithmName.Danish:
            return configClass(algoConfig, DanishAlgorithm)
        elif algoName == WfAlgorithmName.AiDonut:
            return configClass(algoConfig, AiDonutAlgorithm)
        else:
            raise ValueError(f"{algoName} not supported.")
