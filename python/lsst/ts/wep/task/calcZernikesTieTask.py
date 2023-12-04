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

__all__ = ["CalcZernikesTieTask"]

from lsst.ts.wep.estimation import TieAlgorithm
from lsst.ts.wep.task.calcZernikesBase import CalcZernikesBaseTask
from lsst.ts.wep.utils import WfAlgorithmName


class CalcZernikesTieTask(CalcZernikesBaseTask):
    """Calculate Zernikes from pairs of DonutStamps using the TIE algorithm."""

    _DefaultName = "calcZernikesTieTask"

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum."""
        return WfAlgorithmName.TIE

    @property
    def wfAlgo(self) -> TieAlgorithm:
        """Return the WfAlgorithm that is used for Zernike estimation."""
        return TieAlgorithm(configFile="policy/estimation/tie.yaml")
