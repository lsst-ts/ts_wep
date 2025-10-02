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

__all__ = ["DeblendDonutFactory"]

from lsst.ts.wep.deblend.deblendAdapt import DeblendAdapt
from lsst.ts.wep.utils import DeblendDonutType


class DeblendDonutFactory(object):
    """Factory for creating the deblend donut object to deblend the bright star
    donut from neighboring stars."""

    @staticmethod
    def createDeblendDonut(deblendDonutType: DeblendDonutType) -> DeblendAdapt:
        """Create the concrete deblend donut object.

        Parameters
        ----------
        deblendDonutType : enum 'DeblendDonutType'
            Algorithm to deblend the donut.

        Returns
        -------
        DeblendAdapt
            Deblend donut object.

        Raises
        ------
        ValueError
            The deblend donut type is not supported.
        """

        if deblendDonutType == DeblendDonutType.Adapt:
            return DeblendAdapt()
        else:
            raise ValueError("The %s is not supported." % deblendDonutType)
