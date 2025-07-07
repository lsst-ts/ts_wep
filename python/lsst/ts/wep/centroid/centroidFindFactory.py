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

__all__ = ["CentroidFindFactory"]

from lsst.ts.wep.centroid.centroidConvolveTemplate import CentroidConvolveTemplate
from lsst.ts.wep.centroid.centroidOtsu import CentroidOtsu
from lsst.ts.wep.centroid.centroidRandomWalk import CentroidRandomWalk
from lsst.ts.wep.utils import CentroidFindType


class CentroidFindFactory(object):
    """Factory for creating the centroid find object to calculate the centroid
    of donut."""

    @staticmethod
    def createCentroidFind(
        centroidFindType: CentroidFindType,
    ) -> CentroidConvolveTemplate | CentroidOtsu | CentroidRandomWalk:
        """Create the centroid find object.

        Parameters
        ----------
        centroidFindType : enum 'CentroidFindType'
            Algorithm to find the centroid.

        Returns
        -------
        Child class of centroidDefault
            Centroid find object.

        Raises
        ------
        ValueError
            The centroid find type is not supported.
        """

        if centroidFindType == CentroidFindType.RandomWalk:
            return CentroidRandomWalk()
        elif centroidFindType == CentroidFindType.Otsu:
            return CentroidOtsu()
        elif centroidFindType == CentroidFindType.ConvolveTemplate:
            return CentroidConvolveTemplate()
        else:
            raise ValueError("The %s is not supported." % centroidFindType)
