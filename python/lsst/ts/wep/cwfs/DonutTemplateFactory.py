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

__all__ = ["DonutTemplateFactory"]

from lsst.ts.wep.Utility import DonutTemplateType
from lsst.ts.wep.cwfs.DonutTemplateModel import DonutTemplateModel
from lsst.ts.wep.cwfs.DonutTemplatePhosim import DonutTemplatePhosim


class DonutTemplateFactory(object):
    """
    Factory used to create template maker object to generate donut
    template images CentroidConvolveTemplate.
    """

    @staticmethod
    def createDonutTemplate(donutTemplateType):
        """Create the template maker object.

        Parameters
        ----------
        donutTemplateType : enum 'DonutTemplateType'
            Algorithm to create the donut templates.

        Returns
        -------
        Child class of DonutTemplateDefault
            Donut template object.

        Raises
        ------
        ValueError
            The template type is not supported.
        """

        if donutTemplateType == DonutTemplateType.Model:
            return DonutTemplateModel()
        elif donutTemplateType == DonutTemplateType.Phosim:
            return DonutTemplatePhosim()
        else:
            raise ValueError(f"The {donutTemplateType} is not supported.")
