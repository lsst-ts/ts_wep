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

__all__ = ["CentroidDefault"]

from typing import Any

import numpy as np
from scipy.ndimage import center_of_mass


class CentroidDefault(object):
    """Default Centroid class."""

    def getCenterAndR(self, imgDonut: np.ndarray, **kwargs: Any) -> tuple[float, float, float]:
        """Get the centroid data and effective weighting radius.

        Parameters
        ----------
        imgDonut : numpy.ndarray
            Donut image.
        **kwargs : dict[str, any]
            Dictionary of input argument: new value for that input argument.

        Returns
        -------
        float
            Centroid x.
        float
            Centroid y.
        float
            Effective weighting radius.
        """

        imgBinary = self.getImgBinary(imgDonut)

        return self.getCenterAndRfromImgBinary(imgBinary)

    def getCenterAndRfromImgBinary(self, imgBinary: np.ndarray, **kwargs: Any) -> tuple[float, float, float]:
        """Get the centroid data and effective weighting radius from the binary
        image.

        Parameters
        ----------
        imgBinary : numpy.ndarray [int]
            Binary image of donut.
        **kwargs : dict[str, any]
            Dictionary of input argument: new value for that input argument.

        Returns
        -------
        float
            Centroid x.
        float
            Centroid y.
        float
            Effective weighting radius.
        """

        y, x = center_of_mass(imgBinary)
        radius = np.sqrt(np.sum(imgBinary) / np.pi)

        return x, y, radius

    def getImgBinary(self, imgDonut: np.ndarray) -> np.ndarray:
        """Get the binary image.

        Parameters
        ----------
        imgDonut : numpy.ndarray
            Donut image to do the analysis.

        Returns
        -------
        numpy.ndarray [int]
            Binary image of donut.

        Raises
        ------
        NotImplementedError
            Child class should implement this.
        """

        raise NotImplementedError("Child class should implement this.")
