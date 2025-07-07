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

__all__ = ["CentroidOtsu"]

import numpy as np
from lsst.ts.wep.centroid.centroidDefault import CentroidDefault
from skimage.filters import threshold_otsu


class CentroidOtsu(CentroidDefault):
    """CentroidDefault child class to get the centroid of donut by the
    Otsu's method."""

    def __init__(self) -> None:
        # Number of bins in the histogram
        self.numOfBins = 256

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
        """

        threshold = threshold_otsu(imgDonut, nbins=self.numOfBins)
        imgBinary = (imgDonut > threshold).astype(int)

        return imgBinary
