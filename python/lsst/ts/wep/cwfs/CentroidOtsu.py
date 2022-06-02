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

from skimage.filters import threshold_otsu

from lsst.ts.wep.cwfs.CentroidDefault import CentroidDefault


class CentroidOtsu(CentroidDefault):
    """CentroidDefault child class to get the centroid of donut by the
    Otsu's method."""

    def __init__(self):

        # Number of bins in the histogram
        self.numOfBins = 256

    def getImgBinary(self, imgDonut):
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
