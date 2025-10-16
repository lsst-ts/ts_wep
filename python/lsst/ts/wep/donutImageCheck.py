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

__all__ = ["DonutImageCheck"]

import numpy as np
from scipy.stats import entropy


class DonutImageCheck(object):
    def __init__(self, numOfBins: int = 256, entroThres: float = 3.5, returnEntro: bool = False) -> None:
        """Donut image check class to judge the donut image is effective or
        not.

        Parameters
        ----------
        numOfBins : int, optional
            Number of bins in the histogram. (the default is 256.)
        entroThres : float, optional
            Threshold of entropy (the default is 3.5.)
        returnEntro: bool, optional
            Whether to return the calculated entropy value
            (the default is False).
        """

        # Number of bins in the histogram
        self.numOfBins = int(numOfBins)

        # Threshold of entropy
        self.entroThres = entroThres

        # Whether to return the calculated value
        self.returnEntro = returnEntro

    def isEffDonut(self, donutImg: np.ndarray) -> tuple[bool, float] | bool:
        """Is effective donut image or not.

        Parameters
        ----------
        donutImg : numpy.ndarray
            Donut image.

        Returns
        -------
        bool
            True if the donut image is effective.
        """

        array1d = donutImg.flatten()
        hist = np.histogram(array1d, bins=self.numOfBins)[0]

        # Square the distribution to magnify the difference in entropy
        imgEntropy = entropy(hist**2)
        if (imgEntropy < self.entroThres) and (imgEntropy != 0):
            eff = True
        else:
            eff = False

        # Return the actual entropy value if needed
        if self.returnEntro:
            return eff, imgEntropy
        else:
            return eff


if __name__ == "__main__":
    pass
