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

__all__ = ["CentroidRandomWalk"]

import numpy as np
from lsst.ts.wep.centroid.centroidDefault import CentroidDefault


class CentroidRandomWalk(CentroidDefault):
    """CentroidDefault child class to get the centroid of donut by the
    random walk model."""

    def __init__(self) -> None:
        # Minimum effective signal
        self.minEffSignal = 1e-8

        # Number of bins in the histogram
        self.numOfBins = 256

        # Random seed
        self.seed = 1000

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

        imgBinary = np.zeros(imgDonut.shape, dtype=int)

        threshold = self._calcThreshold(imgDonut)
        imgBinary[imgDonut > max(self.minEffSignal, threshold)] = 1

        return imgBinary

    def _calcThreshold(self, imgDonut: np.ndarray) -> float:
        """Calculate the threshold to decide the effective signal.

        Parameters
        ----------
        imgDonut : numpy.ndarray
            Donut image to do the analysis.

        Returns
        -------
        float
            Threshold.
        """

        # Parameters to decide the signal of donut
        slide = int(0.1 * self.numOfBins)
        stepsize = int(0.06 * self.numOfBins)
        nwalk = int(1.56 * self.numOfBins)

        # Reshape the image to 1D array
        array1d = imgDonut.flatten()

        # Generate the histogram of intensity
        hist, binEdges = np.histogram(array1d, bins=self.numOfBins)

        # Parameters for random walk search
        start = int(self.numOfBins / 2.1)
        end = slide + 25  # Go back
        startidx = range(start, end, -15)

        foundvalley = False
        for istartPoint in range(len(startidx)):
            minind = startidx[istartPoint]

            # Check the condition of start index
            if (minind <= 0) or (max(hist[minind - 1 :]) == 0):
                continue
            minval = hist[minind - 1]

            # Do the random walk search
            np.random.seed(seed=self.seed)
            for ii in range(nwalk + 1):
                # if (minind <= slide):
                if minind >= slide:
                    # Find the index of bin that the count is not zero
                    while minval == 0:
                        minind = minind - 1
                        minval = hist[int(minind - 1)]

                    # Generate the thermal fluctuation based on the random
                    # table to give a random walk/ step with a random thermal
                    # fluctuation.
                    ind = np.round(stepsize * (2 * np.random.rand() - 1)).astype(int)
                    thermal = 1 + 0.5 * np.random.rand() * np.exp(
                        1.0 * ii / (nwalk * 0.3)
                    )

                    # Check the index of bin is whithin the range of histogram
                    if (minind + ind < 1) or (minind + ind > (self.numOfBins)):
                        continue

                    # Look for the minimum point
                    if hist[int(minind + ind - 1)] < (minval * thermal):
                        # Add the panality to go to the high intensity position
                        if ind > 0:
                            ind = int(ind / 3)
                        else:
                            ind = int(ind / 2)

                        # Update the value of minind
                        minval = hist[int(minind + ind - 1)]
                        minind = minind + ind

                else:
                    break

            # Find the signal of donut in histogram
            if minind >= slide:
                foundvalley = True
                break

        # Try to close the second peak
        while (minind >= slide) and (foundvalley is True):
            if np.abs(hist[int(minind - 5)] - hist[int(minind)]) < 4 * np.median(
                hist[len(hist) - 20 :]
            ):
                minind = minind - 1
            else:
                break

        # If no valley (signal) is found for the noise, use the value at start
        # index of histogram to be the threshold.
        if not foundvalley:
            minind = start

        # Get the threshold value of donut
        threshold = binEdges[int(minind)]

        return threshold
