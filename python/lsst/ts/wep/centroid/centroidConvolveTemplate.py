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

__all__ = ["CentroidConvolveTemplate"]

from copy import copy

import numpy as np
from lsst.ts.wep.centroid.centroidDefault import CentroidDefault
from lsst.ts.wep.centroid.centroidRandomWalk import CentroidRandomWalk
from scipy.signal import correlate
from sklearn.cluster import DBSCAN, KMeans


class CentroidConvolveTemplate(CentroidDefault):
    """CentroidDefault child class to get the centroid of donut by
    convolution with a template donut image."""

    def __init__(self) -> None:
        super(CentroidConvolveTemplate, self).__init__()
        self._centRandomWalk = CentroidRandomWalk()

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

        return self._centRandomWalk.getImgBinary(imgDonut)

    def getCenterAndR(
        self,
        imgDonut: np.ndarray,
        templateDonut: np.ndarray | None = None,
        peakThreshold: float = 0.95,
    ) -> tuple[float, float, float]:
        """Get the centroid data and effective weighting radius.

        Parameters
        ----------
        imgDonut : numpy.ndarray
            Donut image.
        templateDonut : None or numpy.ndarray, optional
            Template image for a single donut. If set to None
            then the image will be convolved with itself. (The Default is None)
        peakThreshold : float, optional
            This value is a specifies a number between 0 and 1 that is
            the fraction of the highest pixel value in the convolved image.
            The code then sets all pixels with a value below this to 0 before
            running the K-means algorithm to find peaks that represent possible
            donut locations. (The default is 0.95)

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

        if templateDonut is None:
            templateBinary = copy(imgBinary)
        else:
            templateBinary = self.getImgBinary(templateDonut)

        return self.getCenterAndRfromImgBinary(
            imgBinary,
            templateBinary=templateBinary,
            peakThreshold=peakThreshold,
        )

    def getCenterAndRfromImgBinary(
        self,
        imgBinary: np.ndarray,
        templateBinary: np.ndarray | None = None,
        peakThreshold: float = 0.95,
    ) -> tuple[float, float, float]:
        """Get the centroid data and effective weighting radius.

        Parameters
        ----------
        imgBinary : numpy.ndarray
            Binary image of donut.
        templateBinary : None or numpy.ndarray, optional
            Binary image of template for a single donut. If set to None
            then the image will be convolved with itself. (The Default is None)
        peakThreshold : float, optional
            This value is a specifies a number between 0 and 1 that is
            the fraction of the highest pixel value in the convolved image.
            The code then sets all pixels with a value below this to 0 before
            running the K-means algorithm to find peaks that represent possible
            donut locations. (The default is 0.95)

        Returns
        -------
        float
            Centroid x.
        float
            Centroid y.
        float
            Effective weighting radius.
        """

        x, y, radius = self.getCenterAndRfromTemplateConv(
            imgBinary,
            templateImgBinary=templateBinary,
            nDonuts=1,
            peakThreshold=peakThreshold,
        )

        return x[0], y[0], radius

    def getCenterAndRfromTemplateConv(
        self,
        imageBinary: np.ndarray,
        templateImgBinary: np.ndarray | None = None,
        nDonuts: int = 1,
        peakThreshold: float = 0.95,
        dbscanEps: float = 5.0,
    ) -> tuple[list[float], list[float], float]:
        """
        Get the centers of the donuts by convolving a binary template image
        with the binary image of the donut or donuts.

        Peaks will appear as bright spots in the convolved image. Since we
        use binary images the brightness of the stars does not matter and
        the peaks of any stars in the image should have about the same
        brightness if the template is correct.

        Parameters
        ----------
        imageBinary: numpy.ndarray
            Binary image of postage stamp.
        templateImgBinary: None or numpy.ndarray, optional
            Binary image of template donut. If set to None then the image
            is convolved with itself. (The default is None)
        nDonuts: int, optional
            Number of donuts there should be in the binary image. If the number
            is >= 1 then K-Means clustering will be used to return the
            specified number of donut centers. However, this can also be set to
            -1 if the number of donuts is unknown and it will perform DBSCAN
            clustering to find and return a set of donut centers.
            (The default is 1)
        peakThreshold: float, optional
            This value is a specifies a number between 0 and 1 that is
            the fraction of the highest pixel value in the convolved image.
            The code then sets all pixels with a value below this to 0 before
            running the K-means algorithm to find peaks that represent possible
            donut locations. (The default is 0.95)
        dbscanEps: float, optional
            Maximum distance scikit-learn DBSCAN algorithm allows "between two
            samples for one to considered in the neighborhood of the other".
            (The default is 5.0)

        Returns
        -------
        list
            X pixel coordinates for donut centroid.
        list
            Y pixel coordinates for donut centroid.
        float
            Effective weighting radius calculated using the template image.
        """

        if templateImgBinary is None:
            templateImgBinary = copy(imageBinary)

        nDonutsAssertStr = "nDonuts must be an integer >= 1 or -1"
        assert ((nDonuts >= 1) | (nDonuts == -1)) & (type(nDonuts) is int), nDonutsAssertStr
        # ruff: noqa

        # We set the mode to be "same" because we need to return the same
        # size image to the code.
        tempConvolve = correlate(imageBinary, templateImgBinary, mode="same")

        # Then we rank the pixel values keeping only those above
        # some fraction of the highest value.
        rankedConvolve = np.argsort(tempConvolve.flatten())[::-1]
        cutoff = len(np.where(tempConvolve.flatten() > peakThreshold * np.max(tempConvolve))[0])
        rankedConvolveCutoff = rankedConvolve[:cutoff]
        nx, ny = np.unravel_index(rankedConvolveCutoff, np.shape(imageBinary))

        # Donut centers lists
        centX = []
        centY = []

        if nDonuts >= 1:
            # Then to find peaks in the image we use K-Means with the
            # specified number of donuts
            kmeans = KMeans(n_clusters=nDonuts)
            labels = kmeans.fit_predict(np.array([nx, ny]).T)

            # Then in each cluster we take the brightest pixel as the centroid
            for labelNum in range(nDonuts):
                nxLabel, nyLabel = np.unravel_index(
                    rankedConvolveCutoff[labels == labelNum][0], np.shape(imageBinary)
                )
                centX.append(nxLabel)
                centY.append(nyLabel)
        elif nDonuts == -1:
            # Use DBSCAN to find clusters of points when the
            # number of donuts is unknown
            labels = DBSCAN(eps=dbscanEps).fit_predict(np.array([ny, nx]).T)

            # Save the centroid as the brightest pixel
            # within each identified cluster
            for labelNum in np.unique(labels):
                nxLabel, nyLabel = np.unravel_index(
                    rankedConvolveCutoff[labels == labelNum][0], np.shape(imageBinary)
                )
                centX.append(nxLabel)
                centY.append(nyLabel)

        # Get the radius of the donut from the template image
        radius = np.sqrt(np.sum(templateImgBinary) / np.pi)

        return centX, centY, radius
