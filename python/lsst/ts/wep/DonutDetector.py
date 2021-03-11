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

import numpy as np
import pandas as pd
from copy import copy


from lsst.ts.wep.Utility import CentroidFindType
from lsst.ts.wep.cwfs.CentroidFindFactory import CentroidFindFactory
from scipy.spatial.distance import cdist


class DonutDetector(object):

    """Class to detect donuts directly from an out of focus image."""

    def detectDonuts(
        self, expArray, template, blendRadius, peakThreshold=0.95, dbscanEps=5
    ):

        """
        Detect and categorize donut sources as blended/unblended

        Parameters
        -------
        expArray: numpy ndarray
            The input image data
        template: numpy ndarray
            Donut template appropriate for the image
        blendRadius: float
            Minimum distance in pixels two donut centers need to
            be apart in order to be tagged as unblended
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
        pandas dataframe
            Dataframe identifying donut positions and if they
            are blended with other donuts. If blended also identfies
            which donuts are blended with which.
        """

        centroidFinder = CentroidFindFactory.createCentroidFind(
            CentroidFindType.ConvolveTemplate
        )
        binaryExp = centroidFinder.getImgBinary(copy(expArray))
        centroidX, centroidY, donutRad = centroidFinder.getCenterAndRfromTemplateConv(
            binaryExp,
            templateImgBinary=template,
            nDonuts=-1,
            peakThreshold=peakThreshold,
            dbscanEps=dbscanEps,
        )

        donutDf = pd.DataFrame(
            np.array([centroidX, centroidY]).T, columns=["x_center", "y_center"]
        )
        donutDf = self.labelUnblended(donutDf, blendRadius)

        return donutDf

    def labelUnblended(self, donutDf, blendRadius):

        """
        Label donuts as blended/unblended if the centroids are within
        the blendRadius number of pixels.

        Parameters
        ----------
        donutDf: pandas dataframe
            Dataframe identifying donut positions with labels
            'x_center' and 'y_center'.
        blendRadius: float
            Minimum distance in pixels two donut centers need to
            be apart in order to be tagged as unblended

        Returns
        -------
        pandas dataframe
            Dataframe identifying donut positions and if they
            are blended with other donuts. If blended also identfies
            which donuts are blended with which.
        """

        # Find distances between each pair of objects
        donutCenters = [donutDf["x_center"].values, donutDf["y_center"].values]
        donutCenters = np.array(donutCenters).T
        distMatrix = cdist(donutCenters, donutCenters)
        # Don't need repeats of each pair
        distMatrixUpper = np.triu(distMatrix)

        # Identify blended pairs of objects by distance
        blendedPairs = np.array(
            np.where((distMatrixUpper > 0.0) & (distMatrixUpper < blendRadius))
        ).T
        blendedCenters = np.unique(blendedPairs.flatten())

        # Add blended information into dataframe
        donutDf["blended"] = False
        donutDf.loc[blendedCenters, "blended"] = True
        donutDf["blended_with"] = None
        for i, j in blendedPairs:
            if donutDf.loc[i, "blended_with"] is None:
                donutDf.at[i, "blended_with"] = []
            if donutDf.loc[j, "blended_with"] is None:
                donutDf.at[j, "blended_with"] = []
            donutDf.loc[i, "blended_with"].append(j)
            donutDf.loc[j, "blended_with"].append(i)

        # Count the number of other donuts overlapping
        # each donut
        donutDf["num_blended_neighbors"] = 0
        for i in range(len(donutDf)):
            if donutDf["blended_with"].iloc[i] is None:
                continue

            donutDf.at[i, "num_blended_neighbors"] = len(donutDf["blended_with"].loc[i])

        return donutDf
