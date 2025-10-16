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

__all__ = ["DonutDetector"]

from copy import copy

import numpy as np
import pandas as pd
from lsst.ts.wep.centroid import CentroidFindFactory
from lsst.ts.wep.deblend.deblendAdapt import DeblendAdapt
from lsst.ts.wep.utils import CentroidFindType
from scipy.spatial.distance import cdist


class DonutDetector(object):
    """
    Class to detect donuts directly from an out of focus image
    by convolution with a template image.
    """

    def detectDonuts(
        self,
        expArray: np.ndarray,
        template: np.ndarray,
        blendRadius: float,
        peakThreshold: float = 0.95,
        dbscanEps: int = 5,
        binaryChoice: str = "centroid",
    ) -> pd.DataFrame:
        """
        Detect and categorize donut sources as blended/unblended

        Parameters
        ----------
        expArray: numpy.ndarray
            The input image data.
        template: numpy.ndarray
            Donut template appropriate for the image.
        blendRadius: float
            Minimum distance in pixels two donut centers need to
            be apart in order to be tagged as unblended.
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
        binaryChoice: str, optional
            Indicates choice of how to arrive at the binary image passed to
            centroidFinder. Select 'centroid' to use getImgBinary method from
            centroidFinder, use 'deblend' for adaptative image thresholding,
            use 'exposure' to pass an unchanged copy of the exposure array.
            (The default is 'centroid')

        Returns
        -------
        pandas.DataFrame
            Dataframe identifying donut positions and if they
            are blended with other donuts. If blended also identfies
            which donuts are blended with which.

        Raises
        ------
        ValueError
            binaryChoice is not supported.
        """

        centroidFinder = CentroidFindFactory.createCentroidFind(CentroidFindType.ConvolveTemplate)
        if binaryChoice == "centroid":
            binaryExp = centroidFinder.getImgBinary(copy(expArray))

        elif binaryChoice == "deblend":
            deblend = DeblendAdapt()
            binaryExp = deblend._getImgBinaryAdapt(copy(expArray))

        elif binaryChoice == "exposure":
            binaryExp = copy(expArray)

        else:
            raise ValueError(f"binaryChoice {binaryChoice} is not supported.")

        centroidX, centroidY, donutRad = centroidFinder.getCenterAndRfromTemplateConv(
            binaryExp,
            templateImgBinary=template,
            nDonuts=-1,
            peakThreshold=peakThreshold,
            dbscanEps=dbscanEps,
        )

        donutDf = pd.DataFrame(np.array([centroidX, centroidY]).T, columns=["x_center", "y_center"])
        donutDf = self.identifyBlendedDonuts(donutDf, blendRadius)

        return donutDf

    def identifyBlendedDonuts(self, donutDf: pd.DataFrame, blendRadius: float) -> pd.DataFrame:
        """
        Label donuts as blended/unblended if the centroids are within
        the blendRadius number of pixels.

        Parameters
        ----------
        donutDf: pandas.DataFrame
            Dataframe identifying donut positions with labels
            'x_center' and 'y_center'.
        blendRadius: float
            Minimum distance in pixels two donut centers need to
            be apart in order to be tagged as unblended.

        Returns
        -------
        pandas.DataFrame
            Dataframe identifying donut positions and if they
            are blended with other donuts. If blended also identfies
            which donuts are blended with which.
        """

        # Add blend_center columns
        donutDf["x_blend_center"] = [list() for x in range(len(donutDf))]
        donutDf["y_blend_center"] = [list() for y in range(len(donutDf))]

        # Find distances between each pair of objects
        donutCenters = [donutDf["x_center"].values, donutDf["y_center"].values]
        donutCenters = np.array(donutCenters).T
        distMatrix = cdist(donutCenters, donutCenters)
        # Don't need repeats of each pair
        distMatrixUpper = np.triu(distMatrix)

        # Identify blended pairs of objects by distance
        blendedPairs = np.array(np.where((distMatrixUpper > 0.0) & (distMatrixUpper < blendRadius))).T
        blendedCenters = np.unique(blendedPairs.flatten())

        # Add blended information into dataframe
        donutDf["blended"] = False
        donutDf.loc[blendedCenters, "blended"] = True
        donutDf["blended_with"] = None
        for donutOne, donutTwo in blendedPairs:
            if donutDf.loc[donutOne, "blended_with"] is None:
                donutDf.at[donutOne, "blended_with"] = []
            if donutDf.loc[donutTwo, "blended_with"] is None:
                donutDf.at[donutTwo, "blended_with"] = []
            donutDf.loc[donutOne, "blended_with"].append(donutTwo)
            donutDf.loc[donutOne, "x_blend_center"].append(donutDf.loc[donutTwo, "x_center"])
            donutDf.loc[donutOne, "y_blend_center"].append(donutDf.loc[donutTwo, "y_center"])
            donutDf.loc[donutTwo, "blended_with"].append(donutOne)
            donutDf.loc[donutTwo, "x_blend_center"].append(donutDf.loc[donutOne, "x_center"])
            donutDf.loc[donutTwo, "y_blend_center"].append(donutDf.loc[donutOne, "y_center"])

        # Count the number of other donuts overlapping
        # each donut
        donutDf["num_blended_neighbors"] = 0
        for donutIdx in range(len(donutDf)):
            if donutDf["blended_with"].iloc[donutIdx] is None:
                continue

            donutDf.at[donutIdx, "num_blended_neighbors"] = len(donutDf["blended_with"].loc[donutIdx])

        return donutDf
