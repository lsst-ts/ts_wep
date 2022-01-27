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
import astropy.units as u
from sklearn.neighbors import NearestNeighbors

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms.sourceSelector import (
    _getFieldFromCatalog,
)


class DonutSourceSelectorTaskConfig(pexConfig.Config):

    xCoordField = pexConfig.Field(
        dtype=str, default="centroid_x", doc="Name of x-coordinate column."
    )
    yCoordField = pexConfig.Field(
        dtype=str, default="centroid_y", doc="Name of y-coordinate column."
    )
    fluxField = pexConfig.Field(
        dtype=str, default="flux", doc="Name of the source flux field to use."
    )
    donutRadius = pexConfig.Field(
        dtype=float, default=63, doc="Radius of the defocal donuts in pixels."
    )
    isoMagDiff = pexConfig.Field(
        dtype=float,
        default=2,
        doc="Min. difference in magnitude for 'isolated' star.",
    )
    sourceLimit = pexConfig.Field(
        dtype=int,
        default=-1,
        doc="Maximum number of desired sources (default is -1 which will give all in catalog).",
    )
    maxBlended = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Number of blended objects (defined by donutRadius and isoMagDiff) allowed with a bright source.",
    )


class DonutSourceSelectorTask(pipeBase.Task):
    """
    Donut Source Selector that uses a nearest neighbors radius
    query to find all donuts within the pixel radius set in the
    config. Then it goes from the brightest sources down to the faintest
    picking donuts that are at least isoMagDiff brighter than any sources
    with centers within 2 times the donutRadius until reaching numSources
    kept or going through the whole list.
    """

    ConfigClass = DonutSourceSelectorTaskConfig
    _DefaultName = "donutSourceSelectorTask"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

    def run(self, sourceCat, bbox):
        """Select sources and return them.

        Parameters
        ----------
        sourceCat : `lsst.afw.table.SourceCatalog` or `pandas.DataFrame`
                    or `astropy.table.Table`
            Catalog of sources to select from.
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Box which bounds a region in pixel space.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
            - sourceCat : `lsst.afw.table.SourceCatalog` or `pandas.DataFrame`
                          or `astropy.table.Table`
                The catalog of sources that were selected.
                (may not be memory-contiguous)
            - selected : `numpy.ndarray` of `bool`
                Boolean array of sources that were selected, same length as
                sourceCat.
        Raises
        ------
        RuntimeError
            Raised if ``sourceCat`` is not contiguous.
        """
        if hasattr(sourceCat, "isContiguous"):
            # Check for continuity on afwTable catalogs
            if not sourceCat.isContiguous():
                raise RuntimeError(
                    "Input catalogs for source selection must be contiguous."
                )

        result = self.selectSources(sourceCat, bbox)

        return pipeBase.Struct(
            sourceCat=sourceCat[result.selected], selected=result.selected
        )

    @pipeBase.timeMethod
    def selectSources(self, sourceCat, bbox):
        """
        Run the source selection algorithm and return the indices to keep
        in the original catalog.

        Parameters
        ----------
        sourceCat : `lsst.afw.table.SourceCatalog` or `pandas.DataFrame`
                    or `astropy.table.Table`
            Catalog of sources to select from.
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Box which bounds a region in pixel space.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
            - selected : `numpy.ndarray` of `bool``
                Boolean array of sources that were selected, same length as
                sourceCat.

        Raises
        ------
        ValueError
            sourceLimit in config for task must be -1 or a positive integer.
        """

        donutRadius = self.config.donutRadius

        selected = np.zeros(len(sourceCat), dtype=bool)

        if len(selected) == 0:
            return pipeBase.Struct(selected=selected)

        xCoord = _getFieldFromCatalog(sourceCat, self.config.xCoordField)
        yCoord = _getFieldFromCatalog(sourceCat, self.config.yCoordField)
        flux = _getFieldFromCatalog(sourceCat, self.config.fluxField)
        mag = (flux * u.nJy).to_value(u.ABmag)
        df = pd.DataFrame({"x": xCoord, "y": yCoord, "mag": mag})
        # Grab any donut centers within 2 times the donut radius.
        xyNeigh = NearestNeighbors(radius=2 * donutRadius)
        minMagDiff = self.config.isoMagDiff

        # Remove area too close to edge with new bounding box
        # that allows only area at least one donut radius from edges
        trimmedBBox = bbox.erodedBy(int(np.ceil(donutRadius)))

        index = []
        magSortedDf = df.sort_values("mag")
        groupIndices = magSortedDf.index.values
        xyNeigh.fit(magSortedDf[["x", "y"]])
        radDist, radIdx = xyNeigh.radius_neighbors(
            magSortedDf[["x", "y"]], sort_results=True
        )

        errMsg = str(
            "config.sourceLimit must be a positive integer "
            + "or turned off by setting it to '-1'"
        )
        if not ((self.config.sourceLimit == -1) or (self.config.sourceLimit > 0)):
            raise ValueError(errMsg)

        maxBlended = self.config.maxBlended
        sourcesKept = 0
        # Go through catalog with nearest neighbor information
        # and keep sources that match our configuration settings
        for srcOn, idxList in list(enumerate(radIdx)):
            # Move on if source is within donutRadius
            # of the edge of a given exposure
            srcX = magSortedDf["x"].iloc[srcOn]
            srcY = magSortedDf["y"].iloc[srcOn]
            if trimmedBBox.contains(srcX, srcY) is False:
                continue

            # If there is no overlapping source keep
            # the source and move on to next
            if len(idxList) == 1:
                index.append(groupIndices[srcOn])
                sourcesKept += 1
            else:
                # In this case there is at least one overlapping source
                srcMag = magSortedDf["mag"].iloc[srcOn]
                magDiff = magSortedDf["mag"].iloc[idxList[1:]] - srcMag
                # If this is the fainter source of the overlaps move on
                if np.min(magDiff) < 0.0:
                    continue
                # If this source overlaps but is brighter than all its
                # overlapping sources by minMagDiff then keep it
                elif (maxBlended == 0) and (np.min(magDiff) >= minMagDiff):
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                # If the number of overlapping sources is less than or equal to
                # maxBlended then keep this source
                elif len(magDiff) <= maxBlended:
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                # Keep the source if it is blended with up to maxBlended
                # number of sources. To check this we look at the maxBlended+1
                # source in the magDiff list and check that the object
                # is at least minMagDiff brighter than this. Satisfying this
                # criterion means it is blended with maxBlended
                # or fewer sources.
                elif np.partition(magDiff, maxBlended)[maxBlended] > minMagDiff:
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                else:
                    continue

            if (self.config.sourceLimit > 0) and (
                sourcesKept == self.config.sourceLimit
            ):
                break

        selected[index] = True

        return pipeBase.Struct(selected=selected)
