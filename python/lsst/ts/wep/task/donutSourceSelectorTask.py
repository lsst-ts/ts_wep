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

__all__ = ["DonutSourceSelectorTaskConfig", "DonutSourceSelectorTask"]

from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.neighbors import NearestNeighbors

import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Detector
from lsst.meas.algorithms.sourceSelector import _getFieldFromCatalog
from lsst.ts.wep.utils import readConfigYaml
from lsst.utils.timer import timeMethod


class DonutSourceSelectorTaskConfig(pexConfig.Config):
    xCoordField: pexConfig.Field = pexConfig.Field(
        dtype=str, default="centroid_x", doc="Name of x-coordinate column."
    )
    yCoordField: pexConfig.Field = pexConfig.Field(
        dtype=str, default="centroid_y", doc="Name of y-coordinate column."
    )
    useCustomMagLimit: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply user-defined magnitude limit? If this is False then the code"
        + " will default to use the magnitude values in policy:magLimitStar.yaml.",
    )
    magMax: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=99.0,
        doc="Maximum magnitude for selection. Only used if useCustomMagLimit is True.",
    )
    magMin: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=-99.0,
        doc="Minimum magnitude for selection. Only used if useCustomMagLimit is True.",
    )
    # For information on where this default maxFieldDist comes from see details
    # in ts_analysis_notebooks/aos/vignetting.
    maxFieldDist: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=1.808,
        doc="Maximum distance from center of focal plane (in degrees).",
    )
    unblendedSeparation: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=160,
        doc="Distance in pixels between two donut centers for them to be considered unblended. "
        + "This setting and minBlendedSeparation will both be affected by the defocal distance.",
    )
    minBlendedSeparation: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=120,
        doc="Minimum separation in pixels between blended donut centers. "
        + "This setting and unblendedSeparation will both be affected by the defocal distance.",
    )
    isolatedMagDiff: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=2,
        doc="Min. difference in magnitude for 'isolated' star.",
    )
    sourceLimit: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=-1,
        doc="Maximum number of desired sources (default is -1 which will give all in catalog).",
    )
    maxBlended: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Number of blended objects (defined by unblendedSeparation and isolatedMagDiff) "
        + "allowed with a bright source.",
    )


class DonutSourceSelectorTask(pipeBase.Task):
    """
    Donut Source Selector that uses a nearest neighbors radius
    query to find all donuts within the pixel radius set in the
    config. Then it goes from the brightest sources down to the faintest
    picking donuts that are at least isolatedMagDiff brighter than any sources
    with centers within 2 times the unblendedSeparation until reaching
    numSources kept or going through the whole list.
    """

    ConfigClass = DonutSourceSelectorTaskConfig
    _DefaultName = "donutSourceSelectorTask"
    config: DonutSourceSelectorTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def run(
        self,
        sourceCat: lsst.afw.table.SourceCatalog,
        detector: Detector,
        filterName: str,
    ) -> pipeBase.Struct:
        """Select sources and return them.

        Parameters
        ----------
        sourceCat : `lsst.afw.table.SourceCatalog` or `pandas.DataFrame`
        or `astropy.table.Table`
            Catalog of sources to select from.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.
        filterName : `str`
            Name of camera filter.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - sourceCat : `lsst.afw.table.SourceCatalog`
                or `pandas.DataFrame` or `astropy.table.Table`
                    The catalog of sources that were selected.
                    (may not be memory-contiguous)
                - selected : `numpy.ndarray` of `bool`
                    Boolean array of sources that were selected, same length as
                    sourceCat.

        Raises
        ------
        `RuntimeError`
            Raised if ``sourceCat`` is not contiguous.
        """
        if hasattr(sourceCat, "isContiguous"):
            # Check for continuity on afwTable catalogs
            if not sourceCat.isContiguous():
                raise RuntimeError("Input catalogs for source selection must be contiguous.")

        result = self.selectSources(sourceCat, detector, filterName)

        return pipeBase.Struct(
            sourceCat=sourceCat[result.selected],
            selected=result.selected,
            blendCentersX=result.blendCentersX,
            blendCentersY=result.blendCentersY,
        )

    @timeMethod
    def selectSources(
        self,
        sourceCat: lsst.afw.table.SourceCatalog | pd.DataFrame | Table,
        detector: lsst.afw.cameraGeom.Detector,
        filterName: str,
    ) -> pipeBase.Struct:
        """
        Run the source selection algorithm and return the indices to keep
        in the original catalog.

        Parameters
        ----------
        sourceCat : `lsst.afw.table.SourceCatalog` or `pandas.DataFrame`
        or `astropy.table.Table`
            Catalog of sources to select from.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.
        filterName : `str`
            Name of camera filter.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - selected : `numpy.ndarray` of `bool`
                    Boolean array of sources that were selected, same length as
                    sourceCat.

        Raises
        ------
        `ValueError`
            sourceLimit in config for task must be -1 or a positive integer.
        """

        bbox = detector.getBBox()

        selected = np.zeros(len(sourceCat), dtype=bool)
        if len(selected) == 0:
            return pipeBase.Struct(
                selected=selected,
                blendCentersX=None,
                blendCentersY=None,
            )

        fluxField = f"{filterName}_flux"
        flux = _getFieldFromCatalog(sourceCat, fluxField)
        mag = (flux * u.nJy).to_value(u.ABmag)
        minMagDiff = self.config.isolatedMagDiff
        unblendedSeparation = self.config.unblendedSeparation
        minBlendedSeparation = self.config.minBlendedSeparation

        # Use user defined inputs or ts_wep defaults
        # depending on useCustomMagLimit.
        if self.config.useCustomMagLimit:
            magMin = self.config.magMin
            magMax = self.config.magMax
        else:
            magPolicyDefaults = readConfigYaml("policy:magLimitStar.yaml")
            defaultFilterKey = f"filter{filterName.upper()}"
            magMax = magPolicyDefaults[defaultFilterKey]["high"]
            magMin = magPolicyDefaults[defaultFilterKey]["low"]

        magSelected = np.ones(len(sourceCat), dtype=bool)
        magSelected &= mag < (magMax + minMagDiff)
        mag = mag[magSelected]
        if len(mag) == 0:
            return pipeBase.Struct(
                selected=selected,
                blendCentersX=None,
                blendCentersY=None,
            )
        xCoord = _getFieldFromCatalog(sourceCat[magSelected], self.config.xCoordField)
        yCoord = _getFieldFromCatalog(sourceCat[magSelected], self.config.yCoordField)

        df = pd.DataFrame({"x": xCoord, "y": yCoord, "mag": mag})
        # Grab any donut centers within unblended distance.
        xyNeigh = NearestNeighbors(radius=unblendedSeparation)

        # Get distance to center of field
        fieldXY = detector.transform(
            [lsst.geom.Point2D(xPix, yPix) for xPix, yPix in zip(xCoord, yCoord)],
            PIXELS,
            FIELD_ANGLE,
        )
        fieldDist = [np.degrees(np.sqrt(fieldLoc[0] ** 2 + fieldLoc[1] ** 2)) for fieldLoc in fieldXY]
        df["fieldDist"] = fieldDist

        # Remove area too close to edge with new bounding box that allows
        # only area at least distance for unblended separation from edges
        trimmedBBox = bbox.erodedBy(unblendedSeparation)

        index = list()
        magSortedDf = df.sort_values("mag")
        groupIndices = magSortedDf.index.values
        xyNeigh.fit(magSortedDf[["x", "y"]])
        radDist, radIdx = xyNeigh.radius_neighbors(magSortedDf[["x", "y"]], sort_results=True)

        errMsg = str("config.sourceLimit must be a positive integer " + "or turned off by setting it to '-1'")
        if not ((self.config.sourceLimit == -1) or (self.config.sourceLimit > 0)):
            raise ValueError(errMsg)

        maxBlended = self.config.maxBlended
        blendCentersX: list = [list() for _ in range(len(magSortedDf))]
        blendCentersY: list = [list() for _ in range(len(magSortedDf))]
        sourcesKept = 0
        # Go through catalog with nearest neighbor information
        # and keep sources that match our configuration settings
        srcOn = -1
        for nbrDist, idxList in zip(radDist, radIdx):
            srcOn += 1
            # Move on if source is within unblendedSeparation
            # of the edge of a given exposure
            srcX = magSortedDf["x"].iloc[srcOn]
            srcY = magSortedDf["y"].iloc[srcOn]
            if trimmedBBox.contains(srcX, srcY) is False:
                continue

            # If distance from field center is greater than
            # maxFieldDist discard the source and move on
            if magSortedDf["fieldDist"].iloc[srcOn] > self.config.maxFieldDist:
                continue

            # If this source's magnitude is outside our bounds then discard
            srcMag = magSortedDf["mag"].iloc[srcOn]
            if (srcMag > magMax) | (srcMag < magMin):
                continue

            # If there is no overlapping source keep
            # the source and move on to next
            if len(idxList) == 1:
                index.append(groupIndices[srcOn])
                sourcesKept += 1
            # In this case there is at least one overlapping source
            else:
                # Measure magnitude differences with overlapping objects
                magDiff = magSortedDf["mag"].iloc[idxList[1:]] - srcMag
                magTooClose = magDiff.values < minMagDiff

                # Measure distances to overlapping objects
                blendSeparations = nbrDist[1:]
                blendTooClose = blendSeparations < minBlendedSeparation

                # If this is the fainter source of the overlaps move on
                if np.min(magDiff) < 0.0:
                    continue
                # If this source overlaps but is brighter than all its
                # overlapping sources by minMagDiff then keep it
                elif np.min(magDiff) >= minMagDiff:
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                # If the centers of any blended objects with a magnitude
                # within minMagDiff of the source magnitude
                # are closer than minBlendedSeparation move on
                elif np.sum(blendTooClose & magTooClose) > 0:
                    continue
                # If the number of overlapping sources with magnitudes close
                # enough to count as blended is less than or equal to
                # maxBlended then keep this source
                elif len(magDiff) <= maxBlended:
                    index.append(groupIndices[srcOn])
                    # Only include sources bright enough to count as
                    # blended based upon isolatedMagDiff. Otherwise
                    # masks for deblending will include footprints of
                    # all the faint sources that we don't care about
                    # when deblending. Add one to index because
                    # magDiff is all sources in magSortedDf after index=0.
                    blendMagIdx = np.where(magDiff < minMagDiff)[0] + 1
                    blendCentersX[groupIndices[srcOn]] = magSortedDf["x"].iloc[idxList[blendMagIdx]].values
                    blendCentersY[groupIndices[srcOn]] = magSortedDf["y"].iloc[idxList[blendMagIdx]].values
                    sourcesKept += 1
                # Keep the source if it is blended with up to maxBlended
                # number of sources. To check this we look at the maxBlended+1
                # source in the magDiff list and check that the object
                # is at least minMagDiff brighter than this. Satisfying this
                # criterion means it is blended with maxBlended
                # or fewer sources.
                elif np.partition(magDiff, maxBlended)[maxBlended] > minMagDiff:
                    index.append(groupIndices[srcOn])
                    # Same process as above to make sure we only get
                    # the blend centers we care about
                    blendMagIdx = np.where(magDiff < minMagDiff)[0] + 1
                    blendCentersX[groupIndices[srcOn]] = magSortedDf["x"].iloc[idxList[blendMagIdx]].values
                    blendCentersY[groupIndices[srcOn]] = magSortedDf["y"].iloc[idxList[blendMagIdx]].values
                    sourcesKept += 1
                else:
                    continue

            if (self.config.sourceLimit > 0) and (sourcesKept == self.config.sourceLimit):
                break

        # magSelected is a boolean array so we can
        # find indices with True by finding nonzero elements
        magIndex = magSelected.nonzero()[0]
        finalIndex = magIndex[index]
        selected[finalIndex] = True
        sortedIndex = np.sort(index)
        selectedBlendCentersX = [blendCentersX[idx] for idx in sortedIndex]
        selectedBlendCentersY = [blendCentersY[idx] for idx in sortedIndex]

        self.log.info("Selected %d/%d references", selected.sum(), len(sourceCat))

        return pipeBase.Struct(
            selected=selected,
            blendCentersX=selectedBlendCentersX,
            blendCentersY=selectedBlendCentersY,
        )
