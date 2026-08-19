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
from scipy.spatial import KDTree

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
    allowFluxless: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Allow selection on catalogs that lack flux information (e.g. a "
        + "detection catalog with only coordinates)? When True and no flux field "
        + "is present, magnitude cuts are skipped, sources are ordered by field "
        + "distance (center-out), and isolation/blending is decided purely on "
        + "separation (unblendedSeparation, minBlendedSeparation, maxBlended); no "
        + "blend centers are returned. When False (default), a missing flux field "
        + "raises. Flux is always used when it is present, regardless of this "
        + "setting.",
    )
    useCustomMagLimit: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply user-defined magnitude limit? If this is False then the code"
        + " will default to use the magnitude values in policy:magLimitStar.yaml."
        + " Only used when flux is available.",
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
        doc="Min. difference in magnitude for 'isolated' star. Only used when flux is available.",
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

def validate(self) -> None:
    super().validate()
    if self.sourceLimit != -1 and self.sourceLimit <= 0:
        raise pexConfig.FieldValidationError(
            self.__class__.sourceLimit,
            self,
            "sourceLimit must be a positive integer "
            "or turned off by setting it to '-1'",
        )
    if self.minBlendedSeparation > self.unblendedSeparation:
        raise pexConfig.FieldValidationError(
            self.__class__.minBlendedSeparation,
            self,
            "minBlendedSeparation must be <= unblendedSeparation "
            "(neighbors are only found within unblendedSeparation).",
        )
    if self.maxBlended < 0:
        raise pexConfig.FieldValidationError(
            self.__class__.maxBlended,
            self,
            "maxBlended must be >= 0.",
        )


class DonutSourceSelectorTask(pipeBase.Task):
    """
    Donut Source Selector that uses a nearest neighbors radius
    query to find all donuts within the pixel radius set in the
    config. Then it goes from the brightest sources down to the faintest
    picking donuts that are at least isolatedMagDiff brighter than any sources
    with centers within 2 times the unblendedSeparation until reaching
    numSources kept or going through the whole list.

    When the input catalog lacks flux information (and ``config.allowFluxless``
    is True) the selector operates on coordinates only (e.g. a detection catalog
    with only coordinates).  In that mode magnitude cuts are skipped, sources are
    ordered by field distance (center-out), and isolation/blending decisions are
    made purely on separation.  No blend centers are produced.  Flux is always
    used when it is present.
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
            Name of camera filter.  Ignored when the catalog has no flux field
            and ``config.allowFluxless`` is True.

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
            Name of camera filter.  Ignored when the catalog has no flux field
            and ``config.allowFluxless`` is True.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - selected : `numpy.ndarray` of `bool`
                    Boolean array of sources that were selected, same length as
                    sourceCat.

        Raises
        ------
        `RuntimeError`
            Raised if the catalog lacks the flux field ``f"{filterName}_flux"``
            and ``config.allowFluxless`` is False.
        `KeyError`
            Raised if a required coordinate column is missing from ``sourceCat``,
            or (when flux is used and useCustomMagLimit is False) if
            ``filterName`` has no entry in policy:magLimitStar.yaml.
        """

        bbox = detector.getBBox()

        selected = np.zeros(len(sourceCat), dtype=bool)
        if len(selected) == 0:
            return pipeBase.Struct(
                selected=selected,
                blendCentersX=None,
                blendCentersY=None,
            )

        minMagDiff = self.config.isolatedMagDiff
        unblendedSeparation = self.config.unblendedSeparation
        minBlendedSeparation = self.config.minBlendedSeparation
        maxBlended = self.config.maxBlended
        maxFieldDist = self.config.maxFieldDist
        sourceLimit = self.config.sourceLimit
        allowFluxless = self.config.allowFluxless

        # Determine whether flux is available.  Try to read it via the same
        # code path that would be used to consume it, so detection and
        # retrieval can never disagree.  Flux is always used when present.
        fluxField = f"{filterName}_flux"
        try:
            flux = np.asarray(_getFieldFromCatalog(sourceCat, fluxField))
            useFlux = True
        except KeyError:
            useFlux = False

        if not useFlux and not allowFluxless:
            raise RuntimeError(
                f"Flux field '{fluxField}' not found in catalog and "
                "config.allowFluxless is False."
            )

        if useFlux:
            mag = (flux * u.nJy).to_value(u.ABmag)

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
        else:
            # No flux information available (e.g. detection catalog).
            # Keep everything through the "mag" pre-filter and disable
            # magnitude-based cuts downstream.
            mag = np.zeros(len(sourceCat), dtype=float)
            magSelected = np.ones(len(sourceCat), dtype=bool)
            magMin = -np.inf
            magMax = np.inf

        if len(mag) == 0:
            return pipeBase.Struct(
                selected=selected,
                blendCentersX=None,
                blendCentersY=None,
            )

        xCoord = np.asarray(_getFieldFromCatalog(sourceCat[magSelected], self.config.xCoordField))
        yCoord = np.asarray(_getFieldFromCatalog(sourceCat[magSelected], self.config.yCoordField))

        # Distance to center of field (degrees) for each selected source.
        # Vectorized transform: avoid per-point Point2D construction and the
        # Python-level coordinate extraction that followed it.
        xform = detector.getTransform(PIXELS, FIELD_ANGLE)
        mapping = xform.getMapping()
        xyField = mapping.applyForward(np.vstack([xCoord, yCoord]))  # shape (2, N)
        fieldDist = np.degrees(np.hypot(xyField[0], xyField[1]))

        # Ordering.  With flux we sort brightest-first.  Without flux we sort
        # by field distance (center-out) so that sourceLimit keeps the most
        # central sources and the "winner" of an overlapping pair is the more
        # central one.
        if useFlux:
            groupIndices = np.argsort(mag, kind="stable")
        else:
            groupIndices = np.argsort(fieldDist, kind="stable")

        xSorted = xCoord[groupIndices]
        ySorted = yCoord[groupIndices]
        magSorted = mag[groupIndices]
        fieldDistSorted = fieldDist[groupIndices]

        # Remove area too close to edge with new bounding box that allows
        # only area at least distance for unblended separation from edges.
        trimmedBBox = bbox.erodedBy(unblendedSeparation)
        minX = trimmedBBox.getMinX()
        minY = trimmedBBox.getMinY()
        maxX = trimmedBBox.getMaxX()
        maxY = trimmedBBox.getMaxY()
        # NOTE: erodedBy on an integer bbox yields a Box2I, whose contains() is
        # inclusive of the max corner.  Match that with <=.  If trimmedBBox is a
        # Box2D in your build, change the upper comparisons to <.
        inBox = (
            (xSorted >= minX)
            & (xSorted <= maxX)
            & (ySorted >= minY)
            & (ySorted <= maxY)
        )

        # Nearest-neighbor structure on the (sorted) positions.
        xy = np.ascontiguousarray(np.column_stack([xSorted, ySorted]), dtype=np.float64)
        tree = KDTree(xy)
        radIdxList = tree.query_ball_point(xy, r=unblendedSeparation, return_sorted=True)

        index = list()
        # Sparse storage: most sources have no blend centers, so only populate
        # the ones we actually keep-with-blends.  Keyed by sorted-order position.
        blendCentersXMap: dict = {}
        blendCentersYMap: dict = {}
        sourcesKept = 0

        # Go through catalog (brightest first, or center-out when flux-less)
        # with nearest neighbor information and keep sources that match our
        # configuration settings.
        for srcOn, idxList in enumerate(radIdxList):
            # Move on if source is within unblendedSeparation
            # of the edge of a given exposure
            if not inBox[srcOn]:
                continue

            # If distance from field center is greater than
            # maxFieldDist discard the source and move on
            if fieldDistSorted[srcOn] > maxFieldDist:
                continue

            # If this source's magnitude is outside our bounds then discard.
            # (Vacuous when flux-less: magMin/magMax are +/-inf and srcMag=0.)
            srcMag = magSorted[srcOn]
            if (srcMag > magMax) | (srcMag < magMin):
                continue

            # If there is no overlapping source keep
            # the source and move on to next
            if len(idxList) == 1:
                index.append(groupIndices[srcOn])
                sourcesKept += 1

            elif not useFlux:
                # --- Geometry-only isolation/blending (no flux) ---
                # idxList is a plain Python list (distance-sorted, self first).
                # Neighbors excluding the self-match at position 0.
                neighbors = idxList[1:]

                # Because the arrays are sorted center-out, any neighbor with a
                # smaller sorted index is more central than this source.  If one
                # exists, let that (already-considered) source own the overlap so
                # we don't keep both members of a close pair.
                if any(j < srcOn for j in neighbors):
                    continue

                neighborIdx = np.asarray(neighbors)
                # Measure distances to overlapping objects
                dxy = xy[neighborIdx] - xy[srcOn]
                blendSeparations = np.hypot(dxy[:, 0], dxy[:, 1])

                # If anything is closer than the minimum allowed separation,
                # reject this source.
                if np.any(blendSeparations < minBlendedSeparation):
                    continue

                # Otherwise accept if the number of overlapping neighbors is
                # within maxBlended (0 -> require full isolation).
                if len(neighborIdx) <= maxBlended:
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                else:
                    continue

            # In this case there is at least one overlapping source and we have
            # flux information to arbitrate the blend.
            else:
                # idxList is a plain Python list (distance-sorted, self first).
                # Neighbors excluding the self-match at position 0.
                neighbors = idxList[1:]

                # Because the arrays are magnitude-sorted, any neighbor with a
                # smaller sorted index is brighter than this source.  If one
                # exists, this source is the fainter member of the overlap.
                # Short-circuiting pure-Python test avoids building a numpy
                # array for the common rejection path.
                # (Equivalent to the old np.min(magDiff) < 0.0.)
                if any(j < srcOn for j in neighbors):
                    continue

                # Only build arrays for the few sources that survive to here.
                neighborIdx = np.asarray(neighbors)
                # Measure magnitude differences with overlapping objects
                magDiff = magSorted[neighborIdx] - srcMag
                magTooClose = magDiff < minMagDiff

                # Measure distances to overlapping objects
                dxy = xy[neighborIdx] - xy[srcOn]
                blendSeparations = np.hypot(dxy[:, 0], dxy[:, 1])
                blendTooClose = blendSeparations < minBlendedSeparation

                minMagDiffVal = magDiff.min()

                # If this source overlaps but is brighter than all its
                # overlapping sources by minMagDiff then keep it
                if minMagDiffVal >= minMagDiff:
                    index.append(groupIndices[srcOn])
                    sourcesKept += 1
                # If the centers of any blended objects with a magnitude
                # within minMagDiff of the source magnitude
                # are closer than minBlendedSeparation move on
                elif np.any(blendTooClose & magTooClose):
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
                    # magDiff is all sources after index=0.
                    blendMagIdx = np.where(magDiff < minMagDiff)[0] + 1
                    keepIdx = np.asarray(idxList)[blendMagIdx]
                    blendCentersXMap[groupIndices[srcOn]] = xSorted[keepIdx]
                    blendCentersYMap[groupIndices[srcOn]] = ySorted[keepIdx]
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
                    keepIdx = np.asarray(idxList)[blendMagIdx]
                    blendCentersXMap[groupIndices[srcOn]] = xSorted[keepIdx]
                    blendCentersYMap[groupIndices[srcOn]] = ySorted[keepIdx]
                    sourcesKept += 1
                else:
                    continue

            if (sourceLimit > 0) and (sourcesKept == sourceLimit):
                break

        # magSelected is a boolean array so we can
        # find indices with True by finding nonzero elements
        magIndex = magSelected.nonzero()[0]
        finalIndex = magIndex[index]
        selected[finalIndex] = True
        sortedIndex = np.sort(index)
        if useFlux:
            selectedBlendCentersX = [blendCentersXMap.get(idx, np.array([])) for idx in sortedIndex]
            selectedBlendCentersY = [blendCentersYMap.get(idx, np.array([])) for idx in sortedIndex]
        else:
            # No blend centers are produced in the flux-less path.
            selectedBlendCentersX = [np.array([]) for _ in sortedIndex]
            selectedBlendCentersY = [np.array([]) for _ in sortedIndex]

        self.log.info("Selected %d/%d references", selected.sum(), len(sourceCat))

        return pipeBase.Struct(
            selected=selected,
            blendCentersX=selectedBlendCentersX,
            blendCentersY=selectedBlendCentersY,
        )
