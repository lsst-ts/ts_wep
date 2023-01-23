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

__all__ = [
    "GenerateDonutCatalogWcsTaskConnections",
    "GenerateDonutCatalogWcsTaskConfig",
    "GenerateDonutCatalogWcsTask",
]

import os
import typing
import warnings
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.utils.timer import timeMethod
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig
from lsst.meas.algorithms.sourceSelector import ReferenceSourceSelectorTask
from lsst.ts.wep.task.DonutSourceSelectorTask import DonutSourceSelectorTask


class GenerateDonutCatalogWcsTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
):
    """
    Specify the pipeline connections needed for
    GenerateDonutCatalogWcsTask. We
    need the reference catalogs and exposures and
    will produce donut catalogs for a specified instrument.
    """

    refCatalogs = connectionTypes.PrerequisiteInput(
        doc="Reference catalog",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        multiple=True,
        deferLoad=True,
        name="cal_ref_cat",
    )
    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="DataFrame",
        name="donutCatalog",
    )


class GenerateDonutCatalogWcsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogWcsTaskConnections,
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    referenceSelector = pexConfig.ConfigurableField(
        target=ReferenceSourceSelectorTask, doc="How to select reference objects."
    )
    donutSelector = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )


class GenerateDonutCatalogWcsTask(pipeBase.PipelineTask):
    """
    Create a WCS from boresight info and then use this
    with a reference catalog to select sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutCatalogWcsTaskConfig
    _DefaultName = "generateDonutCatalogWcsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The filter in the reference catalog we want to use to find sources.
        self.makeSubtask("referenceSelector")
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

        # TODO: Temporary until DM-24162 is closed at which point we
        # can remove this
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

    def getRefObjLoader(self, refCatalogList):
        """
        Create a `ReferenceObjectLoader` from available reference catalogs
        in the repository.

        Parameters
        ----------
        refCatalogList : `list`
            List of deferred butler references for the reference catalogs.

        Returns
        -------
        `lsst.meas.algorithms.ReferenceObjectsLoader`
            Object to conduct spatial searches through the reference catalogs
        """

        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in refCatalogList],
            refCats=refCatalogList,
            config=LoadReferenceObjectsConfig(),
        )
        # This removes the padding around the border of detector BBox when
        # matching to reference catalog.
        # We remove this since we only want sources within detector.
        refObjLoader.config.pixelMargin = 0

        return refObjLoader

    def runSelection(self, refObjLoader, detector, wcs, filterName):
        """
        Match the detector area to the reference catalog
        and then run the LSST DM reference selection task.
        For configuration parameters on the reference selector
        see `lsst.meas.algorithms.ReferenceSourceSelectorConfig`.

        Parameters
        ----------
        refObjLoader : `meas.algorithms.ReferenceObjectLoader`
            Reference object loader to use in getting reference objects.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.
        wcs : `lsst.afw.geom.SkyWcs`
            Wcs object defining the pixel to sky (and inverse) transform for
            the supplied ``bbox``.
        filterName : `str`
            Name of camera filter.

        Returns
        -------
        referenceCatalog : `lsst.afw.table.SimpleCatalog`
            Catalog containing reference objects inside the specified bounding
            box and with properties within the bounds set by the
            `referenceSelector`.
        """

        bbox = detector.getBBox()
        donutCatalog = refObjLoader.loadPixelBox(bbox, wcs, filterName).refCat

        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            donutSelection = self.donutSelector.run(donutCatalog, detector, filterName)
            return (
                donutCatalog[donutSelection.selected],
                donutSelection.blendCentersX,
                donutSelection.blendCentersY,
            )
        else:
            return donutCatalog, [[]] * len(donutCatalog), [[]] * len(donutCatalog)

    def donutCatalogToDataFrame(
        self, donutCatalog=None, filterName=None, blendCentersX=None, blendCentersY=None
    ):
        """
        Reformat afwCatalog into a pandas dataframe sorted by flux with
        the brightest objects at the top.

        Parameters
        ----------
        donutCatalog : `lsst.afw.table.SimpleCatalog` or `None`, optional
            lsst.afw.table.SimpleCatalog object returned by the
            ReferenceObjectLoader search over the detector footprint.
            If None then it will return an empty dataframe.
            (the default is None.)
        filterName : `str` or `None`, optional
            Name of camera filter. If donutCatalog is not None then
            this cannot be None. (the default is None.)
        blendCentersX : `list` or `None`, optional
             X pixel position of centroids for blended objects. List
             should be the same length as the donutCatalog. If
             blendCentersY is not None then this cannot be None. (the default
             is None.)
        blendCentersY : `list` or `None`, optional
             Y pixel position of centroids for blended objects. List
             should be the same length as the donutCatalog. If
             blendCentersX is not None then this cannot be None. (the default
             is None.)

        Returns
        -------
        `pandas.DataFrame`
            Complete catalog of reference sources in the pointing.

        Raises
        ------
        `ValueError`
            Raised if filterName is None when donutCatalog is not None.
        `ValueError`
            Raised if blendCentersX and blendCentersY are not the same length.
        `ValueError`
            Raised if blendCentersX and blendCentersY are not both
            a list or are not both None.
        """

        ra = []
        dec = []
        centroidX = []
        centroidY = []
        sourceFlux = []
        blendCX = []
        blendCY = []

        if donutCatalog is not None:
            filterErrMsg = "If donutCatalog is not None then filterName cannot be None."
            if filterName is None:
                raise ValueError(filterErrMsg)
            ra = donutCatalog["coord_ra"]
            dec = donutCatalog["coord_dec"]
            centroidX = donutCatalog["centroid_x"]
            centroidY = donutCatalog["centroid_y"]
            sourceFlux = donutCatalog[f"{filterName}_flux"]

            if (blendCentersX is None) and (blendCentersY is None):
                blendCX = list()
                blendCY = list()
                for idx in range(len(donutCatalog)):
                    blendCX.append(list())
                    blendCY.append(list())
            elif isinstance(blendCentersX, list) and isinstance(blendCentersY, list):
                lengthErrMsg = (
                    "blendCentersX and blendCentersY need "
                    + "to be same length as donutCatalog."
                )
                if (len(blendCentersX) != len(donutCatalog)) or (
                    len(blendCentersY) != len(donutCatalog)
                ):
                    raise ValueError(lengthErrMsg)
                xyMismatchErrMsg = (
                    "Each list in blendCentersX must have the same "
                    + "length as the list in blendCentersY at the "
                    + "same index."
                )
                for xList, yList in zip(blendCentersX, blendCentersY):
                    if len(xList) != len(yList):
                        raise ValueError(xyMismatchErrMsg)
                blendCX = blendCentersX
                blendCY = blendCentersY
            else:
                blendErrMsg = (
                    "blendCentersX and blendCentersY must be"
                    + " both be None or both be a list."
                )
                raise ValueError(blendErrMsg)

        fieldObjects = pd.DataFrame([])
        fieldObjects["coord_ra"] = ra
        fieldObjects["coord_dec"] = dec
        fieldObjects["centroid_x"] = centroidX
        fieldObjects["centroid_y"] = centroidY
        fieldObjects["source_flux"] = sourceFlux
        fieldObjects["blend_centroid_x"] = blendCX
        fieldObjects["blend_centroid_y"] = blendCY

        fieldObjects = fieldObjects.sort_values(
            "source_flux", ascending=False
        ).reset_index(drop=True)

        return fieldObjects

    @timeMethod
    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:

        refObjLoader = self.getRefObjLoader(refCatalogs)

        detector = exposure.getDetector()
        detectorWcs = exposure.getWcs()
        filterName = exposure.filter.bandLabel

        try:
            # Match detector layout to reference catalog
            refSelection, blendCentersX, blendCentersY = self.runSelection(
                refObjLoader, detector, detectorWcs, filterName
            )

        # Except RuntimeError caused when no reference catalog
        # available for the region covered by detector
        except RuntimeError:
            warnings.warn(
                "No catalogs cover this detector. Returning empty catalog.",
                RuntimeWarning,
            )
            refSelection = None
            blendCentersX = None
            blendCentersY = None

        fieldObjects = self.donutCatalogToDataFrame(
            refSelection, filterName, blendCentersX, blendCentersY
        )

        return pipeBase.Struct(donutCatalog=fieldObjects)
