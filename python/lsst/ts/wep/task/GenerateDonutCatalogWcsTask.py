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

import os
import typing
import warnings
import pandas as pd
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base.connectionTypes as connectionTypes
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

    filterName = pexConfig.Field(doc="Reference filter", dtype=str, default="g")
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
        self.filterName = self.config.filterName
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

    def runSelection(self, refObjLoader, bbox, wcs, filterName):
        """
        Match the detector area to the reference catalog
        and then run the LSST DM reference selection task.
        For configuration parameters on the reference selector
        see `lsst.meas.algorithms.ReferenceSourceSelectorConfig`.

        Parameters
        ----------
        refObjLoader : `meas.algorithms.ReferenceObjectLoader`
            Reference object loader to use in getting reference objects.
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Box which bounds a region in pixel space.
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

        donutCatalog = refObjLoader.loadPixelBox(bbox, wcs, filterName).refCat

        refSelection = self.referenceSelector.run(donutCatalog)

        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            donutSelection = self.donutSelector.run(donutCatalog, bbox)
            finalSelection = refSelection.selected & donutSelection.selected
            return donutCatalog[finalSelection]
        else:
            return refSelection.sourceCat

    def donutCatalogToDataFrame(self, detectorName, donutCatalog=None):
        """
        Reformat afwCatalog into a pandas dataframe sorted by flux with
        the brightest objects at the top.

        Parameters
        ----------
        detectorName : `str`
            Name of the detectors associated with donutCatalog.
        donutCatalog : `lsst.afw.table.SimpleCatalog` or `None`, optional
            lsst.afw.table.SimpleCatalog object returned by the
            ReferenceObjectLoader search over the detector footprint.
            If None then it will return an empty dataframe.
            (the default is None.)

        Returns
        -------
        `pandas.DataFrame`
            Complete catalog of reference sources in the pointing.
        """

        ra = []
        dec = []
        centroidX = []
        centroidY = []
        sourceFlux = []
        detNames = []

        if donutCatalog is not None:
            ra = donutCatalog["coord_ra"]
            dec = donutCatalog["coord_dec"]
            centroidX = donutCatalog["centroid_x"]
            centroidY = donutCatalog["centroid_y"]
            sourceFlux = donutCatalog[f"{self.filterName}_flux"]
            detNames = [detectorName] * len(donutCatalog)

        fieldObjects = pd.DataFrame([])
        fieldObjects["coord_ra"] = ra
        fieldObjects["coord_dec"] = dec
        fieldObjects["centroid_x"] = centroidX
        fieldObjects["centroid_y"] = centroidY
        fieldObjects["source_flux"] = sourceFlux
        fieldObjects["detector"] = detNames

        fieldObjects = fieldObjects.sort_values(
            "source_flux", ascending=False
        ).reset_index(drop=True)

        return fieldObjects

    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:

        refObjLoader = self.getRefObjLoader(refCatalogs)

        detectorName = exposure.getDetector().getName()
        detectorBBox = exposure.getBBox()
        detectorWcs = exposure.getWcs()

        try:
            # Match detector layout to reference catalog
            refSelection = self.runSelection(
                refObjLoader, detectorBBox, detectorWcs, self.filterName
            )

        # Except RuntimeError caused when no reference catalog
        # available for the region covered by detector
        except RuntimeError:
            warnings.warn(
                "No catalogs cover this detector. Returning empty catalog.",
                RuntimeWarning,
            )
            refSelection = None

        fieldObjects = self.donutCatalogToDataFrame(detectorName, refSelection)

        return pipeBase.Struct(donutCatalog=fieldObjects)
