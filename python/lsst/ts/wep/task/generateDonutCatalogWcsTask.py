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

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    donutCatalogToDataFrame,
    runSelection,
)
from lsst.utils.timer import timeMethod


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

    donutSelector = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )
    # When matching photometric filters are not available in
    # the reference catalog (e.g. Gaia) use anyFilterMapsToThis
    # to get sources out of the catalog.
    anyFilterMapsToThis = LoadReferenceObjectsConfig.anyFilterMapsToThis


class GenerateDonutCatalogWcsTask(pipeBase.PipelineTask):
    """
    Create a WCS from boresight info and then use this
    with a reference catalog to select sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutCatalogWcsTaskConfig
    _DefaultName = "generateDonutCatalogWcsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            config=LoadReferenceObjectsConfig(
                anyFilterMapsToThis=self.config.anyFilterMapsToThis
            ),
        )
        # This removes the padding around the border of detector BBox when
        # matching to reference catalog.
        # We remove this since we only want sources within detector.
        refObjLoader.config.pixelMargin = 0

        return refObjLoader

    @timeMethod
    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:
        refObjLoader = self.getRefObjLoader(refCatalogs)

        detector = exposure.getDetector()
        detectorWcs = exposure.getWcs()
        anyFilterMapsToThis = self.config.anyFilterMapsToThis
        filterName = (
            exposure.filter.bandLabel
            if anyFilterMapsToThis is None
            else anyFilterMapsToThis
        )

        try:
            # Match detector layout to reference catalog
            self.log.info("Running Donut Selector")
            donutSelectorTask = (
                self.donutSelector if self.config.doDonutSelection is True else None
            )
            refSelection, blendCentersX, blendCentersY = runSelection(
                refObjLoader, detector, detectorWcs, filterName, donutSelectorTask
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

        fieldObjects = donutCatalogToDataFrame(
            refSelection, filterName, blendCentersX, blendCentersY
        )

        return pipeBase.Struct(donutCatalog=fieldObjects)
