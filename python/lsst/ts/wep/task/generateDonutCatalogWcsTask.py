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
import warnings
from typing import Any

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.pipe.base.task import TaskError
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    addVisitInfoToCatTable,
    donutCatalogToAstropy,
    runSelection,
)
from lsst.utils.timer import timeMethod


class GenerateDonutCatalogWcsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),  # type: ignore
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
        name="post_isr_image",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="AstropyQTable",
        name="donutTable",
    )


class GenerateDonutCatalogWcsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogWcsTaskConnections,  # type: ignore
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Size of detector edge margin in pixels", dtype=int, default=80
    )
    # When matching photometric filters are not available in
    # the reference catalog (e.g. Gaia) use anyFilterMapsToThis
    # to get sources out of the catalog.
    photoRefFilter: pexConfig.Field = pexConfig.Field(
        doc="Set filter to use in Photometry catalog. "
        + "Cannot set both this and photoRefFilterPrefix. "
        + "If neither is set then will just try to use the name of the exposure filter.",
        dtype=str,
        optional=True,
    )
    photoRefFilterPrefix: pexConfig.Field = pexConfig.Field(
        doc="Set filter prefix to use. "
        + "Will then try to use exposure band label with given catalog prefix. "
        + "Cannot set both this and photoRefFilter. "
        + "If neither is set then will just try to use the name of the exposure filter.",
        dtype=str,
        optional=True,
    )
    catalogFilterList: pexConfig.ListField = pexConfig.ListField(
        dtype=str,
        doc="Filters from reference catalog to include in donut catalog.",
        default=["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"],
    )


class GenerateDonutCatalogWcsTask(pipeBase.PipelineTask):
    """
    Create a WCS from boresight info and then use this
    with a reference catalog to select sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutCatalogWcsTaskConfig
    _DefaultName = "generateDonutCatalogWcsTask"

    config: GenerateDonutCatalogWcsTaskConfig
    donutSelector: DonutSourceSelectorTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

        # TODO: Temporary until DM-24162 is closed at which point we
        # can remove this
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

    def getRefObjLoader(self, refCatalogList: list) -> ReferenceObjectLoader:
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
        )
        # This removes the padding around the border of detector BBox when
        # matching to reference catalog.
        # We remove this since we only want sources within detector.
        refObjLoader.config.pixelMargin = 0

        return refObjLoader

    @timeMethod
    def run(
        self,
        refCatalogs: list[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:
        refObjLoader = self.getRefObjLoader(refCatalogs)

        detector = exposure.getDetector()
        detectorWcs = exposure.getWcs()
        edgeMargin = self.config.edgeMargin
        # Check that specified filter exists in catalogs
        if self.config.photoRefFilter is not None and self.config.photoRefFilterPrefix is not None:
            raise ValueError("photoRefFilter and photoRefFilterConfig cannot both be set.")
        if self.config.photoRefFilter is not None:
            filterName = self.config.photoRefFilter
        elif self.config.photoRefFilterPrefix is not None:
            filterName = f"{self.config.photoRefFilterPrefix}_{exposure.filter.bandLabel}"

        try:
            # Match detector layout to reference catalog
            self.log.info("Running Donut Selector")
            donutSelectorTask = self.donutSelector if self.config.doDonutSelection is True else None
            refSelection, blendCentersX, blendCentersY = runSelection(
                refObjLoader,
                detector,
                detectorWcs,
                filterName,
                donutSelectorTask,
                edgeMargin,
            )
            # Create list of filters to include in final catalog
            filterList = self.config.catalogFilterList
            availableRefFilters = [
                col[:-5] for col in refSelection.schema.getNames() if col.endswith("_flux")
            ]

            # Validate all requested filters exist
            missing_filters = set(filterList) - set(availableRefFilters)
            if missing_filters:
                raise TaskError(
                    f"Filter(s) {missing_filters} not in available columns"
                    " in reference catalog. Check catalogFilterList config "
                    f"(currently set as {filterList}). "
                    f"Available ref catalog filters are {availableRefFilters}."
                )

            # Add photoRefFilter if not present and get its index
            if filterName not in filterList:
                filterList.append(filterName)
            sortFilterIdx = filterList.index(filterName)

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

        fieldObjects = donutCatalogToAstropy(
            refSelection, filterList, blendCentersX, blendCentersY, sortFilterIdx=sortFilterIdx
        )
        fieldObjects["detector"] = np.array([detector.getName()] * len(fieldObjects), dtype=str)

        fieldObjects = addVisitInfoToCatTable(exposure, fieldObjects)

        return pipeBase.Struct(donutCatalog=fieldObjects)
