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

__all__ = ["GenerateDonutCatalogOnlineTaskConfig", "GenerateDonutCatalogOnlineTask"]

import warnings
from typing import Any

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.cameraGeom import Detector
from lsst.afw.geom import SkyWcs
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    donutCatalogToAstropy,
    runSelection,
)
from lsst.utils.timer import timeMethod


class GenerateDonutCatalogOnlineTaskConfig(pexConfig.Config):
    """Configuration for GenerateDonutCatalogOnlineTask."""

    filterName: pexConfig.Field = pexConfig.Field(doc="Reference filter", dtype=str, default="g")
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Size of detector edge margin in pixels", dtype=int, default=80
    )


class GenerateDonutCatalogOnlineTask(pipeBase.Task):
    """
    Construct a source catalog from reference catalogs
    and pointing information.

    Parameters
    ----------
    dataIds : `list`
        List of the dataIds for the reference catalog shards.
    refCats : `list`
        List of the deferred dataset references pointing to the pieces
        of the reference catalog we want in the butler.
    **kwargs : dict[str, any]
        Dictionary of input argument: new value for that input argument.
    """

    ConfigClass = GenerateDonutCatalogOnlineTaskConfig
    _DefaultName = "generateDonutCatalogOnlineTask"
    config: GenerateDonutCatalogOnlineTaskConfig
    donutSelector: DonutSourceSelectorTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

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
            Object to conduct spatial searches through the reference catalogs.
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
        refCatalogs: list,
        detector: Detector,
        detectorWcs: SkyWcs,
    ) -> pipeBase.Struct:
        # refObjLoader handles the interaction with the butler repository
        # needed to get the pieces of the reference catalogs we need.
        refObjLoader = self.getRefObjLoader(refCatalogs)
        edgeMargin = self.config.edgeMargin
        filterName = self.config.filterName

        try:
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

        # Except RuntimeError caused when no reference catalog
        # available for the region covered by detector
        except RuntimeError:
            warnings.warn(
                "No catalogs cover this detector. Returning empty catalog. "
                + f"Double check that filterName: '{filterName}' is in catalog.",
                RuntimeWarning,
            )
            refSelection = None
            blendCentersX = None
            blendCentersY = None

        fieldObjects = donutCatalogToAstropy(refSelection, filterName, blendCentersX, blendCentersY)
        fieldObjects["detector"] = np.array([detector.getName()] * len(fieldObjects), dtype=str)
        return pipeBase.Struct(donutCatalog=fieldObjects)
