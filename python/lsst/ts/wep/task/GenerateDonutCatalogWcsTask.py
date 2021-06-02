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

import typing
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.ts.wep.task.GenerateDonutCatalogBase import (
    GenerateDonutCatalogBaseConnections,
    GenerateDonutCatalogBaseConfig,
    GenerateDonutCatalogBaseTask,
)


class GenerateDonutCatalogWcsConnections(
    GenerateDonutCatalogBaseConnections,
    dimensions=(
        "exposure",
        "instrument",
    ),
):

    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )


class GenerateDonutCatalogWcsTaskConfig(
    GenerateDonutCatalogBaseConfig,
    pipelineConnections=GenerateDonutCatalogWcsConnections,
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    """

    pass


class GenerateDonutCatalogWcsTask(GenerateDonutCatalogBaseTask):
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

    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
        exposures: typing.List[afwImage.Exposure],
    ) -> pipeBase.Struct:

        refObjLoader = self.getRefObjLoader(refCatalogs)

        detectorList = []
        donutCatalogList = []

        for exposure in exposures:
            detectorName = exposure.getDetector().getName()
            detWcs = exposure.getWcs()

            try:
                # Match detector layout to reference catalog
                donutCatalog = refObjLoader.loadPixelBox(
                    exposure.getBBox(), detWcs, filterName=self.filterName
                ).refCat

                detectorList.append(detectorName)
                donutCatalogList.append(donutCatalog)

            except RuntimeError:
                continue

        fieldObjects = self.donutCatalogListToDataFrame(donutCatalogList, detectorList)

        # Return pandas DataFrame with sources in pointing
        # with ra, dec, filter flux, pixel XY information and detector name
        # for each source
        finalSources = self.filterResults(fieldObjects)

        return pipeBase.Struct(donutCatalog=finalSources)
