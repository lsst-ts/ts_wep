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
    "GenDonutCatWcsDirectBackupTaskConnections",
    "GenDonutCatWcsDirectBackupTaskConfig",
    "GenDonutCatWcsDirectBackupTask",
]

from typing import Any

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogWcsTask import (
    GenerateDonutCatalogWcsTask,
)
from lsst.ts.wep.task.generateDonutDirectDetectTask import (
    GenerateDonutDirectDetectTask,
)
from lsst.utils.timer import timeMethod


class GenDonutCatWcsDirectBackupTaskConnections(
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
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )


class GenDonutCatWcsDirectBackupTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenDonutCatWcsDirectBackupTaskConnections,
):
    wcsCatTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=GenerateDonutCatalogWcsTask,
        doc="Task to generate donut catalog with WCS information.",
    )
    directDetectTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=GenerateDonutDirectDetectTask,
        doc="Task to generate direct detection catalog. "
        + "Used as a backup when wcs catalog generation fails.",
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )


class GenDonutCatWcsDirectBackupTask(
    pipeBase.PipelineTask,
):
    """
    Task to generate donut catalog with WCS information.
    If WCS generation fails, falls back to direct detection catalog generation.
    """

    ConfigClass = GenDonutCatWcsDirectBackupTaskConfig
    _DefaultName = "genDonutCatWcsDirectBackupTask"

    config: GenDonutCatWcsDirectBackupTaskConfig
    wcsCatTask: GenerateDonutCatalogWcsTask
    directDetectTask: GenerateDonutDirectDetectTask

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # self.makeSubtask("donutSelector")
        self.config.wcsCatTask.donutSelector = self.config.donutSelector
        self.makeSubtask("wcsCatTask")

    @timeMethod
    def run(
        self,
        refCatalogs: list[afwTable.SimpleCatalog],
        exposure: afwImage.ExposureF,
        camera: Any,
    ) -> pipeBase.Struct:
        """Run the GenDonutCatWcsDirectBackupTask.

        Parameters
        ----------
        refCatalogs : `list` of `lsst.afw.table.SimpleCatalog`
            List of reference catalogs.
        exposure : `lsst.afw.image.ExposureF`
            Input exposure to make measurements on.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object to construct complete exposures.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Struct containing the donut catalog.
        """
        self.metadata["wcs_catalog_success"] = False
        try:
            self.log.info("Attempting WCS-based donut catalog generation.")
            self.wcsCatTask.donutSelector = self.donutSelector
            result = self.wcsCatTask.run(
                refCatalogs=refCatalogs,
                exposure=exposure,
            )
            result.donutCatalog.meta["catalog_method"] = "wcs"
            self.metadata["wcs_catalog_success"] = True
            self.log.info("WCS-based donut catalog generation succeeded.")
            return result
        except RuntimeWarning as e:
            self.makeSubtask("directDetectTask")
            self.directDetectTask.donutSelector = self.donutSelector
            self.log.info(
                f"WCS catalog generation failed with error: {e}. "
                "Falling back to direct detection catalog generation."
            )
            direct_result = self.directDetectTask.run(
                exposure=exposure,
                camera=camera,
            )
            direct_result.donutCatalog.meta["catalog_method"] = "direct"
            self.log.info("Direct detection donut catalog generation succeeded.")
            return direct_result
