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
    "GenerateDonutTaskBaseConnections",
    "GenerateDonutTaskBaseConfig",
    "GenerateDonutTaskBase",
]

from typing import Any

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask


class GenerateDonutTaskBaseConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),  # type: ignore
):
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


class GenerateDonutTaskBaseConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutTaskBaseConnections,  # type: ignore
):
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Size of detector edge margin in pixels", dtype=int, default=80
    )
    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )
    doSubtractBackground: pexConfig.Field = pexConfig.Field(
        doc="Do background subtration?",
        dtype=bool,
        default=True,
    )


class GenerateDonutTaskBase(pipeBase.PipelineTask):
    """Base class for generating donut tables."""

    ConfigClass = GenerateDonutTaskBaseConfig
    _DefaultName = "generateDonutTaskBase"
    config: GenerateDonutTaskBaseConfig
    donutSelector: DonutSourceSelectorTask
    subtractBackground: SubtractBackgroundTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set up background task if we need it
        if self.config.doSubtractBackground:
            self.makeSubtask("subtractBackground")

        # Set up the donut selector task if we need it
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

    def _subtractBackground(self, exposure: afwImage.Exposure) -> None:
        """Subtract the background from the exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
        """
        if self.config.doSubtractBackground:
            self.log.info("Running background subtraction")
            self.subtractBackground.run(exposure=exposure)
