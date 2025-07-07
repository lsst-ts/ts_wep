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
    "CutOutDonutsUnpairedTaskConnections",
    "CutOutDonutsUnpairedTaskConfig",
    "CutOutDonutsUnpairedTask",
]


from typing import Any

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from astropy.table import QTable
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
)
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod


class CutOutDonutsUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("exposure", "instrument"),  # type: ignore
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
        multiple=True,
    )
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="AstropyQTable",
        name="donutTable",
        multiple=True,
    )
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    donutStamps = connectionTypes.Output(
        doc="Defocused Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
        multiple=True,
    )


class CutOutDonutsUnpairedTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutDonutsUnpairedTaskConnections,  # type: ignore
):
    pass


class CutOutDonutsUnpairedTask(CutOutDonutsBaseTask):
    """Cutout donuts without pairing any exposures."""

    ConfigClass = CutOutDonutsUnpairedTaskConfig
    _DefaultName = "CutOutDonutsUnpairedTask"
    config: CutOutDonutsUnpairedTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @timeMethod
    def run(
        self,
        exposures: list[afwImage.Exposure],
        donutCatalog: list[QTable],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:
        # Loop over exposures (and corresponding catalogs)
        stampList = []
        for exposure, catalog in zip(exposures, donutCatalog):
            # Determine which side of focus
            if exposure.visitInfo.focusZ > 0:
                defocalType = DefocalType.Extra
            else:
                defocalType = DefocalType.Intra

            # Cutout the stamps
            stampList.append(
                self.cutOutStamps(
                    exposure,
                    catalog,
                    defocalType,
                    camera.getName(),
                )
            )

        return pipeBase.Struct(donutStamps=stampList)
