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

__all__ = ["CutOutDonutsCwfsTaskConfig", "CutOutDonutsCwfsTask"]


from typing import Any

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from astropy.table import QTable
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
    CutOutDonutsBaseTaskConnections,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod


class CutOutDonutsCwfsTaskConnections(
    CutOutDonutsBaseTaskConnections, dimensions=("exposure", "detector", "instrument")  # type: ignore
):
    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
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
        multiple=False,
    )
    donutStampsOut = connectionTypes.Output(
        doc="Donut Postage Stamp Images (intra and extra)",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsCwfs",
        multiple=False,
    )


class CutOutDonutsCwfsTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutDonutsCwfsTaskConnections,  # type: ignore
):
    pass


class CutOutDonutsCwfsTask(CutOutDonutsBaseTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = CutOutDonutsCwfsTaskConfig
    _DefaultName = "CutOutDonutsCwfsTask"
    config: CutOutDonutsCwfsTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set final size (in pixels) of postage stamp images returned as
        # DonutStamp objects
        self.donutStampSize = self.config.donutStampSize
        # Add this many pixels onto each side of initial
        # cutout stamp beyond the size specified
        # in self.donutStampSize. This makes sure that
        # after recentroiding the donut from the catalog
        # position by convolving a template on the initial
        # cutout stamp we will still have a postage stamp
        # of size self.donutStampSize.
        self.initialCutoutPadding = self.config.initialCutoutPadding
        # Specify optical model
        self.opticalModel = self.config.opticalModel

        # Set which sensors are extra and intra focal
        # See LCA-13381 for definition
        self.extraFocalNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        self.intraFocalNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

    @timeMethod
    def run(
        self,
        exposure: afwImage.Exposure,
        donutCatalog: QTable,
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:
        cameraName = camera.getName()

        # Check which defocal type applies to exposure
        detectorName = exposure.getDetector().getName()
        defocalType = None
        if detectorName in self.extraFocalNames:
            defocalType = DefocalType.Extra
        elif detectorName in self.intraFocalNames:
            defocalType = DefocalType.Intra
        else:
            raise ValueError("Detector provided is not a corner wavefront sensor.")
        # If no donuts are in the donutCatalog for a set of exposures
        # then return empty donut stamps list
        # Catch that case before attempting to do any cutouts
        if len(donutCatalog) == 0:
            donutStampsOut = DonutStamps([])
            donutStampsOut = self.addVisitLevelMetadata(
                exposure,
                donutStampsOut,
                donutCatalog,
                defocalType,
            )
            donutStampsOut.use_archive = False
        else:
            donutStampsOut = self.cutOutStamps(
                exposure,
                donutCatalog,
                defocalType,
                cameraName,
            )

        return pipeBase.Struct(donutStampsOut=donutStampsOut)
