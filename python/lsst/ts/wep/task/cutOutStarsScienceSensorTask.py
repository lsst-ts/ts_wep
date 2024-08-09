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
    "CutOutStarsScienceSensorTaskConnections",
    "CutOutStarsScienceSensorTaskConfig",
    "CutOutStarsScienceSensorTask",
]

import typing

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import pandas as pd
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
    CutOutDonutsBaseTaskConnections,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod


class CutOutStarsScienceSensorTaskConnections(
    CutOutDonutsBaseTaskConnections, dimensions=("detector", "instrument")
):
    starStamps = connectionTypes.Output(
        doc="In-Focus Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="starStamps",
        multiple=True,
    )


class CutOutStarsScienceSensorTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutStarsScienceSensorTaskConnections,
):
    pass


class CutOutStarsScienceSensorTask(CutOutDonutsBaseTask):
    """
    Cut out stamps for in-focus sources.
    """

    ConfigClass = CutOutStarsScienceSensorTaskConfig
    _DefaultName = "CutOutStarsScienceSensorTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We need to be able to take pairs of detectors from the full
        set of detector exposures and run the task. Then we need to put
        the outputs back into the butler repository with
        the appropriate butler dataIds.

        For the `outputZernikesRaw` and `outputZernikesAvg`
        we only have one set of values per pair of wavefront detectors
        so we put this in the dataId associated with the
        extra-focal detector.
        """
        # Get the inputs from butler
        camera = butlerQC.get(inputRefs.camera)
        exposure = butlerQC.get(inputRefs.exposures)
        srcCat = butlerQC.get(inputRefs.donutCatalog)

        # Run the task
        outputs = self.run(exposure, srcCat, camera)

        # Put the outputs in the butler
        butlerQC.put(outputs.starStamps, outputRefs.starStamps)

    @timeMethod
    def run(
        self,
        exposure: typing.List[afwImage.Exposure],
        sourceCatalog: typing.List[pd.DataFrame],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        cameraName = camera.getName()
        # Shall we add here testing  based on focusZ whether
        # the exposure is indeed in-focus?

        # Get the star stamps from in-focus exposure
        starStamps = self.cutOutStamps(
            exposure[0],
            sourceCatalog[0],
            DefocalType.Focus,
            cameraName,
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(starStamps) == 0:
            return pipeBase.Struct(starStamps=DonutStamps([]))

        # Return in-focus stamps as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(starStamps=starStamps)
