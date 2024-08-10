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

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import pandas as pd
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod


class CutOutStarsScienceSensorTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="DataFrame",
        name="donutCatalog",
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
        exposureRefs = [v for v in inputRefs.exposures]
        donutCatalogRefs = [v for v in inputRefs.donutCatalog]

        for i in range(len(exposureRefs)):
            exposure = butlerQC.get(exposureRefs[i])
            srcCat = butlerQC.get(donutCatalogRefs[i])

            # Run the task
            outputs = self.run(exposure, srcCat, camera)

            # Put the outputs in the butler
            butlerQC.put(outputs.starStamps, outputRefs.starStamps[i])

    @timeMethod
    def run(
        self,
        exposure: afwImage.Exposure,
        sourceCatalog: pd.DataFrame,
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        cameraName = camera.getName()
        # Shall we add here testing  based on focusZ whether
        # the exposure is indeed in-focus?

        # Get the star stamps from in-focus exposure
        starStamps = self.cutOutStamps(
            exposure,
            sourceCatalog,
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
