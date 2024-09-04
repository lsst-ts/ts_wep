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
    "CalcZernikesUnpairedTaskConnections",
    "CalcZernikesUnpairedTaskConfig",
    "CalcZernikesUnpairedTask",
]

import lsst.pipe.base as pipeBase
import numpy as np
import pandas as pd
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTaskConfig
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod


class CalcZernikesUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),
):
    donutStamps = connectionTypes.Input(
        doc="Defocused Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
    )
    outputZernikesRaw = connectionTypes.Output(
        doc="Zernike Coefficients from all donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
    )
    outputZernikesAvg = connectionTypes.Output(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
    )
    donutsQuality = connectionTypes.Output(
        doc="Quality information for donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="DataFrame",
        name="donutsQuality",
    )


class CalcZernikesUnpairedTaskConfig(
    CalcZernikesTaskConfig,
    pipelineConnections=CalcZernikesUnpairedTaskConnections,
):
    pass


class CalcZernikesUnpairedTask(pipeBase.PipelineTask):
    """Calculate Zernikes using unpaired donuts."""

    ConfigClass = CalcZernikesUnpairedTaskConfig
    _DefaultName = "calcZernikesUnpairedTask"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Create subtasks
        self.estimateZernikes = self.config.estimateZernikes
        self.makeSubtask("estimateZernikes")

        self.combineZernikes = self.config.combineZernikes
        self.makeSubtask("combineZernikes")

        self.donutStampSelector = self.config.donutStampSelector
        self.makeSubtask("donutStampSelector")

        self.doDonutStampSelector = self.config.doDonutStampSelector

    @timeMethod
    def run(self, donutStamps: DonutStamps) -> pipeBase.Struct:
        # Get jmax
        jmax = self.estimateZernikes.config.maxNollIndex

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStamps) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=np.full(jmax - 4, np.nan),
                outputZernikesAvg=np.full(jmax - 4, np.nan),
                donutsQuality=pd.DataFrame([]),
            )

        # Run donut selection
        if self.doDonutStampSelector:
            self.log.info("Running Donut Stamp Selector")
            selection = self.donutStampSelector.run(donutStamps)

            # If no donuts get selected, return NaNs
            if len(selection.donutStampsSelect) == 0:
                self.log.info("No donut stamps were selected.")
                return pipeBase.Struct(
                    outputZernikesRaw=np.full(jmax - 4, np.nan),
                    outputZernikesAvg=np.full(jmax - 4, np.nan),
                    donutsQuality=pd.DataFrame([]),
                )

            # Save selection and quality
            selectedDonuts = selection.donutStampsSelect
            donutsQuality = selection.donutsQuality
        else:
            selectedDonuts = donutStamps
            donutsQuality = pd.DataFrame([])

        # Assign stamps to either intra or extra
        if selectedDonuts[0].wep_im.defocalType == DefocalType.Extra:
            extraStamps = selectedDonuts
            intraStamps = []
        else:
            extraStamps = []
            intraStamps = selectedDonuts

        # Estimate Zernikes
        zkCoeffRaw = self.estimateZernikes.run(extraStamps, intraStamps)
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)
        return pipeBase.Struct(
            outputZernikesAvg=np.array(zkCoeffCombined.combinedZernikes),
            outputZernikesRaw=np.array(zkCoeffRaw.zernikes),
            donutsQuality=donutsQuality,
        )
