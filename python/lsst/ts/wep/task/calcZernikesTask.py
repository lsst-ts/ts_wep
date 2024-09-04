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
    "CalcZernikesTaskConnections",
    "CalcZernikesTaskConfig",
    "CalcZernikesTask",
]

import abc

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
import pandas as pd
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.donutStampSelectorTask import DonutStampSelectorTask
from lsst.ts.wep.task.estimateZernikesTieTask import EstimateZernikesTieTask
from lsst.utils.timer import timeMethod


class CalcZernikesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),
):
    donutStampsExtra = connectionTypes.Input(
        doc="Extra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
    )
    donutStampsIntra = connectionTypes.Input(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
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
    donutsExtraQuality = connectionTypes.Output(
        doc="Quality information for extra-focal donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="DataFrame",
        name="donutsExtraQuality",
    )
    donutsIntraQuality = connectionTypes.Output(
        doc="Quality information for intra-focal donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="DataFrame",
        name="donutsIntraQuality",
    )


class CalcZernikesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CalcZernikesTaskConnections,
):
    estimateZernikes = pexConfig.ConfigurableField(
        target=EstimateZernikesTieTask,
        doc=str(
            "Choice of task to estimate Zernikes from pairs of donuts. "
            + "(the default is EstimateZernikesTieTask)"
        ),
    )
    combineZernikes = pexConfig.ConfigurableField(
        target=CombineZernikesSigmaClipTask,
        doc=str(
            "Choice of task to combine the Zernikes from pairs of "
            + "donuts into a single value for the detector. (The default "
            + "is CombineZernikesSigmaClipTask.)"
        ),
    )
    donutStampSelector = pexConfig.ConfigurableField(
        target=DonutStampSelectorTask, doc="How to select donut stamps."
    )
    doDonutStampSelector = pexConfig.Field(
        doc="Whether or not to run donut stamp selector.", dtype=bool, default=False
    )


class CalcZernikesTask(pipeBase.PipelineTask, metaclass=abc.ABCMeta):
    """Base class for calculating Zernike coeffs from pairs of DonutStamps.

    This class joins the EstimateZernikes and CombineZernikes subtasks to
    be run on sets of DonutStamps.
    """

    ConfigClass = CalcZernikesTaskConfig
    _DefaultName = "calcZernikesBaseTask"

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
    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
    ) -> pipeBase.Struct:
        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0 or len(donutStampsIntra) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=np.full(19, np.nan),
                outputZernikesAvg=np.full(19, np.nan),
                donutsExtraQuality=pd.DataFrame([]),
                donutsIntraQuality=pd.DataFrame([]),
            )

        # Run donut stamp selection. By default all donut stamps are selected
        # and we are provided with donut quality table containing
        # SN and entropy information if present in donut stamps
        # metadata.
        if self.doDonutStampSelector:
            self.log.info("Running Donut Stamp Selector")
            selectionExtra = self.donutStampSelector.run(donutStampsExtra)
            selectionIntra = self.donutStampSelector.run(donutStampsIntra)

            # If no donuts get selected, also return Zernike
            # coefficients as nan.
            if (
                len(selectionExtra.donutStampsSelect) == 0
                or len(selectionIntra.donutStampsSelect) == 0
            ):
                self.log.info("No donut stamps were selected.")
                return pipeBase.Struct(
                    outputZernikesRaw=np.full(19, np.nan),
                    outputZernikesAvg=np.full(19, np.nan),
                    donutsExtraQuality=pd.DataFrame([]),
                    donutsIntraQuality=pd.DataFrame([]),
                )

            # Estimate Zernikes from the collection of selected stamps
            zkCoeffRaw = self.estimateZernikes.run(
                selectionExtra.donutStampsSelect, selectionIntra.donutStampsSelect
            )
            zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

            return pipeBase.Struct(
                outputZernikesAvg=np.array(zkCoeffCombined.combinedZernikes),
                outputZernikesRaw=np.array(zkCoeffRaw.zernikes),
                donutsExtraQuality=selectionExtra.donutsQuality,
                donutsIntraQuality=selectionIntra.donutsQuality,
            )
        else:
            # Estimate Zernikes from the collection of all stamps,
            # without using entropy or SN
            zkCoeffRaw = self.estimateZernikes.run(donutStampsExtra, donutStampsIntra)
            zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)
            return pipeBase.Struct(
                outputZernikesAvg=np.array(zkCoeffCombined.combinedZernikes),
                outputZernikesRaw=np.array(zkCoeffRaw.zernikes),
                donutsExtraQuality=pd.DataFrame([]),
                donutsIntraQuality=pd.DataFrame([]),
            )
