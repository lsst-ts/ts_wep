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
from astropy.table import QTable
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.utils.timer import timeMethod


class CalcZernikesUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),  # type: ignore
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
    zernikes = connectionTypes.Output(
        doc="Zernike Coefficients for individual donuts and average over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="zernikes",
    )
    donutQualityTable = connectionTypes.Output(
        doc="Quality information for donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
    )


class CalcZernikesUnpairedTaskConfig(
    CalcZernikesTaskConfig,
    pipelineConnections=CalcZernikesUnpairedTaskConnections,  # type: ignore
):
    pass


class CalcZernikesUnpairedTask(CalcZernikesTask):
    """Calculate Zernikes using unpaired donuts."""

    ConfigClass = CalcZernikesUnpairedTaskConfig
    _DefaultName = "calcZernikesUnpairedTask"

    def createUnpairedZkTable(
        self,
        zkCoeffRaw: pipeBase.Struct,
        zkCoeffCombined: pipeBase.Struct,
    ) -> QTable:
        """Create the Zernike table to store Zernike Coefficients.

        Parameters
        ----------
        zkCoeffRaw: pipeBase.Struct
            All zernikes returned by self.estimateZernikes.run(...)
        zkCoeffCombined
            Combined zernikes returned by self.combineZernikes.run(...)

        Returns
        -------
        table : `astropy.table.QTable`
            Table with the Zernike coefficients
        """

        if self.stampsExtra is None:
            extraStamps = DonutStamps([])
            intraStamps = self.stampsIntra
        elif self.stampsIntra is None:
            extraStamps = self.stampsExtra
            intraStamps = DonutStamps([])

        return self.createZkTable(
            extraStamps,
            intraStamps,
            zkCoeffRaw,
            zkCoeffCombined
        )


    @timeMethod
    def run(
        self,
        donutStamps: DonutStamps,
        numCores: int = 1,
    ) -> pipeBase.Struct:

        if len(donutStamps) == 0:
            self.stampsIntra = None
            self.stampsExtra = None
            return self.empty()

        # Run donut selection
        if self.doDonutStampSelector:
            self.log.info("Running Donut Stamp Selector")
            selection = self.donutStampSelector.run(donutStamps)

            # If no donuts get selected, return NaNs
            if len(selection.donutStampsSelect) == 0:
                self.log.info("No donut stamps were selected.")
                return self.empty()

            # Save selection and quality
            selectedDonuts = selection.donutStampsSelect
            donutQualityTable = selection.donutsQuality
        else:
            selectedDonuts = donutStamps
            donutQualityTable = QTable([])

        # Assign stamps to either intra or extra

        defocalType = donutStamps.metadata["DFC_TYPE"]
        if defocalType == "extra":
            self.stampsExtra = selectedDonuts
            self.stampsIntra = None
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "extra"
            zkCoeffRaw = self.estimateZernikes.run(
                self.stampsExtra,
                DonutStamps([]),
                numCores=numCores
            )
        else:
            self.stampsIntra = selectedDonuts
            self.stampsExtra = None
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "intra"
            zkCoeffRaw = self.estimateZernikes.run(
                DonutStamps([]),
                self.stampsIntra,
                numCores=numCores
            )
        # Combine Zernikes
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

        zkTable = self.createUnpairedZkTable(
            zkCoeffRaw, zkCoeffCombined
        )
        zkTable.meta["estimatorInfo"] = zkCoeffRaw.wfEstInfo

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(zkCoeffCombined.combinedZernikes)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
