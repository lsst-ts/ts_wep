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

import numpy as np
from astropy.table import QTable, Table

import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.utils.timer import timeMethod

def lookupIntrinsicTables(
    datasetType: DatasetType,
    registry: Registry,
    dataId: DataCoordinate,
    collections: Sequence[str]
) -> list[DatasetRef]:
    detector = dataId["detector"]
    isCornerChip = (detector in intra_focal_ids) or (detector in extra_focal_ids)

    refs = [registry.findDataset(datasetType, dataId, collections=collections)]
    return refs


class CalcZernikesUnpairedTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument", "physical_filter"),  # type: ignore
):
    donutStamps = connectionTypes.Input(
        doc="Defocused Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
    )
    intrinsicTable = connectionTypes.Input(
        doc="Intrinsic Zernike Map for the instrument",
        dimensions=("detector", "instrument", "physical_filter"),
        storageClass="ArrowAstropy",
        name="intrinsic_aberrations_temp",
        lookupFunction=lookupIntrinsicTables,
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

    @timeMethod
    def run(
        self,
        donutStamps: DonutStamps,
        intrinsicTable: Table,
        numCores: int = 1,
    ) -> pipeBase.Struct:
        if len(donutStamps) == 0:
            self.log.info("No donut stamps available.")
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

        # Assign stamps to either intra or extra, and build intrinsic map
        defocalType = donutStamps.metadata["DFC_TYPE"]
        if defocalType == "extra":
            self.stampsIntra = DonutStamps([])
            self.stampsExtra = selectedDonuts
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "extra"
            self.intrinsicMapExtra = self._createIntrinsicMap(intrinsicTable)
            self.intrinsicMapIntra = None
        else:
            self.stampsIntra = selectedDonuts
            self.stampsExtra = DonutStamps([])
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "intra"
            self.intrinsicMapExtra = None
            self.intrinsicMapIntra = self._createIntrinsicMap(intrinsicTable)

        # Estimate Zernikes
        zkCoeffRaw = self.estimateZernikes.run(
            self.stampsExtra,
            self.stampsIntra,
            numCores=numCores,
        )

        # Save the outputs in the table
        zkTable = self.createZkTable(zkCoeffRaw)
        zkTable.meta["estimatorInfo"] = zkCoeffRaw.wfEstInfo

        # Combine Zernikes
        zkTable = self.combineZernikes.run(zkTable).combinedTable

        avg = zkTable[zkTable["label"] == "average"]
        outputZernikesAvg = np.array([avg[col].to_value("um")[0] for col in avg.meta["opd_columns"]])

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(outputZernikesAvg)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
