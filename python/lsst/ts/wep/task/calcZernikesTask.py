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

import astropy.units as u
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy.table import QTable, vstack
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.donutStampSelectorTask import DonutStampSelectorTask
from lsst.ts.wep.task.estimateZernikesTieTask import EstimateZernikesTieTask
from lsst.utils.timer import timeMethod

pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])


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
    zernikes = connectionTypes.Output(
        doc="Zernike Coefficients for individual donuts and average over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyTable",
        name="zernikes",
    )
    donutQualityTable = connectionTypes.Output(
        doc="Quality information for donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
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
        self.maxNollIndex = self.estimateZernikes.config.maxNollIndex

        self.combineZernikes = self.config.combineZernikes
        self.makeSubtask("combineZernikes")

        self.donutStampSelector = self.config.donutStampSelector
        self.makeSubtask("donutStampSelector")

        self.doDonutStampSelector = self.config.doDonutStampSelector

    def initZkTable(self):
        """Initialize the table to store the Zernike coefficients

        Returns
        -------
        table : `astropy.table.QTable`
            Table to store the Zernike coefficients
        """
        dtype = [
            ("label", "<U12"),
            ("used", np.bool_),
            ("intra_field", pos2f_dtype),
            ("extra_field", pos2f_dtype),
            ("intra_centroid", pos2f_dtype),
            ("extra_centroid", pos2f_dtype),
            ("intra_mag", "<f4"),
            ("extra_mag", "<f4"),
            ("intra_sn", "<f4"),
            ("extra_sn", "<f4"),
            ("intra_entropy", "<f4"),
            ("extra_entropy", "<f4"),
        ]
        for j in range(4, self.maxNollIndex + 1):
            dtype.append((f"Z{j}", "<f4"))

        table = QTable(dtype=dtype)

        # Assign units where appropriate
        table["intra_field"].unit = u.deg
        table["extra_field"].unit = u.deg
        table["intra_centroid"].unit = u.pixel
        table["extra_centroid"].unit = u.pixel
        for j in range(4, self.maxNollIndex + 1):
            table[f"Z{j}"].unit = u.nm

        return table

    def empty(self):
        """Return empty results if no donuts are available."""
        qualityTableCols = [
            "SN",
            "ENTROPY",
            "ENTROPY_SELECT",
            "SN_SELECT",
            "FINAL_SELECT",
            "DEFOCAL_TYPE",
        ]
        return pipeBase.Struct(
            outputZernikesRaw=np.atleast_2d(np.full(self.maxNollIndex - 3, np.nan)),
            outputZernikesAvg=np.atleast_2d(np.full(self.maxNollIndex - 3, np.nan)),
            zernikes=self.initZkTable(),
            donutQualityTable=QTable({name: [] for name in qualityTableCols}),
        )

    @timeMethod
    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
    ) -> pipeBase.Struct:
        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0 or len(donutStampsIntra) == 0:
            return self.empty()

        # Run donut stamp selection. By default all donut stamps are selected
        # and we are provided with donut quality table.
        if self.doDonutStampSelector:
            self.log.info("Running Donut Stamp Selector")
            selectionExtra = self.donutStampSelector.run(donutStampsExtra)
            selectionIntra = self.donutStampSelector.run(donutStampsIntra)
            donutExtraQuality = selectionExtra.donutsQuality
            donutIntraQuality = selectionIntra.donutsQuality
            donutStampsExtra = selectionExtra.donutStampsSelect
            donutStampsIntra = selectionIntra.donutStampsSelect

            donutExtraQuality["DEFOCAL_TYPE"] = "extra"
            donutIntraQuality["DEFOCAL_TYPE"] = "intra"
            donutQualityTable = vstack([donutExtraQuality, donutIntraQuality])

            # If no donuts get selected, also return Zernike
            # coefficients as nan.
            if (
                len(selectionExtra.donutStampsSelect) == 0
                or len(selectionIntra.donutStampsSelect) == 0
            ):
                self.log.info("No donut stamps were selected.")
                return self.empty()
        else:
            donutQualityTable = QTable([])

        # Estimate Zernikes from the collection of selected stamps
        zkCoeffRaw = self.estimateZernikes.run(donutStampsExtra, donutStampsIntra)
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

        zkTable = self.initZkTable()
        zkTable.add_row(
            {
                "label": "average",
                "used": True,
                **{
                    f"Z{j}": zkCoeffCombined.combinedZernikes[j - 4] * u.micron
                    for j in range(4, self.maxNollIndex + 1)
                },
                "intra_field": np.nan,
                "extra_field": np.nan,
                "intra_centroid": np.nan,
                "extra_centroid": np.nan,
                "intra_mag": np.nan,
                "extra_mag": np.nan,
                "intra_sn": np.nan,
                "extra_sn": np.nan,
                "intra_entropy": np.nan,
                "extra_entropy": np.nan,
            }
        )
        for i, (intra, extra, zk, flag) in enumerate(
            zip(
                donutStampsIntra,
                donutStampsExtra,
                zkCoeffRaw.zernikes,
                zkCoeffCombined.flags,
            )
        ):
            row = dict()
            row["label"] = f"pair{i+1}"
            row["used"] = not flag
            row.update(
                {f"Z{j}": zk[j - 4] * u.micron for j in range(4, self.maxNollIndex + 1)}
            )
            row["intra_field"] = (
                np.array(intra.calcFieldXY(), dtype=pos2f_dtype) * u.deg
            )
            row["extra_field"] = (
                np.array(extra.calcFieldXY(), dtype=pos2f_dtype) * u.deg
            )
            row["intra_centroid"] = (
                np.array(
                    (intra.centroid_position.x, intra.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            )
            row["extra_centroid"] = (
                np.array(
                    (extra.centroid_position.x, extra.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            )
            for key in ["MAG", "SN", "ENTROPY"]:
                for stamps, foc in [
                    (donutStampsIntra, "intra"),
                    (donutStampsExtra, "extra"),
                ]:
                    if key in stamps.metadata:
                        val = stamps.metadata.getArray(key)[i]
                    else:
                        val = np.nan
                    row[f"{foc}_{key.lower()}"] = val
            zkTable.add_row(row)

        zkTable.meta["intra"] = {}
        zkTable.meta["extra"] = {}

        for dict_, stamps in [
            (zkTable.meta["intra"], donutStampsIntra),
            (zkTable.meta["extra"], donutStampsExtra),
        ]:
            dict_["det_name"] = stamps.metadata["DET_NAME"]
            dict_["visit"] = stamps.metadata["VISIT"]
            dict_["dfc_dist"] = stamps.metadata["DFC_DIST"]
            dict_["band"] = stamps.metadata["BANDPASS"]

        zkTable.meta["cam_name"] = donutStampsIntra.metadata["CAM_NAME"]
        assert (
            donutStampsIntra.metadata["CAM_NAME"]
            == donutStampsExtra.metadata["CAM_NAME"]
        )

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(zkCoeffCombined.combinedZernikes)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
