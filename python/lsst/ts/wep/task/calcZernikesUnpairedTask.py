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

import astropy.units as u
from astropy.table import QTable
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.calcZernikesTask import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    pos2f_dtype,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod
from itertools import zip_longest


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


class CalcZernikesUnpairedTaskConfig(
    CalcZernikesTaskConfig,
    pipelineConnections=CalcZernikesUnpairedTaskConnections,
):
    pass


class CalcZernikesUnpairedTask(CalcZernikesTask):
    """Calculate Zernikes using unpaired donuts."""

    ConfigClass = CalcZernikesUnpairedTaskConfig
    _DefaultName = "calcZernikesUnpairedTask"

    @timeMethod
    def run(self, donutStamps: DonutStamps) -> pipeBase.Struct:
        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStamps) == 0:
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
        if selectedDonuts[0].wep_im.defocalType == DefocalType.Extra:
            extraStamps = selectedDonuts
            intraStamps = []
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "extra"
        else:
            extraStamps = []
            intraStamps = selectedDonuts
            if len(donutQualityTable) > 0:
                donutQualityTable["DEFOCAL_TYPE"] = "intra"

        # Estimate Zernikes
        zkCoeffRaw = self.estimateZernikes.run(extraStamps, intraStamps)
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

        # Create the Table of results
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
            zip_longest(
                intraStamps,
                extraStamps,
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
                np.array(np.nan, dtype=pos2f_dtype) * u.deg
            ) if intra is None else (
                np.array(intra.calcFieldXY(), dtype=pos2f_dtype) * u.deg
            )
            row["extra_field"] = (
                np.array(np.nan, dtype=pos2f_dtype) * u.deg
            ) if extra is None else (
                np.array(extra.calcFieldXY(), dtype=pos2f_dtype) * u.deg
            )
            row["intra_centroid"] = (
                np.array(
                    (np.nan, np.nan),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            ) if intra is None else (
                np.array(
                    (intra.centroid_position.x, intra.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            )
            row["extra_centroid"] = (
                np.array(
                    (np.nan, np.nan),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            ) if extra is None else (
                np.array(
                    (extra.centroid_position.x, extra.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            )
            for key in ["MAG", "SN", "ENTROPY"]:
                for stamps, foc in [
                    (intraStamps, "intra"),
                    (extraStamps, "extra"),
                ]:
                    if len(stamps) > 0 and key in stamps.metadata:
                        val = stamps.metadata.getArray(key)[i]
                    else:
                        val = np.nan
                    row[f"{foc}_{key.lower()}"] = val
            zkTable.add_row(row)

        zkTable.meta["intra"] = {}
        zkTable.meta["extra"] = {}

        for dict_, stamps in [
            (zkTable.meta["intra"], intraStamps),
            (zkTable.meta["extra"], extraStamps),
        ]:
            if len(stamps) > 0:
                dict_["det_name"] = stamps.metadata["DET_NAME"]
                dict_["visit"] = stamps.metadata["VISIT"]
                dict_["dfc_dist"] = stamps.metadata["DFC_DIST"]
                dict_["band"] = stamps.metadata["BANDPASS"]
            else:
                dict_["det_name"] = ""
                dict_["visit"] = ""
                dict_["dfc_dist"] = ""
                dict_["band"] = ""


        zkTable.meta["cam_name"] = selectedDonuts.metadata["CAM_NAME"]

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(zkCoeffCombined.combinedZernikes)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
