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
from itertools import zip_longest

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
        storageClass="AstropyQTable",
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
        target=DonutStampSelectorTask,
        doc="How to select donut stamps.",
    )
    doDonutStampSelector = pexConfig.Field(
        doc="Whether or not to run donut stamp selector.",
        dtype=bool,
        default=True,
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
        self.nollIndices = self.estimateZernikes.config.nollIndices

        self.combineZernikes = self.config.combineZernikes
        self.makeSubtask("combineZernikes")

        self.donutStampSelector = self.config.donutStampSelector
        self.makeSubtask("donutStampSelector")

        self.doDonutStampSelector = self.config.doDonutStampSelector

    def initZkTable(self) -> QTable:
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
            ("intra_frac_bad_pix", "<f4"),
            ("extra_frac_bad_pix", "<f4"),
        ]
        for j in self.nollIndices:
            dtype.append((f"Z{j}", "<f4"))

        table = QTable(dtype=dtype)

        # Assign units where appropriate
        table["intra_field"].unit = u.deg
        table["extra_field"].unit = u.deg
        table["intra_centroid"].unit = u.pixel
        table["extra_centroid"].unit = u.pixel
        for j in self.nollIndices:
            table[f"Z{j}"].unit = u.nm

        return table

    def createZkTable(
        self,
        extraStamps: DonutStamps,
        intraStamps: DonutStamps,
        zkCoeffRaw: pipeBase.Struct,
        zkCoeffCombined: pipeBase.Struct,
    ) -> QTable:
        """Create the Zernike table to store Zernike Coefficients.

        Note this is written with the assumption that either extraStamps or
        intraStamps MIGHT be empty. This is because calcZernikesUnpairedTask
        also uses this method.

        Parameters
        ----------
        extraStamps: DonutStamps
            The extrafocal stamps
        intraStamps: DonutStamps
            The intrafocal stamps
        zkCoeffRaw: pipeBase.Struct
            All zernikes returned by self.estimateZernikes.run(...)
        zkCoeffCombined
            Combined zernikes returned by self.combineZernikes.run(...)

        Returns
        -------
        table : `astropy.table.QTable`
            Table with the Zernike coefficients
        """
        zkTable = self.initZkTable()
        zkTable.add_row(
            {
                "label": "average",
                "used": True,
                **{
                    f"Z{j}": zkCoeffCombined.combinedZernikes[i] * u.micron
                    for i, j in enumerate(self.nollIndices)
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
            # If zk is None, we need to stop. This can happen when running
            # paired Zernike estimation and the number of intra/extra stamps
            # is not the same
            if zk is None:
                break

            row = dict()
            row["label"] = f"pair{i+1}"
            row["used"] = not flag
            row.update(
                {f"Z{j}": zk[i] * u.micron for i, j in enumerate(self.nollIndices)}
            )
            row["intra_field"] = (
                (np.array(np.nan, dtype=pos2f_dtype) * u.deg)
                if intra is None
                else (np.array(intra.calcFieldXY(), dtype=pos2f_dtype) * u.deg)
            )
            row["extra_field"] = (
                (np.array(np.nan, dtype=pos2f_dtype) * u.deg)
                if extra is None
                else (np.array(extra.calcFieldXY(), dtype=pos2f_dtype) * u.deg)
            )
            row["intra_centroid"] = (
                (
                    np.array(
                        (np.nan, np.nan),
                        dtype=pos2f_dtype,
                    )
                    * u.pixel
                )
                if intra is None
                else (
                    np.array(
                        (intra.centroid_position.x, intra.centroid_position.y),
                        dtype=pos2f_dtype,
                    )
                    * u.pixel
                )
            )
            row["extra_centroid"] = (
                (
                    np.array(
                        (np.nan, np.nan),
                        dtype=pos2f_dtype,
                    )
                    * u.pixel
                )
                if extra is None
                else (
                    np.array(
                        (extra.centroid_position.x, extra.centroid_position.y),
                        dtype=pos2f_dtype,
                    )
                    * u.pixel
                )
            )
            for key in ["MAG", "SN", "ENTROPY", "FRAC_BAD_PIX"]:
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
            dict_["det_name"] = "" if len(stamps) == 0 else stamps.metadata["DET_NAME"]
            dict_["visit"] = "" if len(stamps) == 0 else stamps.metadata["VISIT"]
            dict_["dfc_dist"] = "" if len(stamps) == 0 else stamps.metadata["DFC_DIST"]
            dict_["band"] = "" if len(stamps) == 0 else stamps.metadata["BANDPASS"]
            dict_["boresight_rot_angle_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_ROT_ANGLE_RAD"]
            )
            dict_["boresight_par_angle_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_PAR_ANGLE_RAD"]
            )
            dict_["boresight_alt_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_ALT_RAD"]
            )
            dict_["boresight_az_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_AZ_RAD"]
            )
            dict_["boresight_ra_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_RA_RAD"]
            )
            dict_["boresight_dec_rad"] = (
                "" if len(stamps) == 0 else stamps.metadata["BORESIGHT_DEC_RAD"]
            )
            dict_["mjd"] = "" if len(stamps) == 0 else stamps.metadata["MJD"]

        if len(intraStamps) > 0:
            zkTable.meta["cam_name"] = intraStamps.metadata["CAM_NAME"]
        else:
            zkTable.meta["cam_name"] = extraStamps.metadata["CAM_NAME"]
        if len(intraStamps) > 0 and len(extraStamps) > 0:
            assert intraStamps.metadata["CAM_NAME"] == extraStamps.metadata["CAM_NAME"]

        return zkTable

    def empty(self, qualityTable=None) -> pipeBase.Struct:
        """Return empty results if no donuts are available. If
        it is a result of no quality donuts we still include the
        quality table results instead of an empty quality table.

        Parameters
        ----------
        qualityTable : astropy.table.QTable
            Quality table created with donut stamp input.

        Returns
        -------
        lsst.pipe.base.Struct
            Empty output tables for zernikes. Empty quality table
            if no donuts. Otherwise contains quality table
            with donuts that all failed to pass quality check.
        """
        qualityTableCols = [
            "SN",
            "ENTROPY",
            "ENTROPY_SELECT",
            "SN_SELECT",
            "FINAL_SELECT",
            "DEFOCAL_TYPE",
        ]
        if qualityTable is None:
            donutQualityTable = QTable({name: [] for name in qualityTableCols})
        else:
            donutQualityTable = qualityTable
        return pipeBase.Struct(
            outputZernikesRaw=np.atleast_2d(np.full(len(self.nollIndices), np.nan)),
            outputZernikesAvg=np.atleast_2d(np.full(len(self.nollIndices), np.nan)),
            zernikes=self.initZkTable(),
            donutQualityTable=donutQualityTable,
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
                return self.empty(qualityTable=donutQualityTable)
        else:
            donutQualityTable = QTable([])

        # Estimate Zernikes from the collection of selected stamps
        zkCoeffRaw = self.estimateZernikes.run(donutStampsExtra, donutStampsIntra)
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

        zkTable = self.createZkTable(
            donutStampsExtra,
            donutStampsIntra,
            zkCoeffRaw,
            zkCoeffCombined,
        )

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(zkCoeffCombined.combinedZernikes)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
