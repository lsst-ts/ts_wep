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
from typing import Any, cast, Sequence

import astropy.units as u
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import QTable, Table, vstack
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import DataCoordinate, DatasetType, Registry, DatasetRef

from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
    connectionTypes,
)
from lsst.ts.wep.task.combineZernikesMeanTask import CombineZernikesMeanTask
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamp, DonutStamps
from lsst.ts.wep.task.donutStampSelectorTask import DonutStampSelectorTask
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.utils.timer import timeMethod

pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])
intra_focal_ids = set([192, 196, 200, 204])
extra_focal_ids = set([191, 195, 199, 203])


def lookupIntrinsicTables(
    datasetType: DatasetType,
    registry: Registry,
    dataId: DataCoordinate,
    collections: Sequence[str]
) -> list[DatasetRef]:
    """Assumes that the dataId is always for the extra focal at present
    """
    detector = dataId["detector"]
    isCornerChip = (detector in intra_focal_ids) or (detector in extra_focal_ids)

    refs = [registry.findDataset(datasetType, dataId, collections=collections)]
    if isCornerChip:  # we're running a CWFS pair, not a FAM image
        dataId2 = DataCoordinate.standardize(dataId, detector=dataId["detector"] + 1)
        refs.append(registry.findDataset(datasetType, dataId2, collections=collections))
    return refs


class CalcZernikesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument", "physical_filter"),  # type: ignore
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
    intrinsicTables = connectionTypes.PrerequisiteInput(
        doc="Intrinsic Zernike Map for the instrument",
        dimensions=("detector", "instrument", "physical_filter"),
        storageClass="ArrowAstropy",
        name="intrinsic_aberrations_temp",
        multiple=True,
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


class CalcZernikesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CalcZernikesTaskConnections,  # type: ignore
):
    estimateZernikes: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=EstimateZernikesDanishTask,
        doc=str(
            "Choice of task to estimate Zernikes from pairs of donuts. "
            + "(the default is EstimateZernikesTieTask)"
        ),
    )
    combineZernikes: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=CombineZernikesSigmaClipTask,
        doc=str(
            "Choice of task to combine the Zernikes from pairs of "
            + "donuts into a single value for the detector. (The default "
            + "is CombineZernikesSigmaClipTask.)"
        ),
    )
    donutStampSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutStampSelectorTask,
        doc="How to select donut stamps.",
    )
    doDonutStampSelector: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut stamp selector."
        + "If this is False, then we do not get donutQualityTable."
        + "(The default is True). It is also possible to run"
        + "donut stamp selector (with this config set to True), but"
        + "turn off doSelection config inside the donut stamp selector,"
        + "which would return all donuts as selected, as well as"
        + "returning a quality table.",
        dtype=bool,
        default=True,
    )
    doBlurClip: pexConfig.Field = pexConfig.Field(
        doc="Remove donuts with outlier donut blur fwhm from" + "final averages.", dtype=bool, default=True
    )


class CalcZernikesTask(pipeBase.PipelineTask, metaclass=abc.ABCMeta):
    """Base class for calculating Zernike coeffs from pairs of DonutStamps.

    This class joins the EstimateZernikes and CombineZernikes subtasks to
    be run on sets of DonutStamps.
    """

    ConfigClass = CalcZernikesTaskConfig
    _DefaultName = "calcZernikesBaseTask"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Create subtasks
        config = cast(CalcZernikesTaskConfig, self.config)

        self.estimateZernikes = config.estimateZernikes
        self.makeSubtask("estimateZernikes")
        self.nollIndices = self.estimateZernikes.config.nollIndices

        self.combineZernikes = config.combineZernikes
        self.makeSubtask("combineZernikes")

        self.donutStampSelector = config.donutStampSelector
        self.makeSubtask("donutStampSelector")

        self.doDonutStampSelector = config.doDonutStampSelector
        self.doBlurClip = config.doBlurClip

        # Initialize the donut stamps to empty placeholders
        self.stampsExtra = DonutStamps([])
        self.stampsIntra = DonutStamps([])

    def _createIntrinsicMap(self, intrinsicTable: Table | None) -> LinearNDInterpolator | None:
        """Create a RegularGridInterpolator for the intrinsic Zernike map.

        Parameters
        ----------
        intrinsicTable : astropy.table.Table or None
            Table containing the intrinsic Zernike coefficients.

        Returns
        -------
        interpolator : scipy.interpolate.RegularGridInterpolator or None
            Interpolator for the intrinsic Zernike map,
            or None if no table is provided.
        """
        if intrinsicTable is None:
            return None

        # Extract arrays of field angle (deg)
        x = intrinsicTable["x"].to("deg").value
        y = intrinsicTable["y"].to("deg").value
        x_grid = np.unique(x)
        y_grid = np.unique(y)

        # Extract intrinsic Zernike coefficients (microns)
        zkTable = intrinsicTable[[f"Z{i}" for i in self.nollIndices]]
        zks = np.column_stack([zkTable[col].to("um").value for col in zkTable.colnames])

        # Create the interpolator
        if (len(x_grid) * len(y_grid)) == len(intrinsicTable):
            # If the grid is regular and complete, use RegularGridInterpolator
            values = zks.reshape(y_grid.size, x_grid.size, -1)
            interpolator = RegularGridInterpolator((y_grid, x_grid), values)
        else:
            # Otherwise, use LinearNDInterpolator
            interpolator = LinearNDInterpolator(np.column_stack([y, x]), zks)

        return interpolator

    def _unpackStampData(self, stamp: DonutStamp) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Unpack data from the stamp object, handling None stamps.

        Parameters
        ----------
        stamp : DonutStamp or None
            The DonutStamp object to unpack data from.

        Returns
        -------
        fieldAngle : `astropy.units.Quantity`
            The field angle of the stamp in degrees.
        centroid : `astropy.units.Quantity`
            The centroid position of the stamp in pixels.
        intrinsics : `astropy.units.Quantity`
            The intrinsic Zernike coefficients for the stamp in microns.
        """
        if stamp is None:
            fieldAngle = np.array(np.nan, dtype=pos2f_dtype) * u.deg
            centroid = np.array((np.nan, np.nan), dtype=pos2f_dtype) * u.pixel
            intrinsics = np.full(len(self.nollIndices), np.nan) * u.micron
        else:
            fieldAngle = np.array(stamp.calcFieldXY(), dtype=pos2f_dtype) * u.deg
            centroid = (
                np.array(
                    (stamp.centroid_position.x, stamp.centroid_position.y),
                    dtype=pos2f_dtype,
                )
                * u.pixel
            )
            if stamp.defocal_type == "extra":
                intrinsicMap = self.intrinsicMapExtra
            else:
                intrinsicMap = self.intrinsicMapIntra

            # Note that if you compare to the _createIntrinsicMap method you
            # might think we would need to reverse the fieldAngle here (i.e.
            # swap x and y), however stamp.calcFieldXY() returns coordinates
            # in the DVCS instead of CCS, which is equivalent to already
            # swapping x and y. Therefore we will not reverse the order here.
            intrinsics = intrinsicMap(fieldAngle.value.tolist()) * u.micron  # type: ignore

        return fieldAngle, centroid, intrinsics

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
            ("intra_max_power_grad", "<f4"),
            ("extra_max_power_grad", "<f4"),
            ("intra_donut_id", "<U21"),
            ("extra_donut_id", "<U21"),
        ]
        for j in self.nollIndices:
            dtype.append((f"Z{j}", "<f4"))
        for j in self.nollIndices:
            dtype.append((f"Z{j}_intrinsic", "<f4"))
        for j in self.nollIndices:
            dtype.append((f"Z{j}_deviation", "<f4"))

        table = QTable(dtype=dtype)

        # Assign units where appropriate
        table["intra_field"].unit = u.deg
        table["extra_field"].unit = u.deg
        table["intra_centroid"].unit = u.pixel
        table["extra_centroid"].unit = u.pixel
        for j in self.nollIndices:
            table[f"Z{j}"].unit = u.nm
        for j in self.nollIndices:
            table[f"Z{j}_intrinsic"].unit = u.nm
        for j in self.nollIndices:
            table[f"Z{j}_deviation"].unit = u.nm

        return table

    def createZkTable(self, zkCoeffRaw: pipeBase.Struct) -> QTable:
        """Create the Zernike table to store Zernike Coefficients.

        Note this is written with the assumption that either extraStamps or
        intraStamps MIGHT be empty. This is because calcZernikesUnpairedTask
        also uses this method.

        Parameters
        ----------
        zkCoeffRaw: pipeBase.Struct
            All zernikes returned by self.estimateZernikes.run(...)

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
                **{f"Z{j}": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
                **{f"Z{j}_intrinsic": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
                **{f"Z{j}_deviation": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
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
                "intra_frac_bad_pix": np.nan,
                "extra_frac_bad_pix": np.nan,
                "intra_max_power_grad": np.nan,
                "extra_max_power_grad": np.nan,
                "intra_donut_id": "",
                "extra_donut_id": "",
            }
        )
        for i, (intra, extra, zk) in enumerate(
            zip_longest(
                self.stampsIntra,
                self.stampsExtra,
                zkCoeffRaw.zernikes,
            )
        ):
            # If zk is None, we need to stop. This can happen when running
            # paired Zernike estimation and the number of intra/extra stamps
            # is not the same
            if zk is None:
                break

            # Assign units
            zk = zk * u.micron

            # Unpack data from stamps, handling cases with None stamps
            intraAngle, intraCentroid, intraIntrinsics = self._unpackStampData(intra)
            extraAngle, extraCentroid, extraIntrinsics = self._unpackStampData(extra)

            # Average the intrinsics
            intrinsics = np.nanmean((intraIntrinsics, extraIntrinsics), axis=0) * u.micron

            # Calculate the wavefront deviation
            deviation = zk - intrinsics

            row: dict = dict()
            row["label"] = f"pair{i + 1}"
            row["used"] = False  # Placeholder for now
            row.update({f"Z{j}": zk[i] for i, j in enumerate(self.nollIndices)})
            row.update({f"Z{j}_intrinsic": intrinsics[i] for i, j in enumerate(self.nollIndices)})
            row.update({f"Z{j}_deviation": deviation[i] for i, j in enumerate(self.nollIndices)})
            row["intra_field"] = intraAngle
            row["extra_field"] = extraAngle
            row["intra_centroid"] = intraCentroid
            row["extra_centroid"] = extraCentroid
            for key in ["MAG", "SN", "ENTROPY", "FRAC_BAD_PIX", "MAX_POWER_GRAD", "DONUT_ID"]:
                for stamps, foc in [
                    (self.stampsIntra, "intra"),
                    (self.stampsExtra, "extra"),
                ]:
                    if len(stamps) > 0 and key in stamps.metadata:
                        val = stamps.metadata.getArray(key)[i]
                    else:
                        val = "" if key == "DONUT_ID" else np.nan
                    row[f"{foc}_{key.lower()}"] = val
            zkTable.add_row(row)

        zkTable.meta = self.createZkTableMetadata()

        return zkTable

    def createZkTableMetadata(self) -> dict:
        """Create the metadata for the Zernike table.

        Returns
        -------
        metadata : dict
            Metadata for the Zernike table
        """
        meta: dict = {}
        meta["intra"] = {}
        meta["extra"] = {}
        cam_name = None

        if not self.stampsIntra.metadata and not self.stampsExtra.metadata:
            raise ValueError("No metadata in either DonutStamps object. Cannot create Zk table metadata.")

        for dict_, stamps in [
            (meta["intra"], self.stampsIntra),
            (meta["extra"], self.stampsExtra),
        ]:
            if not stamps.metadata:
                continue
            dict_["det_name"] = stamps.metadata["DET_NAME"]
            dict_["visit"] = stamps.metadata["VISIT"]
            dict_["dfc_dist"] = stamps.metadata["DFC_DIST"]
            dict_["band"] = stamps.metadata["BANDPASS"]
            dict_["boresight_rot_angle_rad"] = stamps.metadata["BORESIGHT_ROT_ANGLE_RAD"]
            dict_["boresight_par_angle_rad"] = stamps.metadata["BORESIGHT_PAR_ANGLE_RAD"]
            dict_["boresight_alt_rad"] = stamps.metadata["BORESIGHT_ALT_RAD"]
            dict_["boresight_az_rad"] = stamps.metadata["BORESIGHT_AZ_RAD"]
            dict_["boresight_ra_rad"] = stamps.metadata["BORESIGHT_RA_RAD"]
            dict_["boresight_dec_rad"] = stamps.metadata["BORESIGHT_DEC_RAD"]
            dict_["mjd"] = stamps.metadata["MJD"]
            if cam_name is None:
                cam_name = stamps.metadata["CAM_NAME"]

        meta["cam_name"] = cam_name
        meta["noll_indices"] = self.nollIndices.list()
        meta["opd_columns"] = [f"Z{j}" for j in self.nollIndices]
        meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in self.nollIndices]
        meta["deviation_columns"] = [f"Z{j}_deviation" for j in self.nollIndices]

        if self.stampsIntra.metadata and self.stampsExtra.metadata:
            assert self.stampsIntra.metadata["CAM_NAME"] == self.stampsExtra.metadata["CAM_NAME"]

        return meta

    def empty(
        self, qualityTable: QTable | None = None, zernikeTable: QTable | None = None
    ) -> pipeBase.Struct:
        """Return empty results if no donuts are available. If
        it is a result of no quality donuts we still include the
        quality table results instead of an empty quality table.

        Parameters
        ----------
        qualityTable : astropy.table.QTable
            Quality table created with donut stamp input.
        zernikeTable : astropy.table.QTable
            Zernike table created with donut stamp input.

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
            "RADIUS",
            "RADIUS_FAIL_FLAG",
            "DEFOCAL_TYPE",
            "DONUT_ID",
        ]
        if qualityTable is None:
            donutQualityTable = QTable({name: [] for name in qualityTableCols})
        else:
            donutQualityTable = qualityTable

        if zernikeTable is None:
            zkTable = self.initZkTable()
            zkTable.meta = self.createZkTableMetadata()
        else:
            zkTable = zernikeTable

        return pipeBase.Struct(
            outputZernikesRaw=np.atleast_2d(np.full(len(self.nollIndices), np.nan)),
            outputZernikesAvg=np.atleast_2d(np.full(len(self.nollIndices), np.nan)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )

    def blurClip(self, zkTable: QTable) -> QTable:
        """
        Look at the donut blur values returned for all values in a sensor and
        use sigma clipping to remove donuts that are outliers.

        Parameters
        ----------
        zkTable : astropy.table.QTable
            Zernike table.

        Returns
        -------
        astropy.table.QTable
            Zernike table where donuts with outlier donut blur values
            have been changed to false and the average recomputed.
        """

        useIdx = np.where((zkTable["used"]) & (zkTable["label"] != "average"))[0]
        # account for average row with "- 1" on index below
        fwhmList = np.array(zkTable.meta["estimatorInfo"]["fwhm"])[useIdx - 1]
        blurMask = sigma_clip(fwhmList, stdfunc="mad_std", sigma_lower=99).mask
        dropIdx = useIdx[np.where(blurMask)[0]]
        zkTable["used"][dropIdx] = False
        zkTable.meta["estimatorInfo"]["blur_clipped"] = np.isin(np.arange(1, len(zkTable)), dropIdx).tolist()
        # Calculate the average correctly
        combineZernikesMean = CombineZernikesMeanTask().combineZernikes(zkTable)

        return combineZernikesMean

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs, numCores=butlerQC.resources.num_cores)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        intrinsicTables: list[Table],
        numCores: int = 1,
    ) -> pipeBase.Struct:
        # If no donuts are in the donutCatalog for a set of exposures
        # in one of the sides of focus, then attempt to get the zernikes
        # by fitting the donut radius. If that fails, return empty struct.
        self.stampsExtra = donutStampsExtra
        self.stampsIntra = donutStampsIntra
        if len(self.stampsExtra) == 0 or len(self.stampsIntra) == 0:
            return self.empty()

        # Run donut stamp selection. By default, doSelection is turned on,
        # and we select only donuts that pass the criteria.
        # We are always provided with donut quality table.
        if self.doDonutStampSelector:
            self.log.info("Running Donut Stamp Selector")
            selectionExtra = self.donutStampSelector.run(self.stampsExtra)
            selectionIntra = self.donutStampSelector.run(self.stampsIntra)
            donutExtraQuality = selectionExtra.donutsQuality
            donutIntraQuality = selectionIntra.donutsQuality
            selectedExtraStamps = selectionExtra.donutStampsSelect
            selectedIntraStamps = selectionIntra.donutStampsSelect
            donutExtraQuality["DEFOCAL_TYPE"] = "extra"
            donutIntraQuality["DEFOCAL_TYPE"] = "intra"
            donutQualityTable = vstack([donutExtraQuality, donutIntraQuality])

            # If no donuts get selected, also attempt to
            # to compute the Z4 from donut radius fit.
            # If unsuccessful, return empty struct.
            if len(selectedExtraStamps) == 0 or len(selectedIntraStamps) == 0:
                self.log.info("No donut stamps were selected.")
                return self.empty(qualityTable=donutQualityTable)
        else:
            self.log.info("Not running Donut Stamp Selector")
            donutQualityTable = QTable([])
            selectedExtraStamps = self.stampsExtra
            selectedIntraStamps = self.stampsIntra

        # Update stampsExtra and stampsIntra with the selected donuts
        self.stampsExtra = selectedExtraStamps
        self.stampsIntra = selectedIntraStamps

        # Set the intrinsic map interpolators
        self.log.info("Creating intrinsic map.")
        if self.stampsExtra[0].detector_name == self.stampsIntra[0].detector_name:
            # If both intra and extra focal donuts are from the same detector,
            # then we only have one intrinsic table to use for both.
            self.intrinsicMapExtra = self._createIntrinsicMap(intrinsicTables[0])
            self.intrinsicMapIntra = self._createIntrinsicMap(intrinsicTables[0])
        else:
            self.intrinsicMapExtra = self._createIntrinsicMap(intrinsicTables[0])
            self.intrinsicMapIntra = self._createIntrinsicMap(intrinsicTables[1])

        # Estimate Zernikes from the collection of selected stamps
        self.log.info("Starting Zernike Estimation")
        zkCoeffRaw = self.estimateZernikes.run(self.stampsExtra, self.stampsIntra, numCores=numCores)

        # Save the outputs in the table
        zkTable = self.createZkTable(zkCoeffRaw)
        zkTable.meta["estimatorInfo"] = zkCoeffRaw.wfEstInfo

        # If we have a fit failure recorded then replace Zernikes
        # with NaNs for those donuts so we don't use them in combining.
        if "fit_success" in zkTable.meta["estimatorInfo"].keys():
            fitSuccess = zkTable.meta["estimatorInfo"]["fit_success"]
            if np.sum(fitSuccess) == 0:
                self.log.info("All donuts had fit failures. Returning empty results.")
                return self.empty(qualityTable=donutQualityTable, zernikeTable=zkTable)
            failIdx = np.where(~np.array(fitSuccess))[0]
            for j in self.nollIndices:
                zkTable[f"Z{j}"][failIdx + 1] = np.nan  # +1 to skip average row
                zkTable[f"Z{j}_deviation"][failIdx + 1] = np.nan

        # Combine Zernikes
        zkTable = self.combineZernikes.run(zkTable).combinedTable

        # Implement Blur Clip
        if self.doBlurClip and ("fwhm" in zkTable.meta["estimatorInfo"].keys()):
            zkTable = self.blurClip(zkTable)

        avg = zkTable[zkTable["label"] == "average"]
        outputZernikesAvg = np.array([avg[col].to_value("um")[0] for col in avg.meta["opd_columns"]])

        return pipeBase.Struct(
            outputZernikesAvg=np.atleast_2d(np.array(outputZernikesAvg)),
            outputZernikesRaw=np.atleast_2d(np.array(zkCoeffRaw.zernikes)),
            zernikes=zkTable,
            donutQualityTable=donutQualityTable,
        )
