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

from typing import Any, Sequence, cast

import astropy.units as u
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import QTable, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import DataCoordinate, DatasetRef, DatasetType, Registry
from lsst.ip.isr import IntrinsicZernikes
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
    connectionTypes,
)
from lsst.ts.wep.task.calcZernikesBaseTask import CalcZernikesBaseTask
from lsst.ts.wep.task.combineZernikesMeanTask import CombineZernikesMeanTask
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamp, DonutStamps
from lsst.ts.wep.task.donutStampSelectorTask import DonutStampSelectorTask
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.utils.timer import timeMethod

pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])
intra_focal_ids = set([192, 196, 200, 204])
extra_focal_ids = set([191, 195, 199, 203])


def lookupIntrinsicZernikes(
    datasetType: DatasetType, registry: Registry, dataId: DataCoordinate, collections: Sequence[str]
) -> list[DatasetRef | None]:
    """Assumes that the dataId is always for the extra focal at present"""
    detector = dataId["detector"]
    isCornerChip = (detector in intra_focal_ids) or (detector in extra_focal_ids)

    refs = [registry.findDataset(datasetType, dataId, collections=collections, timespan=dataId.timespan)]
    if isCornerChip:  # we're running a CWFS pair, not a FAM image
        dataId2 = DataCoordinate.standardize(dataId, detector=int(dataId["detector"]) + 1)
        refs.append(
            registry.findDataset(datasetType, dataId2, collections=collections, timespan=dataId.timespan)
        )
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
    intrinsicZernikes = connectionTypes.PrerequisiteInput(
        doc="Intrinsic Zernike calibration for the instrument",
        dimensions=("detector", "instrument", "physical_filter"),
        storageClass="IsrCalib",
        name="intrinsicZernikes",
        multiple=True,
        isCalibration=True,
        lookupFunction=lookupIntrinsicZernikes,  # type: ignore
        minimum=0,
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


class CalcZernikesTask(CalcZernikesBaseTask):
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

        # Intrinsic Zernike calibrations are assigned per-quantum in run().
        # They stay None when the instrument is allowed to run without them.
        self.intrinsicZernikesExtra: IntrinsicZernikes | None = None
        self.intrinsicZernikesIntra: IntrinsicZernikes | None = None

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
                intrinsicCalib = self.intrinsicZernikesExtra
            else:
                intrinsicCalib = self.intrinsicZernikesIntra

            if intrinsicCalib is None:
                # No intrinsic Zernike calibration available (allowed for
                # instruments that do not require one, e.g. LATISS).
                intrinsics = np.full(len(self.nollIndices), np.nan) * u.micron
            else:
                # stamp.calcFieldXY() returns coordinates in the DVCS, which is
                # equivalent to swapping x and y from CCS. So the first element
                # of fieldAngle is the CCS y-coordinate and the second is the
                # CCS x-coordinate, which is what IntrinsicZernikes expects.
                ccs_y, ccs_x = fieldAngle.value.tolist()
                # getIntrinsicZernikes returns shape (1, nNoll) for a single
                # field point; squeeze to 1-D so it lines up with the
                # per-Zernike row assignment in createZkTable.
                intrinsics = (
                    np.atleast_1d(
                        np.squeeze(
                            intrinsicCalib.getIntrinsicZernikes(
                                field_x=ccs_x,
                                field_y=ccs_y,
                                noll_indices=self.nollIndices,
                            )
                        )
                    )
                    * u.micron
                )

        return fieldAngle, centroid, intrinsics

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

        # The lookup function returns the intrinsic Zernike calibrations in
        # [extra, intra] order: a single shared calibration for full-array
        # mode (same detector for both defocal types), one per detector for
        # corner wavefront sensors, and none for instruments without a
        # calibration (e.g. LATISS). Resolve the ordering here so that ``run``
        # receives the extra- and intra-focal calibrations explicitly, rather
        # than relying on the list ordering inside ``run`` where it could be
        # set incorrectly when running interactively.
        #
        # Pop the loaded calibrations out of ``inputs`` so they are not also
        # forwarded via ``**inputs`` (``run`` has no ``intrinsicZernikes``
        # parameter) and so we work with the actual ``IntrinsicZernikes``
        # objects rather than the ``DatasetRef``s in ``inputRefs``.
        intrinsicZernikes = inputs.pop("intrinsicZernikes")

        if len(intrinsicZernikes) == 0:
            intrinsicZernikesExtra = None
            intrinsicZernikesIntra = None
        elif len(intrinsicZernikes) == 1:
            intrinsicZernikesExtra = intrinsicZernikes[0]
            intrinsicZernikesIntra = intrinsicZernikes[0]
        else:
            detectors = [ref.dataId["detector"] for ref in inputRefs.intrinsicZernikes]
            if detectors[0] < detectors[1]:
                intrinsicZernikes.reverse()
            intrinsicZernikesIntra, intrinsicZernikesExtra = intrinsicZernikes

        outputs = self.run(
            **inputs,
            intrinsicZernikesExtra=intrinsicZernikesExtra,
            intrinsicZernikesIntra=intrinsicZernikesIntra,
            numCores=butlerQC.resources.num_cores,
        )
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        intrinsicZernikesExtra: IntrinsicZernikes | None = None,
        intrinsicZernikesIntra: IntrinsicZernikes | None = None,
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

        # Assign the intrinsic Zernike calibrations. These inputs are optional:
        # LATISS has no intrinsic Zernike calibration and may run without one,
        # but the LSST cameras require it. The extra- and intra-focal
        # calibrations are passed in explicitly (see ``runQuantum``) so we do
        # not rely on any list ordering here.
        if intrinsicZernikesExtra is None and intrinsicZernikesIntra is None:
            cameraName = self.stampsExtra[0].cam_name
            if cameraName == "LATISS":
                self.log.warning(
                    "No intrinsic Zernike calibration found for %s; proceeding "
                    "without it. Intrinsic and deviation columns will be NaN.",
                    cameraName,
                )
            else:
                raise RuntimeError(
                    f"No intrinsic Zernike calibration found for instrument "
                    f"{cameraName!r}, which requires one. Provide an input "
                    f"collection containing the 'intrinsicZernikes' calibration."
                )
        self.intrinsicZernikesExtra = intrinsicZernikesExtra
        self.intrinsicZernikesIntra = intrinsicZernikesIntra

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
                # all donuts are fit failures, so none are blur clipped
                zkTable.meta["estimatorInfo"]["blur_clipped"] = [False] * (len(zkTable) - 1)
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
