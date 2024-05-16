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
from astropy.table import Table
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamps
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
    selectWithEntropy = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use entropy in deciding to use the donut.",
    )

    selectWithSignalToNoise = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use signal to noise ratio in deciding to use the donut.",
    )

    minSignalToNoise = pexConfig.Field(
        dtype=float,
        default=5000,
        doc=str(
            "The S/N threshold to use (keep donuts only above the threshold)."
            + "The default S/N = 5000 corresponds to the default entropy threshold (3.5)"
        ),
    )
    maxEntropy = pexConfig.Field(
        dtype=float,
        default=3.5,
        doc=str("The entropy threshold to use (keep donuts only below the threshold)."),
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

        # Inherit config parameters
        self.selectWithEntropy = self.config.selectWithEntropy
        self.selectWithSignalToNoise = self.config.selectWithSignalToNoise
        self.minSignalToNoise = self.config.minSignalToNoise
        self.maxEntropy = self.config.maxEntropy

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
                donutsExtraQuality=[],
                donutsIntraQuality=[],
            )
        # Which donuts to use for Zernike estimation
        # initiate these by selecting all donuts
        entropySelectExtra = np.ones(len(donutStampsExtra), dtype="bool")
        entropySelectIntra = np.ones(len(donutStampsIntra), dtype="bool")
        entropyExtra = np.zeros(len(donutStampsIntra))
        entropyIntra = np.zeros(len(donutStampsIntra))
        if self.selectWithEntropy:
            entropyExtra = np.asarray(donutStampsExtra.metadata.getArray("ENTROPY"))
            entropyIntra = np.asarray(donutStampsIntra.metadata.getArray("ENTROPY"))
            entropySelectExtra = entropyExtra < self.maxEntropy
            entropySelectIntra = entropyIntra < self.maxEntropy

        snSelectExtra = np.ones(len(donutStampsExtra), dtype="bool")
        snSelectIntra = np.ones(len(donutStampsIntra), dtype="bool")
        snExtra = np.zeros(len(donutStampsIntra))
        snIntra = np.zeros(len(donutStampsIntra))
        if self.selectWithSignalToNoise:
            snExtra = np.asarray(donutStampsExtra.metadata.getArray("SN_VAR"))
            snSelectExtra = self.minSignalToNoise < snExtra

            snIntra = np.asarray(donutStampsIntra.metadata.getArray("SN_VAR"))
            snSelectIntra = self.minSignalToNoise < snIntra

        # AND condition : if both selectWithEntropy
        # and selectWithSignalToNoise, then
        # only donuts that pass with SN criterion as well
        # as entropy criterion are selected
        selectExtra = entropySelectExtra * snSelectExtra
        selectIntra = entropySelectIntra * snSelectIntra

        # store information about which extra-focal donuts were selected
        donutsExtraQuality = Table(
            data=[
                snExtra,
                entropyExtra,
                entropySelectExtra,
                snSelectExtra,
                selectExtra,
            ],
            names=["SN", "ENTROPY", "ENTROPY_SELECT", "SN_SELECT", "FINAL_SELECT"],
        ).to_pandas()

        # store information about which intra-focal donuts were selected
        donutsIntraQuality = Table(
            data=[
                snIntra,
                entropyIntra,
                entropySelectIntra,
                snSelectIntra,
                selectIntra,
            ],
            names=["SN", "ENTROPY", "ENTROPY_SELECT", "SN_SELECT", "FINAL_SELECT"],
        ).to_pandas()

        donutStampsExtraSelect = np.array(donutStampsExtra)[selectExtra]
        donutStampsIntraSelect = np.array(donutStampsIntra)[selectIntra]

        # Estimate Zernikes from the collection of stamps
        zkCoeffRaw = self.estimateZernikes.run(
            donutStampsExtraSelect, donutStampsIntraSelect
        )
        zkCoeffCombined = self.combineZernikes.run(zkCoeffRaw.zernikes)

        return pipeBase.Struct(
            outputZernikesAvg=np.array(zkCoeffCombined.combinedZernikes),
            outputZernikesRaw=np.array(zkCoeffRaw.zernikes),
            donutsExtraQuality=donutsExtraQuality,
            donutsIntraQuality=donutsIntraQuality,
        )
