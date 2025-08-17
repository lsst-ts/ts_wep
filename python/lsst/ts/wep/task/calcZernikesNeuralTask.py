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
    "CalcZernikesNeuralTaskConnections",
    "CalcZernikesNeuralTaskConfig",
    "CalcZernikesNeuralTask",
]

import abc
from typing import Any, Optional
import torch
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy.table import QTable
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod
from NeuralAOS import NeuralActiveOpticsSys

class CalcZernikesNeuralTaskConnections(
    pipeBase.PipelineTaskConnections,
):
    dimensions = ("visit", "detector", "instrument")
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


class CalcZernikesNeuralTaskConfig(
    pipeBase.PipelineTaskConfig,
):
    pipelineConnections = CalcZernikesNeuralTaskConnections

    wavenet_path = pexConfig.Field(
        doc="Model Weights Path for wavenet",
        dtype=str
    )
    alignet_path = pexConfig.Field(
        doc="Model Weights Path for alignet",
        dtype=str
    )
    aggregatornet_path = pexConfig.Field(
        doc="Model Weights Path for aggregatornet",
        dtype=str
    )
    dataset_param_path = pexConfig.Field(
        doc="datasetparam path",
        dtype=str
    )
    device = pexConfig.Field(
        doc="Device to use for calculations",
        dtype=str,
        default="cuda"
    )


class CalcZernikesNeuralTask(pipeBase.PipelineTask, metaclass=abc.ABCMeta):
    """Base class for calculating Zernike coeffs from pairs of DonutStamps.

    This class joins the EstimateZernikes and CombineZernikes subtasks to
    be run on sets of DonutStamps.
    """
    _DefaultName = "calcZernikesNeuralBaseTask"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Define default Noll indices (zk terms 4-23, excl piston, tip, tilt)
        self.nollIndices = np.arange(4, 24)

        self.NAOS = NeuralActiveOpticsSys(
            self.config.dataset_param_path,
            self.config.wavenet_path,
            self.config.alignet_path,
            self.config.aggregatornet_path
        )
        self.NAOS = self.NAOS.eval()
        self.CROP_SIZE = self.NAOS.CROP_SIZE
        self.device = self.config.device

        if self.device == "cpu":
            self.NAOS = self.NAOS.cpu()

            # Update device attributes for all sub-models to CPU
            self.NAOS.device_val = torch.device("cpu")
            if hasattr(self.NAOS.alignnet_model, 'device_val'):
                self.NAOS.alignnet_model.device_val = torch.device("cpu")
            if hasattr(self.NAOS.wavenet_model, 'device_val'):
                self.NAOS.wavenet_model.device_val = torch.device("cpu")
            if hasattr(self.NAOS.aggregatornet_model, 'device_val'):
                self.NAOS.aggregatornet_model.device_val = torch.device("cpu")

            # Update device attributes for the underlying models
            if hasattr(self.NAOS.alignnet_model.alignnet, 'device_val'):
                self.NAOS.alignnet_model.alignnet.device_val = torch.device("cpu")
            if hasattr(self.NAOS.wavenet_model.wavenet, 'device_val'):
                self.NAOS.wavenet_model.wavenet.device_val = torch.device("cpu")

            # Ensure all CNN models within the sub-models are also on CPU
            if hasattr(self.NAOS.alignnet_model.alignnet, 'cnn'):
                self.NAOS.alignnet_model.alignnet.cnn = self.NAOS.alignnet_model.alignnet.cnn.cpu()
            if hasattr(self.NAOS.wavenet_model.wavenet, 'cnn'):
                self.NAOS.wavenet_model.wavenet.cnn = self.NAOS.wavenet_model.wavenet.cnn.cpu()
        else:
            self.NAOS.to("cuda")

    def calc_exposure(self, exposure: Any) -> np.ndarray:
        """
        Calculate the Zernike coefficients for an exposure.
        """
        with torch.no_grad():
            pred = self.NAOS.deploy_run(exposure)
        return pred

    def empty(self, qualityTable: Optional[QTable] = None) -> pipeBase.Struct:
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

    def initZkTable(self) -> QTable:
        """Initialize an empty Zernike coefficient table.

        Returns
        -------
        astropy.table.QTable
            Empty table with columns for Zernike coefficients.
        """
        # Create columns for each Noll index
        columns: dict[str, list] = {}
        for noll_idx in self.nollIndices:
            columns[f"Z{noll_idx}"] = []

        return QTable(columns)

    @timeMethod
    def run(
        self,
        ExtraExposure: Any,
        IntraExposure: Any,
    ) -> pipeBase.Struct:

        pred_intra = self.calc_exposure(IntraExposure) #gives microns
        pred_extra = self.calc_exposure(ExtraExposure) #gives microns

        zernikes_Raw = np.stack([pred_intra, pred_extra], axis=0)
        zernikes_Avg = np.mean(zernikes_Raw, axis=0)

        return pipeBase.Struct(
            outputZernikesAvg=zernikes_Avg,
            outputZernikesRaw=zernikes_Raw
        )
