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

from typing import Any, Optional

import numpy as np
import torch

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from astropy.table import QTable
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod
from TARTS import NeuralActiveOpticsSys

class CalcZernikesNeuralTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("visit", "detector", "instrument")  # type: ignore
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

class CalcZernikesNeuralTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CalcZernikesNeuralTaskConnections,  # type: ignore
):
    """Configuration for CalcZernikesNeuralTask.

    Attributes
    ----------
    wavenet_path : str
        Model Weights Path for wavenet
    alignet_path : str
        Model Weights Path for alignet
    aggregatornet_path : str
        Model Weights Path for aggregatornet
    dataset_param_path : str
        datasetparam path
    device : str
        Device to use for calculations
    noll_indices : list[int]
        List of Noll indices to calculate. Default is Z4-Z23 (4-23),
        excluding piston (Z1), tip (Z2), and tilt (Z3) which are
        typically not measured in wavefront sensing.
    """

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
        doc="datasetparam path for TARTS includes normalization scaling and parameters path",
        dtype=str
    )
    device = pexConfig.Field(
        doc="Device to use for calculations",
        dtype=str,
        default="cuda"
    )
    noll_indices = pexConfig.ListField(
        doc="List of Noll indices to calculate. Default is Z4-Z23 (4-23), "
            "excluding piston (Z1), tip (Z2), and tilt (Z3) which are "
            "typically not measured in wavefront sensing.",
        dtype=int,
        default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    )


class CalcZernikesNeuralTask(pipeBase.PipelineTask):
    """Neural network-based Zernike estimation task using TARTS.

    This class uses the TARTS (Telescope Active Optics Real-Time System)
    neural network models to estimate Zernike coefficients from pairs of
    intra and extra-focal exposures.
    """
    _DefaultName = "calcZernikesNeuralTask"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the neural network-based Zernike estimation task.

        This method sets up the TARTS neural network system, configures the
        device (CPU or CUDA), and initializes the Noll indices for Zernike
        coefficients. The TARTS system is loaded with the specified model
        weights and dataset parameters from the configuration.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the parent PipelineTask
            class.

        Notes
        -----
        The initialization process:
        1. Sets up Noll indices from configuration (default: Z4-Z23)
        2. Loads TARTS neural network models with specified weights
        3. Configures device (CPU/CUDA) and moves models accordingly
        4. Sets the models to evaluation mode for inference

        The TARTS system includes three main components:
        - Wavenet: Estimates Zernike coefficients from donut images
        - Alignnet: Handles image alignment and preprocessing
        - Aggregatornet: Combines results from multiple donuts

        Noll indices are configurable via the config.noll_indices parameter.
        The default excludes Z1-Z3 (piston, tip, tilt) as these are typically
        not measured in wavefront sensing.
        """
        super().__init__(**kwargs)

        # Define default Noll indices (zk terms 4-23, excl piston, tip, tilt)
        self.nollIndices = self.config.noll_indices

        self.tarts = NeuralActiveOpticsSys(
            self.config.dataset_param_path,
            self.config.wavenet_path,
            self.config.alignet_path,
            self.config.aggregatornet_path
        )
        self.tarts = self.tarts.eval()
        self.cropSize = self.tarts.CROP_SIZE
        self.device = self.config.device

        if self.device == "cpu":
            self.tarts = self.tarts.cpu()

            # Update device attributes for all sub-models to CPU
            self.tarts.device_val = torch.device("cpu")
            if hasattr(self.tarts.alignnet_model, 'device_val'):
                self.tarts.alignnet_model.device_val = torch.device("cpu")
            if hasattr(self.tarts.wavenet_model, 'device_val'):
                self.tarts.wavenet_model.device_val = torch.device("cpu")
            if hasattr(self.tarts.aggregatornet_model, 'device_val'):
                self.tarts.aggregatornet_model.device_val = torch.device("cpu")

            # Update device attributes for the underlying models
            if hasattr(self.tarts.alignnet_model.alignnet, 'device_val'):
                self.tarts.alignnet_model.alignnet.device_val = torch.device("cpu")
            if hasattr(self.tarts.wavenet_model.wavenet, 'device_val'):
                self.tarts.wavenet_model.wavenet.device_val = torch.device("cpu")

            # Ensure all CNN models within the sub-models are also on CPU
            if hasattr(self.tarts.alignnet_model.alignnet, 'cnn'):
                self.tarts.alignnet_model.alignnet.cnn = self.tarts.alignnet_model.alignnet.cnn.cpu()
            if hasattr(self.tarts.wavenet_model.wavenet, 'cnn'):
                self.tarts.wavenet_model.wavenet.cnn = self.tarts.wavenet_model.wavenet.cnn.cpu()
        else:
            self.tarts.to("cuda")

    def calcExposure(self, exposure: afwImage.Exposure) -> np.ndarray:
        """Calculate the Zernike coefficients for a single exposure.

        This method processes a single LSST exposure through the TARTS neural
        network to estimate Zernike coefficients representing wavefront
        aberrations.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            The LSST exposure data to process. This should contain the image
            data and metadata needed by the TARTS neural network for wavefront
            estimation.

        Returns
        -------
        np.ndarray
            Array of Zernike coefficients in microns. The coefficients
            represent the estimated wavefront aberrations for the given
            exposure.

        Notes
        -----
        This method runs the neural network in inference mode (no gradients
        computed) for efficiency during prediction. The exposure object
        should contain valid image data and WCS information for proper
        processing.
        """
        with torch.no_grad():
            pred = self.tarts.deploy_run(exposure)
        # Convert PyTorch tensor to numpy array
        if hasattr(pred, 'cpu'):
            pred = pred.cpu().numpy()
        return pred

    def empty(self, qualityTable: Optional[QTable] = None) -> pipeBase.Struct:
        """Return empty results when no donuts are available for processing.

        This method creates empty output structures when the task cannot
        process any donut data. It handles two scenarios: when there are
        no donuts at all, and when there are donuts but they all fail
        quality checks.

        Parameters
        ----------
        qualityTable : astropy.table.QTable, optional
            Quality table created from donut stamp input. If provided, this
            table will be included in the output even if all donuts failed
            quality checks. If None, an empty quality table will be created.

        Returns
        -------
        lsst.pipe.base.Struct
            A struct containing empty or failed results:
            - outputZernikesRaw : np.ndarray
                Array filled with NaN values for all Noll indices
            - outputZernikesAvg : np.ndarray
                Array filled with NaN values for all Noll indices
            - zernikes : astropy.table.QTable
                Empty Zernike coefficient table
            - donutQualityTable : astropy.table.QTable
                Either the provided quality table or an empty one

        Notes
        -----
        The NaN values in the output arrays indicate that no valid Zernike
        coefficients could be calculated. The quality table preserves
        information about why donuts failed, which can be useful for
        debugging.
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

        Creates a table structure for storing Zernike coefficients with columns
        for each configured Noll index from the configuration.

        Returns
        -------
        astropy.table.QTable
            Empty table with columns for each configured Zernike coefficient.
            The default configuration includes:
            - Z4: Defocus
            - Z5: Astigmatism (0°)
            - Z6: Astigmatism (45°)
            - Z7: Coma (0°)
            - Z8: Coma (90°)
            - Z9: Spherical aberration
            - Z10: Trefoil (0°)
            - Z11: Trefoil (30°)
            - Z12: Secondary astigmatism (0°)
            - Z13: Secondary astigmatism (45°)
            - Z14: Secondary coma (0°)
            - Z15: Secondary coma (90°)
            - Z16: Secondary spherical aberration
            - Z17: Quadrafoil (0°)
            - Z18: Quadrafoil (22.5°)
            - Z19: Secondary trefoil (0°)
            - Z20: Secondary trefoil (30°)
            - Z21: Secondary quadrafoil (0°)
            - Z22: Secondary quadrafoil (22.5°)
            - Z23: Tertiary spherical aberration

        Notes
        -----
        The Noll indices are configurable via config.noll_indices. The default
        starts from 4 (not 1) as Z1-Z3 represent piston and tip/tilt, which
        are typically not measured in wavefront sensing. Users can customize
        this list based on their specific requirements.
        """
        # Create columns for each Noll index
        columns: dict[str, list] = {}
        for noll_idx in self.nollIndices:
            columns[f"Z{noll_idx}"] = []

        return QTable(columns)

    @timeMethod
    def run(
        self,
        ExtraExposure: afwImage.Exposure,
        IntraExposure: afwImage.Exposure,
    ) -> pipeBase.Struct:
        """Run the neural network-based Zernike estimation task.

        This method processes pairs of intra and extra-focal LSST exposures
        to estimate Zernike coefficients using the TARTS neural network.
        The method calculates Zernike coefficients for each exposure
        separately and then provides both individual results and averaged
        results.

        Parameters
        ----------
        ExtraExposure : lsst.afw.image.Exposure
            The extra-focal LSST exposure data. This should contain donut
            stamps or image data with proper WCS information for the TARTS
            neural network.
        IntraExposure : lsst.afw.image.Exposure
            The intra-focal LSST exposure data. This should contain donut
            stamps or image data with proper WCS information for the TARTS
            neural network.

        Returns
        -------
        lsst.pipe.base.Struct
            A struct containing:
            - outputZernikesAvg : np.ndarray
                Average Zernike coefficients across both exposures (in
                microns)
            - outputZernikesRaw : np.ndarray
                Raw Zernike coefficients for each exposure separately (in
                microns)

        Notes
        -----
        This implementation assumes separate intra and extra-focal exposures.
        For CWFS mode (single exposure with different corners), this method
        would need to be modified to handle the different data structure.
        The current approach processes each exposure independently through
        the neural network and then combines the results.

        Robustness features:
        - If only one exposure is available, it will be used for both
          intra and extra focal calculations
        - If both exposures are available, they will be processed separately
        - The method gracefully handles missing or invalid exposures

        Both exposure objects should contain:
        - Valid image data (typically donut stamps)
        - Proper WCS (World Coordinate System) information
        - Appropriate metadata for the instrument and observation

        See Also
        --------
        calcExposure : Method that processes individual exposures
        """
        # Check if exposures are valid and handle missing cases
        hasIntra = IntraExposure is not None
        hasExtra = ExtraExposure is not None

        if not hasIntra and not hasExtra:
            # No exposures available - return empty results
            return self.empty()

        if hasIntra and hasExtra:
            # Both exposures available - process normally
            predIntra = self.calcExposure(IntraExposure)[0,:]  # gives microns
            predExtra = self.calcExposure(ExtraExposure)[0,:]  # gives microns

            zernikesRaw = np.stack([predIntra, predExtra], axis=0)
            zernikesAvg = np.mean(zernikesRaw, axis=0)

        elif hasIntra:
            # Only intra-focal available - use it for both
            predIntra = self.calcExposure(IntraExposure)[0,:]  # gives microns
            zernikesRaw = np.stack([predIntra, predIntra], axis=0)
            zernikesAvg = predIntra  # Average of same value is the value itself

        else:  # hasExtra only
            # Only extra-focal available - use it for both
            predExtra = self.calcExposure(ExtraExposure)[0,:]  # gives microns
            zernikesRaw = np.stack([predExtra, predExtra], axis=0)
            zernikesAvg = predExtra  # Average of same value is the value itself

        return pipeBase.Struct(
            outputZernikesAvg=zernikesAvg,
            outputZernikesRaw=zernikesRaw
        )


__all__ = [
    "CalcZernikesNeuralTaskConnections",
    "CalcZernikesNeuralTaskConfig",
    "CalcZernikesNeuralTask",
]
