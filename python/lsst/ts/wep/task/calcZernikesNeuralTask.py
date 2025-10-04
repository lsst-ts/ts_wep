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

from copy import deepcopy
from itertools import zip_longest
from typing import Any, Optional
import logging

import numpy as np
import os
import torch
from astropy.table import QTable

import lsst.afw.image as afwImage
import lsst.geom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.daf.base import PropertyList, DateTime
import astropy.units as u

# Define the position 2D float dtype for the zernikes table
POS2F_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4")])


class CalcZernikesNeuralTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "detector", "instrument")  # type: ignore
):

    exposure = connectionTypes.Input(
        doc="Exposure containing donut stamps",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
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
    donutStampsNeural = connectionTypes.Output(
        doc="Neural network-generated donut stamps",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
    )
    donutTable = connectionTypes.Output(
        doc="Donut source catalog with positions and properties",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutTable",
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
    wavenetPath : str or None
        Model Weights Path for wavenet. If None, TARTS will create a new
        model with random weights (useful for testing).
    alignetPath : str or None
        Model Weights Path for alignet. If None, TARTS will create a new
        model with random weights (useful for testing).
    aggregatornetPath : str or None
        Model Weights Path for aggregatornet. If None, TARTS will create a new
        model with random weights (useful for testing).
    datasetParamPath : str
        Path to TARTS dataset parameters YAML file containing normalization
        scaling factors, image processing parameters (CROP_SIZE, deg_per_pix,
        mm_pix), model hyperparameters, and training data file paths
    device : str
        Device to use for calculations
    nollIndices : list[int]
        List of Noll indices to calculate. Default is Z4-Z23 (4-23),
        excluding piston (Z1), tip (Z2), and tilt (Z3) which are
        typically not measured in wavefront sensing.
    cropSize : int
        Size of donut crop in pixels (width and height). Default is 160 pixels,
        which matches the TARTS neural network training data format.
    """

    wavenetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for wavenet",
        dtype=str,
        default=None,
        optional=True
    )
    alignetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for alignet",
        dtype=str,
        default=None,
        optional=True
    )
    aggregatornetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for aggregatornet",
        dtype=str,
        default=None,
        optional=True
    )
    datasetParamPath: pexConfig.Field = pexConfig.Field(
        doc="Path to TARTS dataset parameters YAML file containing normalization "
            "scaling factors, image processing parameters (CROP_SIZE, deg_per_pix, "
            "mm_pix), model hyperparameters, and training data file paths",
        dtype=str
    )
    device: pexConfig.Field = pexConfig.Field(
        doc="Device to use for calculations",
        dtype=str,
        default="cuda"
    )
    nollIndices: pexConfig.ListField = pexConfig.ListField(
        doc="List of Noll indices to calculate. Default is Z4-Z23 (4-23), "
            "excluding piston (Z1), tip (Z2), and tilt (Z3) which are "
            "typically not measured in wavefront sensing.",
        dtype=int,
        default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    )
    cropSize: pexConfig.Field = pexConfig.Field(
        doc="Size of donut crop in pixels (width and height). Default is 160 pixels, "
            "which matches the TARTS neural network training data format.",
        dtype=int,
        default=160
    )


class CalcZernikesNeuralTask(pipeBase.PipelineTask):
    """Neural network-based Zernike estimation task using TARTS.

    This class uses the TARTS (Telescope Active Optics Real-Time System)
    neural network models to estimate Zernike coefficients from pairs of
    intra and extra-focal exposures. Each exposure contains donut stamps
    from one focal position, and the task processes them separately to
    estimate Zernike coefficients for each side of the focal plane.
    """
    ConfigClass = CalcZernikesNeuralTaskConfig
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
        1. Calls parent class constructor with super().__init__(**kwargs)
        2. Sets up Noll indices from configuration (default: Z4-Z23)
        3. Loads TARTS neural network models with specified weights
        4. Sets models to evaluation mode with .eval()
        5. Extracts crop size from TARTS system
        6. Configures device (CPU/CUDA) and moves models accordingly

        The TARTS system includes three main components:
        - Wavenet: Estimates Zernike coefficients from donut images
        - Alignnet: Handles image alignment and preprocessing
        - Aggregatornet: Combines results from multiple donuts

        Noll indices are configurable via the config.nollIndices parameter.
        The default excludes Z1-Z3 (piston, tip, tilt) as these are typically
        not measured in wavefront sensing.
        """
        super().__init__(**kwargs)

        # Type annotation for mypy to understand the config structure
        self.config: CalcZernikesNeuralTaskConfig

        # Initialize task logger consistent with other tasks
        self.log = logging.getLogger(type(self).__name__)  # type: ignore

        # Define default Noll indices (zk terms 4-23, excl piston, tip, tilt)
        self.nollIndices = self.config.nollIndices
        self.log.debug("Configured Noll indices: %s", self.nollIndices)

        # Deferred import of TARTS to handle cases where it's not in the build
        try:
            from tarts import NeuralActiveOpticsSys
        except ImportError as e:
            raise ImportError(
                "TARTS neural network system not available. "
                "Please ensure TARTS is installed and available in the Python path. "
                f"Original error: {e}"
            ) from e

        # TARTS system handles None paths by creating new models with random
        # weights
        self.tarts = NeuralActiveOpticsSys(
            os.path.expandvars(self.config.datasetParamPath),
            os.path.expandvars(self.config.wavenetPath),
            os.path.expandvars(self.config.alignetPath),
            os.path.expandvars(self.config.aggregatornetPath),
        )
        self.tarts = self.tarts.eval()
        # Use configurable crop size instead of hard-coded TARTS value
        self.cropSize = self.config.cropSize
        self.device = self.config.device

        self.log.info("Initialized TARTS with dataset params: %s", self.config.datasetParamPath)
        self.log.debug(
            "Model paths - wavenet: %s, alignet: %s, aggregatornet: %s",
            self.config.wavenetPath,
            self.config.alignetPath,
            self.config.aggregatornetPath,
        )
        self.log.info("Running on device: %s", self.device)

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
            self.log.debug("Moved all sub-models and CNNs to CPU")
        else:
            self.tarts.to("cuda")
            self.log.debug("Moved models to CUDA")

        self.log.debug("TARTS crop size: %s", self.cropSize)

    def _is_pytorch_tensor(self, obj: Any) -> bool:
        """Check if object is a PyTorch tensor."""
        try:
            # Use getattr to avoid hasattr checks
            cpu_method = getattr(obj, 'cpu', None)
            numpy_method = getattr(obj, 'numpy', None)
            return cpu_method is not None and numpy_method is not None
        except Exception:
            return False

    def _to_numpy(self, obj: Any) -> np.ndarray:
        """Convert PyTorch tensor or other object to numpy array."""
        if self._is_pytorch_tensor(obj):
            return obj.cpu().numpy()
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(obj)

    def _get_tarts_centers(self) -> Optional[np.ndarray]:
        """Get TARTS centers as numpy array, handling PyTorch tensors."""
        try:
            if self.tarts.centers is not None:
                return self._to_numpy(self.tarts.centers)
        except AttributeError:
            pass
        return None

    def _validate_and_normalize_centers(self, centers_array: np.ndarray) -> Optional[np.ndarray]:
        """Validate and normalize TARTS centers array to [n, 2] format.

        Args:
            centers_array: Input centers array

        Returns:
            Normalized centers array in [n, 2] format, or None if invalid

        Raises:
            Warning: Logged when centers array has invalid shape or dimensions
        """
        if centers_array.ndim == 1:
            # Flattened array: [x1, y1, x2, y2, ...] -> reshape to [n, 2]
            if len(centers_array) % 2 == 0:
                centers_array = centers_array.reshape(-1, 2)
                self.log.info("Reshaped flattened centers to shape: %s", centers_array.shape)
                return centers_array
            else:
                self.log.warning("TARTS centers length %d is not divisible by 2", len(centers_array))
                return None
        elif centers_array.ndim == 2:
            # Already in [n, 2] format
            if centers_array.shape[1] != 2:
                self.log.warning("TARTS centers shape %s doesn't have 2 columns", centers_array.shape)
                return None
            return centers_array
        else:
            self.log.warning("Unexpected TARTS centers dimensionality: %d", centers_array.ndim)
            return None

    def _extract_centers_for_stamps(
        self, centers_array: np.ndarray, num_stamps: int
    ) -> tuple[list[float], list[float]]:
        """Extract center coordinates for the specified number of stamps.

        Args:
            centers_array: Validated centers array in [n, 2] format
            num_stamps: Number of stamps needed

        Returns:
            Tuple of (cent_x_list, cent_y_list) for the stamps
        """
        if len(centers_array) == num_stamps:
            cent_x_list = centers_array[:, 0].tolist()
            cent_y_list = centers_array[:, 1].tolist()
            self.log.info(
                "Using TARTS centers for %d donut stamps: first few = %s",
                num_stamps, centers_array[:3].tolist()
            )
        elif len(centers_array) > num_stamps:
            # Take first num_stamps centers
            cent_x_list = centers_array[:num_stamps, 0].tolist()
            cent_y_list = centers_array[:num_stamps, 1].tolist()
            self.log.info(
                "Using first %d TARTS centers out of %d available for stamps",
                num_stamps, len(centers_array)
            )
        else:
            # Not enough centers, repeat the last one
            cent_x_list = [centers_array[-1, 0]] * num_stamps
            cent_y_list = [centers_array[-1, 1]] * num_stamps
            self.log.warning(
                "Only %d TARTS centers available for %d stamps, repeating last center",
                len(centers_array), num_stamps
            )
        return cent_x_list, cent_y_list

    def _get_tarts_fx(self) -> Optional[np.ndarray]:
        """Get TARTS fx values as numpy array."""
        try:
            if self.tarts.fx is not None:
                return self._to_numpy(self.tarts.fx)
        except AttributeError:
            pass
        return None

    def _get_tarts_fy(self) -> Optional[np.ndarray]:
        """Get TARTS fy values as numpy array."""
        try:
            if self.tarts.fy is not None:
                return self._to_numpy(self.tarts.fy)
        except AttributeError:
            pass
        return None

    def _get_tarts_snr(self) -> Optional[np.ndarray]:
        """Get TARTS SNR values as numpy array."""
        try:
            if self.tarts.SNR is not None:
                return self._to_numpy(self.tarts.SNR)
        except AttributeError:
            pass
        return None

    def _get_visit_id(self, exp_id: Any) -> int:
        """Get visit ID from exposure ID, handling different types."""
        try:
            # Use getattr with default to avoid hasattr check
            get_visit_id_method = getattr(exp_id, 'getVisitId', None)
            if get_visit_id_method is not None:
                return get_visit_id_method()
            else:
                return exp_id
        except Exception:
            return -1

    def _get_exposure_metadata(self, exposure: afwImage.Exposure) -> dict[str, float | int]:
        """Extract metadata from exposure as a dictionary
           with safe fallback values.

        Args:
            exposure: The exposure object to extract metadata from

        Returns:
            Dictionary containing metadata with safe fallback values
        """
        metadata: dict[str, float | int] = {}

        # Get exposure info safely
        exp_info = getattr(exposure, 'getInfo', lambda: None)()
        if exp_info is None:
            return metadata

        # Visit ID
        exp_id = getattr(exp_info, 'getId', lambda: None)()
        if exp_id is not None:
            metadata['visit_id'] = self._get_visit_id(exp_id)
        else:
            metadata['visit_id'] = -1

        # Boresight rotation angle
        boresight_rot_angle = getattr(
            exp_info, 'getBoresightRotAngle', lambda: None
        )()
        if boresight_rot_angle is not None:
            metadata['boresight_rot_angle_rad'] = getattr(
                boresight_rot_angle, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_rot_angle_rad'] = float('nan')

        # Boresight par angle
        boresight_par_angle = getattr(
            exp_info, 'getBoresightParAngle', lambda: None
        )()
        if boresight_par_angle is not None:
            metadata['boresight_par_angle_rad'] = getattr(
                boresight_par_angle, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_par_angle_rad'] = float('nan')

        # Boresight altitude
        boresight_alt = getattr(exp_info, 'getBoresightAlt', lambda: None)()
        if boresight_alt is not None:
            metadata['boresight_alt_rad'] = getattr(
                boresight_alt, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_alt_rad'] = float('nan')

        # Boresight azimuth
        boresight_az = getattr(exp_info, 'getBoresightAz', lambda: None)()
        if boresight_az is not None:
            metadata['boresight_az_rad'] = getattr(
                boresight_az, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_az_rad'] = float('nan')

        # Boresight RA
        boresight_ra = getattr(exp_info, 'getBoresightRa', lambda: None)()
        if boresight_ra is not None:
            metadata['boresight_ra_rad'] = getattr(
                boresight_ra, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_ra_rad'] = float('nan')

        # Boresight DEC
        boresight_dec = getattr(exp_info, 'getBoresightDec', lambda: None)()
        if boresight_dec is not None:
            metadata['boresight_dec_rad'] = getattr(
                boresight_dec, 'asRadians', lambda: float('nan')
            )()
        else:
            metadata['boresight_dec_rad'] = float('nan')

        # MJD date
        exp_date = getattr(exp_info, 'getDate', lambda: None)()
        if exp_date is not None:
            mjd = getattr(
                exp_date, 'get', lambda system: float('nan')
            )(system=DateTime.MJD)
            metadata['mjd'] = mjd
        else:
            metadata['mjd'] = float('nan')

        return metadata

    def createDonutStampFromTarts(
        self, exposure: afwImage.Exposure, cropped_image: np.ndarray, defocalType: str
    ) -> DonutStamps:
        """Create DonutStamps from TARTS output - handles multiple donuts."""
        # Log image type and shape for debugging
        try:
            image_shape = cropped_image.shape
            self.log.debug(
                f"Creating DonutStamps; input image shape: {image_shape}, defocalType='{defocalType}'"
            )
        except AttributeError:
            self.log.debug(
                f"Creating DonutStamps; input image type: {type(cropped_image)}, defocalType='{defocalType}'"
            )
        # Extract information from exposure
        detector = exposure.getDetector()
        detectorName = detector.getName()
        cameraName = "LSSTCam"
        bandLabel = exposure.filter.bandLabel

        # Get exposure dimensions for centroid calculation
        bbox = exposure.getBBox()
        center_x = bbox.getCenterX()
        center_y = bbox.getCenterY()

        # Use the cropped image directly as it's guaranteed to be a numpy array
        image_array = cropped_image

        # Handle different input shapes
        if len(image_array.shape) == 2:
            # Single donut: [cropSize, cropSize] -> [1, cropSize, cropSize]
            image_array = image_array.reshape(1, self.cropSize, self.cropSize)
        elif len(image_array.shape) == 3:
            # Multiple donuts: [num_stamps, cropSize, cropSize] - correct shape
            pass
        else:
            raise ValueError(f"Unexpected image array shape: {image_array.shape}")

        num_stamps = image_array.shape[0]
        self.log.debug("Normalized to %d stamp(s) of size %dx%d", num_stamps, self.cropSize, self.cropSize)
        # Get TARTS centers for accurate centroid positions
        centers_array = self._get_tarts_centers()
        if centers_array is not None:
            self.log.info("TARTS centers numpy array shape: %s", centers_array.shape)

            # Validate and normalize centers array
            normalized_centers = self._validate_and_normalize_centers(centers_array)
            if normalized_centers is not None and len(normalized_centers) > 0:
                # Extract coordinates for stamps
                cent_x_list, cent_y_list = self._extract_centers_for_stamps(normalized_centers, num_stamps)
            else:
                # TARTS centers invalid, use exposure center
                cent_x_list = [center_x] * num_stamps
                cent_y_list = [center_y] * num_stamps
                self.log.warning(
                    "TARTS centers invalid, using exposure center [%.1f, %.1f] for %d donut stamps",
                    center_x, center_y, num_stamps
                )
        else:
            # TARTS centers not available, use exposure center
            cent_x_list = [center_x] * num_stamps
            cent_y_list = [center_y] * num_stamps
            self.log.warning(
                "TARTS centers not available, using exposure center [%.1f, %.1f] for %d donut stamps",
                center_x, center_y, num_stamps
            )

        # Create metadata for all donuts
        metadata = PropertyList()
        metadata["RA_DEG"] = [0.0] * num_stamps  # Will be calculated from WCS
        metadata["DEC_DEG"] = [0.0] * num_stamps  # Will be calculated from WCS
        metadata["CENT_X"] = cent_x_list  # Use TARTS centers or exposure center
        metadata["CENT_Y"] = cent_y_list  # Use TARTS centers or exposure center
        metadata["DET_NAME"] = [detectorName] * num_stamps  # Valid detector name
        metadata["CAM_NAME"] = [cameraName] * num_stamps  # Valid camera name
        metadata["DFC_TYPE"] = [defocalType] * num_stamps  # "extra" or "intra"
        metadata["DFC_DIST"] = [1.5] * num_stamps  # Default defocal distance in mm
        metadata["BANDPASS"] = [bandLabel] * num_stamps  # Filter band
        metadata["BLEND_CX"] = ["nan"] * num_stamps  # No blended sources
        metadata["BLEND_CY"] = ["nan"] * num_stamps  # No blended sources

        # Add TARTS-specific metadata
        try:
            # Convert PyTorch tensors to numpy arrays, then to Python lists
            fx_array = self._get_tarts_fx()
            fy_array = self._get_tarts_fy()

            if fx_array is not None and fy_array is not None:
                fx_list = fx_array.tolist()
                fy_list = fy_array.tolist()
            else:
                fx_list = []
                fy_list = []

            # Flatten nested lists if needed
            if fx_list and isinstance(fx_list[0], list):
                fx_list = [item for sublist in fx_list for item in sublist]
            if fy_list and isinstance(fy_list[0], list):
                fy_list = [item for sublist in fy_list for item in sublist]

            # Use fx/fy lists directly if they match the number of stamps
            if len(fx_list) == num_stamps and len(fy_list) == num_stamps:
                metadata["FX"] = fx_list  # TARTS fx values (flat Python list)
                metadata["FY"] = fy_list  # TARTS fy values (flat Python list)
            else:
                # If lengths don't match, pad with zeros or repeat last value
                metadata["FX"] = [fx_list[0] if len(fx_list) > 0 else 0.0] * num_stamps
                metadata["FY"] = [fy_list[0] if len(fy_list) > 0 else 0.0] * num_stamps
        except AttributeError:
            self.log.warning("TARTS fx/fy attributes not available, using default values")
            metadata["FX"] = [0.0] * num_stamps  # Default fx values
            metadata["FY"] = [0.0] * num_stamps  # Default fy values

        # Create list of DonutStamp objects
        donutStamps = []

        for i in range(num_stamps):
            # Create MaskedImageF for this donut
            stamp_im = afwImage.MaskedImageF(self.cropSize, self.cropSize)
            stamp_im.image.array[:] = image_array[i]  # Use the i-th donut image
            stamp_im.setXY0(0, 0)  # Set origin

            # Create linear WCS for this donut stamp using TARTS center if
            # available
            try:
                if self.tarts.centers is not None:
                    # Handle TARTS centers data structure
                    centers_array = self._get_tarts_centers()

                    # Handle different array shapes
                    if centers_array is not None:
                        if centers_array.ndim == 1 and len(centers_array) % 2 == 0:
                            centers_array = centers_array.reshape(-1, 2)
                        elif centers_array.ndim != 2 or centers_array.shape[1] != 2:
                            centers_array = None

                    # Use individual center if available
                    if centers_array is not None and i < len(centers_array):
                        centroid_position = lsst.geom.Point2D(centers_array[i, 0], centers_array[i, 1])
                        self.log.debug(
                            "Using TARTS center [%.2f, %.2f] for donut %d",
                            centers_array[i, 0], centers_array[i, 1], i+1
                        )
                    else:
                        centroid_position = lsst.geom.Point2D(center_x, center_y)
                        self.log.debug(
                            "Using exposure center [%.2f, %.2f] for donut %d", center_x, center_y, i+1
                        )
                else:
                    centroid_position = lsst.geom.Point2D(center_x, center_y)
                    self.log.debug(
                        "Using exposure center [%.2f, %.2f] for donut %d", center_x, center_y, i+1
                    )
            except AttributeError:
                centroid_position = lsst.geom.Point2D(center_x, center_y)
                self.log.debug(
                    "Using exposure center [%.2f, %.2f] for donut %d", center_x, center_y, i+1
                )

            wcs = exposure.wcs
            linearTransform = wcs.linearizePixelToSky(centroid_position, lsst.geom.degrees)
            cdMatrix = linearTransform.getLinear().getMatrix()
            linear_wcs = lsst.afw.geom.makeSkyWcs(
                centroid_position,
                wcs.pixelToSky(centroid_position),
                cdMatrix
            )

            # Create DonutStamp using factory
            donutStamp = DonutStamp.factory(
                stamp_im=stamp_im,
                metadata=metadata,
                index=i,  # Index into the metadata lists
                archive_element=linear_wcs
            )
            donutStamps.append(donutStamp)

        # Create DonutStamps collection
        donutStampsObj = DonutStamps(donutStamps)

        # Set scalar visit-level metadata directly (don't call
        # _refresh_metadata as it overrides with arrays)
        # Use direct assignment to ensure scalar values are stored
        donutStampsObj.metadata["DET_NAME"] = detectorName
        donutStampsObj.metadata["CAM_NAME"] = cameraName
        donutStampsObj.metadata["BANDPASS"] = bandLabel
        donutStampsObj.metadata["DFC_TYPE"] = defocalType
        donutStampsObj.metadata["DFC_DIST"] = 1.5

        self.log.info("Set DonutStamps metadata: DET_NAME='%s', CAM_NAME='%s', DFC_TYPE='%s'",
                     detectorName, cameraName, defocalType)

        # Debug: Check what metadata keys are available after setting
        self.log.debug("Metadata keys after setting: %s", list(donutStampsObj.metadata.names()))
        self.log.debug("DET_NAME value: %s", donutStampsObj.metadata.get("DET_NAME", "NOT_FOUND"))

        # Add exposure metadata using safe extraction
        exp_metadata = self._get_exposure_metadata(exposure)

        # Set metadata with safe fallback values
        donutStampsObj.metadata["VISIT"] = exp_metadata.get('visit_id', -1)
        donutStampsObj.metadata["BORESIGHT_ROT_ANGLE_RAD"] = exp_metadata.get(
            'boresight_rot_angle_rad', float('nan')
        )
        donutStampsObj.metadata["BORESIGHT_PAR_ANGLE_RAD"] = exp_metadata.get(
            'boresight_par_angle_rad', float('nan')
        )
        donutStampsObj.metadata["BORESIGHT_ALT_RAD"] = exp_metadata.get(
            'boresight_alt_rad', float('nan')
        )
        donutStampsObj.metadata["BORESIGHT_AZ_RAD"] = exp_metadata.get(
            'boresight_az_rad', float('nan')
        )
        donutStampsObj.metadata["BORESIGHT_RA_RAD"] = exp_metadata.get(
            'boresight_ra_rad', float('nan')
        )
        donutStampsObj.metadata["BORESIGHT_DEC_RAD"] = exp_metadata.get(
            'boresight_dec_rad', float('nan')
        )
        donutStampsObj.metadata["MJD"] = exp_metadata.get('mjd', float('nan'))

        self.log.info("Constructed %d DonutStamps with WCS and metadata", len(donutStamps))
        return donutStampsObj

    def donutStampsToQTable(self, donutStamps: DonutStamps) -> QTable:
        """Convert DonutStamps to QTable for storage compatibility.

        Parameters
        ----------
        donutStamps : DonutStamps
            The DonutStamps object to convert

        Returns
        -------
        QTable
            A QTable containing the donut stamp metadata
        """
        if len(donutStamps) == 0:
            self.log.warning("No donut stamps available to convert to QTable")
            return QTable()

        # Refresh metadata to ensure it's up to date
        donutStamps._refresh_metadata()

        # Convert PropertyList to regular dictionary
        metadata_dict = {}
        for key in donutStamps.metadata.names():
            metadata_dict[key] = donutStamps.metadata.getArray(key)

        # Convert metadata dictionary to QTable
        return QTable(metadata_dict)

    def initZkTable(self) -> QTable:
        """Initialize the table to store the Zernike coefficients

        Returns
        -------
        table : `astropy.table.QTable`
            Table to store the Zernike coefficients
        """
        # Create table with proper dtype structure
        dtype = [
            ("label", "<U12"),
            ("used", np.bool_),
            ("intra_field_x", "<f4"),
            ("intra_field_y", "<f4"),
            ("extra_field_x", "<f4"),
            ("extra_field_y", "<f4"),
            ("intra_centroid_x", "<f4"),
            ("intra_centroid_y", "<f4"),
            ("extra_centroid_x", "<f4"),
            ("extra_centroid_y", "<f4"),
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
            ("intra_fx", "<f4"),
            ("extra_fx", "<f4"),
            ("intra_fy", "<f4"),
            ("extra_fy", "<f4"),
        ]

        # Add Zernike coefficient columns
        for j in self.config.nollIndices:
            dtype.append((f"Z{j}", "<f4"))

        table = QTable(dtype=dtype)

        # Assign units where appropriate
        table["intra_field_x"].unit = u.deg
        table["intra_field_y"].unit = u.deg
        table["extra_field_x"].unit = u.deg
        table["extra_field_y"].unit = u.deg
        table["intra_centroid_x"].unit = u.pixel
        table["intra_centroid_y"].unit = u.pixel
        table["extra_centroid_x"].unit = u.pixel
        table["extra_centroid_y"].unit = u.pixel
        for j in self.config.nollIndices:
            table[f"Z{j}"].unit = u.um

        self.log.debug("Initialized Zernike table with %d coefficient columns", len(self.config.nollIndices))
        return table

    def createZkTable(
        self,
        extraStamps: DonutStamps,
        intraStamps: DonutStamps,
        zkCoeffRaw: np.ndarray,
        zkCoeffAvg: np.ndarray,
    ) -> QTable:
        """Create the Zernike table to store Zernike Coefficients.

        Parameters
        ----------
        extraStamps: DonutStamps
            The extrafocal stamps
        intraStamps: DonutStamps
            The intrafocal stamps
        zkCoeffRaw: np.ndarray
            Raw zernike coefficients from TARTS
        zkCoeffAvg: np.ndarray
            Averaged zernike coefficients

        Returns
        -------
        table : `astropy.table.QTable`
            Table with the Zernike coefficients
        """
        zkTable = self.initZkTable()
        self.log.debug(
            "Creating Zernike table: intra stamps=%d, extra stamps=%d",
            len(intraStamps),
            len(extraStamps),
        )

        # Add average row
        row_data = {
            "label": "average",
            "used": True,
            "intra_field_x": np.nan,
            "intra_field_y": np.nan,
            "extra_field_x": np.nan,
            "extra_field_y": np.nan,
            "intra_centroid_x": np.nan,
            "intra_centroid_y": np.nan,
            "extra_centroid_x": np.nan,
            "extra_centroid_y": np.nan,
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
            "intra_fx": np.nan,
            "extra_fx": np.nan,
            "intra_fy": np.nan,
            "extra_fy": np.nan,
        }

        # Add Zernike coefficients
        for i, j in enumerate(self.config.nollIndices):
            if i < len(zkCoeffAvg):
                row_data[f"Z{j}"] = zkCoeffAvg[i] * u.um
            else:
                row_data[f"Z{j}"] = 0.0 * u.um

        zkTable.add_row(row_data)
        self.log.debug("Added average row with %d Zernike terms", len(self.config.nollIndices))

        # Add individual donut rows only if we have donut stamps
        # For TARTS, we typically have one set of coefficients per exposure
        # We'll create one row per donut if available, otherwise just the
        # average row
        max_stamps = max(len(extraStamps), len(intraStamps))
        self.log.debug(
            "Creating Zernike table: %d extra stamps, %d intra stamps, "
            "zkCoeffRaw shape: %s",
            len(extraStamps), len(intraStamps), zkCoeffRaw.shape
        )

        # Add individual donut rows
        for i, (intraStamp, extraStamp) in enumerate(zip_longest(intraStamps, extraStamps)):
            # Get the zernike coefficients for this donut
            # For TARTS, we have individual coefficients per donut in
            # total_zernikes
            if zkCoeffRaw.ndim == 2 and zkCoeffRaw.shape[0] > i:
                zk = zkCoeffRaw[i]
                self.log.debug("Using individual coefficients for donut %d", i+1)
            elif zkCoeffRaw.ndim == 2 and zkCoeffRaw.shape[0] == 1:
                # Single set of coefficients for all donuts (fallback)
                zk = zkCoeffRaw[0]
                if i == 0:  # Only log once
                    self.log.debug(
                        "Using single coefficient set for all %d donuts (fallback)",
                        max_stamps,
                    )
            else:
                # If we don't have individual coefficients, use the average
                zk = zkCoeffAvg
                self.log.debug(
                    "Raw coeff shape %s does not cover stamp index %d; using average",
                    getattr(zkCoeffRaw, "shape", None),
                    i,
                )

            row: dict = dict()
            row["label"] = f"donut{i+1}"
            row["used"] = True  # TARTS predictions are always used

            # Add Zernike coefficients
            for idx, j in enumerate(self.config.nollIndices):
                if idx < len(zk):
                    row[f"Z{j}"] = zk[idx] * u.um
                else:
                    row[f"Z{j}"] = 0.0 * u.um

            # Get field positions and centroids from stamps
            if intraStamp is not None:
                intra = intraStamp
                # Use TARTS fx/fy values directly (same as
                # DonutQualityTable)
                try:
                    fx_array = self._get_tarts_fx()
                    fy_array = self._get_tarts_fy()

                    fx_list = fx_array.tolist() if fx_array is not None else []
                    fy_list = fy_array.tolist() if fy_array is not None else []
                    # Flatten nested lists if needed
                    if fx_list and isinstance(fx_list[0], list):
                        fx_list = [item for sublist in fx_list for item in sublist]
                    if fy_list and isinstance(fy_list[0], list):
                        fy_list = [item for sublist in fy_list for item in sublist]

                    if i < len(fx_list) and i < len(fy_list):
                        row["intra_field_x"] = fx_list[i] * u.deg
                        row["intra_field_y"] = fy_list[i] * u.deg
                    else:
                        self.log.warning(
                            "TARTS fx/fy data insufficient for donut %d: "
                            "fx_list length=%d, fy_list length=%d. "
                            "This may indicate a data mismatch between TARTS output and donut stamps.",
                            i + 1, len(fx_list), len(fy_list)
                        )
                        row["intra_field_x"] = np.nan
                        row["intra_field_y"] = np.nan
                except AttributeError:
                    # Fallback to calcFieldXY if TARTS fx/fy not available
                    # Expected when TARTS doesn't provide field positions
                    self.log.debug(
                        "TARTS fx/fy not available for donut %d, using calcFieldXY fallback",
                        i + 1
                    )
                    field_xy = intra.calcFieldXY()
                    row["intra_field_x"] = field_xy[0] * u.deg
                    row["intra_field_y"] = field_xy[1] * u.deg
                # Use TARTS centers if available for more accurate
                # positions
                try:
                    if self.tarts.centers is not None:
                        # Handle TARTS centers data structure
                        centers_array = self._get_tarts_centers()

                        # Handle different array shapes
                        if centers_array is not None:
                            array = centers_array  # Type narrowing for mypy
                            if array.ndim == 1 and len(array) % 2 == 0:
                                centers_array = array.reshape(-1, 2)
                            elif array.ndim != 2 or array.shape[1] != 2:
                                centers_array = None

                        # Use individual center if available
                        if centers_array is not None and i < len(centers_array):
                            row["intra_centroid_x"] = centers_array[i, 0] * u.pixel
                            row["intra_centroid_y"] = centers_array[i, 1] * u.pixel
                        else:
                            row["intra_centroid_x"] = intra.centroid_position.x * u.pixel
                            row["intra_centroid_y"] = intra.centroid_position.y * u.pixel
                    else:
                        row["intra_centroid_x"] = intra.centroid_position.x * u.pixel
                        row["intra_centroid_y"] = intra.centroid_position.y * u.pixel
                except AttributeError:
                    row["intra_centroid_x"] = intra.centroid_position.x * u.pixel
                    row["intra_centroid_y"] = intra.centroid_position.y * u.pixel
                # intraStamp is None (padded by zip_longest when arrays have
                # different lengths)
                self.log.debug("No intra stamp available for donut %d", i + 1)
                row["intra_field_x"] = np.nan
                row["intra_field_y"] = np.nan
                row["intra_centroid_x"] = np.nan
                row["intra_centroid_y"] = np.nan

            if extraStamp is not None:
                extra = extraStamp
                # Use TARTS fx/fy values directly (same as
                # DonutQualityTable)
                try:
                    fx_array = self._get_tarts_fx()
                    fy_array = self._get_tarts_fy()

                    fx_list = fx_array.tolist() if fx_array is not None else []
                    fy_list = fy_array.tolist() if fy_array is not None else []
                    # Flatten nested lists if needed
                    if fx_list and isinstance(fx_list[0], list):
                        fx_list = [item for sublist in fx_list for item in sublist]
                    if fy_list and isinstance(fy_list[0], list):
                        fy_list = [item for sublist in fy_list for item in sublist]

                    if i < len(fx_list) and i < len(fy_list):
                        row["extra_field_x"] = fx_list[i] * u.deg
                        row["extra_field_y"] = fy_list[i] * u.deg
                    else:
                        self.log.warning(
                            "TARTS fx/fy data insufficient for donut %d: "
                            "fx_list length=%d, fy_list length=%d. "
                            "This may indicate a data mismatch between TARTS output and donut stamps.",
                            i + 1, len(fx_list), len(fy_list)
                        )
                        row["extra_field_x"] = np.nan
                        row["extra_field_y"] = np.nan
                except AttributeError:
                    # Fallback to calcFieldXY if TARTS fx/fy not available
                    # Expected when TARTS doesn't provide field positions
                    self.log.debug(
                        "TARTS fx/fy not available for donut %d, using calcFieldXY fallback",
                        i + 1
                    )
                    field_xy = extra.calcFieldXY()
                    row["extra_field_x"] = field_xy[0] * u.deg
                    row["extra_field_y"] = field_xy[1] * u.deg
                # Use TARTS centers if available for more accurate
                # positions
                try:
                    if self.tarts.centers is not None:
                        # Handle TARTS centers data structure
                        centers_array = self._get_tarts_centers()

                        # Handle different array shapes
                        if centers_array is not None:
                            array = centers_array  # Type narrowing for mypy
                            if array.ndim == 1 and len(array) % 2 == 0:
                                centers_array = array.reshape(-1, 2)
                            elif array.ndim != 2 or array.shape[1] != 2:
                                centers_array = None

                        # Use individual center if available
                        if centers_array is not None and i < len(centers_array):
                            row["extra_centroid_x"] = centers_array[i, 0] * u.pixel
                            row["extra_centroid_y"] = centers_array[i, 1] * u.pixel
                        else:
                            row["extra_centroid_x"] = extra.centroid_position.x * u.pixel
                            row["extra_centroid_y"] = extra.centroid_position.y * u.pixel
                    else:
                        row["extra_centroid_x"] = extra.centroid_position.x * u.pixel
                        row["extra_centroid_y"] = extra.centroid_position.y * u.pixel
                except AttributeError:
                    row["extra_centroid_x"] = extra.centroid_position.x * u.pixel
                    row["extra_centroid_y"] = extra.centroid_position.y * u.pixel
            else:
                # extraStamp=None (padded when arrays have different lengths)
                self.log.debug("No extra stamp available for donut %d", i + 1)
                row["extra_field_x"] = np.nan
                row["extra_field_y"] = np.nan
                row["extra_centroid_x"] = np.nan
                row["extra_centroid_y"] = np.nan

            # Get quality metrics from metadata
            for key in ["MAG", "SN", "ENTROPY", "FRAC_BAD_PIX", "MAX_POWER_GRAD", "FX", "FY"]:
                for stamps, foc in [
                    (intraStamps, "intra"),
                    (extraStamps, "extra"),
                ]:
                    if (len(stamps) > 0 and key in stamps.metadata and
                            i < len(stamps.metadata.getArray(key))):
                        val = stamps.metadata.getArray(key)[i]
                    else:
                        val = np.nan
                    row[f"{foc}_{key.lower()}"] = val

            zkTable.add_row(row)

        # Set metadata on the Zernike table
        zkTable.meta = self.createZkTableMetadata()

        # Debug: Log detector names in metadata
        if "intra" in zkTable.meta and "det_name" in zkTable.meta["intra"]:
            self.log.info("Zernike table intra det_name: '%s'", zkTable.meta["intra"]["det_name"])
        if "extra" in zkTable.meta and "det_name" in zkTable.meta["extra"]:
            self.log.info("Zernike table extra det_name: '%s'", zkTable.meta["extra"]["det_name"])

        self.log.info("Created Zernike QTable with %d rows", len(zkTable))
        return zkTable

    def createZkTableMetadata(self) -> dict:
        """Create metadata for the Zernike table.

        This method creates the metadata structure expected by downstream
        tasks,
        including 'intra' and 'extra' dictionaries with detector information.

        Returns
        -------
        dict
            Metadata dictionary with 'intra' and 'extra' keys containing
            detector and visit information.
        """
        meta: dict = {}
        meta["intra"] = {}
        meta["extra"] = {}
        cam_name = None

        # Handle case where both stamp collections are None
        if self.stampsIntra is None and self.stampsExtra is None:
            self.log.warning(
                "Both intra and extra stamp collections are None. "
                "This indicates no donut data is available for Zernike estimation."
            )
            meta["cam_name"] = "LSSTCam"
            meta["intra"]["det_name"] = "Unknown"
            meta["extra"]["det_name"] = "Unknown"
            return meta

        # Process intra and extra stamps
        for dict_, stamps in [
            (meta["intra"], self.stampsIntra),
            (meta["extra"], self.stampsExtra),
        ]:
            if stamps is None:
                # Populate with sentinel values if stamps are None
                dict_["det_name"] = "Unknown"
                dict_["visit"] = -1
                dict_["dfc_dist"] = float('nan')
                dict_["band"] = "Unknown"
                dict_["boresight_rot_angle_rad"] = float('nan')
                dict_["boresight_par_angle_rad"] = float('nan')
                dict_["boresight_alt_rad"] = float('nan')
                dict_["boresight_az_rad"] = float('nan')
                dict_["boresight_ra_rad"] = float('nan')
                dict_["boresight_dec_rad"] = float('nan')
                dict_["mjd"] = float('nan')
                continue

            # Debug: Check what metadata keys are available
            self.log.debug("Available metadata keys: %s", list(stamps.metadata.names()))
            self.log.debug("DET_NAME value: %s", stamps.metadata.get("DET_NAME", "NOT_FOUND"))

            # Extract metadata from stamps (now stored as scalar values)
            try:
                # Get DET_NAME from stamps metadata (should always be present)
                dict_["det_name"] = stamps.metadata.get("DET_NAME", "Unknown")
                if dict_["det_name"] == "Unknown":
                    self.log.warning(
                        "DET_NAME not found in stamps metadata, using 'Unknown'. "
                        "This may indicate a problem with stamp creation."
                    )
                else:
                    self.log.info("Using DET_NAME from stamps metadata: '%s'", dict_["det_name"])

                # Get other metadata with proper sentinel values
                dict_["visit"] = stamps.metadata.get("VISIT", -1)
                dict_["dfc_dist"] = stamps.metadata.get("DFC_DIST", float('nan'))
                dict_["band"] = stamps.metadata.get("BANDPASS", "Unknown")
                dict_["boresight_rot_angle_rad"] = stamps.metadata.get(
                    "BORESIGHT_ROT_ANGLE_RAD", float('nan')
                )
                dict_["boresight_par_angle_rad"] = stamps.metadata.get(
                    "BORESIGHT_PAR_ANGLE_RAD", float('nan')
                )
                dict_["boresight_alt_rad"] = stamps.metadata.get("BORESIGHT_ALT_RAD", float('nan'))
                dict_["boresight_az_rad"] = stamps.metadata.get("BORESIGHT_AZ_RAD", float('nan'))
                dict_["boresight_ra_rad"] = stamps.metadata.get("BORESIGHT_RA_RAD", float('nan'))
                dict_["boresight_dec_rad"] = stamps.metadata.get("BORESIGHT_DEC_RAD", float('nan'))
                dict_["mjd"] = stamps.metadata.get("MJD", float('nan'))

                if cam_name is None:
                    cam_name = stamps.metadata.get("CAM_NAME", "LSSTCam")
            except Exception as e:
                self.log.error("Error accessing metadata: %s", e)
                # Use sentinel values if metadata is missing
                dict_["det_name"] = "Unknown"
                dict_["visit"] = -1
                dict_["dfc_dist"] = float('nan')
                dict_["band"] = "Unknown"
                dict_["boresight_rot_angle_rad"] = float('nan')
                dict_["boresight_par_angle_rad"] = float('nan')
                dict_["boresight_alt_rad"] = float('nan')
                dict_["boresight_az_rad"] = float('nan')
                dict_["boresight_ra_rad"] = float('nan')
                dict_["boresight_dec_rad"] = float('nan')
                dict_["mjd"] = float('nan')
                if cam_name is None:
                    cam_name = "LSSTCam"

        meta["cam_name"] = cam_name if cam_name else "LSSTCam"

        # Ensure both intra and extra have at least basic structure (for
        # downstream compatibility)
        if "det_name" not in meta["intra"]:
            meta["intra"]["det_name"] = "Unknown"
            meta["intra"]["visit"] = -1
            meta["intra"]["dfc_dist"] = float('nan')
            meta["intra"]["band"] = "Unknown"
            meta["intra"]["boresight_rot_angle_rad"] = float('nan')
            meta["intra"]["boresight_par_angle_rad"] = float('nan')
            meta["intra"]["boresight_alt_rad"] = float('nan')
            meta["intra"]["boresight_az_rad"] = float('nan')
            meta["intra"]["boresight_ra_rad"] = float('nan')
            meta["intra"]["boresight_dec_rad"] = float('nan')
            meta["intra"]["mjd"] = float('nan')

        if "det_name" not in meta["extra"]:
            meta["extra"]["det_name"] = "Unknown"
            meta["extra"]["visit"] = -1
            meta["extra"]["dfc_dist"] = float('nan')
            meta["extra"]["band"] = "Unknown"
            meta["extra"]["boresight_rot_angle_rad"] = float('nan')
            meta["extra"]["boresight_par_angle_rad"] = float('nan')
            meta["extra"]["boresight_alt_rad"] = float('nan')
            meta["extra"]["boresight_az_rad"] = float('nan')
            meta["extra"]["boresight_ra_rad"] = float('nan')
            meta["extra"]["boresight_dec_rad"] = float('nan')
            meta["extra"]["mjd"] = float('nan')

        return meta

    def createDonutQualityTable(self, donutStamps: DonutStamps) -> QTable:
        """Create a quality table from TARTS outputs.

        Parameters
        ----------
        donutStamps : DonutStamps
            The donut stamps (used to determine number of donuts)

        Returns
        -------
        QTable
            A table containing quality information for each donut
        """
        if len(donutStamps) == 0:
            # Return empty table with expected columns
            self.log.warning(
                "No donut stamps available for quality table creation. "
                "This may indicate a problem with donut detection or processing."
            )
            qualityTableCols = [
                "SN",
                "ENTROPY",
                "ENTROPY_SELECT",
                "SN_SELECT",
                "FINAL_SELECT",
                "DEFOCAL_TYPE",
                "FX",
                "FY",
            ]
            return QTable({name: [] for name in qualityTableCols})

        num_donuts = len(donutStamps)

        # Extract TARTS quality metrics
        quality_data = {}

        # Get FX and FY from TARTS
        try:
            # Convert PyTorch tensors to numpy arrays, then to Python lists
            fx_array = self._get_tarts_fx()
            fy_array = self._get_tarts_fy()

            fx_list = fx_array.tolist() if fx_array is not None else []
            fy_list = fy_array.tolist() if fy_array is not None else []

            # Flatten nested lists if needed
            if fx_list and isinstance(fx_list[0], list):
                fx_list = [item for sublist in fx_list for item in sublist]
            if fy_list and isinstance(fy_list[0], list):
                fy_list = [item for sublist in fy_list for item in sublist]

            # Use fx/fy lists directly if they match the number of donuts
            if len(fx_list) == num_donuts and len(fy_list) == num_donuts:
                quality_data["FX"] = fx_list
                quality_data["FY"] = fy_list
            else:
                # If lengths don't match, pad with zeros or repeat last value
                quality_data["FX"] = [fx_list[0] if len(fx_list) > 0 else 0.0] * num_donuts
                quality_data["FY"] = [fy_list[0] if len(fy_list) > 0 else 0.0] * num_donuts
        except AttributeError:
            self.log.warning("TARTS fx/fy attributes not available for quality data, using default values")
            quality_data["FX"] = [0.0] * num_donuts
            quality_data["FY"] = [0.0] * num_donuts

        # Get SNR from TARTS
        try:
            snr_array = self._get_tarts_snr()
            snr_list = snr_array.tolist() if snr_array is not None else []

            # Flatten nested lists if needed
            if snr_list and isinstance(snr_list[0], list):
                snr_list = [item for sublist in snr_list for item in sublist]

            if len(snr_list) == num_donuts:
                quality_data["SN"] = snr_list
            else:
                quality_data["SN"] = [snr_list[0] if len(snr_list) > 0 else np.nan] * num_donuts
        except AttributeError:
            self.log.warning("TARTS SNR attribute not available for quality data, using default values")
            quality_data["SN"] = [np.nan] * num_donuts

        # Add NaN values for missing metrics
        quality_data["ENTROPY"] = [np.nan] * num_donuts
        quality_data["ENTROPY_SELECT"] = [np.nan] * num_donuts
        quality_data["SN_SELECT"] = [np.nan] * num_donuts
        quality_data["FINAL_SELECT"] = [True] * num_donuts  # TARTS predictions are always selected
        quality_data["DEFOCAL_TYPE"] = [np.nan] * num_donuts

        # Create the quality table
        qualityTable = QTable(quality_data)

        self.log.debug("Created donut quality table with %d rows using TARTS FX/FY/SNR", len(qualityTable))
        return qualityTable

    def createDonutTable(
        self, donutStamps: DonutStamps, exposure: afwImage.Exposure, defocalType: str = "intra"
    ) -> QTable:
        """Create a donut source catalog table matching
        GenerateDonutCatalogWcsTask structure.

        This method creates an Astropy QTable containing the positions and
        properties
        of donut sources, following the same structure as
        GenerateDonutCatalogWcsTask.

        Parameters
        ----------
        donutStamps : DonutStamps
            The donut stamps collection
        exposure : afwImage.Exposure
            The exposure containing the donut data

        Returns
        -------
        QTable
            A table containing donut source positions and properties matching
            the standard format
        """
        if len(donutStamps) == 0:
            # Return empty table with expected columns matching standard format
            empty_cols = [
                "coord_ra", "coord_dec", "centroid_x", "centroid_y"
            ]
            return QTable({name: [] for name in empty_cols})

        num_donuts = len(donutStamps)

        # Extract detector and camera info
        detector = exposure.getDetector()
        detector_name = detector.getName()
        camera_name = "LSSTCam"
        band_label = exposure.filter.bandLabel

        self.log.info("Extracted detector info: name='%s', camera='%s', band='%s'",
                     detector_name, camera_name, band_label)

        # Initialize data arrays
        donut_data = {}

        # Get positions from donut stamps - use standard column names and units
        try:
            sky_positions = donutStamps.getSkyPositions()
            donut_data["coord_ra"] = [
                pos.getRa().asRadians() for pos in sky_positions
            ]  # Use radians like standard
            donut_data["coord_dec"] = [
                pos.getDec().asRadians() for pos in sky_positions
            ]  # Use radians like standard
        except AttributeError:
            # Fallback: use exposure center
            bbox = exposure.getBBox()
            center_pos = exposure.wcs.pixelToSky(bbox.getCenterX(), bbox.getCenterY())
            donut_data["coord_ra"] = [center_pos.getRa().asRadians()] * num_donuts
            donut_data["coord_dec"] = [center_pos.getDec().asRadians()] * num_donuts

        # Get centroid positions from TARTS centers (most accurate)
        try:
            if self.tarts.centers is not None:
                self.log.debug(
                    "TARTS centers type: %s, shape: %s",
                    type(self.tarts.centers), getattr(self.tarts.centers, 'shape', 'no shape')
                )

            # Handle different data types for TARTS centers
            centers_array = self._get_tarts_centers()
            if centers_array is not None:
                self.log.info("TARTS centers numpy array shape: %s", centers_array.shape)

                # Handle different array shapes
                if centers_array.ndim == 1:
                    # Flattened array -> reshape to [n,2]
                    if len(centers_array) % 2 == 0:
                        centers_array = centers_array.reshape(-1, 2)
                        self.log.info("Reshaped flattened centers to shape: %s", centers_array.shape)
                    else:
                        self.log.warning("TARTS centers length %d is not divisible by 2", len(centers_array))
                        centers_array = None
                elif centers_array.ndim == 2:
                    # Already in [n, 2] format
                    if centers_array.shape[1] != 2:
                        self.log.warning("TARTS centers shape %s doesn't have 2 columns", centers_array.shape)
                        centers_array = None
                else:
                    self.log.warning("Unexpected TARTS centers dimensionality: %d", centers_array.ndim)
                    centers_array = None

            # Extract coordinates if we have valid centers
            if centers_array is not None and len(centers_array) > 0:
                if len(centers_array) == num_donuts:
                    donut_data["centroid_x"] = centers_array[:, 0].tolist()
                    donut_data["centroid_y"] = centers_array[:, 1].tolist()
                    self.log.info(
                        "Using TARTS centers for %d donuts: first few = %s",
                        num_donuts, centers_array[:3].tolist()
                    )
                elif len(centers_array) > num_donuts:
                    # Take first num_donuts centers
                    donut_data["centroid_x"] = centers_array[:num_donuts, 0].tolist()
                    donut_data["centroid_y"] = centers_array[:num_donuts, 1].tolist()
                    self.log.info(
                        "Using first %d TARTS centers out of %d available",
                        num_donuts, len(centers_array)
                    )
                else:
                    # Not enough centers, repeat the last one
                    donut_data["centroid_x"] = [centers_array[-1, 0]] * num_donuts
                    donut_data["centroid_y"] = [centers_array[-1, 1]] * num_donuts
                    self.log.info(
                        "Only %d TARTS centers available for %d donuts, repeating last center",
                        len(centers_array), num_donuts
                    )
            else:
                # Fallback: use exposure center
                bbox = exposure.getBBox()
                center_x = bbox.getCenterX()
                center_y = bbox.getCenterY()
                donut_data["centroid_x"] = [center_x] * num_donuts
                donut_data["centroid_y"] = [center_y] * num_donuts
                self.log.info(
                    "TARTS centers invalid, using exposure center [%.1f, %.1f] for %d donuts",
                    center_x, center_y, num_donuts
                )
        except AttributeError:
            # TARTS centers not available, use exposure center
            bbox = exposure.getBBox()
            center_x = bbox.getCenterX()
            center_y = bbox.getCenterY()
            donut_data["centroid_x"] = [center_x] * num_donuts
            donut_data["centroid_y"] = [center_y] * num_donuts
            self.log.info(
                "TARTS centers not available, using exposure center [%.1f, %.1f] for %d donuts",
                center_x, center_y, num_donuts
            )
        try:
            centroid_positions = donutStamps.getCentroidPositions()
            donut_data["centroid_x"] = [pos.getX() for pos in centroid_positions]
            donut_data["centroid_y"] = [pos.getY() for pos in centroid_positions]
            self.log.info("Using donut stamps centroids for %d donuts", num_donuts)
        except AttributeError:
            # Final fallback: use exposure center
            bbox = exposure.getBBox()
            center_x = bbox.getCenterX()
            center_y = bbox.getCenterY()
            donut_data["centroid_x"] = [center_x] * num_donuts
            donut_data["centroid_y"] = [center_y] * num_donuts
            self.log.info("Using exposure center final fallback for %d donuts", num_donuts)

        # Get TARTS-specific data for additional columns if available
        try:
            # Convert PyTorch tensors to numpy arrays, then to Python lists
            fx_array = self._get_tarts_fx()
            fy_array = self._get_tarts_fy()

            fx_list = fx_array.tolist() if fx_array is not None else []
            fy_list = fy_array.tolist() if fy_array is not None else []

            # Flatten nested lists if needed
            if fx_list and isinstance(fx_list[0], list):
                fx_list = [item for sublist in fx_list for item in sublist]
            if fy_list and isinstance(fy_list[0], list):
                fy_list = [item for sublist in fy_list for item in sublist]

            # Use fx/fy lists if they match the number of donuts
            if len(fx_list) == num_donuts and len(fy_list) == num_donuts:
                donut_data["fx"] = fx_list
                donut_data["fy"] = fy_list
            else:
                # If lengths don't match, pad with zeros
                donut_data["fx"] = [fx_list[0] if len(fx_list) > 0 else 0.0] * num_donuts
                donut_data["fy"] = [fy_list[0] if len(fy_list) > 0 else 0.0] * num_donuts
        except AttributeError:
            self.log.warning("TARTS fx/fy attributes not available for donut data, using default values")
            donut_data["fx"] = [0.0] * num_donuts
            donut_data["fy"] = [0.0] * num_donuts

        # Get SNR from TARTS
        try:
            snr_array = self._get_tarts_snr()
            snr_list = snr_array.tolist() if snr_array is not None else []

            # Flatten nested lists if needed
            if snr_list and isinstance(snr_list[0], list):
                snr_list = [item for sublist in snr_list for item in sublist]

            if len(snr_list) == num_donuts:
                donut_data["snr"] = snr_list
            else:
                donut_data["snr"] = [snr_list[0] if len(snr_list) > 0 else np.nan] * num_donuts
        except AttributeError:
            self.log.warning("TARTS SNR attribute not available for donut data, using default values")
            donut_data["snr"] = [np.nan] * num_donuts

        # Add detector information as a column (required by downstream
        # aggregation)
        # Ensure detector name is consistent with expected format
        donut_data["detector"] = [detector_name] * num_donuts
        self.log.info("Using detector name: '%s' for %d donuts (defocal type: '%s')",
                     detector_name, num_donuts, defocalType)

        # Add missing telescope coordinate columns required by downstream
        # aggregation
        # These are not generated by neural network, so use default values
        donut_data["thx_CCS"] = [np.nan] * num_donuts  # Telescope X in Camera Coordinate System
        donut_data["thy_CCS"] = [np.nan] * num_donuts  # Telescope Y in Camera Coordinate System
        donut_data["thx_OCS"] = [np.nan] * num_donuts  # Telescope X in Observatory Coordinate System
        donut_data["thy_OCS"] = [np.nan] * num_donuts  # Telescope Y in Observatory Coordinate System
        donut_data["th_N"] = [np.nan] * num_donuts      # Telescope North coordinate
        donut_data["th_W"] = [np.nan] * num_donuts      # Telescope West coordinate

        # Create the table
        donut_table = QTable(donut_data)

        # Add units matching standard format
        donut_table["coord_ra"].unit = u.rad  # Use radians like standard
        donut_table["coord_dec"].unit = u.rad  # Use radians like standard
        # Note: centroid_x and centroid_y are kept as plain floats to avoid
        # Point2D constructor issues
        donut_table["fx"].unit = u.deg
        donut_table["fy"].unit = u.deg

        # Add comprehensive visit_info metadata matching standard format
        try:
            visitInfo = exposure.visitInfo

            # Get visit ID using safe extraction
            exp_metadata = self._get_exposure_metadata(exposure)
            visit_id = exp_metadata.get('visit_id', -1)
            if visit_id == -1:
                visit_id = visitInfo.id

            # Create visit_info matching the standard structure from
            # addVisitInfoToCatTable
            catVisitInfo = {}

            # Boresight coordinates
            visitRaDec = visitInfo.boresightRaDec
            catVisitInfo["boresight_ra"] = visitRaDec.getRa().asDegrees() * u.deg
            catVisitInfo["boresight_dec"] = visitRaDec.getDec().asDegrees() * u.deg

            # Boresight altitude/azimuth
            visitAzAlt = visitInfo.boresightAzAlt
            catVisitInfo["boresight_alt"] = visitAzAlt.getLatitude().asDegrees() * u.deg
            catVisitInfo["boresight_az"] = visitAzAlt.getLongitude().asDegrees() * u.deg

            # Rotation angles
            catVisitInfo["boresight_rot_angle"] = visitInfo.boresightRotAngle.asDegrees() * u.deg
            catVisitInfo["rot_type_name"] = visitInfo.rotType.name
            catVisitInfo["rot_type_value"] = visitInfo.rotType.value
            catVisitInfo["boresight_par_angle"] = visitInfo.boresightParAngle.asDegrees() * u.deg

            # Focus and timing
            catVisitInfo["focus_z"] = visitInfo.focusZ * u.mm
            catVisitInfo["mjd"] = visitInfo.date.toAstropy().tai.mjd
            catVisitInfo["visit_id"] = visit_id
            catVisitInfo["instrument_label"] = visitInfo.instrumentLabel

            # Observatory info
            catVisitInfo["observatory_elevation"] = visitInfo.observatory.getElevation() * u.m
            catVisitInfo["observatory_latitude"] = visitInfo.observatory.getLatitude().asDegrees() * u.deg
            catVisitInfo["observatory_longitude"] = visitInfo.observatory.getLongitude().asDegrees() * u.deg
            catVisitInfo["ERA"] = visitInfo.era.asDegrees() * u.deg
            catVisitInfo["exposure_time"] = visitInfo.exposureTime * u.s

            # Estimate donut diameter using DonutSizeCorrelator
            try:
                from lsst.ts.wep.donutSizeCorrelator import DonutSizeCorrelator
                correlator = DonutSizeCorrelator()
                img = correlator.prepButlerExposure(exposure)
                diameter, *_ = correlator.getDonutDiameter(img)
                catVisitInfo["donut_radius"] = 0.5 * diameter
            except Exception as e:
                self.log.warning("Could not estimate donut diameter: %s", e)
                catVisitInfo["donut_radius"] = 1.0  # Default value

            donut_table.meta["visit_info"] = catVisitInfo

        except Exception as e:
            self.log.warning("Could not extract comprehensive visit_info metadata: %s", e)
            # Provide minimal visit_info structure
            try:
                visit_id = exposure.visitInfo.id
            except Exception:
                visit_id = 0

            donut_table.meta["visit_info"] = {
                "visit_id": visit_id,
                "focus_z": 0.0 * u.mm,
                "boresight_ra": 0.0 * u.deg,
                "boresight_dec": 0.0 * u.deg,
                "boresight_rot_angle": 0.0 * u.deg,
                "boresight_par_angle": 0.0 * u.deg,
                "boresight_alt": 0.0 * u.deg,
                "boresight_az": 0.0 * u.deg,
                "mjd": 0.0,
                "donut_radius": 1.0,
            }

        # Add detector metadata
        donut_table.meta["detector"] = detector_name
        donut_table.meta["camera"] = camera_name
        donut_table.meta["band"] = band_label

        self.log.debug("Created donut table with %d sources matching standard format", len(donut_table))
        return donut_table

    def calcZernikesFromExposure(self, exposure: afwImage.Exposure, defocalType: str) -> np.ndarray:
        """Calculate Zernike coefficients from a single focal position
        exposure.

        This method processes either an intra-focal OR extra-focal exposure
        through the TARTS neural network to estimate Zernike coefficients
        representing wavefront aberrations. The exposure contains donut
        stamps from one side of the focal plane (not both).

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            An exposure containing donut stamps from either the intra-focal
            OR extra-focal position. This is NOT a full frame exposure
            with both focal positions - it's specifically one side of
            the focal pair. Must have valid image data and WCS information.

        Returns
        -------
        np.ndarray
            Array of Zernike coefficients in microns, representing the
            estimated wavefront aberrations for this focal position.
            The array length matches the configured Noll indices
            (default: Z4-Z23).

        Notes
        -----
        This method processes one side of the focal pair (intra OR extra).
        In production, a single "exposure" typically contains both intra
        and extra-focal donuts in different corners, but this method
        expects an exposure with donut stamps from only one focal position.

        The exposure should contain donut stamps that the TARTS models
        are trained to process, not full frame images.
        """
        self.log.debug("Calculating Zernikes from exposure; defocalType='%s'", defocalType)
        with torch.no_grad():
            pred = self.tarts.deploy_run(exposure)
        # Convert PyTorch tensor to numpy array
        pred = self._to_numpy(pred)
        self.log.debug("TARTS prediction array shape: %s", np.shape(pred))

        # Ensure cropped_image is a numpy array before passing to function
        cropped_image_np = self.tarts.cropped_image
        cropped_image_np = self._to_numpy(cropped_image_np)

        # Determine defocal type based on exposure
        donutStamps = self.createDonutStampFromTarts(exposure, cropped_image_np, defocalType)
        self.log.debug(
            "Returning pred shape: %s and total_zernikes shape: %s",
            np.shape(pred),
            np.shape(self.tarts.total_zernikes),
        )
        return pred, donutStamps, deepcopy(self.tarts.total_zernikes)

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
            - donutStampsNeural : astropy.table.QTable
                Empty neural network-generated donut stamps collection
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
            "FX",
            "FY",
        ]
        self.log.info("Producing empty results; quality table provided: %s", qualityTable is not None)
        if qualityTable is None:
            donutQualityTable = QTable({name: [] for name in qualityTableCols})
        else:
            donutQualityTable = qualityTable

        # Set stamp attributes to None for empty case
        self.stampsIntra = None
        self.stampsExtra = None

        # Create empty Zernike table with metadata
        emptyZkTable = self.initZkTable()
        emptyZkTable.meta = self.createZkTableMetadata()

        # Create empty donut table matching standard format
        empty_donut_table = QTable({
            "coord_ra": [], "coord_dec": [], "centroid_x": [], "centroid_y": [],
            "detector": [],
            "thx_CCS": [], "thy_CCS": [], "thx_OCS": [], "thy_OCS": [], "th_N": [],
            "th_W": []
        })

        # Add default visit_info metadata for empty table
        # Use proper units for serialization compatibility
        empty_donut_table.meta["visit_info"] = {
            "visit_id": 0,
            "focus_z": 0.0 * u.mm,
            "boresight_ra": 0.0 * u.deg,
            "boresight_dec": 0.0 * u.deg,
            "boresight_rot_angle": 0.0 * u.deg,
            "boresight_par_angle": 0.0 * u.deg,
            "boresight_alt": 0.0 * u.deg,
            "boresight_az": 0.0 * u.deg,
            "mjd": 0.0,
            "donut_radius": 1.0,
        }

        # Create properly structured arrays for empty case
        # Ensure arrays have shape (1, n_zernikes) to match expected structure
        empty_zernikes = np.full((1, len(self.nollIndices)), np.nan)

        struct = pipeBase.Struct(
            outputZernikesRaw=empty_zernikes,
            outputZernikesAvg=empty_zernikes,
            donutStampsNeural=DonutStamps([]),
            zernikes=emptyZkTable,
            donutTable=empty_donut_table,
            donutQualityTable=donutQualityTable,
        )
        self.log.debug("Empty outputs have %d NaN Zernike terms", len(self.nollIndices))
        return struct

    @timeMethod
    def run(
        self,
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:
        """Run the neural network-based Zernike estimation task.

        This method processes a single LSST exposure to estimate Zernike
        coefficients using the TARTS neural network. The method determines
        whether the exposure contains intra-focal or extra-focal donuts
        and processes them accordingly.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            The LSST exposure data containing donut stamps. This should
            contain proper WCS information for the TARTS neural network.

        Returns
        -------
        lsst.pipe.base.Struct
            A struct containing:
            - outputZernikesAvg : np.ndarray
                Zernike coefficients from the exposure (in microns)
            - outputZernikesRaw : np.ndarray
                Raw Zernike coefficients from the exposure (in microns)
            - donutStampsNeural : astropy.table.QTable
                Neural network-generated donut stamps metadata from TARTS
                output
            - zernikes : astropy.table.QTable
                Zernike coefficients table with individual donut and
                average values
            - donutQualityTable : astropy.table.QTable
                Quality information for donuts

        Notes
        -----
        This implementation processes a single exposure containing donut
        stamps from either the intra-focal or extra-focal position.
        The TARTS neural network is used to estimate Zernike coefficients
        representing wavefront aberrations.

        The exposure should contain:
        - Valid image data (typically donut stamps)
        - Proper WCS (World Coordinate System) information
        - Appropriate metadata for the instrument and observation

        See Also
        --------
        calcZernikesFromExposure : Method that processes individual exposures
        """
        # Check if exposure is valid
        self.log.info("Starting Zernike estimation for a single exposure")
        if exposure is None:
            # No exposure available - return empty results
            self.log.warning("No exposure provided; returning empty results")
            return self.empty()

        # Store exposure for metadata creation
        self._current_exposure = exposure

        # Determine defocal type from exposure metadata
        # Try to get defocal type from exposure metadata first
        try:
            # Check if exposure has defocal type in metadata
            if exposure.getMetadata() is not None:
                defocalType = exposure.getMetadata().get("DFC_TYPE", "intra")
                self.log.info("Found DFC_TYPE in exposure metadata: '%s'", defocalType)
            else:
                # Fallback: try to determine from focus Z value if available
                try:
                    focusZ = exposure.visitInfo.focusZ
                    # For LSSTCam, positive focusZ is typically extra-focal
                    # This is a heuristic - may need adjustment based on actual
                    # data
                    if focusZ > 0:
                        defocalType = "extra"
                    else:
                        defocalType = "intra"
                    self.log.info("Determined defocal type from focusZ=%.3f: '%s'", focusZ, defocalType)
                except (AttributeError, NameError):
                    defocalType = "intra"
                    self.log.info(
                        "Could not determine defocal type from metadata or focusZ; defaulting to '%s'",
                        defocalType
                    )
        except Exception as e:
            defocalType = "intra"
            self.log.warning("Error determining defocal type: %s; defaulting to '%s'", e, defocalType)
        pred, donutStamps, zk = self.calcZernikesFromExposure(exposure, defocalType)
        self.log.debug(
            "Pred shape pre-squeeze: %s, donut stamps: %d, total zernikes shape: %s",
            np.shape(pred),
            len(donutStamps),
            np.shape(zk),
        )
        # Convert PyTorch tensors to NumPy arrays
        # pred and zk should already be numpy arrays
        pred_np = pred if isinstance(pred, np.ndarray) else np.array(pred)
        zk_np = zk if isinstance(zk, np.ndarray) else np.array(zk)

        # Ensure pred is 1D array of Zernike coefficients
        if pred_np.ndim > 1:
            pred_np = pred_np[0, :]  # Take first row if multi-dimensional

        # For outputZernikesRaw and outputZernikesAvg, use the same structure
        # as standard tasks
        # Both should be 2D arrays with shape (1, n_zernikes) for single
        # exposure
        zernikesRaw = np.atleast_2d(pred_np)  # Single prediction per exposure
        zernikesAvg = np.atleast_2d(pred_np)  # Single prediction per exposure

        # Store individual donut coefficients separately for the table
        individualZernikes = zk_np  # zk_np is converted numpy array

        # Create zernikes table
        # Since we only have one exposure, put stamps in the appropriate slot
        if defocalType == "extra":
            extraStamps = donutStamps
            intraStamps = DonutStamps([])
        else:  # intra or unknown
            extraStamps = DonutStamps([])
            intraStamps = donutStamps

        # Set instance attributes for metadata creation
        self.stampsIntra = intraStamps
        self.stampsExtra = extraStamps

        # Store detector name for metadata creation
        detector = exposure.getDetector()
        self._detector_name = detector.getName()

        self.log.debug(
            "Using defocalType='%s' -> intra stamps: %d, extra stamps: %d",
            defocalType,
            len(intraStamps),
            len(extraStamps),
        )
        zernikesTable = self.createZkTable(
            extraStamps=extraStamps,
            intraStamps=intraStamps,
            zkCoeffRaw=individualZernikes,
            zkCoeffAvg=pred_np,
        )

        # Create quality table from donut stamps
        donutQualityTable = self.createDonutQualityTable(donutStamps)

        # Create donut table from donut stamps
        donutTable = self.createDonutTable(donutStamps, exposure, defocalType)

        self.log.info("Estimated %d Zernike terms for exposure", zernikesAvg.shape[1])
        self.log.debug("ZernikesAvg (first 5): %s", np.array2string(zernikesAvg[0, :5], precision=3))
        self.log.info("Finished Zernike estimation; table rows: %d", len(zernikesTable))
        return pipeBase.Struct(
            outputZernikesAvg=zernikesAvg,  # Already 2D
            outputZernikesRaw=zernikesRaw,  # Already 2D
            donutStampsNeural=donutStamps,
            zernikes=zernikesTable,
            donutTable=donutTable,
            donutQualityTable=donutQualityTable,
        )
