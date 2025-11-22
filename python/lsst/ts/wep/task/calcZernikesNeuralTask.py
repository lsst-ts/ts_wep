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

import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import astropy.units as u
from astropy.table import QTable

import lsst.afw.image as afwImage
import lsst.geom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.daf.base import PropertyList
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod

from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable

# Define the position 2D float dtype for the zernikes table
POS2F_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4")])

# Module-level sentinel values for missing data
UNKNOWN_STRING = "Unknown"
UNKNOWN_INT = -1
UNKNOWN_FLOAT = float("nan")
UNKNOWN_ANGLE = float("nan")

# Module-level constant for empty donut table columns
EMPTY_DONUT_TABLE_COLUMNS = [
    "coord_ra",
    "coord_dec",
    "centroid_x",
    "centroid_y",
    "detector",
    "thx_ccs",
    "thy_ccs",
    "thx_ocs",
    "thy_ocs",
    "th_n",
    "th_w",
    "fx",
    "fy",
    "snr",
    "ood_score",
]

# Module-level constant for donut quality table columns
DONUT_QUALITY_TABLE_COLUMNS = [
    "SN",
    "ENTROPY",
    "ENTROPY_SELECT",
    "SN_SELECT",
    "FINAL_SELECT",
    "DEFOCAL_TYPE",
    "FX",
    "FY",
]


class CalcZernikesNeuralTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("exposure", "detector", "instrument"),  # type: ignore
):
    exposure = connectionTypes.Input(
        doc="Input post_isr exposure to run wavefront estimation on",
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
    CalcZernikesTaskConfig,
    pipelineConnections=CalcZernikesNeuralTaskConnections,  # type: ignore
):
    """Configuration for CalcZernikesNeuralTask.

    Attributes
    ----------
    wavenetPath : str or None
        Model weights path for wavenet. If None, TARTS will create a new
        model with random weights (useful for testing).
    alignetPath : str or None
        Model weights path for alignet. If None, TARTS will create a new
        model with random weights (useful for testing).
    aggregatornetPath : str or None
        Model weights path for aggregatornet. If None, TARTS will create a new
        model with random weights (useful for testing).
    oodModelPath : str or None
        Directory path for the OOD model to be used by TARTS for out-of-
        distribution detection. If None, OOD checks are disabled.
    datasetParamPath : str
        Path to TARTS dataset parameters YAML file containing normalization
        scaling factors, image processing parameters (CROP_SIZE, deg_per_pix,
        mm_pix), model hyperparameters, and training data file paths.
    device : str
        Device to use for calculations. Options: 'cpu' for CPU computation,
        'cuda' for GPU computation (default). Any value other than 'cpu'
        will use CUDA.
    nollIndices : list[int]
        List of Noll indices to calculate. Default is Z4-Z22 (4-22),
        excluding piston (Z1), tip (Z2), and tilt (Z3) which are
        typically not measured in wavefront sensing.
    cropSize : int
        Size of donut crop in pixels (width and height). Default is 160 pixels,
        which matches the TARTS neural network training data format.
    intraDfcDist : float
        Defocal distance for intra-focal images in mm. Negative value indicates
        inward defocus. Default is -1.5 mm.
    extraDfcDist : float
        Defocal distance for extra-focal images in mm. Positive value indicates
        outward defocus. Default is 1.5 mm.
    """

    wavenetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for wavenet", dtype=str, default=None, optional=True
    )
    alignetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for alignet", dtype=str, default=None, optional=True
    )
    aggregatornetPath: pexConfig.Field = pexConfig.Field(
        doc="Model Weights Path for aggregatornet", dtype=str, default=None, optional=True
    )
    oodModelPath: pexConfig.Field = pexConfig.Field(
        doc="Directory path for OOD model used by TARTS (optional)",
        dtype=str,
        default=None,
        optional=True,
    )
    datasetParamPath: pexConfig.Field = pexConfig.Field(
        doc="Path to TARTS dataset parameters YAML file containing normalization "
        "scaling factors, image processing parameters (CROP_SIZE, deg_per_pix, "
        "mm_pix), model hyperparameters, and training data file paths",
        dtype=str,
    )
    device: pexConfig.Field = pexConfig.Field(
        doc="Device to use for calculations. Options: 'cpu' for CPU computation, "
        "'cuda' for GPU computation (default). Any value other than 'cpu' will use CUDA.",
        dtype=str,
        default="cuda",
        optional=True,
    )
    nollIndices: pexConfig.Field = pexConfig.ListField(
        doc="List of Noll indices to calculate. Default is Z4-Z22 (4-22), "
        "excluding piston (Z1), tip (Z2), and tilt (Z3) which are "
        "typically not measured in wavefront sensing.",
        dtype=int,
        default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    )
    cropSize: pexConfig.Field = pexConfig.Field(
        doc="Size of donut crop in pixels (width and height). Default is 160 pixels, "
        "which matches the TARTS neural network training data format.",
        dtype=int,
        default=160,
        optional=True,
    )
    intraDfcDist: pexConfig.Field = pexConfig.Field(
        doc="Defocal distance for intra-focal images in mm. Negative value indicates inward defocus.",
        dtype=float,
        default=-1.5,
        optional=True,
    )
    extraDfcDist: pexConfig.Field = pexConfig.Field(
        doc="Defocal distance for extra-focal images in mm. Positive value indicates outward defocus.",
        dtype=float,
        default=1.5,
        optional=True,
    )


class CalcZernikesNeuralTask(CalcZernikesTask):
    """Neural network-based Zernike estimation task using TARTS.

    This class uses the TARTS (Triple-stage Alignment and Reconstruction using
    Transformer Systems for Active Optics) neural network models to estimate
    Zernike coefficients from pairs of intra and extra-focal exposures. Each
    exposure contains donut stamps from one focal position, and the task
    processes them separately to estimate Zernike coefficients for each side
    of the focal plane.

    TARTS is a PyTorch package with a triple-stage design: (1) AlignNet for
    donut alignment and normalization, (2) WaveNet for per-donut Zernike
    regression, and (3) AggregatorNet (transformer-based) for fusing multiple
    donut predictions.

    The TARTS system includes three main components:
    - Wavenet: Estimates Zernike coefficients from donut images
    - Alignnet: Handles image alignment and preprocessing
    - Aggregatornet: Combines results from multiple donuts

    Detector Support
    ---------------
    This task only supports corner wavefront sensors with detector names ending
    with '_SW0' (extra-focal) or '_SW1' (intra-focal). Full array mode is not
    currently supported.
    """

    # Class constants for processing
    EXPECTED_IMAGE_DIMENSIONS = 3  # Expected dimensions for TARTS output
    LOG_PRECISION = 3  # Decimal precision for logging Zernike values

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
        2. Sets up Noll indices from configuration
        3. Validates configuration parameters
        4. Loads TARTS neural network models with specified weights
        5. Sets models to evaluation mode with .eval()
        6. Extracts crop size from TARTS system
        7. Configures device (CPU/CUDA) and moves models accordingly

        Noll indices are configurable via the config.nollIndices parameter.
        The default excludes Z1-Z3 (piston, tip, tilt) as these are typically
        not measured in wavefront sensing.
        """
        super().__init__(**kwargs)

        # Type annotation for mypy to understand the config structure
        self.config: CalcZernikesNeuralTaskConfig

        # Define default Noll indices to be used for wavefront estimation
        self.nollIndices = self.config.nollIndices
        self.log.debug("Configured Noll indices: %s", self.nollIndices)

        # Deferred import of TARTS to handle cases where it's not in the build
        from tarts import NeuralActiveOpticsSys

        # TARTS system handles None paths by creating new models with random
        # weights
        self.tarts = NeuralActiveOpticsSys(
            os.path.expandvars(self.config.datasetParamPath),
            os.path.expandvars(self.config.wavenetPath),
            os.path.expandvars(self.config.alignetPath),
            os.path.expandvars(self.config.aggregatornetPath),
            ood_model_path=(
                os.path.expandvars(self.config.oodModelPath) if self.config.oodModelPath is not None else None
            ),
        )
        if self.config.oodModelPath is not None:
            self.log.info(
                "OOD model enabled; path: %s",
                os.path.expandvars(self.config.oodModelPath),
            )
        else:
            self.log.warning("OOD scoring disabled (no oodModelPath provided)")
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
        self.log.info(
            "Defocal distances - intra: %.1f mm, extra: %.1f mm",
            self.config.intraDfcDist,
            self.config.extraDfcDist,
        )

        if self.device == "cpu":
            self.tarts = self.tarts.cpu()

            # Update device attributes for all sub-models to CPU
            self.tarts.device_val = torch.device("cpu")
            self.tarts.alignnet_model.device_val = torch.device("cpu")
            self.tarts.wavenet_model.device_val = torch.device("cpu")
            self.tarts.aggregatornet_model.device_val = torch.device("cpu")

            self.tarts.alignnet_model.alignnet.cnn = self.tarts.alignnet_model.alignnet.cnn.cpu()
            self.tarts.wavenet_model.wavenet.cnn = self.tarts.wavenet_model.wavenet.cnn.cpu()
            self.log.debug("Moved all sub-models and CNNs to CPU")
        else:
            self.tarts.to("cuda")
            self.log.debug("Moved models to CUDA")

        self.log.debug("TARTS crop size: %s", self.cropSize)

        # Validate configuration parameters
        self.validate()

        # Initialize cache for per-donut OOD scores
        self._lastOodScores: list[float] = []

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate Noll indices
        if len(self.nollIndices) == 0:
            raise ValueError("nollIndices cannot be empty")
        if any(idx < 1 for idx in self.nollIndices):
            raise ValueError("Noll indices must be >= 1")

    def _isPytorchTensor(self, obj: Any) -> bool:
        """Check if object is a PyTorch tensor."""
        return isinstance(obj, torch.Tensor)

    def _toNumpy(self, obj: Any) -> np.ndarray:
        """Convert PyTorch tensor or other object to numpy array."""
        if self._isPytorchTensor(obj):
            return obj.cpu().numpy()
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(obj)

    def _getMetadataArray(self, metadata: PropertyList, key: str, numStamps: int, default: Any = 0.0) -> list:
        """Extract donut stamps metadata array safely with default fallback.

        Parameters
        ----------
        metadata : PropertyList
            The donut stamps metadata object to extract from.
        key : str
            The metadata key to retrieve.
        numStamps : int
            Number of donut stamps (for creating default list if key missing).
        default : Any
            Default value to use if key doesn't exist. Default is 0.0.

        Returns
        -------
        list
            List of donut stamps metadata values or defaults.
        """
        if metadata.exists(key):
            return list(metadata.getArray(key))
        else:
            return [default] * numStamps

    def _getDefaultVisitInfoDict(self, visitId: int | None = None) -> dict:
        """Get a dictionary with default visit info values.

        Parameters
        ----------
        visitId : int, optional
            Visit ID to use. If None, uses UNKNOWN_INT.

        Returns
        -------
        dict
            Dictionary containing default visit info values.
        """
        return {
            "visit_id": visitId if visitId is not None else UNKNOWN_INT,
            "focus_z": UNKNOWN_FLOAT * u.mm,
            "boresight_ra": UNKNOWN_FLOAT * u.deg,
            "boresight_dec": UNKNOWN_FLOAT * u.deg,
            "boresight_rot_angle": UNKNOWN_FLOAT * u.deg,
            "boresight_par_angle": UNKNOWN_FLOAT * u.deg,
            "boresight_alt": UNKNOWN_FLOAT * u.deg,
            "boresight_az": UNKNOWN_FLOAT * u.deg,
            "mjd": UNKNOWN_FLOAT,
            "donut_radius": UNKNOWN_FLOAT,
        }

    def _determineDefocalType(self, exposure: afwImage.Exposure) -> str:
        """Determine defocal type (intra/extra) from detector name.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            The exposure to analyze.

        Returns
        -------
        str
            String indicating defocal type: "intra" or "extra".

        Raises:
            ValueError: If defocal type cannot be determined from detector
                name.
        """
        detectorName = exposure.getDetector().getName()
        self.log.debug("Detector name: '%s'", detectorName)

        # Determine defocal type based on detector name pattern
        # SW0 detectors are extra-focal, SW1 detectors are intra-focal
        # This matches the pattern used in CutOutDonutsCwfsTask
        if detectorName.endswith("_SW0"):
            defocalType = "extra"
            self.log.debug("Detector '%s' is extra-focal (SW0)", detectorName)
        elif detectorName.endswith("_SW1"):
            defocalType = "intra"
            self.log.info("Detector '%s' is intra-focal (SW1)", detectorName)
        else:
            # Only corner wavefront sensors (SW0/SW1) are supported
            raise ValueError(
                f"Cannot determine defocal type for detector '{detectorName}'. "
                "This task only supports corner wavefront sensors with detector names "
                "ending with '_SW0' (extra-focal) or '_SW1' (intra-focal). "
                "Full array mode is not currently supported."
            )

        return defocalType

    def createDonutStampFromTarts(
        self,
        exposure: afwImage.Exposure,
        croppedImage: np.ndarray,
        defocalType: str,
        tartsInternalData: list,
    ) -> DonutStamps:
        """Create DonutStamps from TARTS neural network output.

        This method processes the cropped image output from TARTS and converts
        it into DonutStamps objects that are compatible with the LSST WEP
        framework. It handles multiple donuts detected in the image and creates
        the appropriate data structures for downstream processing.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            The original exposure containing metadata, WCS, and other
            information needed for creating proper DonutStamp objects.
        croppedImage : np.ndarray
            The cropped image array output from TARTS neural network
            containing the processed donut regions.
        defocalType : str
            The defocal type indicating which side of focus the donuts
            represent. Should be either "intra" or "extra".
        tartsInternalData : list
            Internal data from TARTS get_internal_data() method containing
            per-donut information like centers, field positions, SNR, etc.
            Required for proper donut position and metadata extraction.

        Returns
        -------
        DonutStamps
            A collection of DonutStamp objects representing the detected
            and processed donuts from the TARTS output. Each DonutStamp
            contains the necessary metadata and image data for wavefront
            estimation.

        Notes
        -----
        This method bridges the gap between TARTS neural network output
        and the LSST WEP framework by converting the processed image data
        into the expected DonutStamp format. The method handles the
        coordinate transformations and metadata extraction needed to
        create valid DonutStamp objects.
        """
        # Log image type and shape for debugging
        if not isinstance(croppedImage, np.ndarray):
            raise ValueError(f"croppedImage must be a numpy array, got {type(croppedImage)}")

        self.log.debug(
            "Processing TARTS output; image shape: %s, defocalType='%s'", croppedImage.shape, defocalType
        )
        # Extract information from exposure
        detector = exposure.getDetector()
        detectorName = detector.getName()
        cameraName = exposure.metadata["LSST BUTLER DATAID INSTRUMENT"]
        bandLabel = exposure.filter.bandLabel

        # Validate that TARTS returned the expected 3D format
        if len(croppedImage.shape) != self.EXPECTED_IMAGE_DIMENSIONS:
            raise ValueError(
                f"TARTS should return 3D array [numStamps, cropSize, cropSize], "
                f"got shape: {croppedImage.shape}"
            )

        numStamps = croppedImage.shape[0]
        self.log.info(
            "Processing %d donut stamp(s) of size %dx%d pixels", numStamps, self.cropSize, self.cropSize
        )

        # Get centroid positions and SNR from TARTS internal data
        centXList = []
        centYList = []
        fxList = []
        fyList = []
        snrList = []

        for i in range(numStamps):
            centers = tartsInternalData[i]["centers"]
            # Convert tensor to numpy and extract coordinates
            centersNp = self._toNumpy(centers)
            centXList.append(float(centersNp[0]))
            centYList.append(float(centersNp[1]))
            fxList.append(float(tartsInternalData[i].get("fx", np.nan)))
            fyList.append(float(tartsInternalData[i].get("fy", np.nan)))
            # Prefer key 'SNR' provided by TARTS; fallback to NaN if missing
            snrVal = float(tartsInternalData[i].get("SNR", np.nan))
            snrList.append(snrVal)

        self.log.debug("Using TARTS internal data for %d centroid positions", numStamps)

        # Create metadata for all donuts
        metadata = PropertyList()
        metadata["RA_DEG"] = [0.0] * numStamps  # Will be calculated from WCS
        metadata["DEC_DEG"] = [0.0] * numStamps  # Will be calculated from WCS
        metadata["CENT_X"] = centXList
        metadata["CENT_Y"] = centYList
        metadata["DET_NAME"] = [detectorName] * numStamps
        metadata["CAM_NAME"] = [cameraName] * numStamps
        metadata["DFC_TYPE"] = [defocalType] * numStamps
        # Set defocal distance using configurable values
        if defocalType == "intra":
            defocalDistance = self.config.intraDfcDist
        elif defocalType == "extra":
            defocalDistance = self.config.extraDfcDist
        else:
            raise ValueError(f"Invalid defocalType: {defocalType}. Must be 'intra' or 'extra'")
        metadata["DFC_DIST"] = [defocalDistance] * numStamps
        metadata["BANDPASS"] = [bandLabel] * numStamps
        metadata["BLEND_CX"] = ["nan"] * numStamps
        metadata["BLEND_CY"] = ["nan"] * numStamps
        metadata["FX"] = fxList
        metadata["FY"] = fyList

        # Add quality metric placeholders (not calculated by neural network)
        # These fields are expected by calcZernikesTask.createZkTable() and
        # DonutStampSelectorTask. Set to NaN since TARTS doesn't calculate
        # per-stamp quality metrics in the same way as cutOutDonutsBase.
        metadata["MAG"] = [np.nan] * numStamps
        metadata["SN"] = snrList
        metadata["ENTROPY"] = [np.nan] * numStamps
        metadata["FRAC_BAD_PIX"] = [np.nan] * numStamps
        metadata["MAX_POWER_GRAD"] = [np.nan] * numStamps

        # Create DonutStamp objects
        donutStamps = []
        wcs = exposure.wcs
        for i in range(numStamps):
            # Create MaskedImageF for this donut
            stampIm = afwImage.MaskedImageF(self.cropSize, self.cropSize)
            stampIm.image.array = croppedImage[i]  # Use the i-th donut image
            stampIm.setXY0(0, 0)  # Set origin - this is a cropped image

            # Create linear WCS using centroid from metadata
            centroidPosition = lsst.geom.Point2D(centXList[i], centYList[i])
            linearTransform = wcs.linearizePixelToSky(centroidPosition, lsst.geom.degrees)
            cdMatrix = linearTransform.getLinear().getMatrix()
            linearWcs = lsst.afw.geom.makeSkyWcs(centroidPosition, wcs.pixelToSky(centroidPosition), cdMatrix)

            # Create DonutStamp using factory
            donutStamp = DonutStamp.factory(
                stamp_im=stampIm,
                metadata=metadata,
                index=i,  # Index into the metadata lists
                archive_element=linearWcs,
            )
            donutStamps.append(donutStamp)

        # Add exposure metadata using addVisitInfoToCatTable directly
        emptyTable = addVisitInfoToCatTable(exposure, QTable())
        comprehensiveMetadata = emptyTable.meta.get("visit_info", {})

        # Extract visit_id as plain int
        metadata["VISIT"] = comprehensiveMetadata.get("visit_id", UNKNOWN_INT)

        # Convert angles from degrees to radians for PropertyList compatibility
        metadata["BORESIGHT_ROT_ANGLE_RAD"] = comprehensiveMetadata.get(
            "boresight_rot_angle", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)
        metadata["BORESIGHT_PAR_ANGLE_RAD"] = comprehensiveMetadata.get(
            "boresight_par_angle", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)
        metadata["BORESIGHT_ALT_RAD"] = comprehensiveMetadata.get(
            "boresight_alt", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)
        metadata["BORESIGHT_AZ_RAD"] = comprehensiveMetadata.get(
            "boresight_az", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)
        metadata["BORESIGHT_RA_RAD"] = comprehensiveMetadata.get(
            "boresight_ra", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)
        metadata["BORESIGHT_DEC_RAD"] = comprehensiveMetadata.get(
            "boresight_dec", UNKNOWN_FLOAT * u.deg
        ).to_value(u.rad)

        # Extract MJD as plain float
        metadata["MJD"] = comprehensiveMetadata.get("mjd", UNKNOWN_FLOAT)

        # Create DonutStamps collection with metadata
        donutStampsObj = DonutStamps(donutStamps, metadata=metadata, use_archive=True)

        self.log.info(
            "Created DonutStamps with metadata: DET_NAME='%s', CAM_NAME='%s', DFC_TYPE='%s'",
            detectorName,
            cameraName,
            defocalType,
        )
        self.log.info("Constructed %d DonutStamps with WCS and metadata", len(donutStamps))
        return donutStampsObj

    def createDonutQualityTable(self, donutStamps: DonutStamps) -> QTable:
        """Create a quality table from donut stamps metadata.

        Parameters
        ----------
        donutStamps : DonutStamps
            The donut stamps with metadata already populated by
            createDonutStampsFromTarts().

        Returns
        -------
        QTable
            A table containing quality information for each donut.
        """
        if len(donutStamps) == 0:
            # Return empty table with expected columns
            self.log.warning(
                "No donut stamps available for quality table creation. "
                "This may indicate a problem with donut detection or processing."
            )
            return QTable({name: [] for name in DONUT_QUALITY_TABLE_COLUMNS})

        # Read quality metrics from stamps metadata (already populated by
        # createDonutStampsFromTarts)
        numDonuts = len(donutStamps)
        qualityData = {}
        qualityData["FX"] = self._getMetadataArray(donutStamps.metadata, "FX", numDonuts, 0.0)
        qualityData["FY"] = self._getMetadataArray(donutStamps.metadata, "FY", numDonuts, 0.0)
        qualityData["SN"] = self._getMetadataArray(donutStamps.metadata, "SN", numDonuts, np.nan)
        qualityData["ENTROPY"] = self._getMetadataArray(donutStamps.metadata, "ENTROPY", numDonuts, np.nan)

        # Add selection flags
        qualityData["ENTROPY_SELECT"] = [False] * len(donutStamps)
        qualityData["SN_SELECT"] = [True] * len(donutStamps)
        qualityData["FINAL_SELECT"] = [True] * len(donutStamps)  # All predictions are selected
        qualityData["DEFOCAL_TYPE"] = self._getMetadataArray(
            donutStamps.metadata, "DFC_TYPE", numDonuts, "unknown"
        )

        # Create the quality table
        qualityTable = QTable(qualityData)

        self.log.debug("Created donut quality table with %d rows from stamps metadata", len(qualityTable))
        return qualityTable

    def createDonutTable(
        self,
        donutStamps: DonutStamps,
        exposure: afwImage.Exposure,
        defocalType: str = "intra",
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
            The donut stamps collection.
        exposure : afwImage.Exposure
            The exposure containing the donut data.

        Returns
        -------
        QTable
            A table containing donut source positions and properties matching
            the standard format.
        """
        if len(donutStamps) == 0:
            # Return empty table with expected columns matching standard format
            emptyTable = QTable({name: [] for name in EMPTY_DONUT_TABLE_COLUMNS})
            # Add required metadata for empty table
            emptyTable.meta["detector"] = UNKNOWN_STRING
            emptyTable.meta["camera"] = exposure.metadata["LSST BUTLER DATAID INSTRUMENT"]
            emptyTable.meta["band"] = UNKNOWN_STRING
            emptyTable.meta["visit_info"] = self._getDefaultVisitInfoDict()
            return emptyTable

        numDonuts = len(donutStamps)

        # Extract detector and camera info
        detector = exposure.getDetector()
        detectorName = detector.getName()
        cameraName = exposure.metadata["LSST BUTLER DATAID INSTRUMENT"]
        bandLabel = exposure.filter.bandLabel

        self.log.info(
            "Extracted detector info: name='%s', camera='%s', band='%s'", detectorName, cameraName, bandLabel
        )

        # Initialize data arrays
        donutData = {}

        # Get positions from donut stamps - use standard column names and units
        try:
            skyPositions = donutStamps.getSkyPositions()
            donutData["coord_ra"] = [pos.getRa().asRadians() for pos in skyPositions]
            donutData["coord_dec"] = [pos.getDec().asRadians() for pos in skyPositions]
        except AttributeError as e:
            # Without valid per-stamp sky positions, downstream consumers
            # that rely on RA/Dec are likely to misbehave. Fail fast so
            # callers can correct missing WCS or stamp metadata.
            raise ValueError(
                "Donut stamps are missing sky positions; ensure WCS and stamp metadata are present"
            ) from e

        # Get centroid positions from stamps metadata (already populated by
        # createDonutStampsFromTarts)
        centroidPositions = donutStamps.getCentroidPositions()
        donutData["centroid_x"] = [pos.getX() for pos in centroidPositions]
        donutData["centroid_y"] = [pos.getY() for pos in centroidPositions]

        # Get fx, fy, snr from stamps metadata (already populated by
        # createDonutStampsFromTarts)
        donutData["fx"] = self._getMetadataArray(donutStamps.metadata, "FX", numDonuts, 0.0)
        donutData["fy"] = self._getMetadataArray(donutStamps.metadata, "FY", numDonuts, 0.0)
        donutData["snr"] = self._getMetadataArray(donutStamps.metadata, "SN", numDonuts, np.nan)
        # Add OOD scores if available from prior neural inference
        if hasattr(self, "_lastOodScores"):
            oodScores = list(self._lastOodScores)
            if len(oodScores) < numDonuts:
                oodScores = oodScores + [np.nan] * (numDonuts - len(oodScores))
            elif len(oodScores) > numDonuts:
                oodScores = oodScores[:numDonuts]
            donutData["ood_score"] = oodScores
        else:
            # OOD scores are expected when the OOD model is active; if missing,
            # we fall back to NaNs and log a warning to aid diagnostics.
            self.log.warning(
                "No OOD scores available for %d donuts; populating 'ood_score' with NaN. "
                "If the OOD model is intended to be active, this indicates missing OOD inference.",
                numDonuts,
            )
            donutData["ood_score"] = [np.nan] * numDonuts

        # Add detector information as a column (required by downstream
        # aggregation)
        # Ensure detector name is consistent with expected format
        donutData["detector"] = [detectorName] * numDonuts
        self.log.info(
            "Using detector name: '%s' for %d donuts (defocal type: '%s')",
            detectorName,
            numDonuts,
            defocalType,
        )

        # Calculate field angles from pixel centroids using calcFieldXY
        # from donut stamps. This ensures we have accurate field_x and
        # field_y values
        field_x_list = []
        field_y_list = []
        for idx, stamp in enumerate(donutStamps):
            field_x, field_y = stamp.calcFieldXY()
            field_x_list.append(field_x)
            field_y_list.append(field_y)

        # Calculate telescope coordinate columns from field angles
        # CCS: Camera Coordinate System - field_x is thx_ccs,
        # field_y is thy_ccs
        donutData["thx_ccs"] = field_x_list
        donutData["thy_ccs"] = field_y_list

        # OCS: rotate CCS by rtp = q - rot - pi/2
        # q: boresight parallactic angle; rot: boresight rotation
        try:
            q_parAngle_rad = exposure.visitInfo.boresightParAngle.asRadians()
            rot_angle_rad = exposure.visitInfo.boresightRotAngle.asRadians()
        except (AttributeError, RuntimeError) as e:
            raise ValueError(
                "Missing required boresight angles in visitInfo. "
                "Both boresightParAngle and boresightRotAngle must be present."
            ) from e

        # Convert field angles (deg) to radians for rotation
        thx_ccs_rad = np.deg2rad(field_x_list)
        thy_ccs_rad = np.deg2rad(field_y_list)

        # Rotate CCS -> OCS using rtp
        rtp = q_parAngle_rad - rot_angle_rad - np.pi / 2.0
        cos_rtp = np.cos(rtp)
        sin_rtp = np.sin(rtp)
        donutData["thx_ocs"] = np.rad2deg(thx_ccs_rad * cos_rtp + thy_ccs_rad * sin_rtp)
        donutData["thy_ocs"] = np.rad2deg(-thx_ccs_rad * sin_rtp + thy_ccs_rad * cos_rtp)

        # Telescope N/E from CCS using q; store N and W (W = -E)
        cos_q = np.cos(q_parAngle_rad)
        sin_q = np.sin(q_parAngle_rad)
        th_n_rad = thx_ccs_rad * cos_q + thy_ccs_rad * sin_q
        th_e_rad = -thx_ccs_rad * sin_q + thy_ccs_rad * cos_q
        donutData["th_n"] = np.rad2deg(th_n_rad)
        donutData["th_w"] = np.rad2deg(-th_e_rad)

        # Create the table
        donutTable = QTable(donutData)

        # Add units matching standard format
        donutTable["coord_ra"].unit = u.rad  # Use radians like standard
        donutTable["coord_dec"].unit = u.rad  # Use radians like standard
        # Note: centroidX and centroidY are kept as plain floats to avoid
        # Point2D constructor issues
        donutTable["fx"].unit = u.deg
        donutTable["fy"].unit = u.deg

        # Add comprehensive visitInfo metadata using utility function
        try:
            donutTable = addVisitInfoToCatTable(exposure, donutTable)
        except (AttributeError, RuntimeError) as e:
            # Missing visitInfo/attributes or runtime errors
            # (e.g., from DonutSizeCorrelator)
            self.log.warning(
                "Could not extract comprehensive visitInfo metadata: %s. Using minimal visitInfo structure.",
                e,
            )
            # Provide minimal visitInfo structure
            visitId = int(exposure.visitInfo.id) if exposure.visitInfo else UNKNOWN_INT

            donutTable.meta["visit_info"] = self._getDefaultVisitInfoDict(visitId)

        # Add detector metadata
        donutTable.meta["detector"] = detectorName
        donutTable.meta["camera"] = cameraName
        donutTable.meta["band"] = bandLabel

        self.log.debug("Created donut table with %d sources matching standard format", len(donutTable))
        return donutTable

    def calcZernikesFromExposure(
        self, exposure: afwImage.Exposure, defocalType: str
    ) -> tuple[np.ndarray, DonutStamps, np.ndarray]:
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
        defocalType : str
            The defocal type indicating which side of focus the exposure
            represents. Should be either "intra" or "extra".

        Returns
        -------
        tuple[np.ndarray, DonutStamps, np.ndarray]
            A tuple containing three elements:
            - aggregatedZernikes : np.ndarray
                Aggregated Zernike coefficients from TARTS deploy_run() in
                microns, representing the final estimated wavefront aberrations
                for this focal position after neural network processing. This
                is the primary output from the TARTS system's AggregatorNet
                component.
            - donutStamps : DonutStamps
                DonutStamps object containing the processed donut stamp data
                created from TARTS output, including positions, images, and
                metadata for downstream analysis.
            - rawZernikes : np.ndarray
                Individual Zernike coefficients per donut from TARTS (a deep
                copy of self.tarts.total_zernikes). This contains the raw
                per-donut predictions before aggregation, useful for detailed
                analysis and debugging of individual donut contributions.

        Notes
        -----
        This method processes one side of the focal pair (intra OR extra).
        In production, a single "exposure" typically contains both intra
        and extra-focal donuts in different corners, but this method
        expects an exposure with donut stamps from only one focal position.

        The complete wavefront estimation workflow involves:
        1. Processing intra-focal donuts separately (this method)
        2. Processing extra-focal donuts separately (this method)
        3. Combining results from both sides using AggregatorNet

        The exposure should a single detector image that the TARTS models
        are trained to process, not full frame images.

        The difference between 'aggregatedZernikes' and 'rawZernikes':
        - 'aggregatedZernikes' is the final aggregated result from TARTS's
          AggregatorNet
        - 'rawZernikes' contains individual per-donut predictions before
          aggregation, useful for analyzing individual donut contributions
        """
        self.log.debug("Calculating Zernikes from exposure; defocalType='%s'", defocalType)
        with torch.no_grad():
            aggregatedZernikes = self.tarts.deploy_run(exposure)
        # Convert PyTorch tensor to numpy array
        aggregatedZernikes = self._toNumpy(aggregatedZernikes)
        self.log.debug("TARTS Zernike coefficients prediction array shape: %s", np.shape(aggregatedZernikes))

        # Ensure croppedImage is a numpy array before passing to function
        croppedImageNp = self._toNumpy(self.tarts.cropped_image)

        # Get TARTS internal data for more reliable field positions and SNR
        tartsInternalData = self.tarts.get_internal_data()
        self.log.debug("Retrieved %d donuts from TARTS internal data", len(tartsInternalData))

        # Extract OOD scores if available; fallback to NaN when missing
        # or when the OOD model is not loaded
        oodScores = [float(d.get("ood_score", np.nan)) for d in tartsInternalData]
        # Cache for later use when creating the zernikes table
        self._lastOodScores = oodScores

        # Count valid OOD scores for logging
        validCount = int(np.sum(np.isfinite(oodScores)))
        if validCount > 0:
            self.log.info(
                "Obtained %d OOD score(s) out of %d donuts",
                validCount,
                len(oodScores),
            )
        else:
            self.log.warning("No valid OOD scores available; using NaN placeholders")

        # Determine defocal type based on exposure
        donutStamps = self.createDonutStampFromTarts(exposure, croppedImageNp, defocalType, tartsInternalData)

        # Log number of donuts detected and processed
        self.log.info(
            "Detected and processed %d donut(s) for defocalType='%s'", len(donutStamps), defocalType
        )
        return aggregatedZernikes, donutStamps, deepcopy(self.tarts.total_zernikes)

    def empty(self, qualityTable: QTable | None = None) -> pipeBase.Struct:
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
                Array filled with NaN values for all Noll indices.
            - outputZernikesAvg : np.ndarray
                Array filled with NaN values for all Noll indices.
            - donutStampsNeural : astropy.table.QTable
                Empty neural network-generated donut stamps collection.
            - zernikes : astropy.table.QTable
                Empty Zernike coefficient table.
            - donutQualityTable : astropy.table.QTable
                Either the provided quality table or an empty one.

        Notes
        -----
        The NaN values in the output arrays indicate that no valid Zernike
        coefficients could be calculated. The quality table preserves
        information about why donuts failed, which can be useful for
        debugging.
        """
        self.log.info("Producing empty results; quality table provided: %s", qualityTable is not None)
        if qualityTable is None:
            self.log.warning(
                "No donut quality table supplied to empty(); constructing an empty table. "
                "If a previous step computed quality metrics (e.g., donuts existed but all failed), "
                "pass that table here to preserve failure diagnostics."
            )
            donutQualityTable = QTable({name: [] for name in DONUT_QUALITY_TABLE_COLUMNS})
        else:
            donutQualityTable = qualityTable

        # Set stamp attributes to None for empty case
        self.stampsIntra = None
        self.stampsExtra = None

        # Create empty Zernike table with metadata
        emptyZkTable = self.initZkTable()
        emptyZkTable.meta = self.createZkTableMetadata()
        # Add empty OOD score column
        emptyZkTable["ood_score"] = []

        # Create empty donut table matching standard format
        emptyDonutTable = QTable({name: [] for name in EMPTY_DONUT_TABLE_COLUMNS})

        # Add default visitInfo metadata for empty table
        emptyDonutTable.meta["visit_info"] = self._getDefaultVisitInfoDict()

        # Create properly structured arrays for empty case
        # Ensure arrays have shape (1, nZernikes) to match expected structure
        emptyZernikes = np.full((1, len(self.nollIndices)), np.nan)

        struct = pipeBase.Struct(
            outputZernikesRaw=emptyZernikes,
            outputZernikesAvg=emptyZernikes,
            donutStampsNeural=DonutStamps([]),
            zernikes=emptyZkTable,
            donutTable=emptyDonutTable,
            donutQualityTable=donutQualityTable,
        )
        self.log.debug(
            "Empty output zernike arrays created with shape (1, %d), all values NaN",
            len(self.nollIndices),
        )
        return struct

    @timeMethod
    def run(
        self,
        exposure: afwImage.Exposure,
        numCores: int = 1,
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
                Zernike coefficients from the exposure (in microns).
            - outputZernikesRaw : np.ndarray
                Raw Zernike coefficients from the exposure (in microns).
            - donutStampsNeural : astropy.table.QTable
                Neural network-generated donut stamps metadata from TARTS
                output.
            - zernikes : astropy.table.QTable
                Zernike coefficients table with individual donut and
                average values.
            - donutQualityTable : astropy.table.QTable
                Quality information for donuts.

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
        calcZernikesFromExposure : Method that processes individual exposures.
        """
        # Check if exposure is valid
        exposure_id = int(exposure.visitInfo.id)
        self.log.info("Starting Zernike estimation for exposure id=%s", exposure_id)

        # Determine defocal type using helper function
        defocalType = self._determineDefocalType(exposure)
        aggregatedZernikes, donutStamps, zk = self.calcZernikesFromExposure(exposure, defocalType)

        # Log final donut count and processing summary
        self.log.info(
            "Final processing summary: %d donut(s) processed, defocalType='%s'", len(donutStamps), defocalType
        )
        self.log.debug(
            "Aggregated Zernikes shape: %s, donut stamps: %d, raw zernikes shape: %s",
            np.shape(aggregatedZernikes),
            len(donutStamps),
            np.shape(zk),
        )
        # aggregatedZernikes and zk are already numpy arrays from
        # calcZernikesFromExposure

        # For outputZernikesRaw and outputZernikesAvg, use the same structure
        # as standard tasks
        # Both should be 2D arrays with shape (1, nZernikes) for single
        # exposure
        zernikesRaw = np.atleast_2d(zk)  # Single prediction per donut
        zernikesAvg = np.atleast_2d(aggregatedZernikes)  # Aggregated prediction per exposure

        # Create zernikes table
        # Since we only have one exposure, put stamps in the appropriate slot
        if defocalType == "extra":
            extraStamps = donutStamps
            intraStamps = DonutStamps([])
        else:
            extraStamps = DonutStamps([])
            intraStamps = donutStamps

        # Set instance attributes for metadata creation
        self.stampsIntra = intraStamps
        self.stampsExtra = extraStamps

        self.log.debug(
            "Using defocalType='%s' -> intra stamps: %d, extra stamps: %d",
            defocalType,
            len(intraStamps),
            len(extraStamps),
        )

        # Convert neural task arrays to base class Struct format
        zkCoeffRaw = pipeBase.Struct(
            zernikes=zernikesRaw,  # 2D array of shape (numDonuts, nZernikes)
        )
        zernikesTable = self.createZkTable(zkCoeffRaw=zkCoeffRaw)
        self._updateAverageRowWithAggregatedZernikes(zernikesTable, aggregatedZernikes)
        # Add OOD scores to zernikes table as a per-donut column
        # OOD scores are extracted in the same order as donuts from TARTS
        # internal data, and both come from the same TARTS run, so they
        # must align 1:1 with zernikesRaw rows
        numDonuts = zernikesRaw.shape[0]
        oodScores = list(self._lastOodScores)

        if len(oodScores) == 0:
            # OOD model was not available - pad with NaN for all donuts
            oodScores = [np.nan] * numDonuts
        else:
            # OOD scores should always match donut count since both come from
            # the same TARTS run (tartsInternalData and total_zernikes)
            if len(oodScores) != numDonuts:
                raise ValueError(
                    f"OOD scores count ({len(oodScores)}) doesn't match donut count ({numDonuts}). "
                    "This indicates a bug - scores and donuts should align 1:1 from TARTS."
                )
        # Compute nanmedian for the 'average' row at index 0
        avgOod = float(np.nanmedian(np.array(oodScores, dtype=float)))
        fullOodColumn = [avgOod] + oodScores
        # The OOD column must align 1:1 with zernikesTable rows (zippable):
        # index 0 is the aggregate row, followed by one entry per donut. The
        # pad/truncate below is a defensive guard for unexpected mismatches.
        # Padding uses NaN sentinels to preserve shape without inventing data.
        if len(fullOodColumn) < len(zernikesTable):
            fullOodColumn += [np.nan] * (len(zernikesTable) - len(fullOodColumn))
        elif len(fullOodColumn) > len(zernikesTable):
            fullOodColumn = fullOodColumn[: len(zernikesTable)]
        zernikesTable["ood_score"] = fullOodColumn

        # Create quality table from donut stamps
        donutQualityTable = self.createDonutQualityTable(donutStamps)

        # Create donut table from donut stamps
        donutTable = self.createDonutTable(donutStamps, exposure, defocalType)

        self.log.info("Estimated %d Zernike terms for exposure", zernikesAvg.shape[1])
        self.log.debug(
            "ZernikesAvg (first 5): %s", np.array2string(zernikesAvg[0, :5], precision=self.LOG_PRECISION)
        )
        self.log.info("Finished Zernike estimation; table rows: %d", len(zernikesTable))
        return pipeBase.Struct(
            outputZernikesAvg=zernikesAvg,
            outputZernikesRaw=zernikesRaw,
            donutStampsNeural=donutStamps,
            zernikes=zernikesTable,
            donutTable=donutTable,
            donutQualityTable=donutQualityTable,
        )

    def _unpackStampData(self, stamp: DonutStamp | None) -> tuple:
        """Override parent method to handle missing intrinsic maps.

        The neural task does not use intrinsic Zernike tables, so this method
        returns NaN for intrinsic values instead of trying to access
        intrinsicMapIntra or intrinsicMapExtra attributes.

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
            The intrinsic Zernike coefficients (always NaN for neural task).
        """
        if stamp is None:
            fieldAngle = np.array(np.nan, dtype=POS2F_DTYPE) * u.deg
            centroid = np.array((np.nan, np.nan), dtype=POS2F_DTYPE) * u.pixel
            intrinsics = np.full_like(self.nollIndices, np.nan) * u.micron
        else:
            fieldAngle = np.array(stamp.calcFieldXY(), dtype=POS2F_DTYPE) * u.deg
            centroid = (
                np.array(
                    (stamp.centroid_position.x, stamp.centroid_position.y),
                    dtype=POS2F_DTYPE,
                )
                * u.pixel
            )
            # Neural task doesn't use intrinsic maps, so return NaN intrinsics
            intrinsics = np.full_like(self.nollIndices, np.nan) * u.micron

        return fieldAngle, centroid, intrinsics

    def _updateAverageRowWithAggregatedZernikes(
        self, zkTable: QTable, aggregatedZernikes: np.ndarray
    ) -> None:
        """Populate the average row of `zkTable` with neural aggregated data.

        Parameters
        ----------
        zkTable : `astropy.table.QTable`
            Table returned by ``createZkTable`` containing placeholder values.
            The first row corresponds to the aggregate (average) entry.
        aggregatedZernikes : `numpy.ndarray`
            One-dimensional array of aggregated Zernike coefficients produced
            by TARTS (in microns).
        """
        if len(zkTable) == 0:
            return

        opd_columns = zkTable.meta["opd_columns"]
        intrinsic_columns = zkTable.meta["intrinsic_columns"]
        deviation_columns = zkTable.meta["deviation_columns"]
        agg = aggregatedZernikes[0]
        if len(agg) != len(opd_columns):
            raise ValueError(
                "Neural aggregated Zernike vector length "
                f"{len(agg)} does not match expected "
                f"{len(opd_columns)} OPD columns."
            )

        agg_quant = (agg * u.micron).to(u.nm)
        avg_row = zkTable[0]

        for value, column in zip(agg_quant, opd_columns):
            avg_row[column] = value
        for column in intrinsic_columns:
            avg_row[column] = np.nan * u.nm
        for value, column in zip(agg_quant, deviation_columns):
            avg_row[column] = value
