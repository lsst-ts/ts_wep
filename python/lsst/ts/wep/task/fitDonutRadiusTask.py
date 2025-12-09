from typing import Any

import numpy as np
from astropy.table import QTable
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from skimage.measure import profile_line

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import binArray
from lsst.utils.timer import timeMethod

__all__ = [
    "FitDonutRadiusTaskConfig",
    "FitDonutRadiusTask",
]


class FitDonutRadiusTaskConfig(pexConfig.Config):
    widthMultiplier: pexConfig.Field = pexConfig.Field[float](
        doc="Multiplier used to convert the width of peaks fitted \
         to donut cross-section to donut edge.",
        default=0.8,
    )
    filterSigma: pexConfig.Field = pexConfig.Field[float](
        doc="Standard deviation of the Gaussian kernel \
        used to smooth out the donut cross-section prior to \
        using peak finder (in pixels).",
        default=3,
    )
    minPeakWidth: pexConfig.Field = pexConfig.Field[float](
        doc="Required minimum width of peaks (in pixels) in \
        donut cross-section.",
        default=5,
    )
    minPeakHeight: pexConfig.Field = pexConfig.Field[float](
        doc="Required minimum height of peaks in normalized \
        donut cross-section (i.e. must be between 0 and 1).",
        default=0.3,
    )
    leftDefault: pexConfig.Field = pexConfig.Field[float](
        doc="Default position of the left edge of the donut \
        expressed as a fraction of image length (i.e. between \
        0 and 1).",
        default=0.1,
    )
    rightDefault: pexConfig.Field = pexConfig.Field[float](
        doc="Default position of the right edge of the donut \
        expressed as a fraction of image length (i.e. between \
        0 and 1).",
        default=0.8,
    )
    nAngles = pexConfig.Field[int](
        doc="Number of angles for cross-sections.",
        default=10,
    )
    binning = pexConfig.Field[int](
        doc="Binning applied to each donut postage stamp.",
        default=1,
    )


class FitDonutRadiusTask(pipeBase.Task):
    ConfigClass = FitDonutRadiusTaskConfig
    _DefaultName = "FitDonutRadius"

    config: FitDonutRadiusTaskConfig
    """
    Run donut radius fitting.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.widthMultiplier = self.config.widthMultiplier
        self.filterSigma = self.config.filterSigma
        self.minPeakWidth = self.config.minPeakWidth
        self.minPeakHeight = self.config.minPeakHeight
        self.leftDefault = self.config.leftDefault
        self.rightDefault = self.config.rightDefault
        self.nAngles = self.config.nAngles
        self.binning = self.config.binning

    def empty(self) -> pipeBase.Struct:
        """Return empty table if no donut stamps are available.

        Parameters
        ----------
        donutRadiiTable : astropy.table.QTable
            Donut radius table.

        Returns
        -------
        lsst.pipe.base.Struct
            Empty donut radii table.
        """

        donutRadiiTable = QTable(
            {
                "VISIT": [],
                "DFC_TYPE": [],
                "DET_NAME": [],
                "DFC_DIST": [],
                "RADIUS": [],
                "X_PIX_LEFT_EDGE": [],
                "X_PIX_RIGHT_EDGE": [],
                "FAIL_FLAG": [],
            }
        )

        return pipeBase.Struct(donutRadiiTable=donutRadiiTable)

    @timeMethod
    def run(
        self,
        stampSet: DonutStamps,
    ) -> pipeBase.Struct:
        # If no donuts are in the DonutStamps
        # Then return an empty table
        if not stampSet:
            return self.empty()

        radiiArray = []
        leftEdgeArray = []
        rightEdgeArray = []
        detNameArray = []
        defocalDistArray = []
        failedFlagArray = []

        for stamp in stampSet:
            xLeft, xRight, donutRadius, flag = self.fit_radius(stamp.stamp_im.image.array)
            radiiArray.append(donutRadius)
            leftEdgeArray.append(xLeft)
            rightEdgeArray.append(xRight)
            detNameArray.append(stamp.detector_name)
            defocalDistArray.append(stamp.defocal_distance)
            failedFlagArray.append(flag)

        donutRadiiTable = QTable(
            {
                "DET_NAME": detNameArray,
                "DFC_DIST": defocalDistArray,
                "RADIUS": radiiArray,
                "X_PIX_LEFT_EDGE": leftEdgeArray,
                "X_PIX_RIGHT_EDGE": rightEdgeArray,
                "FAIL_FLAG": failedFlagArray,
            }
        )

        return pipeBase.Struct(donutRadiiTable=donutRadiiTable)

    def fit_radius(self, image: np.ndarray) -> tuple[float, float, float, int]:
        """
        Find peaks and widths of smoothed donut cross-section.
        The donut cross-section is normalized, and
        smoothed out with Gaussian kernel
        prior to peak finding. The location and width
        of each peak is used to determine the left
        and right edge of the donut.

        Parameters
        ----------
        image : numpy.ndarray
            The donut stamp image.

        Returns
        -------
        left_edge: float
            The position of the left donut edge in pixels.
        right_edge: float
            The position of the right donut edge in pixels.
        radius: float
            The donut radius in pixels.
        fail_flag: int
            Flag set to 1 if any fit failure encountered
            (in which case default values are stored, and
            a warning is logged).

        """
        # apply image binning if needed
        if self.binning > 1:
            image = binArray(image, self.binning)
        y_cross = self.get_median_profile(image, nangles=self.nAngles)
        y_cross_norm = np.array(y_cross) / max(y_cross)
        fail_flag = 0
        # convolve with Gaussian filter to smooth out the cross-section
        filtered_x_profile = gaussian_filter(y_cross_norm, sigma=self.filterSigma)

        # set the defaults used in case of fit failure,
        # so that the returned radius will be reasonable
        left_default_edge = self.leftDefault * len(image)
        right_default_edge = self.rightDefault * len(image)

        # detect sources
        peak_locations, peak_information = find_peaks(
            filtered_x_profile,
            height=self.minPeakHeight,
            width=self.minPeakWidth,
        )
        if len(peak_locations) > 0:
            # choose left and right peaks
            index_of_right = np.argmax(peak_locations)
            index_of_left = np.argmin(peak_locations)

            left_width = peak_information["widths"][index_of_left]
            right_width = peak_information["widths"][index_of_right]
            left_peak = peak_locations[index_of_left]
            right_peak = peak_locations[index_of_right]

            left_edge = left_peak - left_width * self.widthMultiplier
            right_edge = right_peak + right_width * self.widthMultiplier
        else:
            left_edge = left_default_edge
            right_edge = right_default_edge
            fail_flag = 1
            self.log.warning(f"Setting left edge to {left_edge} and right edge to {right_edge}")

        # Catch successful fit with bad values
        if left_edge < 0:
            self.log.warning(f"warning:  left_edge: {left_edge}")
            left_edge = left_default_edge
            fail_flag = 1
        if right_edge > len(image):
            self.log.warning(f"warning:  right_edge: {right_edge}")
            right_edge = right_default_edge
            fail_flag = 1

        # donut radius is half of the distance
        # between two edges
        radius = (right_edge - left_edge) / 2.0

        # if the image was binned, need to return to original scale
        if self.binning > 1:
            radius = radius * self.binning
            left_edge = left_edge * self.binning
            right_edge = right_edge * self.binning
        return left_edge, right_edge, radius, fail_flag

    def get_line_profile(self, image: np.ndarray, angle_deg: float = 0) -> np.ndarray:
        # Center of image
        center = np.array(image.shape) // 2

        # Length of the cross-section line
        length = len(image)  # number of pixels to sample

        # Angle in degrees
        angle_rad = np.deg2rad(angle_deg)

        # Compute endpoints
        dx = np.cos(angle_rad) * length / 2
        dy = np.sin(angle_rad) * length / 2

        start = (center[0] - dy, center[1] - dx)
        end = (center[0] + dy, center[1] + dx)

        # Extract profile
        profile = profile_line(
            image,
            start,
            end,
            mode="constant",
        )

        if len(profile) > length:
            profile = profile[:length]
        return profile

    def get_median_profile(self, image: np.ndarray, nangles: int = 20) -> np.ndarray:
        angles = np.linspace(0, 180, nangles)
        profiles = []
        for angle in angles:
            profile = self.get_line_profile(image, angle)
            profiles.append(profile)
        profiles = np.array(profiles)
        return np.median(profiles, axis=0)
