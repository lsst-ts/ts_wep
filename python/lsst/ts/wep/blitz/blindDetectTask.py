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

"""Template cross-correlation donut detection."""

__all__ = ["BlindDetectConfig", "BlindDetect"]

import numpy as np
from astropy.table import QTable
from scipy.signal import correlate
from skimage.feature import peak_local_max

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure

from .utils import _INSTRUMENT


def _buildAnnularTemplate(radius: float, innerFrac: float) -> np.ndarray:
    """Return a binary annular stamp for cross-correlation donut detection."""
    r_int = round(radius)
    cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
    r = np.hypot(cx, cy)
    return np.where((r < radius) & (r >= radius * innerFrac), 1.0, 0.0)


class BlindDetectConfig(pexConfig.Config):
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Width of detector edge region to exclude from detection, in pixels.",
        dtype=int,
        default=80,
    )
    detectionBinning: pexConfig.Field = pexConfig.Field(
        doc=("Integer factor by which to bin the image before running the cross-correlation detection step."),
        dtype=int,
        default=8,
    )
    peakMinDistanceFactor: pexConfig.Field = pexConfig.Field(
        doc="Multiplier applied to the binned donut radius to set min_distance in peak_local_max.",
        dtype=float,
        default=1.6,
    )
    peakExcludeBorderFactor: pexConfig.Field = pexConfig.Field(
        doc="Multiplier applied to the binned donut radius to set exclude_border in peak_local_max.",
        dtype=float,
        default=1.15,
    )


class BlindDetect(pipeBase.Task):
    ConfigClass = BlindDetectConfig
    _DefaultName = "blindDetectTask"
    config: BlindDetectConfig

    """Detect donuts via annular template cross-correlation.

    Erodes the post-ISR exposure border by ``edgeMargin`` pixels, then calls
    `_detectPeaks`.

    Parameters
    ----------
    exposure : Exposure
        Science exposure; background is subtracted in-place.
    donutRadius : float or None
        Donut radius in pixels.  If None, uses the instrument's configured
        donut radius.

    Returns
    -------
    QTable
        Columns ``id``, ``centroid_x``, ``centroid_y`` in full-exposure pixel
        coordinates.  Empty table if no peaks are found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(
        self,
        exposure: Exposure,
        donutRadius: float | None = None,
    ) -> pipeBase.Struct:
        config = self.config
        if donutRadius is None:
            donutRadius = _INSTRUMENT.donutRadius

        trimmedBBox = exposure.getBBox().erodedBy(config.edgeMargin)
        binning = config.detectionBinning
        binned_donut_radius = donutRadius / binning
        template = _buildAnnularTemplate(
            binned_donut_radius,
            innerFrac=_INSTRUMENT.obscuration
        )

        if binning > 1:
            binnedImg = afwMath.binImage(exposure[trimmedBBox].image, binning)
            arr = binnedImg.array
        else:
            arr = exposure[trimmedBBox].image.array

        # Detect on the histogram equalized image
        heq = np.digitize(arr, np.nanquantile(arr, np.linspace(0, 1, 256)))
        det = correlate(heq.astype(float), template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(config.peakMinDistanceFactor * binned_donut_radius),
            exclude_border=int(config.peakExcludeBorderFactor * binned_donut_radius),
        )
        peaks = peaks * float(binning)
        return pipeBase.Struct(
            detections=QTable(
                {
                    "id": np.arange(1, len(peaks) + 1, dtype=np.int64),
                    "centroid_x": peaks[:, 1] + trimmedBBox.getMinX(),
                    "centroid_y": peaks[:, 0] + trimmedBBox.getMinY(),
                }
            )
        )
