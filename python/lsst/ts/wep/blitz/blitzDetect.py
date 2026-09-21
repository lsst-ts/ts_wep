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

__all__ = ["BlitzDetectConfig", "BlitzDetectTask"]

import numpy as np
from astropy.table import QTable
from scipy.signal import correlate
from skimage.feature import peak_local_max

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure

from .utils import _INSTRUMENT


def _build_annular_template(radius: float, inner_frac: float) -> np.ndarray:
    """Return a binary annular stamp for cross-correlation donut detection."""
    r_int = round(radius)
    cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
    r = np.hypot(cx, cy)
    return np.where((r < radius) & (r >= radius * inner_frac), 1.0, 0.0)


class BlitzDetectConfig(pexConfig.Config):
    """Config for template cross-correlation donut detection."""

    edgeMargin: pexConfig.Field[int] = pexConfig.Field[int](
        doc="Width of detector edge region to exclude from detection, in pixels.",
        default=80,
    )
    detectionBinning: pexConfig.Field[int] = pexConfig.Field[int](
        doc=("Integer factor by which to bin the image before running the cross-correlation detection step."),
        default=8,
    )
    peakMinDistanceFactor: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Multiplier applied to the binned donut radius to set min_distance in peak_local_max.",
        default=1.6,
    )
    peakExcludeBorderFactor: pexConfig.Field[float] = pexConfig.Field[float](
        doc="Multiplier applied to the binned donut radius to set exclude_border in peak_local_max.",
        default=1.15,
    )


class BlitzDetectTask(pipeBase.Task):
    """Detect donuts via annular template cross-correlation.

    Erodes the exposure border by ``edgeMargin`` pixels, bins the remainder by
    ``detectionBinning``, and cross-correlates a binary annular template
    against the histogram-equalized image. Peaks of the correlation are the
    detections; no flux, shape or quality cut is applied here.

    The template's central hole comes from the module-level instrument
    (`_INSTRUMENT.obscuration`), not config -- it is fixed geometry, not
    tunable.
    """

    ConfigClass = BlitzDetectConfig
    _DefaultName = "blitzDetect"
    config: BlitzDetectConfig

    def run(
        self,
        exposure: Exposure,
        donut_radius: float | None = None,
    ) -> pipeBase.Struct:
        """Detect donut candidates by annular template cross-correlation.

        Parameters
        ----------
        exposure : Exposure
            Background-subtracted post-ISR science exposure, in un-binned
            pixel coordinates. Not modified.
        donut_radius : float or None, optional
            Measured donut radius in un-binned pixels. If None, the nominal
            `_INSTRUMENT.donutRadius` is used.

        Returns
        -------
        pipeBase.Struct
            ``detections`` : QTable
                Columns ``donut_id`` (1-based), ``centroid_x``,
                ``centroid_y``, in full-exposure un-binned pixel coordinates.
                Empty table if no peaks are found.
        """
        config = self.config
        if donut_radius is None:
            donut_radius = _INSTRUMENT.donutRadius

        trimmed_bbox = exposure.getBBox().erodedBy(config.edgeMargin)
        binning = config.detectionBinning
        binned_donut_radius = donut_radius / binning
        template = _build_annular_template(binned_donut_radius, inner_frac=_INSTRUMENT.obscuration)

        if binning > 1:
            binned_img = afwMath.binImage(exposure[trimmed_bbox].image, binning)
            arr = binned_img.array
        else:
            arr = exposure[trimmed_bbox].image.array

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
                    "donut_id": np.arange(1, len(peaks) + 1, dtype=np.int64),
                    "centroid_x": peaks[:, 1] + trimmed_bbox.getMinX(),
                    "centroid_y": peaks[:, 0] + trimmed_bbox.getMinY(),
                }
            )
        )
