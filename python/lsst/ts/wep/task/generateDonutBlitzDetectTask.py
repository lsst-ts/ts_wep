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
    "GenerateDonutBlitzDetectTaskConnections",
    "GenerateDonutBlitzDetectTaskConfig",
    "GenerateDonutBlitzDetectTask",
]

from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import QTable
from scipy.signal import correlate
from skimage.feature import peak_local_max

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.afw.cameraGeom import Camera
from lsst.afw.image import Exposure
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable
from lsst.ts.wep.utils import getTaskInstrument
from lsst.utils.timer import timeMethod


class GenerateDonutBlitzDetectTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),  # type: ignore
):
    """
    Specify the pipeline connections needed for
    GenerateDonutBlitzDetectTask. We need the defocal exposure,
    and will produce donut catalogs for a specified instrument.
    """

    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="AstropyQTable",
        name="donutTable",
    )
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )


class GenerateDonutBlitzDetectTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutBlitzDetectTaskConnections,  # type: ignore
):
    """
    Configuration settings for GenerateDonutBlitzDetectTask.
    Controls detection thresholds, quality cuts, and subtasks.
    """

    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction before detection.",
    )
    instConfigFile: pexConfig.Field = pexConfig.Field(
        doc=(
            "Path to an instrument configuration file to override the default. "
            "If begins with 'policy:' the path is relative to the ts_wep policy "
            "directory. If not provided, the default instrument for the camera "
            "will be loaded."
        ),
        dtype=str,
        optional=True,
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Width of detector edge region to exclude from detection, in pixels.",
        dtype=int,
        default=80,
    )
    detectionBinning: pexConfig.Field = pexConfig.Field(
        doc=(
            "Integer factor by which to bin the image before running the "
            "cross-correlation detection step. Larger values speed up detection "
            "at the cost of centroid precision. Peaks are measured at full "
            "resolution regardless of this setting."
        ),
        dtype=int,
        default=8,
    )
    peakMinDistanceFactor: pexConfig.Field = pexConfig.Field(
        doc=(
            "Multiplier applied to the binned donut radius to set "
            "``min_distance`` in ``peak_local_max``. Prevents splitting a "
            "single donut into multiple peaks."
        ),
        dtype=float,
        default=1.6,
    )
    peakExcludeBorderFactor: pexConfig.Field = pexConfig.Field(
        doc=(
            "Multiplier applied to the binned donut radius to set "
            "``exclude_border`` in ``peak_local_max``. Suppresses detections "
            "too close to the image edge."
        ),
        dtype=float,
        default=1.15,
    )
    innerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc=(
            "Maximum allowed |inner_flux / flux| for a candidate to be kept. "
            "The inner zone covers the dark central hole of the donut (r < 0.4 * radius). "
            "A large value indicates the central hole is filled (likely a bad candidate)."
        ),
        dtype=float,
        default=0.005,
    )
    outerFracThreshold: pexConfig.Field = pexConfig.Field(
        doc=(
            "Maximum allowed |outer_flux / flux| for a candidate to be kept. "
            "The outer zone is a background annulus just outside the donut ring. "
            "A large value indicates significant flux outside the expected donut boundary."
        ),
        dtype=float,
        default=0.005,
    )
    snrThreshold: pexConfig.Field = pexConfig.Field(
        doc="Minimum signal-to-noise ratio for a candidate to be kept.",
        dtype=float,
        default=100.0,
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask,
        doc=(
            "Subtask to filter candidates by magnitude, field angle, and blending. "
            "Applied after the blitz quality cuts when doDonutSelection is True."
        ),
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether to run the donutSelector subtask after blitz quality cuts.",
        dtype=bool,
        default=True,
    )


class GenerateDonutBlitzDetectTask(pipeBase.PipelineTask):
    """
    Detect donuts using histogram equalization and annular template
    cross-correlation, then measure flux in three concentric zones
    to assess donut quality.

    Quality filtering uses physically motivated metrics (emptiness
    of the central hole, absence of a surrounding halo,
    signal-to-noise ratio).
    """

    ConfigClass = GenerateDonutBlitzDetectTaskConfig
    _DefaultName = "generateDonutBlitzDetectTask"
    config: GenerateDonutBlitzDetectTaskConfig
    donutSelector: DonutSourceSelectorTask
    subtractBackground: SubtractBackgroundTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("subtractBackground")
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

    def _buildAnnularTemplate(
        self,
        radius: float,
        innerFrac: float,
    ) -> np.ndarray:
        """Build a binary circular annulus template.

        Parameters
        ----------
        radius : float
            Outer radius of the annulus in pixels.
        innerFrac : float
            Inner-hole radius as a fraction of ``radius``.
            Typically ``instrument.obscuration`` (~0.612 for Rubin).

        Returns
        -------
        np.ndarray
            2-D float array of shape ``(2*r_int+1, 2*r_int+1)`` with 1.0
            inside the annulus and 0.0 elsewhere, where
            ``r_int = int(radius)``.
        """
        r_int = int(radius)
        cy, cx = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
        r = np.hypot(cx, cy)
        template = np.where((r < radius) & (r >= radius * innerFrac), 1.0, 0.0)
        return template

    def _detectPeaks(
        self,
        exposureTrim: Exposure,
        donutRadius: float,
        obscuration: float,
    ) -> np.ndarray:
        """Detect donut candidates via histogram equalization and
        cross-correlation with an annular template.

        Parameters
        ----------
        exposureTrim : lsst.afw.image.Exposure
            Background-subtracted, edge-trimmed exposure.
        donutRadius : float
            Expected donut radius in full-resolution pixels.
        obscuration : float
            Fractional pupil obscuration (inner-hole radius / outer radius).
            Typically ``instrument.obscuration`` (~0.612 for Rubin).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 2)`` containing ``[row, col]`` peak
            positions in the full-resolution numpy array coordinate system
            of ``exposureTrim``.  Returns an empty array if no peaks are
            found.
        """
        binning = self.config.detectionBinning
        radius_binned = donutRadius / binning

        template = self._buildAnnularTemplate(radius_binned, innerFrac=obscuration)

        if binning > 1:
            binnedImg = afwMath.binImage(exposureTrim.image, binning)
            arr = binnedImg.array
        else:
            arr = exposureTrim.image.array

        # Histogram equalization: rank-based transform that emphasises the
        # connected ring-shaped pattern over raw flux values.
        heq = np.digitize(arr, np.nanquantile(arr, np.linspace(0, 1, 256)))

        det = correlate(heq.astype(float), template, mode="same")

        peaks = peak_local_max(
            det,
            min_distance=int(self.config.peakMinDistanceFactor * radius_binned),
            exclude_border=int(self.config.peakExcludeBorderFactor * radius_binned),
        )

        if binning > 1:
            peaks = peaks * binning

        self.log.info("Detected %d donut candidates", len(peaks))
        return peaks

    def _measureFlux(
        self,
        peaks: np.ndarray,
        exposureTrim: Exposure,
        donutRadius: float,
        obscuration: float,
    ) -> QTable:
        """Measure per-donut flux in three concentric zones.

        For each candidate the stamp is background-subtracted using the
        median of the inner hole and outer annulus regions, then flux is
        summed over the main donut ring.  A robust noise estimate is
        derived from the IQR of pixel-to-pixel differences in the
        background zones.

        Parameters
        ----------
        peaks : np.ndarray
            ``(N, 2)`` array of ``[row, col]`` positions in the full-
            resolution numpy array coordinate system of ``exposureTrim``
            (as returned by ``_detectPeaks``).
        exposureTrim : lsst.afw.image.Exposure
            Full-resolution trimmed exposure (same as passed to
            ``_detectPeaks``).
        donutRadius : float
            Expected donut radius in full-resolution pixels.
        obscuration : float
            Fractional pupil obscuration (inner-hole radius / outer radius).
            Typically ``instrument.obscuration`` (~0.612 for Rubin).

        Returns
        -------
        astropy.table.QTable
            Table with columns: ``centroid_x``, ``centroid_y`` (numpy
            array coordinates in ``exposureTrim``), ``flux``,
            ``inner_flux``, ``outer_flux``, ``std``, ``snr``.
        """
        arr = exposureTrim.image.array
        radius = donutRadius

        # Stamp size: slightly larger than the donut to include the
        # outer-background annulus used for quality checks.
        half = int(radius * 1.4)

        gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
        r = np.hypot(gx, gy)

        # Three-zone masks — main ring inner boundary set by obscuration
        main_mask = (r < radius * 1.05) & (r > radius * obscuration)
        # Well inside the central hole (0.67 * obscuration keeps this zone
        # safely within the dark region for any instrument)
        inner_mask = r < radius * obscuration * 0.67
        outer_mask = (r > radius * 1.25) & (r < radius * 1.4)
        bkg_mask = inner_mask | outer_mask

        n_main = np.sum(main_mask)

        peak_rows = peaks[:, 0]
        peak_cols = peaks[:, 1]

        flux_list, inner_flux_list, outer_flux_list, std_list = [], [], [], []

        for row, col in zip(peak_rows, peak_cols):
            rmin = row - half
            rmax = row + half + 1
            cmin = col - half
            cmax = col + half + 1

            if rmin < 0 or rmax > arr.shape[0] or cmin < 0 or cmax > arr.shape[1]:
                flux_list.append(np.nan)
                inner_flux_list.append(np.nan)
                outer_flux_list.append(np.nan)
                std_list.append(np.nan)
                continue

            stamp = arr[rmin:rmax, cmin:cmax]
            bkg = np.nanmedian(stamp[bkg_mask])
            stamp_sub = stamp - bkg

            flux_list.append(float(np.sum(stamp_sub[main_mask])))
            inner_flux_list.append(float(np.sum(stamp_sub[inner_mask])))
            outer_flux_list.append(float(np.sum(stamp_sub[outer_mask])))

            # IQR-based noise from row-difference of background pixels
            diff = (stamp_sub - np.roll(stamp_sub, 1, axis=0))[bkg_mask]
            q75, q25 = np.nanpercentile(diff, [75, 25])
            std_list.append(float((q75 - q25) / 1.349 / np.sqrt(2)))

        table = QTable()
        table["centroid_x"] = np.array(peak_cols, dtype=float)
        table["centroid_y"] = np.array(peak_rows, dtype=float)
        table["flux"] = np.array(flux_list, dtype=float)
        table["inner_flux"] = np.array(inner_flux_list, dtype=float)
        table["outer_flux"] = np.array(outer_flux_list, dtype=float)
        table["std"] = np.array(std_list, dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            table["snr"] = (table["flux"] / table["std"]) / np.sqrt(n_main)

        return table

    def emptyTable(self, exposure: Exposure) -> QTable:
        """Return an empty donut table with correct columns and metadata.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            The exposure being processed. Used to populate ``visit_info``
            metadata via ``addVisitInfoToCatTable``.

        Returns
        -------
        astropy.table.QTable
            Empty table suitable for returning when no donuts are found
            or all candidates fail quality cuts.
        """
        donutColumns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
        ]
        donutTable = QTable(names=donutColumns)
        donutTable.meta["blend_centroid_x"] = ""
        donutTable.meta["blend_centroid_y"] = ""
        return addVisitInfoToCatTable(exposure, donutTable)

    @timeMethod
    def run(self, exposure: Exposure, camera: Camera) -> pipeBase.Struct:
        camName = camera.getName()
        detectorName = exposure.getDetector().getName()
        bandLabel = exposure.filter.bandLabel

        self.log.info("Loading instrument %s for detector %s", camName, detectorName)
        instrument = getTaskInstrument(
            camName,
            detectorName,
            self.config.instConfigFile,
        )
        donutRadius = instrument.donutRadius

        if donutRadius < 5:
            self.log.warning(
                "Donut radius %.1f px is too small; exposure may be near focus. Returning empty catalog.",
                donutRadius,
            )
            return pipeBase.Struct(donutCatalog=self.emptyTable(exposure))

        self.log.info("Expected donut radius: %.1f px", donutRadius)

        # Global background subtraction (in-place)
        self.log.info("Running background subtraction")
        self.subtractBackground.run(exposure=exposure)

        # Trim detector edges to avoid edge artifacts
        self.log.info("Trimming exposure edges by %d px", self.config.edgeMargin)
        trimmedBBox = exposure.getBBox().erodedBy(self.config.edgeMargin)
        exposureTrim = exposure[trimmedBBox].clone()

        # Step 1: Detect peaks via histogram equalization + cross-correlation
        self.log.info("Running blitz donut detection")
        peaks = self._detectPeaks(exposureTrim, donutRadius, instrument.obscuration)

        if len(peaks) == 0:
            self.log.warning("No donut candidates found. Returning empty catalog.")
            return pipeBase.Struct(donutCatalog=self.emptyTable(exposure))

        # Step 2: Measure flux in three zones around each candidate
        self.log.info("Measuring flux for %d candidates", len(peaks))
        measTable = self._measureFlux(peaks, exposureTrim, donutRadius, instrument.obscuration)

        # Step 3: Blitz quality cuts
        validFlux = np.isfinite(measTable["flux"]) & (measTable["flux"] > 0)
        measTable = measTable[validFlux]

        if len(measTable) > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                innerOk = np.abs(measTable["inner_flux"] / measTable["flux"]) < self.config.innerFracThreshold
                outerOk = np.abs(measTable["outer_flux"] / measTable["flux"]) < self.config.outerFracThreshold
            snOk = measTable["snr"] > self.config.snrThreshold
            blitzSelected = innerOk & outerOk & snOk
            measTable = measTable[blitzSelected]
            self.log.info(
                "%d of %d candidates passed blitz quality cuts",
                np.sum(blitzSelected),
                len(blitzSelected),
            )

        if len(measTable) == 0:
            self.log.warning("No candidates passed blitz quality cuts. Returning empty catalog.")
            return pipeBase.Struct(donutCatalog=self.emptyTable(exposure))

        # Step 4: Convert numpy array coords in trimmed exposure to detector
        # pixel coordinates (add back the bounding-box origin offset).
        xOffset = trimmedBBox.getMinX()
        yOffset = trimmedBBox.getMinY()
        centroid_x = np.array(measTable["centroid_x"]) + xOffset
        centroid_y = np.array(measTable["centroid_y"]) + yOffset

        # Step 5: Convert detector pixel coords to sky coords via WCS
        wcs = exposure.getWcs()
        ra_arr, dec_arr = wcs.pixelToSkyArray(
            centroid_x.astype(float),
            centroid_y.astype(float),
            degrees=False,
        )

        # Step 6: Build the pre-selection catalog
        donutTable = QTable()
        donutTable["centroid_x"] = centroid_x
        donutTable["centroid_y"] = centroid_y
        donutTable["coord_ra"] = ra_arr * u.rad
        donutTable["coord_dec"] = dec_arr * u.rad
        donutTable[f"{bandLabel}_flux"] = np.array(measTable["flux"]) * u.nJy
        donutTable.meta["blend_centroid_x"] = ""
        donutTable.meta["blend_centroid_y"] = ""

        # Step 7: Optionally run DonutSourceSelectorTask
        if self.config.doDonutSelection:
            self.log.info("Running DonutSourceSelectorTask")
            donutSelection = self.donutSelector.run(donutTable, exposure.detector, bandLabel)
            donutCatSelected = donutTable[donutSelection.selected]
            donutCatSelected.meta["blend_centroid_x"] = donutSelection.blendCentersX
            donutCatSelected.meta["blend_centroid_y"] = donutSelection.blendCentersY
        else:
            donutCatSelected = donutTable

        if len(donutCatSelected) == 0:
            self.log.warning("No sources after selection. Returning empty catalog.")
            return pipeBase.Struct(donutCatalog=self.emptyTable(exposure))

        # Step 8: Sort by flux, rename flux column, add detector column
        donutCatSelected.rename_column(f"{bandLabel}_flux", "source_flux")
        fluxSort = np.argsort(np.array(donutCatSelected["source_flux"]))[::-1]
        donutCatUpd = donutCatSelected[fluxSort]
        donutCatUpd["detector"] = np.array([detectorName] * len(donutCatUpd), dtype=str)

        if self.config.doDonutSelection:
            donutCatUpd.meta["blend_centroid_x"] = [
                donutCatSelected.meta["blend_centroid_x"][i] for i in fluxSort
            ]
            donutCatUpd.meta["blend_centroid_y"] = [
                donutCatSelected.meta["blend_centroid_y"][i] for i in fluxSort
            ]

        # Final column subset to match the required catalog schema
        finalCat = donutCatUpd[
            ["coord_ra", "coord_dec", "centroid_x", "centroid_y", "detector", "source_flux"]
        ]
        finalCat.meta["blend_centroid_x"] = donutCatUpd.meta["blend_centroid_x"]
        finalCat.meta["blend_centroid_y"] = donutCatUpd.meta["blend_centroid_y"]

        # Attach visit_info metadata and generate donut_id values
        finalCat = addVisitInfoToCatTable(exposure, finalCat)

        return pipeBase.Struct(donutCatalog=finalCat)
