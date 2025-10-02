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
    "CutOutDonutsBaseTaskConnections",
    "CutOutDonutsBaseTaskConfig",
    "CutOutDonutsBaseTask",
]

from copy import copy
from typing import Any

import astropy.units as u
import lsst.afw.cameraGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy.table import QTable
from lsst.afw.geom import makeSkyWcs
from lsst.afw.image import Exposure
from lsst.daf.base import PropertyList
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.geom import Point2D, degrees
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.donutImageCheck import DonutImageCheck
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    DefocalType,
    calcStampPowerSpectrum,
    createTemplateForDetector,
    getTaskInstrument,
)
from scipy.ndimage import binary_dilation
from scipy.signal import correlate


class CutOutDonutsBaseTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "instrument")  # type: ignore
):
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="AstropyQTable",
        name="donutTable",
        multiple=True,
    )
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )


class CutOutDonutsBaseTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=CutOutDonutsBaseTaskConnections  # type: ignore
):
    # Config setting for pipeline task with defaults
    donutStampSize: pexConfig.Field = pexConfig.Field(
        doc="Size of donut stamps in pixels", dtype=int, default=160
    )
    initialCutoutPadding: pexConfig.Field = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of the initial "
            + "postage stamp of donutStampSize to make sure we have a "
            + "stamp of donutStampSize after recentroiding donut. Also "
            + "used as the padding of the donut template for centroiding."
        ),
        dtype=int,
        default=5,
    )
    opticalModel: pexConfig.Field = pexConfig.Field(
        doc="Specify the optical model (offAxis, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instConfigFile: pexConfig.Field = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded.",
        dtype=str,
        optional=True,
    )
    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )
    maxRecenterDistance: pexConfig.Field = pexConfig.Field(
        doc="Maximum distance (in pixels) to shift donut stamp centroid when "
        + "performing recentering step with convolution.",
        dtype=int,
        default=20,
    )
    sourceErosionIter: pexConfig.Field = pexConfig.Field(
        doc="How many iterations of binary erosion to run on the image mask "
        + "to calculate mean donut signal (The default is 1).",
        dtype=int,
        default=1,
    )
    bkgDilationIter: pexConfig.Field = pexConfig.Field(
        doc="How many iterations of binary dilation to run on the image mask "
        + "to calculate the background variance (The default is 3).",
        dtype=int,
        default=3,
    )
    badPixelMaskDefinitions: pexConfig.ListField = pexConfig.ListField(
        doc="List of mask values flagged as 'bad' for Zernike estimation.",
        dtype=str,
        default=["SAT", "BAD", "NO_DATA", "INTRP"],
    )


class CutOutDonutsBaseTask(pipeBase.PipelineTask):
    """
    Base class for CutOutDonuts tasks.

    Subclasses must implement _DefaultName.
    """

    ConfigClass = CutOutDonutsBaseTaskConfig
    # _DefaultName implemented here in subclass
    config: CutOutDonutsBaseTaskConfig
    subtractBackground: lsst.meas.algorithms.SubtractBackgroundTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set final size (in pixels) of postage stamp images returned as
        # DonutStamp objects
        self.donutStampSize = self.config.donutStampSize
        # Add this many pixels onto each side of initial
        # cutout stamp beyond the size specified
        # in self.donutStampSize. This makes sure that
        # after recentroiding the donut from the catalog
        # position by convolving a template on the initial
        # cutout stamp we will still have a postage stamp
        # of size self.donutStampSize.
        self.initialCutoutPadding = self.config.initialCutoutPadding
        # Specify optical model
        self.opticalModel = self.config.opticalModel
        # Set instrument configuration info
        self.instConfigFile = self.config.instConfigFile
        # Set the amount of mask dilation for background
        self.bkgDilationIter = self.config.bkgDilationIter
        # Set up background subtraction task
        self.makeSubtask("subtractBackground")
        # Set max recentering distance in pixels
        self.maxRecenterDistance = self.config.maxRecenterDistance
        # Set Variance Plane Warning only once
        self.varianceWarningSet = False

    def shiftCenters(
        self, centerArr: np.ndarray, boundary: float, distance: float
    ) -> np.ndarray:
        """Shift the centers of sources if the distance to
        boundary is less than required.

        Parameters
        ----------
        centerArr : np.ndarray
            Center points.
        boundary : float
            Boundary point.
        distance : float
            Required distance.

        Returns
        -------
        np.ndarray
            Shifted center points.
        """

        # Distance between the center points and boundary
        centerArrCopy = copy(centerArr)
        delta = boundary - centerArrCopy

        # Shift the center if needed
        shiftNeeded = np.where(np.abs(delta) < distance)
        centerArrCopy[shiftNeeded] = boundary - np.sign(delta[shiftNeeded]) * distance

        return centerArrCopy

    def calculateFinalCentroids(
        self,
        exposure: Exposure,
        template: np.ndarray,
        xCenters: np.ndarray,
        yCenters: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Recentroid donuts from catalog values by convolving with template.
        Also return the appropriate corner values for the final donutStamps
        taking into account donuts possibly being near the edges of the
        exposure and compensating appropriately.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Exposure with the donut image.
        template : numpy.ndarray
            Donut template for the exposure.
        xCenters : np.ndarray
            X pixel donut centers from donutCatalog.
        yCenters : np.ndarray
            Y pixel donut centers from donutCatalog.

        Returns
        -------
        np.ndarray
            Final donut x centroid pixel positions on exposure.
        np.ndarray
            Final donut y centroid pixel positions on exposure.
        np.ndarray
            Final x corner positions on exposure for donutStamp BBox.
        np.ndarray
            Final y corner positions on exposure for donutStamp BBox.
        np.ndarray
            Original x corner positions.
        np.ndarray
            Original y corner positions.
        float
            The height of the max point in the convolved image.
        """

        expDim = exposure.getDimensions()
        initialCutoutSize = self.donutStampSize + (2 * self.initialCutoutPadding)
        initialHalfWidth = int(initialCutoutSize / 2)
        stampHalfWidth = int(self.donutStampSize / 2)

        # Shift stamp center if necessary
        xCentersShifted = copy(xCenters)
        yCentersShifted = copy(yCenters)
        xCentersShifted = self.shiftCenters(
            xCentersShifted, expDim.getX(), initialHalfWidth
        )
        xCentersShifted = self.shiftCenters(xCentersShifted, 0, initialHalfWidth)
        yCentersShifted = self.shiftCenters(
            yCentersShifted, expDim.getY(), initialHalfWidth
        )
        yCentersShifted = self.shiftCenters(yCentersShifted, 0, initialHalfWidth)

        # Stamp BBox defined by corner pixel and extent
        initXCorners = xCentersShifted - initialHalfWidth
        initYCorners = yCentersShifted - initialHalfWidth

        newX = list()
        newY = list()
        peakHeight = list()

        for initX, initY in zip(initXCorners, initYCorners):
            # Define BBox and get cutout from exposure
            initCornerPoint = lsst.geom.Point2I(initX, initY)
            initBBox = lsst.geom.Box2I(
                initCornerPoint, lsst.geom.Extent2I(initialCutoutSize)
            )
            initialCutout = exposure[initBBox]

            # Find the centroid by finding the max point in an initial
            # cutout convolved with a template
            correlatedImage = correlate(initialCutout.image.array, template)
            maxIdx = np.argmax(correlatedImage)
            maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))
            peakHeight.append(correlatedImage[maxLoc])

            # The actual donut location is at the center of the template
            # But the peak of correlation will correspond to the [0, 0]
            # corner of the template
            templateHalfWidth = int(len(template) / 2)
            newX.append(maxLoc[1] - templateHalfWidth)
            newY.append(maxLoc[0] - templateHalfWidth)

        finalDonutX = xCentersShifted + (np.array(newX) - initialHalfWidth)
        finalDonutY = yCentersShifted + (np.array(newY) - initialHalfWidth)

        # Shift stamp centers if necessary but not final centroid definition
        xStampCenters = self.shiftCenters(finalDonutX, expDim.getX(), stampHalfWidth)
        xStampCenters = self.shiftCenters(xStampCenters, 0, stampHalfWidth)
        yStampCenters = self.shiftCenters(finalDonutY, expDim.getY(), stampHalfWidth)
        yStampCenters = self.shiftCenters(yStampCenters, 0, stampHalfWidth)

        # Define corner for final stamp BBox
        xCorners = xStampCenters - stampHalfWidth
        yCorners = yStampCenters - stampHalfWidth
        origXCorners = xCenters - stampHalfWidth
        origYCorners = yCenters - stampHalfWidth

        return (
            finalDonutX,
            finalDonutY,
            xCorners,
            yCorners,
            origXCorners,
            origYCorners,
            np.array(peakHeight),
        )

    def calculateSN(self, stamp: DonutStamp) -> dict:
        """
        Calculate signal-to-noise ratio.

        Parameters
        ----------
        stamp : lsst.ts.wep.task.donutStamp
            A stamp containing donut image.

        Returns
        -------
        dict
             A dictionary of calculated quantities
        """

        imageArray = stamp.stamp_im.image.array
        mask = stamp.stamp_im.mask
        varianceArray = stamp.stamp_im.variance.array

        # The following are example donut mask values:
        # maskPlaneDict={'BAD': 0, 'BLEND': 10, 'CR': 3, 'DETECTED': 5,
        # 'DETECTED_NEGATIVE': 6, 'DONUT': 9, 'EDGE': 4, 'INTRP': 2,
        # 'NO_DATA': 8, 'SAT': 1, 'SUSPECT': 7}

        # The total mask value per pixel reflects that,
        # So that each mask pixel has a value of
        # eg.0, 512, 1024 for LSSTCam,
        # or 0, 2048 for auxTel
        # Thus to find out the number of pixels
        # taken by the donut mask we sum all
        # the nonzero mask pixels.
        donutMaskPlane = mask.getMaskPlane("DONUT")
        donutMask = mask.array == np.power(2, donutMaskPlane)

        # Number of pixels taken by the donut in the original donut mask
        nPxMask = np.sum(donutMask)

        # Signal estimate based on the donut mean
        signalMean = imageArray[donutMask].mean()  # per pixel
        ttlSignalMean = nPxMask * signalMean

        # Signal estimate based on the sum of donut pixels
        ttlSignalSum = np.sum(imageArray[donutMask])

        # Background noise estimate:
        # expand the inverted mask to remove donut contribution
        # the amount of default dilation was matched
        # to produce SN comparable to when using the
        # variance plane
        bkgndMask = ~binary_dilation(donutMask, iterations=self.bkgDilationIter)
        # Test whether the mask is not too large.
        # If cross-section reached the edge of the
        # image, reduce until it's not the case.
        width, height = np.shape(bkgndMask)
        xsection = bkgndMask[:, width // 2]
        while (~xsection[0]) and (self.bkgDilationIter > 1):
            self.bkgDilationIter -= 1
            self.log.warning(
                f"Binary dilation of donut mask reached the edge of the image; \
reducing the amount of donut mask dilation to {self.bkgDilationIter}"
            )
            bkgndMask = ~binary_dilation(donutMask, iterations=self.bkgDilationIter)
            xsection = bkgndMask[:, width // 2]
        # Remove any other masked pixels including blends
        bkgndMask[mask.array > 0] = 0

        backgroundImageStdev = imageArray[bkgndMask].std()  # per pixel
        sqrtMeanVariance = np.sqrt(np.mean(varianceArray[bkgndMask]))

        # Per-pixel variance based on the image region
        # outside of the dilated donut mask
        backgroundImageVariance = imageArray[bkgndMask].var()

        # The mean image value  in the background region
        backgroundImageMean = np.mean(imageArray[bkgndMask])

        # Total noise based on the variance of the image background
        ttlNoiseBkgndVariance = np.sqrt(backgroundImageVariance * nPxMask)

        # Noise based on the sum of variance plane pixels inside the donut mask
        ttlNoiseDonutVariance = np.sqrt(varianceArray[donutMask].sum())

        # Avoid zero division in case variance plane doesn't exist
        if ttlNoiseDonutVariance > 0:
            sn = ttlSignalSum / ttlNoiseDonutVariance
        # Legacy behavior: if variance plance was not calculated,
        # use the image background variance
        else:
            sn = ttlSignalSum / ttlNoiseBkgndVariance
            # Only warn about missing variance plane once per task
            if self.varianceWarningSet is False:
                self.log.warning(
                    "Missing variance plane; \
    using the variance of image background for noise estimate."
                )
                self.varianceWarningSet = True
        snDict = {
            "SN": sn,
            "signal_mean": ttlSignalMean,
            "signal_sum": ttlSignalSum,
            "n_px_mask": nPxMask,
            "background_image_stdev": backgroundImageStdev,
            "sqrt_mean_variance": sqrtMeanVariance,
            "background_image_variance": backgroundImageVariance,
            "background_image_mean": backgroundImageMean,
            "ttl_noise_bkgnd_variance": ttlNoiseBkgndVariance,
            "ttl_noise_donut_variance": ttlNoiseDonutVariance,
        }

        return snDict

    def filterBadRecentering(
        self, xShifts: np.ndarray, yShifts: np.ndarray
    ) -> np.ndarray:
        """Filter out donuts that are recentered far away from the median
        shift of all donuts. The median is subtracted to account for a constant
        shift due to any constant offsets from the WCS used to calculate
        the pixel positions.

        Parameters
        ----------
        xShifts : np.ndarray
            Shifts of all donut sources in the x-direction in units of pixels.
        yShifts : np.ndarray
            Shifts of all donut sources in the y-direction in units of pixels.

        Returns
        -------
        np.ndarray
            Indices where total shift after median subtraction is more than
            the value allowed by self.maxRecenterDistance
        """

        # Calculate median shifts and subtract them out to remove WCS offsets
        medXShift = np.median(xShifts)
        medYShift = np.median(yShifts)
        self.metadata["medianXShift"] = medXShift
        self.metadata["medianYShift"] = medYShift
        self.log.info(f"Median Recentering Shift: ({medXShift}, {medYShift})")
        finalXShifts = xShifts - medXShift
        finalYShifts = yShifts - medYShift

        # If shift is greater than maxRecenteringDistance
        # then use no shift at all
        totalShifts = np.sqrt(finalXShifts**2.0 + finalYShifts**2.0)
        shiftFailureIdx = np.where(totalShifts > self.maxRecenterDistance)[0]
        return shiftFailureIdx

    def addVisitLevelMetadata(
        self,
        exposure: Exposure,
        inputDonutStamps: DonutStamps,
        donutCatalog: QTable,
        defocalType: DefocalType,
    ) -> DonutStamps:
        """
        Get visit level metadata and save in the donutStamps object.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Exposure to get metadata from.
        inputDonutStamps : DonutStamps
            DonutStamps object to save metadata in.
        donutCatalog : astropy.table.QTable
            Source catalog for the pointing.
        defocalType : enum 'DefocalType'
            Defocal type of the donut image.

        Returns
        -------
        DonutStamps
            DonutStamps object with metadata added.
        """
        if inputDonutStamps.metadata is None:
            inputDonutStamps.metadata = PropertyList()

        visitInfo = exposure.getInfo().getVisitInfo()
        detector = exposure.getDetector()
        detectorName = detector.getName()
        bandLabel = exposure.filter.bandLabel
        cameraName = donutCatalog.meta["visit_info"]["instrument_label"]

        instrument = getTaskInstrument(
            cameraName,
            detectorName,
            self.instConfigFile,
        )
        inputDonutStamps.metadata["BANDPASS"] = bandLabel
        inputDonutStamps.metadata["DFC_DIST"] = instrument.defocalOffset * 1e3
        inputDonutStamps.metadata["DFC_TYPE"] = defocalType.value
        inputDonutStamps.metadata["DET_NAME"] = detectorName
        inputDonutStamps.metadata["VISIT"] = donutCatalog.meta["visit_info"]["visit_id"]
        inputDonutStamps.metadata["MJD"] = visitInfo.date.toAstropy().tai.mjd
        inputDonutStamps.metadata["CAM_NAME"] = cameraName

        # Save visit info
        inputDonutStamps.metadata["BORESIGHT_ROT_ANGLE_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_rot_angle"].to(u.rad).value
        )
        inputDonutStamps.metadata["BORESIGHT_PAR_ANGLE_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_par_angle"].to(u.rad).value
        )
        inputDonutStamps.metadata["BORESIGHT_ALT_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_alt"].to(u.rad).value
        )
        inputDonutStamps.metadata["BORESIGHT_AZ_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_az"].to(u.rad).value
        )
        inputDonutStamps.metadata["BORESIGHT_RA_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_ra"].to(u.rad).value
        )
        inputDonutStamps.metadata["BORESIGHT_DEC_RAD"] = (
            donutCatalog.meta["visit_info"]["boresight_dec"].to(u.rad).value
        )
        inputDonutStamps.metadata["RADIUS"] = (
            donutCatalog.meta["visit_info"]["donut_radius"]
        )

        return inputDonutStamps

    def cutOutStamps(
        self,
        exposure: Exposure,
        donutCatalog: QTable,
        defocalType: DefocalType,
        cameraName: str,
    ) -> DonutStamps:
        """
        Cut out postage stamps for sources in catalog.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Post-ISR image with defocal donuts sources.
        donutCatalog : astropy.table.QTable
            Source catalog for the pointing.
        defocalType : enum 'DefocalType'
            Defocal type of the donut image.
        cameraName : str
            Name of camera for the exposure. Can accept "LSSTCam",
            "LSSTComCam", "LATISS".

        Returns
        -------
        DonutStamps
            Collection of postage stamps as
            lsst.afw.image.MaskedImage with additional metadata.
        """
        detector = exposure.getDetector()
        detectorName = detector.getName()
        bandLabel = exposure.filter.bandLabel
        catalogMeta = donutCatalog.meta

        # Run background subtraction
        self.subtractBackground.run(exposure=exposure).background

        # Load the instrument
        instrument = getTaskInstrument(
            cameraName,
            detectorName,
            self.instConfigFile,
        )

        # Create the image template for the detector
        template = createTemplateForDetector(
            detector=detector,
            defocalType=defocalType,
            bandLabel=bandLabel,
            instrument=instrument,
            opticalModel=self.opticalModel,
            padding=self.initialCutoutPadding,
            isBinary=True,
        )

        # Initialize donut quality check
        donutCheck = DonutImageCheck(returnEntro=True)

        # Final list of DonutStamp objects
        finalStamps = list()

        # Final locations of blended sources in pixels
        finalBlendXList = list()
        finalBlendYList = list()

        # Keep track of recentering failures
        recenterFlags = np.zeros(len(donutCatalog), dtype=int)
        xCenters = donutCatalog["centroid_x"].value
        yCenters = donutCatalog["centroid_y"].value
        (
            finalDonutX,
            finalDonutY,
            xCorner,
            yCorner,
            initXCorners,
            initYCorners,
            peakHeight,
        ) = self.calculateFinalCentroids(exposure, template, xCenters, yCenters)
        xShifts = finalDonutX - xCenters
        yShifts = finalDonutY - yCenters
        recenterFailureIdx = self.filterBadRecentering(xShifts, yShifts)

        finalDonutX[recenterFailureIdx] = xCenters[recenterFailureIdx]
        finalDonutY[recenterFailureIdx] = yCenters[recenterFailureIdx]
        xCorner[recenterFailureIdx] = initXCorners[recenterFailureIdx]
        yCorner[recenterFailureIdx] = initYCorners[recenterFailureIdx]
        recenterFlags[recenterFailureIdx] = 1

        donutCatalog["finalDonutX"] = np.array(finalDonutX, dtype=int)
        donutCatalog["finalDonutY"] = np.array(finalDonutY, dtype=int)
        donutCatalog["xCorner"] = np.array(xCorner, dtype=int)
        donutCatalog["yCorner"] = np.array(yCorner, dtype=int)
        donutCatalog["xShift"] = np.array(xShifts, dtype=int)
        donutCatalog["yShift"] = np.array(yShifts, dtype=int)
        donutCatalog["recenterFlags"] = recenterFlags

        # Calculation of SN quantities
        snQuant = list()

        # Measure of donut entropy
        isEffective = list()

        # Value of entropy
        stampsEntropy = list()

        # Fraction of bad pixels
        fracBadPixels = list()

        # Max gradient in the stamp power spectrum for k < 10
        maxPowerGradKLess10 = list()

        for idx, donutRow in enumerate(donutCatalog):
            # Make an initial cutout larger than the actual final stamp
            # so that we can centroid to get the stamp centered exactly
            # on the donut

            # Adjust the centroid coordinates from the catalog by convolving
            # the postage stamp with the donut template and return
            # the new centroid position as well as the corners of the
            # postage stamp to cut out of the exposure.
            if donutRow["recenterFlags"] == 1:
                self.log.warning(
                    "Donut Recentering Failed. Flagging and "
                    + f"not shifting center of stamp for {defocalType.value}-focal "
                    + f"source at {(donutRow['centroid_x'].item(), donutRow['centroid_y'].item())}. "
                    + f"Catalog row: {donutRow.index}. "
                    + f"Proposed Shift: {(donutRow['xShift'].item(), donutRow['yShift'].item())}."
                )

            # Get the final cutout
            finalCorner = lsst.geom.Point2I(donutRow["xCorner"], donutRow["yCorner"])
            finalBBox = lsst.geom.Box2I(
                finalCorner, lsst.geom.Extent2I(self.donutStampSize)
            )
            finalCutout = exposure[finalBBox].clone()

            # Save MaskedImage to stamp
            finalStamp = finalCutout.getMaskedImage()

            # Set that as default, unless overwritten
            blendCentroidPositions = np.array([["nan"], ["nan"]], dtype=float).T

            if len(catalogMeta["blend_centroid_x"]) > 0:
                # Save centroid positions as str so we can store in header
                blendStrX = ""
                blendStrY = ""
                for blend_cx, blend_cy in zip(
                    catalogMeta["blend_centroid_x"][idx],
                    catalogMeta["blend_centroid_y"][idx],
                ):
                    blend_final_x = blend_cx + donutRow["xShift"]
                    blend_final_y = blend_cy + donutRow["yShift"]
                    blendStrX += f"{blend_final_x:.2f},"
                    blendStrY += f"{blend_final_y:.2f},"
                # Remove comma from last entry
                if len(blendStrX) > 0:
                    blendStrX = blendStrX[:-1]
                    blendStrY = blendStrY[:-1]
                finalBlendXList.append(blendStrX)
                finalBlendYList.append(blendStrY)

                # Prepare blend centroid position information
                if len(catalogMeta["blend_centroid_x"][idx]) > 0:
                    blendCentroidPositions = np.array(
                        [
                            catalogMeta["blend_centroid_x"][idx] + donutRow["xShift"],
                            catalogMeta["blend_centroid_y"][idx] + donutRow["yShift"],
                        ]
                    ).T

            # Get the local linear WCS for the donut stamp
            # Be careful to get the cd matrix from the linearized WCS instead
            # of the one from the full WCS.
            wcs = exposure.wcs
            centroid_position = Point2D(
                donutRow["finalDonutX"], donutRow["finalDonutY"]
            )
            linearTransform = wcs.linearizePixelToSky(centroid_position, degrees)
            cdMatrix = linearTransform.getLinear().getMatrix()
            linear_wcs = makeSkyWcs(
                centroid_position, wcs.pixelToSky(centroid_position), cdMatrix
            )

            donutStamp = DonutStamp(
                stamp_im=finalStamp,
                sky_position=lsst.geom.SpherePoint(
                    donutRow["coord_ra"].value,
                    donutRow["coord_dec"].value,
                    lsst.geom.radians,
                ),
                centroid_position=centroid_position,
                blend_centroid_positions=blendCentroidPositions,
                detector_name=detectorName,
                cam_name=cameraName,
                defocal_type=defocalType.value,
                # Save defocal offset in mm.
                defocal_distance=instrument.defocalOffset * 1e3,
                bandpass=bandLabel,
                archive_element=linear_wcs,
            )

            # Create image mask
            donutStamp.makeMask(instrument, self.opticalModel)

            # Calculate the S/N per stamp
            snQuant.append(self.calculateSN(donutStamp))

            # Store entropy-based measure of donut quality
            eff, entro = donutCheck.isEffDonut(donutStamp.stamp_im.image.array)
            isEffective.append(eff)
            stampsEntropy.append(entro)

            # Calculate fraction of bad pixels
            bits = finalStamp.mask.getPlaneBitMask(self.config.badPixelMaskDefinitions)
            badPixels = np.bitwise_and(finalStamp.mask.array, bits) > 0
            fracBadPixels.append(np.mean(badPixels))

            # Calculate the max gradient in the stamp power spectrum for k < 10
            _, spectrum = calcStampPowerSpectrum(donutStamp.wep_im.image)
            spectrum /= spectrum[0]  # Normalize the spectrum
            # Max gradient below k=10
            maxPowerGradKLess10.append(np.max(np.diff(spectrum[:10])))

            finalStamps.append(donutStamp)

        # Add additional information into metadata
        stampsMetadata = PropertyList()

        # Save the donut flux as magnitude
        fluxLabel = next(
            (
                colName
                for colName in donutCatalog.columns
                if colName.endswith(f"{bandLabel}_flux")
            ),
            None,
        )
        if fluxLabel is not None and len(donutCatalog[fluxLabel]) > 0:
            stampsMetadata["MAG"] = (donutCatalog[fluxLabel].value * u.nJy).to_value(
                u.ABmag
            )
        else:
            stampsMetadata["MAG"] = np.array([])
        # Save the original centroid values
        stampsMetadata["CENT_X0"] = np.array(donutCatalog["centroid_x"].value)
        stampsMetadata["CENT_Y0"] = np.array(donutCatalog["centroid_y"].value)
        # Save the centroid shift
        stampsMetadata["CENT_DX"] = donutCatalog["xShift"]
        stampsMetadata["CENT_DY"] = donutCatalog["yShift"]
        stampsMetadata["CENT_DR"] = np.sqrt(
            donutCatalog["xShift"] ** 2 + donutCatalog["yShift"] ** 2
        )

        if len(finalStamps) > 0:
            self.metadata[f"recenterFlags{defocalType.value.capitalize()}"] = list(
                recenterFlags
            )

        # Save the S/N values
        stampsMetadata["SN"] = np.array(
            [snQuant[i]["SN"] for i in range(len(snQuant))], dtype=float
        )
        stampsMetadata["SIGNAL_MEAN"] = np.array(
            [snQuant[i]["signal_mean"] for i in range(len(snQuant))]
        )
        stampsMetadata["SIGNAL_SUM"] = np.array(
            [snQuant[i]["signal_sum"] for i in range(len(snQuant))], dtype=float
        )
        stampsMetadata["NPX_MASK"] = np.array(
            [snQuant[i]["n_px_mask"] for i in range(len(snQuant))], dtype=float
        )
        stampsMetadata["BKGD_STDEV"] = np.array(
            [snQuant[i]["background_image_stdev"] for i in range(len(snQuant))],
            dtype=float,
        )
        stampsMetadata["SQRT_MEAN_VAR"] = np.array(
            [snQuant[i]["sqrt_mean_variance"] for i in range(len(snQuant))], dtype=float
        )
        stampsMetadata["BKGD_VAR"] = np.array(
            [snQuant[i]["background_image_variance"] for i in range(len(snQuant))],
            dtype=float,
        )
        stampsMetadata["BACKGROUND_IMAGE_MEAN"] = np.array(
            [snQuant[i]["background_image_mean"] for i in range(len(snQuant))],
            dtype=float,
        )
        stampsMetadata["NOISE_VAR_BKGD"] = np.array(
            [snQuant[i]["ttl_noise_bkgnd_variance"] for i in range(len(snQuant))]
        )
        stampsMetadata["NOISE_VAR_DONUT"] = np.array(
            [snQuant[i]["ttl_noise_donut_variance"] for i in range(len(snQuant))],
            dtype=float,
        )

        # Save the entropy-based quality measure
        stampsMetadata["EFFECTIVE"] = np.array(isEffective).astype(int)
        stampsMetadata["ENTROPY"] = np.array(stampsEntropy)

        # Save the peak of the correlated image
        stampsMetadata["PEAK_HEIGHT"] = peakHeight

        # Save the fraction of bad pixels
        stampsMetadata["FRAC_BAD_PIX"] = np.array(fracBadPixels).astype(float)

        # Save max gradient in the stamp power spectrum for k < 10
        maxPowerGradKLess10 = np.array(maxPowerGradKLess10).astype(float)
        stampsMetadata["MAX_POWER_GRAD"] = maxPowerGradKLess10

        finalDonutStamps = DonutStamps(
            finalStamps, metadata=stampsMetadata, use_archive=True
        )
        # Refresh to pull original metadata into stamps
        # Necessary when running full pipeline interactively.
        finalDonutStamps._refresh_metadata()

        finalDonutStamps = self.addVisitLevelMetadata(
            exposure,
            finalDonutStamps,
            donutCatalog,
            defocalType,
        )

        return finalDonutStamps
