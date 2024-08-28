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

import astropy.units as u
import lsst.afw.cameraGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.geom import makeSkyWcs
from lsst.daf.base import PropertyList
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.geom import Point2D, degrees
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.donutImageCheck import DonutImageCheck
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    createTemplateForDetector,
    getOffsetFromExposure,
    getTaskInstrument,
)
from scipy.ndimage import binary_dilation
from scipy.signal import correlate


class CutOutDonutsBaseTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="DataFrame",
        name="donutCatalog",
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
    donutStampsExtra = connectionTypes.Output(
        doc="Extra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
    )
    donutStampsIntra = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )


class CutOutDonutsBaseTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=CutOutDonutsBaseTaskConnections
):
    # Config setting for pipeline task with defaults
    donutStampSize = pexConfig.Field(
        doc="Size of donut stamps in pixels", dtype=int, default=160
    )
    initialCutoutPadding = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of the initial "
            + "postage stamp of donutStampSize to make sure we have a "
            + "stamp of donutStampSize after recentroiding donut. Also "
            + "used as the padding of the donut template for centroiding."
        ),
        dtype=int,
        default=5,
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instConfigFile = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded, and the defocal offset will "
        + "be determined from the focusZ value in the exposure header.",
        dtype=str,
        optional=True,
    )
    subtractBackground = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )
    maxRecenterDistance = pexConfig.Field(
        doc="Maximum distance (in pixels) to shift donut stamp centroid when "
        + "performing recentering step with convolution.",
        dtype=int,
        default=20,
    )
    sourceErosionIter = pexConfig.Field(
        doc="How many iterations of binary erosion to run on the image mask "
        + "to calculate mean donut signal (The default is 1).",
        dtype=int,
        default=1,
    )
    bkgDilationIter = pexConfig.Field(
        doc="How many iterations of binary dilation to run on the image mask "
        + "to calculate the background variance (The default is 3).",
        dtype=int,
        default=3,
    )


class CutOutDonutsBaseTask(pipeBase.PipelineTask):
    """
    Base class for CutOutDonuts tasks.

    Subclasses must implement _DefaultName.
    """

    ConfigClass = CutOutDonutsBaseTaskConfig
    # _DefaultName implemented here in subclass

    def __init__(self, **kwargs):
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

    def shiftCenters(self, centerArr, boundary, distance):
        """Shift the center if its distance to boundary is less than required.

        Parameters
        ----------
        centerArr : float
            Center point.
        boundary : float
            Boundary point.
        distance : float
            Required distance.

        Returns
        -------
        float
            Shifted center.
        """

        # Distance between the center and boundary
        centerArrCopy = copy(centerArr)
        delta = boundary - centerArrCopy

        # Shift the center if needed
        shiftNeeded = np.where(np.abs(delta) < distance)
        centerArrCopy[shiftNeeded] = boundary - np.sign(delta[shiftNeeded]) * distance

        return centerArrCopy

    def calculateFinalCentroid(self, exposure, template, xCent, yCent):
        """
        Recentroid donut from catalog values by convolving with template.
        Also return the appropriate corner values for the final donutStamp
        taking into account donut possibly being near the edges of the
        exposure and compensating appropriately.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Exposure with the donut image.
        template : numpy.ndarray
            Donut template for the exposure.
        xCent : int
            X pixel donut center from donutCatalog.
        yCent : int
            Y pixel donut center from donutCatalog.

        Returns
        -------
        int
            Final donut x centroid pixel position on exposure.
        int
            Final donut y centroid pixel position on exposure.
        int
            Final x corner position on exposure for donutStamp BBox.
        int
            Final y corner position on exposure for donutStamp BBox.
        float
            The height of the max point in the convolved image.
        """

        expDim = exposure.getDimensions()
        initialCutoutSize = self.donutStampSize + (2 * self.initialCutoutPadding)
        initialHalfWidth = int(initialCutoutSize / 2)
        stampHalfWidth = int(self.donutStampSize / 2)

        # Shift stamp center if necessary
        xCentShifted = copy(xCent)
        yCentShifted = copy(yCent)
        xCentShifted = self.shiftCenters(xCentShifted, expDim.getX(), initialHalfWidth)
        xCentShifted = self.shiftCenters(xCentShifted, 0, initialHalfWidth)
        yCentShifted = self.shiftCenters(yCentShifted, expDim.getY(), initialHalfWidth)
        yCentShifted = self.shiftCenters(yCentShifted, 0, initialHalfWidth)

        # Stamp BBox defined by corner pixel and extent
        initXCorner = xCentShifted - initialHalfWidth
        initYCorner = yCentShifted - initialHalfWidth

        newX = list()
        newY = list()
        peakHeight = list()

        for initX, initY in zip(initXCorner, initYCorner):
            # Define BBox and get cutout from exposure
            initCornerPoint = lsst.geom.Point2I(initX, initY)
            initBBox = lsst.geom.Box2I(
                initCornerPoint, lsst.geom.Extent2I(initialCutoutSize)
            )
            initialCutout = copy(exposure[initBBox])

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

        finalDonutX = xCentShifted + (np.array(newX) - initialHalfWidth)
        finalDonutY = yCentShifted + (np.array(newY) - initialHalfWidth)

        # Shift stamp center if necessary but not final centroid definition
        xStampCent = self.shiftCenters(finalDonutX, expDim.getX(), stampHalfWidth)
        xStampCent = self.shiftCenters(xStampCent, 0, stampHalfWidth)
        yStampCent = self.shiftCenters(finalDonutY, expDim.getY(), stampHalfWidth)
        yStampCent = self.shiftCenters(yStampCent, 0, stampHalfWidth)

        # Define corner for final stamp BBox
        xCorner = xStampCent - stampHalfWidth
        yCorner = yStampCent - stampHalfWidth
        origXCorner = xCent - stampHalfWidth
        origYCorner = yCent - stampHalfWidth

        return (
            finalDonutX,
            finalDonutY,
            xCorner,
            yCorner,
            origXCorner,
            origYCorner,
            np.array(peakHeight),
        )

    def calculateSN(self, stamp):
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

        stamp.makeMask(self.instConfigFile, self.opticalModel)
        image = stamp.stamp_im.image.array
        variance = stamp.stamp_im.variance.array

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
        donut_mask = stamp.stamp_im.mask.array > 0

        # Number of pixels taken by the donut in the original donut mask
        n_px_mask = np.sum(donut_mask)

        # Signal estimate based on the donut mean
        signal_mean = image[donut_mask].mean()  # per pixel
        ttl_signal_mean = n_px_mask * signal_mean

        # Signal estimate based on the sum of donut pixels
        ttl_signal_sum = np.sum(image[donut_mask])

        # Background noise estimate:
        # expand the inverted mask to remove donut contribution
        # the amount of default dilation was matched
        # to produce SN comparable to when using the
        # variance plane
        bkgnd_mask = ~binary_dilation(donut_mask, iterations=self.bkgDilationIter)
        # Test whether the mask is not too large.
        # If cross-section reached the edge of the
        # image, reduce until it's not the case.
        width, height = np.shape(bkgnd_mask)
        xsection = bkgnd_mask[:, width // 2]
        while ~xsection[0]:
            self.bkgDilationIter -= 2
            self.log.warning(
                f"Binary dilation of donut mask reached the edge of the image; \
reducing the amount of donut mask dilation to {self.bkgDilationIter}"
            )
            bkgnd_mask = ~binary_dilation(donut_mask, iterations=self.bkgDilationIter)
            xsection = bkgnd_mask[:, width // 2]

        background_image_stdev = image[bkgnd_mask].std()  # per pixel
        sqrt_mean_variance = np.sqrt(np.mean(variance[bkgnd_mask]))

        # Per-pixel variance based on the image region
        # outside of the dilated donut mask
        background_image_variance = image[bkgnd_mask].var()

        # The mean image value  in the background region
        background_image_mean = np.mean(image[bkgnd_mask])

        # Total noise based on the variance of the image background
        ttl_noise_bkgnd_variance = np.sqrt(background_image_variance * n_px_mask)

        # Noise based on the sum of variance plane pixels inside the donut mask
        ttl_noise_donut_variance = np.sqrt(variance[donut_mask].sum())

        # Avoid zero division in case variance plane doesn't exist
        if ttl_noise_donut_variance > 0:
            sn = ttl_signal_sum / ttl_noise_donut_variance
        # Legacy behavior: if variance plance was not calculated,
        # use the image background variance
        else:
            sn = ttl_signal_sum / ttl_noise_bkgnd_variance
            self.log.warning(
                "Missing variance plane; \
using the variance of image background for noise estimate."
            )
        sn_dic = {
            "SN": sn,
            "signal_mean": ttl_signal_mean,
            "signal_sum": ttl_signal_sum,
            "n_px_mask": n_px_mask,
            "background_image_stdev": background_image_stdev,
            "sqrt_mean_variance": sqrt_mean_variance,
            "background_image_variance": background_image_variance,
            "background_image_mean": background_image_mean,
            "ttl_noise_bkgnd_variance": ttl_noise_bkgnd_variance,
            "ttl_noise_donut_variance": ttl_noise_donut_variance,
        }
        return sn_dic

    def filterBadRecentering(self, xShift, yShift):

        # Calculate median shifts and subtract them out to remove WCS offsets
        medXShift = np.median(xShift)
        medYShift = np.median(yShift)
        self.metadata["medianXShift"] = medXShift
        self.metadata["medianYShift"] = medYShift
        self.log.info(f"Median Recentering Shift: ({medXShift}, {medYShift})")
        finalXShift = xShift - medXShift
        finalYShift = yShift - medYShift

        # If shift is greater than maxRecenteringDistance
        # then use no shift at all
        totalShift = np.sqrt(finalXShift**2.0 + finalYShift**2.0)
        shiftFailureIdx = np.where(totalShift > self.maxRecenterDistance)[0]
        return shiftFailureIdx

    def cutOutStamps(self, exposure, donutCatalog, defocalType, cameraName):
        """
        Cut out postage stamps for sources in catalog.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Post-ISR image with defocal donuts sources.
        donutCatalog : pandas.DataFrame
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
        visitId = exposure.getInfo().getVisitInfo().id

        # Run background subtraction
        self.subtractBackground.run(exposure=exposure).background

        # Get the offset
        offset = getOffsetFromExposure(exposure, cameraName, defocalType)

        # Load the instrument
        instrument = getTaskInstrument(
            cameraName,
            detectorName,
            offset,
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

        # Final locations of BBox corners for DonutStamp images
        xCornerList = list()
        yCornerList = list()

        # Keep track of recentering failures
        recenterFlags = np.zeros(len(donutCatalog), dtype=int)
        xCent = donutCatalog["centroid_x"].values
        yCent = donutCatalog["centroid_y"].values
        (
            finalDonutX,
            finalDonutY,
            xCorner,
            yCorner,
            initXCorner,
            initYCorner,
            peakHeight,
        ) = self.calculateFinalCentroid(exposure, template, xCent, yCent)
        xShift = finalDonutX - xCent
        yShift = finalDonutY - yCent
        recenterFailureIdx = self.filterBadRecentering(xShift, yShift)

        finalDonutX[recenterFailureIdx] = xCent[recenterFailureIdx]
        finalDonutY[recenterFailureIdx] = yCent[recenterFailureIdx]
        xCorner[recenterFailureIdx] = initXCorner[recenterFailureIdx]
        yCorner[recenterFailureIdx] = initYCorner[recenterFailureIdx]
        recenterFlags[recenterFailureIdx] = 1

        donutCatalog["finalDonutX"] = np.array(finalDonutX, dtype=int)
        donutCatalog["finalDonutY"] = np.array(finalDonutY, dtype=int)
        donutCatalog["xCorner"] = np.array(xCorner, dtype=int)
        donutCatalog["yCorner"] = np.array(yCorner, dtype=int)
        donutCatalog["xShift"] = np.array(xShift, dtype=int)
        donutCatalog["yShift"] = np.array(yShift, dtype=int)
        donutCatalog["recenterFlags"] = recenterFlags

        # Calculation of SN quantities
        snQuant = list()

        # Measure of donut entropy
        isEffective = list()

        # Value of entropy
        stampsEntropy = list()

        for donutRow in donutCatalog.to_records():
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
                    + f"source at {(donutRow['centroid_x'], donutRow['centroid_y'])}. "
                    + f"Catalog index: {donutRow.index}. "
                    + f"Proposed Shift: {(donutRow['xShift'], donutRow['yShift'])}."
                )

            # Get the final cutout
            finalCorner = lsst.geom.Point2I(donutRow["xCorner"], donutRow["yCorner"])
            finalBBox = lsst.geom.Box2I(
                finalCorner, lsst.geom.Extent2I(self.donutStampSize)
            )
            xCornerList.append(donutRow["xCorner"])
            yCornerList.append(donutRow["yCorner"])
            finalCutout = exposure[finalBBox].clone()

            # Save MaskedImage to stamp
            finalStamp = finalCutout.getMaskedImage()

            # Save centroid positions as str so we can store in header
            blendStrX = ""
            blendStrY = ""

            for blend_cx, blend_cy in zip(
                donutRow["blend_centroid_x"], donutRow["blend_centroid_y"]
            ):
                blend_final_x = blend_cx + donutRow["xShift"]
                blend_final_y = blend_cy + donutRow["yShift"]
                blendStrX += f"{blend_final_x:.2f},"
                blendStrY += f"{blend_final_y:.2f},"
            # Remove comma from last entry
            if len(blendStrX) > 0:
                blendStrX = blendStrX[:-1]
                blendStrY = blendStrY[:-1]
            else:
                blendStrX = None
                blendStrY = None
            finalBlendXList.append(blendStrX)
            finalBlendYList.append(blendStrY)

            # Prepare blend centroid position information
            if len(donutRow["blend_centroid_x"]) > 0:
                blendCentroidPositions = np.array(
                    [
                        donutRow["blend_centroid_x"] + donutRow["xShift"],
                        donutRow["blend_centroid_y"] + donutRow["yShift"],
                    ]
                ).T
            else:
                blendCentroidPositions = np.array([["nan"], ["nan"]], dtype=float).T

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
                    donutRow["coord_ra"],
                    donutRow["coord_dec"],
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

            # Calculate the S/N per stamp
            snQuant.append(self.calculateSN(donutStamp))

            # Store entropy-based measure of donut quality
            eff, entro = donutCheck.isEffDonut(donutStamp.stamp_im.image.array)
            isEffective.append(eff)
            stampsEntropy.append(entro)

            finalStamps.append(donutStamp)

        # Calculate the difference between original centroid and final centroid
        catalogLength = len(donutCatalog)
        stampsMetadata = PropertyList()
        stampsMetadata["RA_DEG"] = np.degrees(donutCatalog["coord_ra"].values)
        stampsMetadata["DEC_DEG"] = np.degrees(donutCatalog["coord_dec"].values)
        stampsMetadata["DET_NAME"] = np.array([detectorName] * catalogLength, dtype=str)
        stampsMetadata["CAM_NAME"] = np.array([cameraName] * catalogLength, dtype=str)
        stampsMetadata["VISIT"] = np.array([visitId] * catalogLength, dtype=int)
        stampsMetadata["DFC_TYPE"] = np.array(
            [defocalType.value] * catalogLength, dtype=str
        )
        stampsMetadata["DFC_DIST"] = np.array(
            [instrument.defocalOffset * 1e3] * catalogLength
        )
        # Save the donut flux as magnitude
        if len(donutCatalog["source_flux"]) > 0:
            stampsMetadata["MAG"] = (
                donutCatalog["source_flux"].values * u.nJy
            ).to_value(u.ABmag)
        else:
            stampsMetadata["MAG"] = np.array([])
        # Save the original centroid values
        stampsMetadata["CENT_X0"] = np.array(donutCatalog["centroid_x"].values)
        stampsMetadata["CENT_Y0"] = np.array(donutCatalog["centroid_y"].values)
        # Save the centroid values
        stampsMetadata["CENT_X"] = donutCatalog["finalDonutX"]
        stampsMetadata["CENT_Y"] = donutCatalog["finalDonutY"]
        # Save the centroid shift
        stampsMetadata["CENT_DX"] = donutCatalog["xShift"]
        stampsMetadata["CENT_DY"] = donutCatalog["yShift"]
        stampsMetadata["CENT_DR"] = np.sqrt(
            donutCatalog["xShift"] ** 2 + donutCatalog["yShift"] ** 2
        )
        # Save the centroid positions of blended sources
        stampsMetadata["BLEND_CX"] = np.array(finalBlendXList, dtype=str)
        stampsMetadata["BLEND_CY"] = np.array(finalBlendYList, dtype=str)
        # Save the corner values
        stampsMetadata["X0"] = np.array(xCornerList)
        stampsMetadata["Y0"] = np.array(yCornerList)

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
        return DonutStamps(finalStamps, metadata=stampsMetadata, use_archive=True)
