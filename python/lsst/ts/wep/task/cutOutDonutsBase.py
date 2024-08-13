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

import lsst.afw.cameraGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.geom import makeSkyWcs
from lsst.daf.base import PropertyList
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.geom import Point2D, degrees
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    createTemplateForDetector,
    getOffsetFromExposure,
    getTaskInstrument,
)
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
        # Set up background subtraction task
        self.makeSubtask("subtractBackground")
        # Set max recentering distance in pixels
        self.maxRecenterDistance = self.config.maxRecenterDistance

    def shiftCenter(self, center, boundary, distance):
        """Shift the center if its distance to boundary is less than required.

        Parameters
        ----------
        center : float
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
        delta = boundary - center

        # Shift the center if needed
        if abs(delta) < distance:
            return boundary - np.sign(delta) * distance
        else:
            return center

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
        """

        expDim = exposure.getDimensions()
        initialCutoutSize = self.donutStampSize + (2 * self.initialCutoutPadding)
        initialHalfWidth = int(initialCutoutSize / 2)
        stampHalfWidth = int(self.donutStampSize / 2)

        # Shift stamp center if necessary
        xCentShifted = copy(xCent)
        yCentShifted = copy(yCent)
        xCentShifted = self.shiftCenter(xCentShifted, expDim.getX(), initialHalfWidth)
        xCentShifted = self.shiftCenter(xCentShifted, 0, initialHalfWidth)
        yCentShifted = self.shiftCenter(yCentShifted, expDim.getY(), initialHalfWidth)
        yCentShifted = self.shiftCenter(yCentShifted, 0, initialHalfWidth)

        # Stamp BBox defined by corner pixel and extent
        initXCorner = xCentShifted - initialHalfWidth
        initYCorner = yCentShifted - initialHalfWidth

        # Define BBox and get cutout from exposure
        initCornerPoint = lsst.geom.Point2I(initXCorner, initYCorner)
        initBBox = lsst.geom.Box2I(
            initCornerPoint, lsst.geom.Extent2I(initialCutoutSize)
        )
        initialCutout = exposure[initBBox]

        # Find the centroid by finding the max point in an initial
        # cutout convolved with a template
        correlatedImage = correlate(initialCutout.image.array, template)
        maxIdx = np.argmax(correlatedImage)
        maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))

        # The actual donut location is at the center of the template
        # But the peak of correlation will correspond to the [0, 0]
        # corner of the template
        templateHalfWidth = int(len(template) / 2)
        newX = maxLoc[1] - templateHalfWidth
        newY = maxLoc[0] - templateHalfWidth
        finalDonutX = xCentShifted + (newX - initialHalfWidth)
        finalDonutY = yCentShifted + (newY - initialHalfWidth)

        # Shift stamp center if necessary but not final centroid definition
        xStampCent = self.shiftCenter(finalDonutX, expDim.getX(), stampHalfWidth)
        xStampCent = self.shiftCenter(xStampCent, 0, stampHalfWidth)
        yStampCent = self.shiftCenter(finalDonutY, expDim.getY(), stampHalfWidth)
        yStampCent = self.shiftCenter(yStampCent, 0, stampHalfWidth)

        # Define corner for final stamp BBox
        xCorner = xStampCent - stampHalfWidth
        yCorner = yStampCent - stampHalfWidth
        origXCorner = xCent - stampHalfWidth
        origYCorner = yCent - stampHalfWidth

        return finalDonutX, finalDonutY, xCorner, yCorner, origXCorner, origYCorner

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

        # Final list of DonutStamp objects
        finalStamps = list()

        # Final locations of donut centroids in pixels
        finalXCentList = list()
        finalYCentList = list()

        # Final locations of blended sources in pixels
        finalBlendXList = list()
        finalBlendYList = list()

        # Final locations of BBox corners for DonutStamp images
        xCornerList = list()
        yCornerList = list()

        # Keep track of recentering failures
        recenterFlags = list()

        for donutRow in donutCatalog.to_records():
            # Make an initial cutout larger than the actual final stamp
            # so that we can centroid to get the stamp centered exactly
            # on the donut
            xCent = int(donutRow["centroid_x"])
            yCent = int(donutRow["centroid_y"])

            # Adjust the centroid coordinates from the catalog by convolving
            # the postage stamp with the donut template and return
            # the new centroid position as well as the corners of the
            # postage stamp to cut out of the exposure.
            finalDonutX, finalDonutY, xCorner, yCorner, initXCorner, initYCorner = (
                self.calculateFinalCentroid(exposure, template, xCent, yCent)
            )
            xShift = finalDonutX - xCent
            yShift = finalDonutY - yCent
            # If shift is greater than maxRecenteringDistance
            # then use no shift at all
            recenterDist = np.sqrt(xShift**2.0 + yShift**2.0)
            recenterFlag = 0
            if recenterDist > self.maxRecenterDistance:
                self.log.warning(
                    "Donut Recentering Failed. Flagging and "
                    + f"not shifting center of stamp for {defocalType.value}-focal "
                    + f"source at {(xCent, yCent)}. Catalog index: {donutRow.index}. "
                    + f"Proposed Shift: {(xShift, yShift)}."
                )
                # Overwrite Shifts
                finalDonutX = xCent
                finalDonutY = yCent
                xCorner = initXCorner
                yCorner = initYCorner
                recenterFlag = 1

            finalXCentList.append(finalDonutX)
            finalYCentList.append(finalDonutY)
            recenterFlags.append(recenterFlag)

            # Get the final cutout
            finalCorner = lsst.geom.Point2I(xCorner, yCorner)
            finalBBox = lsst.geom.Box2I(
                finalCorner, lsst.geom.Extent2I(self.donutStampSize)
            )
            xCornerList.append(xCorner)
            yCornerList.append(yCorner)
            finalCutout = exposure[finalBBox].clone()

            # Save MaskedImage to stamp
            finalStamp = finalCutout.getMaskedImage()

            # Save centroid positions as str so we can store in header
            blendStrX = ""
            blendStrY = ""

            for blend_cx, blend_cy in zip(
                donutRow["blend_centroid_x"], donutRow["blend_centroid_y"]
            ):
                blend_final_x = blend_cx + xShift
                blend_final_y = blend_cy + yShift
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
                        donutRow["blend_centroid_x"] + xShift,
                        donutRow["blend_centroid_y"] + yShift,
                    ]
                ).T
            else:
                blendCentroidPositions = np.array([["nan"], ["nan"]], dtype=float).T

            # Get the local linear WCS for the donut stamp
            # Be careful to get the cd matrix from the linearized WCS instead
            # of the one from the full WCS.
            wcs = exposure.wcs
            centroid_position = Point2D(finalDonutX, finalDonutY)
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

            finalStamps.append(donutStamp)

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
        # Save the centroid values
        stampsMetadata["CENT_X"] = np.array(finalXCentList)
        stampsMetadata["CENT_Y"] = np.array(finalYCentList)
        # Save the centroid positions of blended sources
        stampsMetadata["BLEND_CX"] = np.array(finalBlendXList, dtype=str)
        stampsMetadata["BLEND_CY"] = np.array(finalBlendYList, dtype=str)
        # Save the corner values
        stampsMetadata["X0"] = np.array(xCornerList)
        stampsMetadata["Y0"] = np.array(yCornerList)

        if len(finalStamps) > 0:
            self.metadata[f"recenterFlags{defocalType.value.capitalize()}"] = (
                recenterFlags
            )

        return DonutStamps(finalStamps, metadata=stampsMetadata, use_archive=True)
