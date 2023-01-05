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

import numpy as np

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.daf.base import PropertyList
from lsst.pipe.base import connectionTypes
from lsst.cp.pipe._lookupStaticCalibration import lookupStaticCalibration

from lsst.ts.wep.Utility import (
    DonutTemplateType,
    createInstDictFromConfig,
    getCamTypeFromButlerName,
)
from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.task.DonutStamps import DonutStamp, DonutStamps
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from scipy.signal import correlate
from scipy.ndimage import shift, binary_dilation


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
        lookupFunction=lookupStaticCalibration,
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
    donutTemplateSize = pexConfig.Field(
        doc="Size of Template in pixels", dtype=int, default=160
    )
    donutStampSize = pexConfig.Field(
        doc="Size of donut stamps in pixels", dtype=int, default=160
    )
    initialCutoutPadding = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of initial "
            + "postage stamp of donutStampSize "
            + "to make sure we have a stamp of donutStampSize after recentroiding donut"
        ),
        dtype=int,
        default=5,
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, paraxial, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instObscuration = pexConfig.Field(
        doc="Obscuration (inner_radius / outer_radius of M1M3)",
        dtype=float,
        default=0.61,
    )
    instFocalLength = pexConfig.Field(
        doc="Instrument Focal Length in m", dtype=float, default=10.312
    )
    instApertureDiameter = pexConfig.Field(
        doc="Instrument Aperture Diameter in m", dtype=float, default=8.36
    )
    instDefocalOffset = pexConfig.Field(
        doc="Instrument defocal offset in mm. \
        If None then will get this from the focusZ value in exposure visitInfo. \
        (The default is None.)",
        dtype=float,
        default=None,
        optional=True,
    )
    instPixelSize = pexConfig.Field(
        doc="Instrument Pixel Size in m", dtype=float, default=10.0e-6
    )
    multiplyMask = pexConfig.Field(
        doc="Multiply the mask with the postage stamp before saving.",
        dtype=bool,
        default=False,
    )
    maskGrowthIter = pexConfig.Field(
        doc="How many iterations of binary dilation to run on the mask model.",
        dtype=int,
        default=6,
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

        # Set size (in pixels) of donut template image used for
        # final centroiding by convolution of initial cutout with template
        self.donutTemplateSize = self.config.donutTemplateSize
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
        # Set up instrument configuration dict
        self.instParams = createInstDictFromConfig(self.config)
        # Parameters for mask multiplication (for deblending)
        self.multiplyMask = self.config.multiplyMask
        self.maskGrowthIter = self.config.maskGrowthIter

    def _checkAndSetOffset(self, dataOffsetValue):
        """Check offset in instParams dictionary and if it
        is not yet defined set to data defined value.

        Parameters
        ----------
        dataOffsetValue : float
            The defocal offset amount defined in the
            data. (An exposure or donutStamp).
        """

        if self.instParams["offset"] is None:
            self.instParams["offset"] = dataOffsetValue

    def getTemplate(
        self,
        detectorName,
        defocalType,
        donutTemplateSize,
        camType,
        opticalModel="offAxis",
        pixelScale=0.2,
    ):
        """
        Get the templates for the detector.

        Parameters
        ----------
        detectorName : str
            Name of the CCD (e.g. 'R22_S11').
        defocalType : enum 'DefocalType'
            Defocal type of the donut image.
        donutTemplateSize : int
            Size of Template in pixels
        camType : enum 'CamType'
            Camera type.
        opticalModel : str, optional
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
            (The default is "offAxis")
        pixelScale : float, optional
            The pixels to arcseconds conversion factor. (The default is 0.2)

        Returns
        -------
        numpy.ndarray
            Template donut for the detector and defocal type.
        """

        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )

        template = templateMaker.makeTemplate(
            detectorName,
            defocalType,
            donutTemplateSize,
            camType=camType,
            opticalModel=opticalModel,
            pixelScale=pixelScale,
            instParams=self.instParams,
        )

        return template

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
        xCent = self.shiftCenter(xCent, expDim.getX(), initialHalfWidth)
        xCent = self.shiftCenter(xCent, 0, initialHalfWidth)
        yCent = self.shiftCenter(yCent, expDim.getY(), initialHalfWidth)
        yCent = self.shiftCenter(yCent, 0, initialHalfWidth)

        # Stamp BBox defined by corner pixel and extent
        initXCorner = xCent - initialHalfWidth
        initYCorner = yCent - initialHalfWidth

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
        templateHalfWidth = int(self.donutTemplateSize / 2)
        newX = maxLoc[1] - templateHalfWidth
        newY = maxLoc[0] - templateHalfWidth
        finalDonutX = xCent + (newX - initialHalfWidth)
        finalDonutY = yCent + (newY - initialHalfWidth)

        # Shift stamp center if necessary but not final centroid definition
        xStampCent = self.shiftCenter(finalDonutX, expDim.getX(), stampHalfWidth)
        xStampCent = self.shiftCenter(xStampCent, 0, stampHalfWidth)
        yStampCent = self.shiftCenter(finalDonutY, expDim.getY(), stampHalfWidth)
        yStampCent = self.shiftCenter(yStampCent, 0, stampHalfWidth)

        # Define corner for final stamp BBox
        xCorner = xStampCent - stampHalfWidth
        yCorner = yStampCent - stampHalfWidth

        return finalDonutX, finalDonutY, xCorner, yCorner

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
            lsst.afw.image.maskedImage.MaskedImage with additional metadata.
        """

        detectorName = exposure.getDetector().getName()
        detectorType = exposure.getDetector().getType()
        pixelScale = exposure.getWcs().getPixelScale().asArcseconds()
        camType = getCamTypeFromButlerName(cameraName, detectorType)
        # Get template
        template = self.getTemplate(
            detectorName,
            defocalType,
            self.donutTemplateSize,
            camType,
            self.opticalModel,
            pixelScale,
        )

        # If offset not yet set then use exposure value.
        self._checkAndSetOffset(np.abs(exposure.visitInfo.focusZ))

        # Set up instrument for masks
        inst = Instrument()
        inst.configFromDict(self.instParams, self.donutTemplateSize, camType)

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
            finalDonutX, finalDonutY, xCorner, yCorner = self.calculateFinalCentroid(
                exposure, template, xCent, yCent
            )
            xShift = finalDonutX - xCent
            yShift = finalDonutY - yCent
            finalXCentList.append(finalDonutX)
            finalYCentList.append(finalDonutY)

            # Get the final cutout
            finalCorner = lsst.geom.Point2I(xCorner, yCorner)
            finalBBox = lsst.geom.Box2I(
                finalCorner, lsst.geom.Extent2I(self.donutStampSize)
            )
            xCornerList.append(xCorner)
            yCornerList.append(yCorner)
            finalCutout = exposure[finalBBox]

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

            donutStamp = DonutStamp(
                stamp_im=finalStamp,
                sky_position=lsst.geom.SpherePoint(
                    donutRow["coord_ra"],
                    donutRow["coord_dec"],
                    lsst.geom.radians,
                ),
                centroid_position=lsst.geom.Point2D(finalDonutX, finalDonutY),
                blend_centroid_positions=np.array(
                    [
                        donutRow["blend_centroid_x"] + xShift,
                        donutRow["blend_centroid_y"] + yShift,
                    ]
                ).T
                if len(donutRow["blend_centroid_x"]) > 0
                else np.array([[], []]),
                detector_name=detectorName,
                cam_name=cameraName,
                defocal_type=defocalType.value,
                # Save defocal offset in mm.
                defocal_distance=self.instParams["offset"],
            )
            boundaryT = 1
            maskScalingFactorLocal = 1
            donutStamp.makeMasks(
                inst, self.opticalModel, boundaryT, maskScalingFactorLocal
            )
            donutStamp.stamp_im.setMask(donutStamp.mask_comp)

            # Create shifted mask from non-blended mask
            blendExists = len(donutRow["blend_centroid_x"]) > 0
            if self.multiplyMask and blendExists:
                donutStamp.comp_im.makeMask(
                    inst, self.opticalModel, boundaryT, maskScalingFactorLocal
                )
                shiftedMask = shift(
                    donutStamp.comp_im.mask_pupil,
                    np.array(
                        [
                            donutStamp.comp_im.blendOffsetY,
                            donutStamp.comp_im.blendOffsetX,
                        ]
                    ),
                )
                shiftedMask = binary_dilation(
                    shiftedMask, iterations=self.maskGrowthIter
                ).astype(int)
                shiftedMask[shiftedMask == 0] += 2
                shiftedMask -= 1
                donutStamp.stamp_im.image.array *= shiftedMask

            finalStamps.append(donutStamp)

        catalogLength = len(donutCatalog)
        stampsMetadata = PropertyList()
        stampsMetadata["RA_DEG"] = np.degrees(donutCatalog["coord_ra"].values)
        stampsMetadata["DEC_DEG"] = np.degrees(donutCatalog["coord_dec"].values)
        stampsMetadata["DET_NAME"] = np.array([detectorName] * catalogLength, dtype=str)
        stampsMetadata["CAM_NAME"] = np.array([cameraName] * catalogLength, dtype=str)
        stampsMetadata["DFC_TYPE"] = np.array([defocalType.value] * catalogLength)
        stampsMetadata["DFC_DIST"] = np.array(
            [self.instParams["offset"]] * catalogLength
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

        return DonutStamps(finalStamps, metadata=stampsMetadata)
