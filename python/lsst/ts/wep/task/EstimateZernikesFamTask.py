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

import os
import typing
import numpy as np
import pandas as pd

import lsst.geom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
from lsst.daf.base import PropertyList

from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.Utility import getConfigDir, DonutTemplateType, DefocalType
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from scipy.signal import correlate

from lsst.ts.wep.task.DonutStamps import DonutStamp, DonutStamps

from lsst.pipe.base import connectionTypes


class EstimateZernikesFamTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("detector", "instrument")
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
        dimensions=("instrument",),
        storageClass="DataFrame",
        name="donutCatalog",
    )
    donutStampsExtra = connectionTypes.Output(
        doc="Extra-focal Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
    )
    donutStampsIntra = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
    )
    outputZernikesRaw = connectionTypes.Output(
        doc="Zernike Coefficients from all donuts",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
    )
    outputZernikesAvg = connectionTypes.Output(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
    )


class EstimateZernikesFamTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=EstimateZernikesFamTaskConnections
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
        default=40,
    )


class EstimateZernikesFamTask(pipeBase.PipelineTask):
    """
    Run Zernike Estimation in full-array mode (FAM)
    """

    ConfigClass = EstimateZernikesFamTaskConfig
    _DefaultName = "EstimateZernikesFamTask"

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

    def assignExtraIntraIdx(self, focusZVal0, focusZVal1):
        """
        Identify which exposure in the list is the extra-focal and which
        is the intra-focal based upon `FOCUSZ` parameter in header.

        Parameters
        ----------
        focusZVal0 : float
            The `FOCUSZ` parameter from the first exposure.
        focusZVal1 : float
            The `FOCUSZ` parameter from the second exposure.

        Returns
        -------
        int
            Index in list which is extra-focal image.
        int
            Index in list which is intra-focal image.

        Raises
        ------
        ValueError
            Exposures must be a pair with one intra-focal
            and one extra-focal image.
        """

        errorStr = "Must have one extra-focal and one intra-focal image."
        if focusZVal0 < 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 <= 0:
                raise ValueError(errorStr)
            extraExpIdx = 1
            intraExpIdx = 0
        elif focusZVal0 > 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 >= 0:
                raise ValueError(errorStr)
            extraExpIdx = 0
            intraExpIdx = 1
        else:
            # Need to be defocal images ('FOCUSZ != 0')
            raise ValueError(errorStr)

        return extraExpIdx, intraExpIdx

    def getTemplate(self, detectorName, defocalType):
        """
        Get the templates for the detector.

        Parameters
        ----------
        detectorName: str
            Name of the CCD (e.g. 'R22_S11').
        defocalType: enum 'DefocalType'
            Defocal type of the donut image.

        Returns
        -------
        numpy.ndarray
            Template donut for the detector and defocal type.
        """

        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )

        template = templateMaker.makeTemplate(
            detectorName, defocalType, self.donutTemplateSize
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
        exposure: lsst.afw.image.Exposure
            Exposure with the donut image.
        template: numpy ndarray
            Donut template for the exposure.
        xCent: int
            X pixel donut center from donutCatalog.
        yCent: int
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

    def cutOutStamps(self, exposure, donutCatalog, defocalType):
        """
        Cut out postage stamps for sources in catalog.

        Parameters
        ----------
        exposure: lsst.afw.image.Exposure
            Post-ISR image with defocal donuts sources.
        donutCatalog: pandas DataFrame
            Source catalog for the pointing.
        defocalType: enum 'DefocalType'
            Defocal type of the donut image.

        Returns
        -------
        DonutStamps
            Collection of postage stamps as
            lsst.afw.image.maskedImage.MaskedImage with additional metadata.
        """

        detectorName = exposure.getDetector().getName()
        template = self.getTemplate(detectorName, defocalType)
        detectorCatalog = donutCatalog.query(f'detector == "{detectorName}"')

        # Final list of DonutStamp objects
        finalStamps = []

        # Final locations of donut centroids in pixels
        finalXCentList = []
        finalYCentList = []

        # Final locations of BBox corners for DonutStamp images
        xCornerList = []
        yCornerList = []

        for donutRow in detectorCatalog.to_records():
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
            finalStamps.append(
                DonutStamp(
                    stamp_im=finalStamp,
                    sky_position=lsst.geom.SpherePoint(
                        donutRow["coord_ra"],
                        donutRow["coord_dec"],
                        lsst.geom.radians,
                    ),
                    centroid_position=lsst.geom.Point2I(finalDonutX, finalDonutY),
                    detector_name=detectorName,
                )
            )

        stampsMetadata = PropertyList()
        stampsMetadata["RA_DEG"] = np.degrees(detectorCatalog["coord_ra"].values)
        stampsMetadata["DEC_DEG"] = np.degrees(detectorCatalog["coord_dec"].values)
        stampsMetadata["DET_NAME"] = np.array(
            [detectorName] * len(detectorCatalog), dtype=str
        )
        # Save the centroid values
        stampsMetadata["CENT_X"] = np.array(finalXCentList)
        stampsMetadata["CENT_Y"] = np.array(finalYCentList)
        # Save the corner values
        stampsMetadata["X0"] = np.array(xCornerList)
        stampsMetadata["Y0"] = np.array(yCornerList)

        return DonutStamps(finalStamps, metadata=stampsMetadata)

    def estimateZernikes(self, donutStampsExtra, donutStampsIntra):
        """
        Take the donut postage stamps and estimate the Zernike coefficients.

        Parameters
        ----------
        donutStampsExtra: DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra: DonutStamps
            Intra-focal donut postage stamps.

        Returns
        -------
        numpy.ndarray
            Zernike coefficients for the exposure. Will return one set of
            coefficients per set of stamps, not one set of coefficients
            per detector so this will be a 2-D numpy array with
            the number of rows equal to the number of donut stamps and
            the number of columns equal to the number of Zernike coefficients.
        """

        zerArray = []

        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")
        wfEsti = WfEstimator(instDir, algoDir)
        wfEsti.config(sizeInPix=self.donutStampSize)
        sourProc = SourceProcessor()
        sourProc.config(sensorName=donutStampsExtra[0].detector_name)

        for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra):
            centroidXY = donutExtra.centroid_position

            # NOTE: TS_WEP expects these images to be transposed
            # TODO: Look into this
            fieldXY = sourProc.camXYtoFieldXY(centroidXY.getY(), centroidXY.getX())
            wfEsti.setImg(
                fieldXY,
                DefocalType.Extra,
                image=donutExtra.stamp_im.getImage().getArray().T,
            )
            wfEsti.setImg(
                fieldXY,
                DefocalType.Intra,
                image=donutIntra.stamp_im.getImage().getArray().T,
            )
            wfEsti.reset()
            zer4UpNm = wfEsti.calWfsErr()
            zer4UpMicrons = zer4UpNm * 1e-3

            zerArray.append(zer4UpMicrons)

        return zerArray

    def combineZernikes(self, zernikeArray):
        """
        Combine the Zernike coefficients from stamp pairs on the
        CCD to create one final value for the CCD.

        Parameters
        ----------
        zernikeArray: numpy ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        finalZernikes: numpy ndarray
            The final combined Zernike coefficients from the CCD.
        """

        # NOTE: Currently just unweighted mean of all donuts.
        # But may change as this is an active research topic.
        return np.mean(zernikeArray, axis=0)

    def run(
        self, exposures: typing.List[afwImage.Exposure], donutCatalog: pd.DataFrame
    ) -> pipeBase.Struct:

        # Get exposure metadata to find which is extra and intra
        focusZ0 = exposures[0].getMetadata()["FOCUSZ"]
        focusZ1 = exposures[1].getMetadata()["FOCUSZ"]

        extraExpIdx, intraExpIdx = self.assignExtraIntraIdx(focusZ0, focusZ1)

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx], donutCatalog, DefocalType.Extra
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx], donutCatalog, DefocalType.Intra
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=np.ones(19) * np.nan,
                outputZernikesAvg=np.ones(19) * np.nan,
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernikeCoeffsAvg = self.combineZernikes(zernikeCoeffsRaw)

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            outputZernikesAvg=np.array(zernikeCoeffsAvg),
            outputZernikesRaw=np.array(zernikeCoeffsRaw),
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
