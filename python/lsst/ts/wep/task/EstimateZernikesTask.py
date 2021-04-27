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


class EstimateZernikesTaskConnections(
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
    outputZernikes = connectionTypes.Output(
        doc="Zernike Coefficients",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimate",
    )


class EstimateZernikesTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=EstimateZernikesTaskConnections
):
    donutTemplateSize = pexConfig.Field(doc="Size of Template", dtype=int, default=160)
    donutStampSize = pexConfig.Field(doc="Size of donut stamps", dtype=int, default=160)
    initialCutoutSize = pexConfig.Field(
        doc="Size of initial donut cutout used to centroid", dtype=int, default=240
    )


class EstimateZernikesTask(pipeBase.PipelineTask):

    ConfigClass = EstimateZernikesTaskConfig
    _DefaultName = "EstimateZernikesTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up configs
        self.donutTemplateSize = self.config.donutTemplateSize
        self.donutStampSize = self.config.donutStampSize
        self.initialCutoutSize = self.config.initialCutoutSize

    def getTemplate(self, detectorName, defocalType):
        """
        Get the templates for the detector

        Parameters
        ----------
        detectorName: str
            Name of the CCD (e.g. 'R22_S11').
        defocalType: enum 'DefocalType'
            Defocal type of the donut image

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

    def cutOutStamps(self, exposure, donutCatalog, defocalType):
        """
        Cut out postage stamps for sources in catalog.

        Parameters
        ----------
        exposure: afwImage.Exposure
            Post-ISR image with defocal donuts sources.
        donutCatalog: pandas DataFrame
            Source catalog for the pointing.
        defocalType: enum 'DefocalType'
            Defocal type of the donut image

        Returns
        -------
        DonutStamps
            Collection of postage stamps as afwImage.MaskedImages
            with additional metadata
        """

        detectorName = exposure.getDetector().getName()
        template = self.getTemplate(detectorName, defocalType)
        detectorCatalog = donutCatalog.query(f'detector == "{detectorName}"')

        initialHalfWidth = int(self.initialCutoutSize / 2)
        stampHalfWidth = int(self.donutStampSize / 2)
        finalStamps = []
        finalXList = []
        finalYList = []
        xLowList = []
        yLowList = []

        for donutRow in detectorCatalog.to_records():
            # Make an initial cutout larger than the actual final stamp
            # so that we can centroid to get the stamp centered exactly
            # on the donut
            # NOTE: Switched x and y because when loading exposure transpose occurs
            yCent = int(donutRow["centroid_x"])
            xCent = int(donutRow["centroid_y"])
            initialCutout = exposure.image.array[
                xCent - initialHalfWidth : xCent + initialHalfWidth,
                yCent - initialHalfWidth : yCent + initialHalfWidth,
            ]

            # Find the centroid by finding the max point in an initial
            # cutout convolved with a template
            correlatedImage = correlate(initialCutout, template)
            maxIdx = np.argmax(correlatedImage)
            maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))

            # The actual donut location is at the center of the template
            # But the peak of correlation will correspond to the [0, 0]
            # corner of the template
            templateHalfWidth = int(self.donutTemplateSize / 2)
            newX = maxLoc[0] - templateHalfWidth
            newY = maxLoc[1] - templateHalfWidth
            finalDonutX = xCent + (newX - initialHalfWidth)
            finalDonutY = yCent + (newY - initialHalfWidth)
            finalXList.append(finalDonutX)
            finalYList.append(finalDonutY)

            # Get the final cutout
            xLow = finalDonutX - stampHalfWidth
            xHigh = finalDonutX + stampHalfWidth
            yLow = finalDonutY - stampHalfWidth
            yHigh = finalDonutY + stampHalfWidth
            xLowList.append(xLow)
            yLowList.append(yLow)
            # Transpose image back
            finalCutout = exposure.image.array[xLow:xHigh, yLow:yHigh].T.copy()
            finalMask = exposure.mask.array[xLow:xHigh, yLow:yHigh].T.copy()
            finalVariance = exposure.variance.array[xLow:xHigh, yLow:yHigh].T.copy()

            # Turn into MaskedImage object to add into a Stamp object for reading by butler
            finalStamp = afwImage.maskedImage.MaskedImageF(
                self.donutStampSize, self.donutStampSize
            )
            finalStamp.setImage(afwImage.ImageF(finalCutout))
            finalStamp.setMask(afwImage.Mask(finalMask))
            finalStamp.setVariance(afwImage.ImageF(finalVariance))
            finalStamp.setXY0(lsst.geom.Point2I(xLow, yLow))
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
        stampsMetadata["CENT_X"] = np.array(finalXList)
        stampsMetadata["CENT_Y"] = np.array(finalYList)
        # Save the corner values
        stampsMetadata["X0"] = np.array(xLowList)
        stampsMetadata["Y0"] = np.array(yLowList)

        return DonutStamps(finalStamps, metadata=stampsMetadata)

    def estimateZernikes(self, donutStampsExtra, donutStampsIntra):
        """
        Take the donut postage stamps and estimate the Zernike coefficients.

        Parameters
        ----------
        donutStampsExtra: DonutStamps
            Extra-focal donut postage stamps
        donutStampsIntra: DonutStamps
            Intra-focal donut postage stamps

        Returns
        -------
        numpy.ndarray
            Zernike coefficients for the exposure. Will return one set of
            coefficients per set of stamps, not one set of coefficients
            per detector so this array will have shape
            (# of stamps, # of zernike coefficients).
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
            fieldXY = sourProc.camXYtoFieldXY(centroidXY.getX(), centroidXY.getY())

            wfEsti.setImg(
                fieldXY,
                DefocalType.Extra,
                image=donutExtra.stamp_im.getImage().getArray(),
            )
            wfEsti.setImg(
                fieldXY,
                DefocalType.Intra,
                image=donutIntra.stamp_im.getImage().getArray(),
            )
            wfEsti.reset()
            zer4UpNm = wfEsti.calWfsErr()
            zer4UpMicrons = zer4UpNm * 1e-3

            zerArray.append(zer4UpMicrons)

        return zerArray

    def run(
        self, exposures: typing.List[afwImage.Exposure], donutCatalog: pd.DataFrame
    ) -> pipeBase.Struct:

        # TODO: For now use dataId to sort extra and intrafocal.
        # In closed loop currently with full array mode the lower
        # exposure id is the extra focal image.
        # Chris is adding new header metadata so we don't have to
        # resort to this temporary hack in the future. (BK 4/26/21)
        expId0 = exposures[0].getInfo().getVisitInfo().getExposureId()
        expId1 = exposures[1].getInfo().getVisitInfo().getExposureId()
        if expId0 < expId1:
            extraExpIdx = 0
            intraExpIdx = 1
        else:
            extraExpIdx = 1
            intraExpIdx = 0

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx], donutCatalog, DefocalType.Extra
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx], donutCatalog, DefocalType.Intra
        )

        # If no sources in exposure exit with default values
        if len(donutStampsExtra) == 0:
            return pipeBase.Struct(
                outputZernikes=np.ones(19) * -9999,
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffs = self.estimateZernikes(donutStampsExtra, donutStampsIntra)

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            outputZernikes=np.array(zernikeCoeffs),
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
