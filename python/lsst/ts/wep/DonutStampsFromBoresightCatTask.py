import numpy as np

import lsst.geom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.daf.base import PropertyList

from lsst.ts.wep.Utility import DonutTemplateType, DefocalType
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from scipy.signal import correlate

from lsst.ts.wep.DonutStamps import DonutStamp, DonutStamps

from lsst.pipe.base import connectionTypes


class DonutStampsFromBoresightCatTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "detector", "instrument")
):
    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
    )
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=("instrument",),
        storageClass="DataFrame",
        name="donutCatalog",
    )
    donutStamps = connectionTypes.Output(
        doc="Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
    )


class DonutStampsFromBoresightCatTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=DonutStampsFromBoresightCatTaskConnections
):
    donutTemplateSize = pexConfig.Field(doc="Size of Template", dtype=int, default=160)
    donutStampSize = pexConfig.Field(doc="Size of donut stamps", dtype=int, default=160)
    initialCutoutSize = pexConfig.Field(
        doc="Size of initial donut cutout used to centroid", dtype=int, default=240
    )


class DonutStampsFromBoresightCatTask(pipeBase.PipelineTask):

    ConfigClass = DonutStampsFromBoresightCatTaskConfig
    _DefaultName = "DonutStampsFromBoresightCatTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.donutTemplateSize = self.config.donutTemplateSize
        self.donutStampSize = self.config.donutStampSize
        self.initialCutoutSize = self.config.initialCutoutSize

    def run(
        self, exposure: afwImage.Exposure, donutCatalog: afwTable.SimpleCatalog
    ) -> pipeBase.Struct:

        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )
        detectorName = exposure.getDetector().getName()
        template = templateMaker.makeTemplate(
            detectorName, DefocalType.Extra, self.donutTemplateSize
        )

        initialHalfWidth = int(self.initialCutoutSize / 2)
        stampHalfWidth = int(self.donutStampSize / 2)
        finalStamps = []
        xLowList = []
        yLowList = []
        finalXList = []
        finalYList = []
        detectorCatalog = donutCatalog.query(f'detector == "{detectorName}"').reset_index(drop=True)
        for idx in np.arange(len(detectorCatalog)):
            # Make an initial cutout larger than the actual final stamp
            # so that we can centroid to get the stamp centered exactly
            # on the donut
            # NOTE: Switched x and y because when loading exposure transpose occurs
            yCent = int(detectorCatalog.iloc[idx]["centroid_x"])
            xCent = int(detectorCatalog.iloc[idx]["centroid_y"])
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
                        detectorCatalog.iloc[idx]["coord_ra"],
                        detectorCatalog.iloc[idx]["coord_dec"],
                        lsst.geom.radians
                    ),
                    centroid_position=lsst.geom.Point2I(finalDonutX, finalDonutY),
                    detector_name=detectorName,
                )
            )

        stampsMetadata = PropertyList()
        stampsMetadata["RA_DEG"] = np.degrees(detectorCatalog["coord_ra"])
        stampsMetadata["DEC_DEG"] = np.degrees(detectorCatalog["coord_dec"])
        stampsMetadata["DET_NAME"] = np.array(
            [detectorName] * len(donutCatalog), dtype=str
        )
        stampsMetadata["CENT_X"] = np.array(finalXList)
        stampsMetadata["CENT_Y"] = np.array(finalYList)
        stampsMetadata["X0"] = np.array(xLowList)
        stampsMetadata["Y0"] = np.array(yLowList)

        return pipeBase.Struct(
            donutStamps=DonutStamps(finalStamps, metadata=stampsMetadata)
        )
