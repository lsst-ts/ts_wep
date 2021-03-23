import numpy as np

import lsst.geom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.meas.algorithms import Stamp, Stamps
from lsst.daf.base import PropertyList

from lsst.ts.wep.Utility import DonutTemplateType, DefocalType
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from scipy.signal import correlate

from lsst.pipe.base import connectionTypes


class CutOutDonutStampsTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "detector", "instrument")
):
    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements " "on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
    )
    donutCatalog = connectionTypes.Input(
        doc="Donut Locations",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="SimpleCatalog",
        name="donutCatalog",
    )
    donutStamps = connectionTypes.Output(
        doc="Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Stamps",
        name="donutStamps",
    )


class CutOutDonutStampsTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=CutOutDonutStampsTaskConnections
):
    donutTemplateSize = pexConfig.Field(doc="Size of Template", dtype=int, default=160)
    donutStampSize = pexConfig.Field(doc="Size of donut stamps", dtype=int, default=160)
    initialCutoutSize = pexConfig.Field(doc="Size of initial donut cutout used to centroid",
                                        dtype=int, default=240)


class CutOutDonutStampsTask(pipeBase.PipelineTask):

    ConfigClass = CutOutDonutStampsTaskConfig
    _DefaultName = "CutOutDonutStampsTask"

    def __init__(self, config: pexConfig.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.donutTemplateSize = self.config.donutTemplateSize
        self.donutStampSize = self.config.donutStampSize
        self.initialCutoutSize = self.config.initialCutoutSize

    def run(self, exposure: afwImage.Exposure,
            donutCatalog: afwTable.SimpleCatalog) -> pipeBase.Struct:

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
        for donutRow in donutCatalog:
            # Make an initial cutout larger than the actual final stamp
            # so that we can centroid to get the stamp centered exactly
            # on the donut
            xCent = int(donutRow['centroid_x'])
            yCent = int(donutRow['centroid_y'])
            initialCutout = exposure.image.array[xCent-initialHalfWidth:
                                                 xCent+initialHalfWidth,
                                                 yCent-initialHalfWidth:
                                                 yCent+initialHalfWidth]

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

            # Get the final cutout
            xLow = finalDonutX-stampHalfWidth
            xHigh = finalDonutX+stampHalfWidth
            yLow = finalDonutY-stampHalfWidth
            yHigh = finalDonutY+stampHalfWidth
            finalCutout = exposure.image.array[xLow:xHigh, yLow:yHigh]
            finalMask = exposure.mask.array[xLow:xHigh, yLow:yHigh]
            finalVariance = exposure.variance.array[xLow:xHigh, yLow:yHigh]

            # Turn into MaskedImage object to add into a Stamp object for reading by butler
            finalStamp = afwImage.maskedImage.MaskedImageF(self.donutStampSize, self.donutStampSize)
            finalStamp.image.array = finalCutout
            finalStamp.mask.array = finalMask
            finalStamp.variance.array = finalVariance
            finalStamps.append(Stamp(stamp_im=finalStamp, position=lsst.geom.SpherePoint(donutRow['coord_ra'], donutRow['coord_dec'])))

        stampsMetadata = PropertyList()
        stampsMetadata['RA_DEG'] = np.degrees(donutCatalog['coord_ra'])
        stampsMetadata['DEC_DEG'] = np.degrees(donutCatalog['coord_dec'])

        return pipeBase.Struct(donutStamps=Stamps(finalStamps, metadata=stampsMetadata))
