import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage

from lsst.ts.wep.Utility import DonutTemplateType, DefocalType
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory
from scipy.signal import correlate

from lsst.pipe.base import connectionTypes


class DetectDonutsTaskConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("exposure", "detector")):
    exposure = connectionTypes.Input(doc="Input exposure to make measurements "
                                          "on",
                                     dimensions=("exposure", "detector"),
                                     storageClass="Exposure",
                                     name="postISRCCD")
    outputCatalog = connectionTypes.Output(doc="Donut Locations",
                                           dimensions=("exposure", "detector"),
                                           storageClass="SourceCatalog",
                                           name="donutCatalog")


class DetectDonutsTaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=DetectDonutsTaskConnections):
    donutImageSize = pexConfig.Field(doc="Size of Template", dtype=int, default=160)


class DetectDonutsTask(pipeBase.PipelineTask):

    ConfigClass = DetectDonutsTaskConfig
    _DefaultName = "DetectDonutsTask"

    def __init__(self, config: pexConfig.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.donutImageSize = self.config.donutImageSize

        self.outputSchema = afwTable.SourceTable.makeMinimalSchema()
        self.donutX = self.outputSchema.addField("xPixel", type=np.float64,
                                                 doc="X Pixel Location")
        self.donutY = self.outputSchema.addField("yPixel", type=np.float64,
                                                 doc="Y Pixel Location")
        self.outputCatalog = afwTable.SourceCatalog(self.outputSchema)

    def run(self, exposure: afwImage.Exposure) -> pipeBase.Struct:

        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )
        detectorName = exposure.getDetector().getName()
        template = templateMaker.makeTemplate(detectorName, DefocalType.Extra,
                                              self.donutImageSize)

        correlatedImage = correlate(exposure.image.array.T, template)

        maxIdx = np.argmax(correlatedImage)
        maxLoc = np.unravel_index(maxIdx, np.shape(correlatedImage))
        # The actual donut location is at the center of the template
        # But the peak of correlation will correspond to the [0, 0] corner of the template
        templateHalfWidth = self.donutImageSize / 2

        tmpRecord = self.outputCatalog.addNew()
        tmpRecord.set(self.donutX, maxLoc[0]-templateHalfWidth)
        tmpRecord.set(self.donutY, maxLoc[1]-templateHalfWidth)

        return pipeBase.Struct(outputCatalog=self.outputCatalog)
