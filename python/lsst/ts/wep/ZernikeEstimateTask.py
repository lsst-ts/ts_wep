import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage

import os
import typing
from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.Utility import getConfigDir, DefocalType

from lsst.pipe.base import connectionTypes


class ZernikeEstimateTaskConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("detector", "instrument")):
    exposures = connectionTypes.Input(doc="Input exposure to make measurements "
                                          "on",
                                      dimensions=("exposure", "detector", "instrument"),
                                      storageClass="Exposure",
                                      name="postISRCCD",
                                      multiple=True)
    inputCatalogs = connectionTypes.Input(doc="Input donut location catalog",
                                          dimensions=("exposure", "detector", "instrument"),
                                          storageClass="SourceCatalog",
                                          name="donutCatalog",
                                          multiple=True)
    outputZernikes = connectionTypes.Output(doc="Zernike Coefficients",
                                            dimensions=("exposure", "detector", "instrument"),
                                            storageClass="NumpyArray",
                                            name="zernikeEstimate")


class ZernikeEstimateTaskConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=ZernikeEstimateTaskConnections):
    donutImageSize = pexConfig.Field(doc="Size of Template", dtype=int, default=160)


class ZernikeEstimateTask(pipeBase.PipelineTask):

    ConfigClass = ZernikeEstimateTaskConfig
    _DefaultName = "ZernikeEstimateTask"

    def __init__(self, config: pexConfig.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.donutImageSize = self.config.donutImageSize

    def run(self, exposures: typing.List[afwImage.Exposure],
            inputCatalogs: typing.List[afwTable.SourceCatalog]) -> pipeBase.Struct:

        configDir = getConfigDir()
        instDir = os.path.join(configDir, 'cwfs', 'instData')
        algoDir = os.path.join(configDir, 'cwfs', 'algo')
        wfEsti = WfEstimator(instDir, algoDir)
        wfEsti.config(sizeInPix=self.donutImageSize)
        sourProc = SourceProcessor()
        sourProc.config(sensorName=exposures[0].getDetector().getName())
        fieldXY = sourProc.camXYtoFieldXY(inputCatalogs[0][0]['xPixel'],
                                          inputCatalogs[0][0]['yPixel'])

        stampWidth = int(self.donutImageSize / 2)
        donutStamps = []
        for exposure, inputCatalog in zip(exposures, inputCatalogs):
            xLoc = int(inputCatalog[0]['xPixel'])
            yLoc = int(inputCatalog[0]['yPixel'])
            donutCutout = exposure.image.array.T[xLoc-stampWidth:xLoc+stampWidth,
                                                 yLoc-stampWidth:yLoc+stampWidth]
            donutStamps.append(donutCutout)

        wfEsti.setImg(fieldXY, DefocalType.Extra, image=donutStamps[0])
        wfEsti.setImg(fieldXY, DefocalType.Intra, image=donutStamps[1])
        wfEsti.reset()
        zer4UpNm = wfEsti.calWfsErr()
        zer4UpMicrons = zer4UpNm * 1e-3

        return pipeBase.Struct(outputZernikes=np.array(zer4UpMicrons))
