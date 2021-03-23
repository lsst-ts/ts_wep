import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.meas.algorithms import Stamps

import os
import typing
from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.Utility import getConfigDir, DefocalType

from lsst.pipe.base import connectionTypes


class ZernikeEstimateTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("detector", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )
    donutStamps = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Stamps",
        name="donutStamps",
        multiple=True,
    )
    donutCatalogs = connectionTypes.Input(
        doc="Input donut location catalog",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="SimpleCatalog",
        name="donutCatalog",
        multiple=True,
    )
    outputZernikes = connectionTypes.Output(
        doc="Zernike Coefficients",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimate",
    )


class ZernikeEstimateTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ZernikeEstimateTaskConnections
):
    donutImageSize = pexConfig.Field(doc="Size of Stamp", dtype=int, default=160)


class ZernikeEstimateTask(pipeBase.PipelineTask):

    ConfigClass = ZernikeEstimateTaskConfig
    _DefaultName = "ZernikeEstimateTask"

    def __init__(self, config: pexConfig.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.donutImageSize = self.config.donutImageSize

    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutStamps: typing.List[Stamps],
        donutCatalogs: typing.List[afwTable.SourceCatalog],
    ) -> pipeBase.Struct:

        zerArray = []

        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")
        wfEsti = WfEstimator(instDir, algoDir)
        wfEsti.config(sizeInPix=self.donutImageSize)
        sourProc = SourceProcessor()
        sourProc.config(sensorName=exposures[0].getDetector().getName())

        extraImages = donutStamps[0].getMaskedImages()
        intraImages = donutStamps[1].getMaskedImages()

        for idx in range(len(donutCatalogs[0])):
            fieldXY = sourProc.camXYtoFieldXY(
                donutCatalogs[0][idx]["centroid_x"], donutCatalogs[0][idx]["centroid_y"]
            )

            wfEsti.setImg(fieldXY, DefocalType.Extra,
                          image=extraImages[idx].image.array)
            wfEsti.setImg(fieldXY, DefocalType.Intra,
                          image=intraImages[idx].image.array)
            wfEsti.reset()
            zer4UpNm = wfEsti.calWfsErr()
            zer4UpMicrons = zer4UpNm * 1e-3

            zerArray.append(zer4UpMicrons)

        return pipeBase.Struct(outputZernikes=np.array(zerArray))
