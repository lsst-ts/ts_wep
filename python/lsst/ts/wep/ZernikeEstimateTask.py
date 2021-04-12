import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage

# from lsst.meas.algorithms import Stamps

import os
import typing
from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.Utility import getConfigDir, DefocalType
from lsst.ts.wep.DonutStamps import DonutStamps

from lsst.pipe.base import connectionTypes


class ZernikeEstimateTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("detector", "instrument")
):
    donutStamps = connectionTypes.Input(
        doc="Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStamps",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.donutImageSize = self.config.donutImageSize

    def run(
        self,
        donutStamps: typing.List[DonutStamps],
    ) -> pipeBase.Struct:

        zerArray = []

        if len(donutStamps[0]) == 0:
            return pipeBase.Struct(outputZernikes=np.ones(19)*-9999)

        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")
        wfEsti = WfEstimator(instDir, algoDir)
        wfEsti.config(sizeInPix=self.donutImageSize)
        sourProc = SourceProcessor()
        sourProc.config(sensorName=donutStamps[0][0].detector_name)

        donutStampsExtra = donutStamps[0]
        donutStampsIntra = donutStamps[1]

        for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra):
            centroidXY = donutExtra.centroid_position
            fieldXY = sourProc.camXYtoFieldXY(
                centroidXY.getX(), centroidXY.getY()
            )

            wfEsti.setImg(
                fieldXY, DefocalType.Extra, image=donutExtra.stamp_im.getImage().getArray()
            )
            wfEsti.setImg(
                fieldXY, DefocalType.Intra, image=donutIntra.stamp_im.getImage().getArray()
            )
            wfEsti.reset()
            zer4UpNm = wfEsti.calWfsErr()
            zer4UpMicrons = zer4UpNm * 1e-3

            zerArray.append(zer4UpMicrons)

        return pipeBase.Struct(outputZernikes=np.array(zerArray))
