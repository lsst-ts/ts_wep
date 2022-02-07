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

import typing
import numpy as np
import pandas as pd

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.obs.lsst as obs_lsst
from lsst.utils.timer import timeMethod
from lsst.pipe.base import connectionTypes

from lsst.ts.wep.Utility import DefocalType
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.task.EstimateZernikesBase import (
    EstimateZernikesBaseConnections,
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)


class EstimateZernikesCwfsTaskConnections(
    EstimateZernikesBaseConnections, dimensions=("exposure", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
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
    outputZernikesRaw = connectionTypes.Output(
        doc="Zernike Coefficients from all donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
        multiple=True,
    )
    outputZernikesAvg = connectionTypes.Output(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
        multiple=True,
    )


class EstimateZernikesCwfsTaskConfig(
    EstimateZernikesBaseConfig, pipelineConnections=EstimateZernikesCwfsTaskConnections
):
    pass


class EstimateZernikesCwfsTask(EstimateZernikesBaseTask):
    """
    Run Zernike Estimation on corner wavefront sensors (CWFS)
    """

    ConfigClass = EstimateZernikesCwfsTaskConfig
    _DefaultName = "EstimateZernikesCwfsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set which sensors are extra and intra focal
        # See LCA-13381 for definition
        self.extraFocalNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        self.intraFocalNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We need to be able to take pairs of detectors from the full
        set of detector exposures and run the task. Then we need to put
        the outputs back into the butler repository with
        the appropriate butler dataIds.

        For the `outputZernikesRaw` and `outputZernikesAvg`
        we only have one set of values per pair of wavefront detectors
        so we put this in the dataId associated with the
        extra-focal detector.
        """

        camera = butlerQC.get(inputRefs.camera)

        # Get the detector IDs for the wavefront sensors so
        # that we can appropriately match up pairs of detectors
        if camera.getName() == "LSSTCam":
            detectorMap = (
                obs_lsst.translators.lsstCam.LsstCamTranslator.detector_mapping()
            )
        else:
            raise ValueError(f"{camera.getName()} is not a valid camera name.")

        extraFocalIds = [detectorMap[detName][0] for detName in self.extraFocalNames]
        intraFocalIds = [detectorMap[detName][0] for detName in self.intraFocalNames]

        detectorIdArr = np.array(
            [exp.dataId["detector"] for exp in inputRefs.exposures]
        )
        donutCatIdArr = np.array(
            [dCat.dataId["detector"] for dCat in inputRefs.donutCatalog]
        )

        for extraId, intraId in zip(extraFocalIds, intraFocalIds):
            extraListIdx = np.where(detectorIdArr == extraId)[0][0]
            intraListIdx = np.where(detectorIdArr == intraId)[0][0]
            dCatExtraIdx = np.where(donutCatIdArr == extraId)[0][0]
            dCatIntraIdx = np.where(donutCatIdArr == intraId)[0][0]
            expInputs = butlerQC.get(
                [inputRefs.exposures[extraListIdx], inputRefs.exposures[intraListIdx]]
            )
            dCatInputs = butlerQC.get(
                [
                    inputRefs.donutCatalog[dCatExtraIdx],
                    inputRefs.donutCatalog[dCatIntraIdx],
                ]
            )
            outputs = self.run(expInputs, dCatInputs, camera)

            butlerQC.put(
                outputs.donutStampsExtra, outputRefs.donutStampsExtra[extraListIdx]
            )
            butlerQC.put(
                outputs.donutStampsIntra, outputRefs.donutStampsIntra[intraListIdx]
            )
            butlerQC.put(
                outputs.outputZernikesRaw, outputRefs.outputZernikesRaw[extraListIdx]
            )
            butlerQC.put(
                outputs.outputZernikesAvg, outputRefs.outputZernikesAvg[extraListIdx]
            )

    @timeMethod
    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalogs: typing.List[pd.DataFrame],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        cameraName = camera.getName()
        extraCatalog, intraCatalog = donutCatalogs

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = DonutStamps([])
        donutStampsIntra = DonutStamps([])

        for exposure in exposures:
            detectorName = exposure.getDetector().getName()
            if detectorName in self.extraFocalNames:
                donutStampsExtraExp = self.cutOutStamps(
                    exposure, extraCatalog, DefocalType.Extra, cameraName
                )
                donutStampsExtra.extend([stamp for stamp in donutStampsExtraExp])
            elif detectorName in self.intraFocalNames:
                donutStampsIntraExp = self.cutOutStamps(
                    exposure, intraCatalog, DefocalType.Intra, cameraName
                )
                donutStampsIntra.extend([stamp for stamp in donutStampsIntraExp])
            else:
                continue

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if (len(donutStampsExtra) == 0) or (len(donutStampsIntra) == 0):
            return pipeBase.Struct(
                outputZernikesRaw=np.ones(19) * np.nan,
                outputZernikesAvg=np.ones(19) * np.nan,
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernikeCoeffsCombined = self.getCombinedZernikes(zernikeCoeffsRaw)
        zernikeCoeffsAvg = zernikeCoeffsCombined.combinedZernikes

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            outputZernikesAvg=np.array(zernikeCoeffsAvg),
            outputZernikesRaw=np.array(zernikeCoeffsRaw),
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
