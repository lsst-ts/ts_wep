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

import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.obs.lsst as obs_lsst
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
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
    )
    donutStampsIntra = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )
    outputZernikesRaw = connectionTypes.Output(
        doc="Zernike Coefficients from all donuts",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
        multiple=True,
    )
    outputZernikesAvg = connectionTypes.Output(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("exposure", "detector", "instrument"),
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

    def selectCwfsSources(self, donutCatalog, expDim):
        """
        Select the sources from the corner wavefront sensors to use. This
        includes arranging the intra and extra catalogs to create donut pairs
        when looking at the same index in each catalog (e.g., row 0 in
        intraCatalog and row 0 in extraCatalog will be paired together in
        wavefront estimation).

        Parameters
        ----------
        donutCatalog: pandas DataFrame
            Source catalog for the pointing.
        expDim: list
            Dimensions of the exposures in pixels.

        Returns
        -------
        pandas DataFrame
            Extra-focal donut sources for wavefront estimation.
        pandas DataFrame
            Intra-focal donut sources for wavefront estimation.
        """
        dimX, dimY = expDim

        # Get sources on corner wavefront sensors
        extraCatalog = donutCatalog.query("detector in @self.extraFocalNames")
        intraCatalog = donutCatalog.query("detector in @self.intraFocalNames")

        # For now we will just sort by flux to pair donuts
        extraCatalog = extraCatalog.sort_values(
            "source_flux", ascending=False
        ).reset_index(drop=True)
        intraCatalog = intraCatalog.sort_values(
            "source_flux", ascending=False
        ).reset_index(drop=True)

        # For now take as many pairs of sources as possible
        catLength = np.min([len(extraCatalog), len(intraCatalog)])

        return extraCatalog[:catLength], intraCatalog[:catLength]

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

        instrument = inputRefs.exposures[0].dataId["instrument"]

        # Get the detector IDs for the wavefront sensors so
        # that we can appropriately match up pairs of detectors
        if instrument == "LSSTCam":
            detectorMap = (
                obs_lsst.translators.lsstCam.LsstCamTranslator.detector_mapping()
            )
        else:
            raise ValueError(f"{instrument} is not a valid camera name.")

        extraFocalIds = [detectorMap[detName][0] for detName in self.extraFocalNames]
        intraFocalIds = [detectorMap[detName][0] for detName in self.intraFocalNames]

        detectorIdArr = np.array(
            [exp.dataId["detector"] for exp in inputRefs.exposures]
        )
        donutCat = butlerQC.get(inputRefs.donutCatalog)

        for extraId, intraId in zip(extraFocalIds, intraFocalIds):
            extraListIdx = np.where(detectorIdArr == extraId)[0][0]
            intraListIdx = np.where(detectorIdArr == intraId)[0][0]
            expInputs = butlerQC.get(
                [inputRefs.exposures[extraListIdx], inputRefs.exposures[intraListIdx]]
            )
            outputs = self.run(expInputs, donutCat, instrument)

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

    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalog: pd.DataFrame,
        cameraName: str,
    ) -> pipeBase.Struct:

        # Create intra and extra focal catalogs
        expDim = None
        for exposure in exposures:
            detectorName = exposure.getDetector().getName()
            if detectorName in self.extraFocalNames:
                expDim = exposure.getDimensions()
                break
        extraCatalog, intraCatalog = self.selectCwfsSources(
            donutCatalog, [expDim.getX(), expDim.getY()]
        )

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
