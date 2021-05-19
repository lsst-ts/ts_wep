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
        expDim:

        Returns
        -------
        pandas DataFrame
            Extra-focal donut sources for wavefront estimation
        pandas DataFrame
            Intra-focal donut sources for wavefront estimation
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

        # Adjust for display of corner wavefront sensors
        extraCatalog["centroid_x"] = dimX - extraCatalog["centroid_x"]
        extraCatalog["centroid_y"] = dimY - extraCatalog["centroid_y"]
        intraCatalog["centroid_x"] = dimX - intraCatalog["centroid_x"]
        intraCatalog["centroid_y"] = dimY - intraCatalog["centroid_y"]

        # For now take as many pairs of sources as possible
        catLength = np.min([len(extraCatalog), len(intraCatalog)])

        return extraCatalog[:catLength], intraCatalog[:catLength]

    def run(
        self, exposures: typing.List[afwImage.Exposure], donutCatalog: pd.DataFrame
    ) -> pipeBase.Struct:

        # Create intra and extra focal catalogs
        expDim = None
        for exposure in exposures:
            detectorName = exposure.getDetector().getName()
            if detectorName in self.extraFocalNames:
                expDim = exposure.getDimensions()
                break
        extraCatalog, intraCatalog = self.selectCwfsSources(
            donutCatalog, (expDim.getX(), expDim.getY())
        )

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = DonutStamps([])
        donutStampsIntra = DonutStamps([])

        for exposure in exposures:
            detectorName = exposure.getDetector().getName()
            if detectorName in self.extraFocalNames:
                donutStampsExtraExp = self.cutOutStamps(
                    exposure, extraCatalog, DefocalType.Extra
                )
                donutStampsExtra.extend([stamp for stamp in donutStampsExtraExp])
            elif detectorName in self.intraFocalNames:
                donutStampsIntraExp = self.cutOutStamps(
                    exposure, intraCatalog, DefocalType.Intra
                )
                donutStampsIntra.extend([stamp for stamp in donutStampsIntraExp])
            else:
                continue

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtraExp) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=np.ones(19) * np.nan,
                outputZernikesAvg=np.ones(19) * np.nan,
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(
            donutStampsExtraExp, donutStampsIntraExp
        )
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
