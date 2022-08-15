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

__all__ = [
    "EstimateZernikesScienceSensorTaskConnections",
    "EstimateZernikesScienceSensorTaskConfig",
    "EstimateZernikesScienceSensorTask",
]

import typing
import numpy as np
import pandas as pd

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
from lsst.utils.timer import timeMethod
from lsst.pipe.base import connectionTypes

from lsst.ts.wep.Utility import DefocalType
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.task.EstimateZernikesBase import (
    EstimateZernikesBaseConnections,
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)


class EstimateZernikesScienceSensorTaskConnections(
    EstimateZernikesBaseConnections, dimensions=("detector", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )


class EstimateZernikesScienceSensorTaskConfig(
    EstimateZernikesBaseConfig,
    pipelineConnections=EstimateZernikesScienceSensorTaskConnections,
):
    pass


class EstimateZernikesScienceSensorTask(EstimateZernikesBaseTask):
    """
    Run Zernike Estimation in full-array mode (FAM)
    """

    ConfigClass = EstimateZernikesScienceSensorTaskConfig
    _DefaultName = "EstimateZernikesScienceSensorTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set size (in pixels) of donut template image used for
        # final centroiding by convolution of initial cutout with template
        self.donutTemplateSize = self.config.donutTemplateSize
        # Set final size (in pixels) of postage stamp images returned as
        # DonutStamp objects
        self.donutStampSize = self.config.donutStampSize
        # Add this many pixels onto each side of initial
        # cutout stamp beyond the size specified
        # in self.donutStampSize. This makes sure that
        # after recentroiding the donut from the catalog
        # position by convolving a template on the initial
        # cutout stamp we will still have a postage stamp
        # of size self.donutStampSize.
        self.initialCutoutPadding = self.config.initialCutoutPadding

    def assignExtraIntraIdx(self, focusZVal0, focusZVal1):
        """
        Identify which exposure in the list is the extra-focal and which
        is the intra-focal based upon `FOCUSZ` parameter in header.

        Parameters
        ----------
        focusZVal0 : float
            The `FOCUSZ` parameter from the first exposure.
        focusZVal1 : float
            The `FOCUSZ` parameter from the second exposure.

        Returns
        -------
        int
            Index in list which is extra-focal image.
        int
            Index in list which is intra-focal image.

        Raises
        ------
        ValueError
            Exposures must be a pair with one intra-focal
            and one extra-focal image.
        """

        errorStr = "Must have one extra-focal and one intra-focal image."
        if focusZVal0 < 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 <= 0:
                raise ValueError(errorStr)
            extraExpIdx = 1
            intraExpIdx = 0
        elif focusZVal0 > 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 >= 0:
                raise ValueError(errorStr)
            extraExpIdx = 0
            intraExpIdx = 1
        else:
            # Need to be defocal images ('FOCUSZ != 0')
            raise ValueError(errorStr)

        return extraExpIdx, intraExpIdx

    @timeMethod
    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalog: typing.List[pd.DataFrame],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        # Get exposure visitInfo to find which is extra and intra
        visitInfo_0 = exposures[0].getInfo().getVisitInfo()
        visitInfo_1 = exposures[1].getInfo().getVisitInfo()
        focusZ0 = visitInfo_0.focusZ
        focusZ1 = visitInfo_1.focusZ

        extraExpIdx, intraExpIdx = self.assignExtraIntraIdx(focusZ0, focusZ1)

        # Get the donut stamps from extra and intra focal images
        # The donut catalogs for each exposure should be the same
        # so we just pick the one for the first exposure
        cameraName = camera.getName()
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx], donutCatalog[0], DefocalType.Extra, cameraName
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx], donutCatalog[0], DefocalType.Intra, cameraName
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=[np.ones(19) * np.nan] * 2,
                outputZernikesAvg=[np.ones(19) * np.nan] * 2,
                donutStampsExtra=[DonutStamps([])] * 2,
                donutStampsIntra=[DonutStamps([])] * 2,
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernikeCoeffsCombined = self.getCombinedZernikes(zernikeCoeffsRaw)

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            outputZernikesAvg=[np.array(zernikeCoeffsCombined.combinedZernikes)] * 2,
            outputZernikesRaw=[np.array(zernikeCoeffsRaw)] * 2,
            donutStampsExtra=[donutStampsExtra] * 2,
            donutStampsIntra=[donutStampsIntra] * 2,
        )
