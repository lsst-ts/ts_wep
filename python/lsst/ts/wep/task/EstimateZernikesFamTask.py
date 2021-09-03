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


class EstimateZernikesFamTaskConnections(
    EstimateZernikesBaseConnections, dimensions=("detector", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )


class EstimateZernikesFamTaskConfig(
    EstimateZernikesBaseConfig, pipelineConnections=EstimateZernikesFamTaskConnections
):
    pass


class EstimateZernikesFamTask(EstimateZernikesBaseTask):
    """
    Run Zernike Estimation in full-array mode (FAM)
    """

    ConfigClass = EstimateZernikesFamTaskConfig
    _DefaultName = "EstimateZernikesFamTask"

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

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We implement a runQuantum method to make sure our configured
        task runs with the instrument required by the pipeline.
        """

        # Get the instrument we are running the pipeline with
        cameraName = inputRefs.exposures[0].dataId["instrument"]

        # Get the input reference objects for the task
        exposures = butlerQC.get(inputRefs.exposures)
        donutCat = butlerQC.get(inputRefs.donutCatalog)

        # Run task on specified instrument
        outputs = self.run(exposures, donutCat, cameraName)

        # Use butler to store output in repository
        butlerQC.put(outputs, outputRefs)

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

    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalog: pd.DataFrame,
        cameraName: str,
    ) -> pipeBase.Struct:

        # Get exposure metadata to find which is extra and intra
        focusZ0 = exposures[0].getMetadata()["FOCUSZ"]
        focusZ1 = exposures[1].getMetadata()["FOCUSZ"]

        extraExpIdx, intraExpIdx = self.assignExtraIntraIdx(focusZ0, focusZ1)

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx], donutCatalog, DefocalType.Extra, cameraName
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx], donutCatalog, DefocalType.Intra, cameraName
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0:
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
