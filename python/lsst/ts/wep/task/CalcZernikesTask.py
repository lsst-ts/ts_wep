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

__all__ = ["CalcZernikesTaskConnections", "CalcZernikesTaskConfig", "CalcZernikesTask"]

import os
import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod

from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.Utility import (
    getConfigDir,
    DefocalType,
    getCamTypeFromButlerName,
    createInstDictFromConfig,
    rotMatrix,
)
from lsst.ts.wep.task.CombineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from scipy.ndimage import rotate


class CalcZernikesTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("visit", "detector", "instrument")
):
    donutStampsExtra = connectionTypes.Input(
        doc="Extra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
    )
    donutStampsIntra = connectionTypes.Input(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
    )
    outputZernikesRaw = connectionTypes.Output(
        doc="Zernike Coefficients from all donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateRaw",
    )
    outputZernikesAvg = connectionTypes.Output(
        doc="Zernike Coefficients averaged over donuts",
        dimensions=("visit", "detector", "instrument"),
        storageClass="NumpyArray",
        name="zernikeEstimateAvg",
    )


class CalcZernikesTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=CalcZernikesTaskConnections
):
    # Config setting for pipeline task with defaults
    combineZernikes = pexConfig.ConfigurableField(
        target=CombineZernikesSigmaClipTask,
        doc=str(
            "Choice of task to combine the Zernikes from pairs of "
            + "donuts into a single value for the detector. (The default "
            + "is CombineZernikesSigmaClipTask.)"
        ),
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, paraxial, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instObscuration = pexConfig.Field(
        doc="Obscuration (inner_radius / outer_radius of M1M3)",
        dtype=float,
        default=0.61,
    )
    instFocalLength = pexConfig.Field(
        doc="Instrument Focal Length in m", dtype=float, default=10.312
    )
    instApertureDiameter = pexConfig.Field(
        doc="Instrument Aperture Diameter in m", dtype=float, default=8.36
    )
    instDefocalOffset = pexConfig.Field(
        doc="Instrument defocal offset in mm. \
        If None then will get this from the focusZ value in exposure visitInfo. \
        (The default is None.)",
        dtype=float,
        default=None,
        optional=True,
    )
    instPixelSize = pexConfig.Field(
        doc="Instrument Pixel Size in m", dtype=float, default=10.0e-6
    )
    transposeImages = pexConfig.Field(
        doc="Specify whether to transpose the intra- and extra-focal images. \
        (The default is to do the transpose).",
        dtype=bool,
        default=True,
        optional=True,
    )


class CalcZernikesTask(pipeBase.PipelineTask):
    """
    Run Zernike Estimation on corner wavefront sensors (CWFS)
    """

    ConfigClass = CalcZernikesTaskConfig
    _DefaultName = "CalcZernikesTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choice of task to combine the Zernike coefficients
        # from individual pairs of donuts into a single array
        # for the detector.
        self.combineZernikes = self.config.combineZernikes
        self.makeSubtask("combineZernikes")
        # Specify whether to transpose images
        self.transposeImages = self.config.transposeImages
        # Specify optical model
        self.opticalModel = self.config.opticalModel
        # Set up instrument configuration dict
        self.instParams = createInstDictFromConfig(self.config)

    def calcBlendOffsets(self, donutStamp, eulerAngle):
        """
        Calculate the offsets between the center of the donutStamp
        image and the centers of blended donuts appearing on the
        stamp image. Include rotations for rotated wavefront
        sensors.

        Parameters
        ----------
        donutStamp : DonutStamp
            Extra or intra-focal DonutStamp object.
        eulerAngle : float
            Angle of rotation of sensor compared to the
            standard alignment of the focal plane.

        Returns
        -------
        numpy.ndarray
            Offsets of blended donuts compared to center of
            DonutStamp postage stamp image.
        """

        blendCentroids = donutStamp.blend_centroid_positions
        # If there are blends the array will not have nans
        if np.sum(np.isnan(blendCentroids)) == 0:
            blendOffsets = blendCentroids - donutStamp.centroid_position
            blendOffsets = np.dot(blendOffsets, rotMatrix(eulerAngle))
            # Exchange X,Y since we transpose the image below
            blendOffsets = blendOffsets.T[::-1]
        else:
            # If no blend then pass nan array.
            # CompensableImage understands nans means no blend.
            blendOffsets = blendCentroids.T

        return blendOffsets

    def estimateZernikes(self, donutStampsExtra, donutStampsIntra):
        """
        Take the donut postage stamps and estimate the Zernike coefficients.

        Parameters
        ----------
        donutStampsExtra : DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra : DonutStamps
            Intra-focal donut postage stamps.

        Returns
        -------
        numpy.ndarray
            Zernike coefficients for the exposure. Will return one set of
            coefficients per set of stamps, not one set of coefficients
            per detector so this will be a 2-D numpy array with
            the number of rows equal to the number of donut stamps and
            the number of columns equal to the number of Zernike coefficients.
        """

        zerArray = []

        configDir = getConfigDir()
        algoDir = os.path.join(configDir, "cwfs", "algo")
        wfEsti = WfEstimator(algoDir)
        detectorNames = donutStampsExtra.getDetectorNames()
        camera = donutStampsExtra[0].getCamera()
        detectorType = camera[detectorNames[0]].getType()
        self.instParams["offset"] = donutStampsExtra.getDefocalDistances()[0]

        camType = getCamTypeFromButlerName(camera.getName(), detectorType)
        donutStampSize = np.shape(donutStampsExtra[0].stamp_im.getImage().getArray())[0]

        wfEsti.config(
            self.instParams,
            sizeInPix=donutStampSize,
            camType=camType,
            opticalModel=self.opticalModel,
        )

        for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra):

            fieldXYExtra = donutExtra.calcFieldXY()
            fieldXYIntra = donutIntra.calcFieldXY()

            camera = donutExtra.getCamera()
            detectorExtra = camera.get(donutExtra.detector_name)
            detectorIntra = camera.get(donutIntra.detector_name)

            # Rotate any sensors that are not lined up with the focal plane.
            # Mostly just for the corner wavefront sensors. The negative sign
            # creates the correct rotation based upon closed loop tests
            # with R04 and R40 corner sensors.
            eulerZExtra = -detectorExtra.getOrientation().getYaw().asDegrees()
            eulerZIntra = -detectorIntra.getOrientation().getYaw().asDegrees()

            # NOTE: TS_WEP expects these images to be transposed
            # TODO: Look into this
            blendOffsetsExtra = self.calcBlendOffsets(donutExtra, eulerZExtra)
            blendOffsetsIntra = self.calcBlendOffsets(donutIntra, eulerZIntra)

            if self.transposeImages:
                imageExtra = rotate(donutExtra.stamp_im.getImage().getArray(), eulerZExtra).T
                imageIntra = rotate(donutIntra.stamp_im.getImage().getArray(), eulerZIntra).T
            else:
                imageExtra = rotate(donutExtra.stamp_im.getImage().getArray(), eulerZExtra)
                imageIntra = rotate(donutIntra.stamp_im.getImage().getArray(), eulerZIntra)

            wfEsti.setImg(
                fieldXYExtra,
                DefocalType.Extra,
                image=imageExtra,
                blendOffsets=blendOffsetsExtra.tolist(),
            )
            wfEsti.setImg(
                fieldXYIntra,
                DefocalType.Intra,
                image=imageIntra,
                blendOffsets=blendOffsetsIntra.tolist(),
            )
            wfEsti.reset()
            zer4UpNm = wfEsti.calWfsErr()
            zer4UpMicrons = zer4UpNm * 1e-3

            zerArray.append(zer4UpMicrons)

        return np.array(zerArray)

    def getCombinedZernikes(self, zernikeArray):
        """
        Combine the Zernike coefficients from stamp pairs on the
        CCD to create one final value for the CCD.

        Parameters
        ----------
        zernikeArray : numpy.ndarray
            The full set of zernike coefficients for each pair
            of donuts on the CCD. Each row of the array should
            be the set of Zernike coefficients for a single
            donut pair.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:

            - combinedZernikes : numpy.ndarray
                The final combined Zernike coefficients from the CCD.
            - combineFlags : numpy.ndarray
                Flag indicating a particular set of Zernike
                coefficients was not used in the final estimate.
                If the values in a row in the `zernikeArray`
                were used then its index is 0.
                A value of 1 means the coefficients from that row
                in the input `zernikeArray` were not used.
        """

        return self.combineZernikes.run(zernikeArray)

    @timeMethod
    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
    ) -> pipeBase.Struct:

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if (len(donutStampsExtra) == 0) or (len(donutStampsIntra) == 0):
            return pipeBase.Struct(
                outputZernikesRaw=np.ones(19) * np.nan,
                outputZernikesAvg=np.ones(19) * np.nan,
            )

        # Estimate Zernikes from collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernikeCoeffsCombined = self.getCombinedZernikes(zernikeCoeffsRaw)

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            outputZernikesAvg=np.array(zernikeCoeffsCombined.combinedZernikes),
            outputZernikesRaw=np.array(zernikeCoeffsRaw),
        )
