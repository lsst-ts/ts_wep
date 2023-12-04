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

import abc

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.pipe.base import connectionTypes
from lsst.ts.wep.estimation import WfAlgorithm, WfEstimator
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import WfAlgorithmName, getInstrumentFromButlerName
from lsst.utils.timer import timeMethod

__all__ = [
    "CalcZernikesTaskConnections",
    "CalcZernikesTaskConfig",
    "CalcZernikesBaseTask",
]


class CalcZernikesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),
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
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CalcZernikesTaskConnections,
):
    combineZernikes = pexConfig.ConfigurableField(
        target=CombineZernikesSigmaClipTask,
        doc=str(
            "Choice of task to combine the Zernikes from pairs of "
            + "donuts into a single value for the detector. (The default "
            + "is CombineZernikesSigmaClipTask.)"
        ),
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
    maxNollIndex = pexConfig.Field(
        doc="The maximum Zernike Noll index estimated.",
        dtype=int,
        default=22,
    )


class CalcZernikesBaseTask(pipeBase.PipelineTask, metaclass=abc.ABCMeta):
    """Base class for calculating Zernike coefficients from pairs of DonutStamps."""

    ConfigClass = CalcZernikesTaskConfig
    _DefaultName = "calcZernikesBaseTask"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Choice of task to combine the Zernike coefficients
        # from individual pairs of donuts into a single array
        # for the detector.
        self.combineZernikes = self.config.combineZernikes
        self.makeSubtask("combineZernikes")

        # Set instrument configuration info
        self.instObscuration = self.config.instObscuration
        self.instFocalLength = self.config.instFocalLength
        self.instApertureDiameter = self.config.instApertureDiameter
        self.instDefocalOffset = self.config.instDefocalOffset
        self.instPixelSize = self.config.instPixelSize

        # Set the maximum Zernike Noll index estimated
        self.maxNollIndex = self.config.maxNollIndex

    @property
    @abc.abstractmethod
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum from the subclass."""
        pass

    @property
    @abc.abstractmethod
    def wfAlgo(self) -> WfAlgorithm:
        """Return the WfAlgorithm that is used for Zernike estimation."""
        pass

    def estimateZernikes(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
    ) -> np.ndarray:
        """Estimate Zernike coefficients from the donut stamps.

        Parameters
        ----------
        donutStampsExtra : DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra : DonutStamps
            Intra-focal donut postage stamps.

        Returns
        -------
        np.ndarray
            2D numpy array. The first axis indexes the pair of DonutStamps,
            and the second axis indexes the Zernike coefficients.
        """
        # Load the default instrument for the camera
        instrument = getInstrumentFromButlerName(donutStampsExtra[0].cam_name)

        # Determine the defocal offset
        if self.instDefocalOffset is None:
            defocalOffset = donutStampsExtra.getDefocalDistances()[0]
        else:
            defocalOffset = self.instDefocalOffset

        # Update the instrument with the config values
        instrument.obscuration = self.instObscuration
        instrument.focalLength = self.instFocalLength
        instrument.diameter = self.instApertureDiameter
        instrument.defocalOffset = defocalOffset * 1e-3
        instrument.pixelSize = self.instPixelSize

        # Create the wavefront estimator
        wfEst = WfEstimator(
            configFile=None,
            algoName=self.wfAlgoName,
            algoConfig=self.wfAlgo,
            instConfig=instrument,
            jmax=self.maxNollIndex,
            units="um",  # Return Zernikes in microns
        )

        # Loop over donut stamps and estimate Zernikes
        zkList = []
        for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra):
            zk = wfEst.estimateZk(donutExtra.wep_im, donutIntra.wep_im)
            zkList.append(zk)

        return np.array(zkList)

    def getCombinedZernikes(self, zernikeArray: np.ndarray) -> np.ndarray:
        """Combine Zernike coefficients from all pairs to create single estimate.

        Parameters
        ----------
        zernikeArray : np.ndarray
            Array of Zernike coefficients for each pair of donuts.
            Each row of the array corresponds to a single pair.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - combinedZernikes : np.ndarray
                    The final combined Zernike coefficients
                - combineFlags : np.ndarray
                    Flag indicating whether each set of coefficients was
                    rejected when combining all the Zernikes. A value of
                    1 means that set of Zernikes was not used, while a 0
                    indicates that set of Zernikes was used.
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
        if len(donutStampsExtra) == 0 or len(donutStampsIntra) == 0:
            return pipeBase.Struct(
                outputZernikesRaw=np.full(19, np.nan),
                outputZernikesAvg=np.full(19, np.nan),
            )

        # Estimate Zernikes from the collection of stamps
        zernikeCoeffsRaw = self.estimateZernikes(donutStampsExtra, donutStampsIntra)
        zernikeCoeffsCombined = self.getCombinedZernikes(zernikeCoeffsRaw)

        return pipeBase.Struct(
            outputZernikesAvg=np.array(zernikeCoeffsCombined.combinedZernikes),
            outputZernikesRaw=np.array(zernikeCoeffsRaw),
        )
