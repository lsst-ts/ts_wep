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

__all__ = ["EstimateZernikesBaseConfig", "EstimateZernikesBaseTask"]

import abc

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.ts.wep.estimation import WfAlgorithm, WfAlgorithmFactory, WfEstimator
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import WfAlgorithmName, getInstrumentFromButlerName


class EstimateZernikesBaseConfig(pexConfig.Config):
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
        dtype=int,
        default=22,
        doc="The maximum Zernike Noll index estimated.",
    )
    startWithIntrinsic = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to start Zernike estimation from the intrinsic Zernikes.",
    )
    returnWfDev = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="If True, returns wavefront deviation. If False, returns full OPD.",
    )
    return4Up = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="If True, the returned Zernike coefficients start with Noll index 4. "
        + "If False, they follow the Galsim convention of starting with index 0 "
        + "(which is meaningless), so the array index of the output corresponds "
        + "to the Noll index. In this case, indices 0-3 are always set to zero, "
        + "because they are not estimated by our pipeline.",
    )
    units = pexConfig.ChoiceField(
        dtype=str,
        default="um",
        doc="Units in which the Zernike coefficients are returned.",
        allowed={
            "m": "meters",
            "um": "microns",
            "nm": "nanometers",
            "arcsecs": "quadrature contribution to the PSF FWHM in arcseconds",
        },
    )


class EstimateZernikesBaseTask(pipeBase.Task, metaclass=abc.ABCMeta):
    """Base class for estimating Zernike coefficients from DonutStamps."""

    ConfigClass = EstimateZernikesBaseConfig
    _DefaultName = "estimateZernikes"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        """# Set instrument configuration info
        self.instObscuration = self.config.instObscuration
        self.instFocalLength = self.config.instFocalLength
        self.instApertureDiameter = self.config.instApertureDiameter
        self.instDefocalOffset = self.config.instDefocalOffset
        self.instPixelSize = self.config.instPixelSize"""

    @property
    @abc.abstractmethod
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum from the subclass."""
        ...

    @property
    def wfAlgoConfig(self) -> WfAlgorithm:
        """Return the configuration for the WfAlgorithm."""
        algoConfig = {
            key: val
            for key, val in self.config.toDict().items()
            if key not in EstimateZernikesBaseConfig._fields.keys()
        }

        return WfAlgorithmFactory.createWfAlgorithm(self.wfAlgoName, algoConfig)

    def run(
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
        `lsst.pipe.base.Struct`
            A struct containing "zernikes", which is a 2D numpy array,
            where the first axis indexes the pair of DonutStamps and the
            second axis indexes the Zernikes coefficients.
        """
        # Load the default instrument for the camera
        instrument = getInstrumentFromButlerName(donutStampsExtra[0].cam_name)

        """# Determine the defocal offset
        if self.instDefocalOffset is None:
            defocalOffset = donutStampsExtra.getDefocalDistances()[0]
        else:
            defocalOffset = self.instDefocalOffset"""

        """# Update the instrument with the config values
        instrument.obscuration = self.instObscuration
        instrument.focalLength = self.instFocalLength
        instrument.diameter = self.instApertureDiameter
        instrument.defocalOffset = defocalOffset * 1e-3
        instrument.pixelSize = self.instPixelSize"""

        # Create the wavefront estimator
        wfEst = WfEstimator(
            algoName=self.wfAlgoName,
            algoConfig=self.wfAlgoConfig,
            instConfig=instrument,
            jmax=self.config.maxNollIndex,
            startWithIntrinsic=self.config.startWithIntrinsic,
            returnWfDev=self.config.returnWfDev,
            return4Up=self.config.return4Up,
            units=self.config.units,
            saveHistory=False,
        )

        # Loop over donut stamps and estimate Zernikes
        zkList = []
        for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra):
            zk = wfEst.estimateZk(donutExtra.wep_im, donutIntra.wep_im)
            zkList.append(zk)

        return pipeBase.Struct(zernikes=np.array(zkList))
