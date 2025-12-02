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

__all__ = ["EstimateZernikesTieConfig", "EstimateZernikesTieTask"]

import lsst.pex.config as pexConfig
from lsst.ts.wep.task.estimateZernikesBase import (
    EstimateZernikesBaseConfig,
    EstimateZernikesBaseTask,
)
from lsst.ts.wep.utils import WfAlgorithmName


class EstimateZernikesTieConfig(EstimateZernikesBaseConfig):
    """TIE-specific configuration parameters for Zernike estimation."""

    opticalModel: pexConfig.ChoiceField = pexConfig.ChoiceField(
        dtype=str,
        default="offAxis",
        doc="The optical model to use for mapping between the image and pupil"
        + "planes. Can be 'offAxis', 'onAxis', or 'paraxial'. offAxis is a"
        + "numerical model that is valid for all optical systems, but requires"
        + "an accurate Batoid model. onAxis is an analytic model that is valid"
        + "for all optical systems near the optical axis. paraxial is an"
        + "analytic model that is valid for slow optical systems near the"
        + "optical axis. offAxis is recommended when you have a Batoid model"
        + "and onAxis is recommended when you do not. paraxial is primarily"
        + "meant for testing (the default is 'offAxis')",
        allowed={
            "offAxis": "Numerical model fit by Batoid telescope model.",
            "onAxis": "Analytic model only suitable for small field angles.",
            "paraxial": "Analytic model only suitable for slow optical systems.",
        },
    )
    maxIter: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=30,
        doc="Maximum number of iterations for the TIE loop. (the default is 30)",
    )
    compSequence: pexConfig.ListField = pexConfig.ListField(
        dtype=int,
        default=[4, 4, 6, 6, 13, 13, 13, 13, 22, 22, 22, 22],
        doc="Max Noll index to compensate during each iteration of TIE. "
        + "Once the end of the sequence is reached, all Zernike coefficients "
        + "are used during compensation. "
        + "(the default is [4, 4, 6, 6, 13, 13, 13, 13, 22, 22, 22, 22])",
    )
    compGain: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.6,
        doc="Gain used to update Zernikes for image compensation. " + "(the default is 0.6)",
    )
    centerTol: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=1e-9,
        doc="Maximum absolute change in any Zernike coefficient (in meters) "
        + "for which the images will be recentered. If 0, the images are "
        + "recentered on every iteration. (the default is 1e-9)",
    )
    centerBinary: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to use a binary template when centering the image. " + "(the default is True)",
    )
    convergeTol: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=1e-9,
        doc="The maximum absolute change in any Zernike amplitude (in meters) "
        + "between subsequent TIE iterations below which convergence is declared. "
        + "(the default is 1e-9)",
    )
    maskKwargs: pexConfig.DictField = pexConfig.DictField(
        keytype=str,
        default=dict(),
        doc="Dictionary of mask keyword arguments to pass to mask creation. "
        + "To see possibilities, see docstring for "
        + "lsst.ts.wep.imageMapper.ImageMapper.createPupilMasks(). "
        + "(the default is an emtpy dictionary)",
    )
    modelPupilKernelSize: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=2,
        doc="The size of the Gaussian kernel to convolve with the model pupil "
        + "when estimating Zernikes with a single donut. "
        + "(the default is 2)",
    )
    binning: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=1,
        doc="Binning factor to apply to the donut stamps before estimating "
        + "Zernike coefficients. A value of 1 means no binning.",
    )
    requireConverge: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to require that the TIE converges. "
        + "If True, and the TIE did not converge, the TIE returns NaNs. ",
    )


class EstimateZernikesTieTask(EstimateZernikesBaseTask):
    """Estimate Zernike coefficients using the TIE algorithm."""

    ConfigClass = EstimateZernikesTieConfig

    @property
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum."""
        return WfAlgorithmName.TIE
