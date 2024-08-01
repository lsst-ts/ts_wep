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

__all__ = ["forwardModelPair"]

from typing import Tuple, Union

import galsim
import numpy as np
from lsst.ts.wep.image import Image
from lsst.ts.wep.imageMapper import ImageMapper
from lsst.ts.wep.instrument import Instrument
from scipy.ndimage import shift
from scipy.signal import convolve


def forwardModelPair(
    seed: int = 1234,
    zkCoeff: Union[np.ndarray, None] = None,
    zkNorm: float = 1e-6,
    zkMax: float = 1e-6,
    jmax: int = 22,
    fluxIntra: float = 1e8,
    fluxExtra: float = 1e8,
    seeing: Union[float, None] = None,
    skyLevel: Union[float, None] = None,
    bandLabel: str = "r",
    fieldAngleIntra: tuple = (0.23, -0.6),
    fieldAngleExtra: tuple = (0.23, -0.6),
    blendOffsetsIntra: Union[np.ndarray, tuple, list, None] = None,
    blendOffsetsExtra: Union[np.ndarray, tuple, list, None] = None,
    instConfig: Union[str, dict, Instrument] = "policy:instruments/LsstCam.yaml",
    nPix: int = 180,
) -> Tuple[np.ndarray, Image, Image]:
    """Forward model a pair of donuts.

    Parameters
    ----------
    seed : int, optional
        The random seed. (the default is 1234)
    zkCoeff : np.ndarray or None, optional
        A set of Zernike coefficients, in meters. If None, a random
        set of Zernikes are generated using zkNorm, zkMax, jmax.
    zkNorm : float, optional
        Normalization for random Zernike coefficients, in meters. Note
        this parameter is ignored if zkCoeff is not None.
        (the default is 1e-5)
    zkMax : float, optional
        The max absolute value of any Zernike coefficient. Note
        this parameter is ignored if zkCoeff is not None.
        (the default is 1e-6)
    jmax : int, optional
        The maximum Noll index for the random Zernike coefficients. Note
        this parameter is ignored if zkCoeff is not None.
        (the default is 22)
    fluxIntra : int, optional
        Flux of the intrafocal donut.
        (the default is 1e8)
    fluxExtra : int, optional
        Flux of the extrafocal donut.
        (the default is 1e8)
    seeing : float or None, optional
        The FWHM of the Kolmogorov kernel in arcseconds. If None, a random
        value is chosen between 0.3 and 1.5. (the default is None)
    skyLevel : float or None, optional
        Noise level in counts / arcsec^2. If None, a random value is chosen
        between 100 and 10^9 (log-uniform)
        (the default is None)
    bandLabel : str, optional
        The name of the band to simulate donuts in.
        (the default is "r")
    fieldAngleIntra : tuple, optional
        Field angle, in degrees, of the intrafocal donut.
        (the default is (0.23, -0.6))
    fieldAngleExtra : tuple, optional
        Field angle, in degrees, of the extrafocal donut.
        (the default is (0.23, -0.6))
    blendOffsetsIntra : Iterable or None, optional
        The blend offsets of the intrafocal donut.
        (the default is None)
    blendOffsetsExtra : Iterable or None, optional
        The blend offsets of the extrafocal donut.
        (the default is None)
    instConfig : str, dict, or Instrument, optional
        The instrument config for the image mapper.
        (the default is "policy:instruments/LsstCam.yaml")
    nPix : int, optional
        The size of the images. (the default is 180)

    Returns
    -------
    np.ndarray
        Zernike coefficients in meters, for Noll indices 4-jmax (inclusive)
    Image
        The intrafocal image
    Image
        The extrafocal image
    """
    # Create the ImageMapper that will forward model the images
    mapper = ImageMapper(instConfig=instConfig)

    # Generate random Zernikes?
    if zkCoeff is None:
        rng = np.random.default_rng(seed)
        zkCoeff = rng.normal(0, zkNorm / np.arange(1, jmax - 2) ** 1.5, size=jmax - 3)
        zkCoeff = np.clip(zkCoeff, -zkMax, +zkMax)

    # Sample random seeing and skyLevel?
    _seeing = rng.uniform(0.3, 1.5)
    seeing = _seeing if seeing is None else seeing
    _skyLevel = 10 ** rng.uniform(2, 9)
    skyLevel = _skyLevel if skyLevel is None else skyLevel

    # Create a Kolmogorov kernel
    atm = galsim.Kolmogorov(fwhm=seeing)
    atmKernel = atm.drawImage(
        nx=nPix // 2,
        ny=nPix // 2,
        scale=mapper.instrument.pixelScale,
    ).array

    # Calculate background per pixel level from skyLevel
    background = skyLevel * mapper.instrument.pixelScale**2

    # First create the intrafocal image
    intraStamp = mapper.mapPupilToImage(
        Image(
            np.zeros((nPix, nPix)),
            fieldAngleIntra,
            "intra",
            bandLabel,
            blendOffsets=blendOffsetsIntra,
        ),
        zkCoeff,
    )
    # Add blends
    if blendOffsetsIntra is None:
        nBlends = 0
    else:
        nBlends = len(blendOffsetsIntra)
        centralDonut = intraStamp.image.copy()
        for blendShift in blendOffsetsIntra:
            intraStamp.image += shift(centralDonut, blendShift[::-1])
    # Convolve with Kolmogorov atmosphere
    intraStamp.image = convolve(intraStamp.image, atmKernel, mode="same")
    # Normalize the flux
    intraStamp.image *= fluxIntra * (1 + nBlends) / intraStamp.image.sum()
    # Poisson noise
    intraStamp.image += background
    intraStamp.image = rng.poisson(intraStamp.image).astype(float)
    intraStamp.image -= background

    # Now the extrafocal image
    extraStamp = mapper.mapPupilToImage(
        Image(
            np.zeros((nPix, nPix)),
            fieldAngleExtra,
            "extra",
            bandLabel,
            blendOffsets=blendOffsetsExtra,
        ),
        zkCoeff,
    )
    # Add blends
    if blendOffsetsExtra is None:
        nBlends = 0
    else:
        nBlends = len(blendOffsetsExtra)
        centralDonut = extraStamp.image.copy()
        for blendShift in blendOffsetsExtra:
            extraStamp.image += shift(centralDonut, blendShift[::-1])
    # Convolve with Kolmogorov atmosphere
    extraStamp.image = convolve(extraStamp.image, atmKernel, mode="same")
    # Normalize the flux
    extraStamp.image *= fluxExtra / extraStamp.image.sum()
    # Poisson noise
    extraStamp.image += background
    extraStamp.image = rng.poisson(extraStamp.image).astype(float)
    extraStamp.image -= background

    return zkCoeff, intraStamp, extraStamp
