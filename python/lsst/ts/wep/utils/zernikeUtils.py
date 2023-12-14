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
    "createGalsimZernike",
    "createZernikeBasis",
    "createZernikeGradBasis",
    "zernikeEval",
    "zernikeGradEval",
    "getPsfGradPerZernike",
    "convertZernikesToPsfWidth",
    "getZernikeParity",
]

import re
from typing import Tuple

import galsim
import numpy as np


def createGalsimZernike(
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> galsim.zernike.Zernike:
    """Create a GalSim Zernike object with the given coefficients.

    Parameters
    ----------
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    galsim.zernike.Zernike
        A GalSim Zernike object
    """
    return galsim.zernike.Zernike(
        np.concatenate([np.zeros(4), zkCoeff]), R_inner=obscuration
    )


def createZernikeBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmax: int = 28,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Create a basis of Zernike polynomials for Noll indices >= 4.

    This function is evaluated on a grid of normalized pupil coordinates,
    where these coordinates are normalized pupil coordinates. Normalized
    pupil coordinates are defined such that u^2 + v^2 = 1 is the edge of
    the pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.
    """
    # Create the basis
    return galsim.zernike.zernikeBasis(jmax, u, v, R_inner=obscuration)[4:]


def createZernikeGradBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmax: int = 28,
    obscuration: float = 0.61,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a basis of Zernike gradient polynomials for Noll indices >= 4."""
    # Get the Noll coefficient array for the u derivates
    nollCoeffU = galsim.zernike._noll_coef_array_xy_gradx(jmax, obscuration)
    nollCoeffU = nollCoeffU[:, :, 3:]  # Keep only Noll indices >= 4

    # Evaluate the polynomials
    dzkdu = np.array(
        [
            galsim.utilities.horner2d(u, v, nc, dtype=float)
            for nc in nollCoeffU.transpose(2, 0, 1)
        ]
    )

    # Repeat for v
    nollCoeffV = galsim.zernike._noll_coef_array_xy_grady(jmax, obscuration)
    nollCoeffV = nollCoeffV[:, :, 3:]  # Keep only Noll indices >= 4
    dzkdv = np.array(
        [
            galsim.utilities.horner2d(u, v, nc, dtype=float)
            for nc in nollCoeffV.transpose(2, 0, 1)
        ]
    )

    return dzkdu, dzkdv


def zernikeEval(
    u: np.ndarray,
    v: np.ndarray,
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> None:
    """Evaluate the Zernike series.

    This function is evaluated at the provided u and v coordinates, where
    these coordinates are normalized pupil coordinates. Normalized pupil
    coordinates are defined such that u^2 + v^2 = 1 is the edge of the
    pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, obscuration)

    # Evaluate on the grid
    return galsimZernike(u, v)


def zernikeGradEval(
    u: np.ndarray,
    v: np.ndarray,
    uOrder: int,
    vOrder: int,
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Evaluate the gradient of the Zernike series.

    This function is evaluated at the provided u and v coordinates, where
    these coordinates are normalized pupil coordinates. Normalized pupil
    coordinates are defined such that u^2 + v^2 = 1 is the edge of the
    pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    uOrder : int
        The number of u derivatives to apply.
    vOrder : int
        The number of v derivatives to apply.
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, obscuration)

    # Apply derivatives
    for _ in range(uOrder):
        galsimZernike = galsimZernike.gradX
    for _ in range(vOrder):
        galsimZernike = galsimZernike.gradY

    # Evaluate on the grid
    return galsimZernike(u, v)


def getPsfGradPerZernike(
    diameter: float = 8.36,
    obscuration: float = 0.612,
    jmin: int = 4,
    jmax: int = 22,
) -> np.ndarray:
    """Get the gradient of the PSF FWHM with respect to each Zernike.

    This function takes no positional arguments. All parameters must be passed
    by name (see the list of parameters below).

    Parameters
    ----------
    diameter : float
        The diameter of the telescope aperture, in meters.
        (the default, 8.36, corresponds to the LSST primary mirror)
    obscuration : float
        The central obscuration of the telescope aperture (i.e. R_outer / R_inner).
        (the default, 0.612, corresponds to the LSST primary mirror)
    jmin : int
        The minimum Zernike Noll index, inclusive.
        (the default, 4, ignores piston, x & y offsets, and tilt.)
    jmax : int
        The max Zernike Noll index, inclusive. (the default is 22.)

    Returns
    -------
    np.ndarray
        Gradient of the PSF FWHM with respect to the corresponding Zernike.
        Units are arcsec / micron.
    """
    # Calculate the conversion factors
    conversion_factors = np.zeros(jmax + 1)
    for i in range(jmin, jmax + 1):
        # Set coefficients for this Noll index: coefs = [0, 0, ..., 1]
        # Note the first coefficient is Noll index 0, which does not exist and
        # is therefore always ignored by galsim
        coefs = [0] * i + [1]

        # Create the Zernike polynomial with these coefficients
        R_outer = diameter / 2
        R_inner = R_outer * obscuration
        Z = galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)

        # We can calculate the size of the PSF from the RMS of the gradient of
        # the wavefront. The gradient of the wavefront perturbs photon paths.
        # The RMS quantifies the size of the collective perturbation.
        # If we expand the wavefront gradient in another series of Zernike
        # polynomials, we can exploit the orthonormality of the Zernikes to
        # calculate the RMS from the Zernike coefficients.
        rms_tilt = np.sqrt(np.sum(Z.gradX.coef**2 + Z.gradY.coef**2) / 2)

        # Convert to arcsec per micron
        rms_tilt = np.rad2deg(rms_tilt * 1e-6) * 3600

        # Convert rms -> fwhm
        fwhm_tilt = 2 * np.sqrt(2 * np.log(2)) * rms_tilt

        # Save this conversion factor
        conversion_factors[i] = fwhm_tilt

    return conversion_factors[jmin:]


def convertZernikesToPsfWidth(
    zernikes: np.ndarray,
    diameter: float = 8.36,
    obscuration: float = 0.612,
    jmin: int = 4,
) -> np.ndarray:
    """Convert Zernike amplitudes to quadrature contribution to the PSF FWHM.

    Parameters
    ----------
    zernikes : np.ndarray
        Zernike amplitudes (in microns), starting with Noll index `jmin`.
        Either a 1D array of zernike amplitudes, or a 2D array, where each row
        corresponds to a different set of amplitudes.
    diameter : float
        The diameter of the telescope aperture, in meters.
        (the default, 8.36, corresponds to the LSST primary mirror)
    obscuration : float
        The central obscuration of the telescope aperture (i.e. R_outer / R_inner).
        (the default, 0.612, corresponds to the LSST primary mirror)
    jmin : int
        The minimum Zernike Noll index, inclusive. The maximum Noll index is
        inferred from `jmin` and the length of `zernikes`.
        (the default is 4, which ignores piston, x & y offsets, and tilt.)

    Returns
    -------
    dFWHM: np.ndarray
        Quadrature contribution of each Zernike vector to the PSF FWHM
        (in arcseconds).

    Notes
    -----
    Converting Zernike amplitudes to their quadrature contributions to the PSF
    FWHM allows for easier physical interpretation of Zernike amplitudes and
    the performance of the AOS system.

    For example, image we have a true set of zernikes, [Z4, Z5, Z6], such that
    ConvertZernikesToPsfWidth([Z4, Z5, Z6]) = [0.1, -0.2, 0.3] arcsecs.
    These Zernike perturbations increase the PSF FWHM by
    sqrt[(0.1)^2 + (-0.2)^2 + (0.3)^2] ~ 0.37 arcsecs.

    If the AOS perfectly corrects for these perturbations, the PSF FWHM will
    not increase in size. However, imagine the AOS estimates zernikes, such
    that ConvertZernikesToPsfWidth([Z4, Z5, Z6]) = [0.1, -0.3, 0.4] arcsecs.
    These estimated Zernikes, do not exactly match the true Zernikes above.
    Therefore, the post-correction PSF will still be degraded with respect to
    the optimal PSF. In particular, the PSF FWHM will be increased by
    sqrt[(0.1 - 0.1)^2 + (-0.2 - (-0.3))^2 + (0.3 - 0.4)^2] ~ 0.14 arcsecs.

    This conversion depends on a linear approximation that begins to break down
    for RSS(dFWHM) > 0.20 arcsecs. Beyond this point, the approximation tends
    to overestimate the PSF degradation. In other words, if
    sqrt(sum( dFWHM^2 )) > 0.20 arcsec, it is likely that dFWHM is
    over-estimated. However, the point beyond which this breakdown begins
    (and whether the approximation over- or under-estimates dFWHM) can change,
    depending on which Zernikes have large amplitudes. In general, if you have
    large Zernike amplitudes, proceed with caution!
    Note that if the amplitudes Z_est and Z_true are large, this is okay, as
    long as |Z_est - Z_true| is small.

    For a notebook demonstrating where the approximation breaks down:
    https://gist.github.com/jfcrenshaw/24056516cfa3ce0237e39507674a43e1
    """
    # Calculate jmax from jmin and the length of the zernike array
    jmax = jmin + zernikes.shape[-1] - 1

    # Calculate the conversion factors for each zernike
    conversion_factors = getPsfGradPerZernike(
        jmin=jmin,
        jmax=jmax,
        diameter=diameter,
        obscuration=obscuration,
    )

    # Convert the Zernike amplitudes from microns to their quadrature
    # contribution to the PSF FWHM
    dFWHM = conversion_factors * zernikes

    return dFWHM


def getZernikeParity(jmax, axis="x"):
    """Return the parity of the Zernike polynomials (Noll index >= 4).

    Parameters
    ----------
    jmax : int
        The maximum Noll index
    axis : str, optional
        The axis for which to return the parity. Can be "x" or "y".

    Returns
    -------
    np.ndarray
        The numpy array of parities, starting with Noll index 4.
        +1 corresponds to even parity, while -1 corresponds to odd parity.

    Raises
    ------
    ValueError
        If axis is not one of "x" or "y"
    ValueError
        If one of the Noll indices <= jmax has mixed parity
    """
    if axis not in ["x", "y"]:
        raise ValueError("axis must be either 'x' or 'y'.")

    parity = []
    for j in range(4, jmax + 1):
        # Get the cartesian string
        zkStr = galsim.zernike.describe_zernike(j)

        # Get all exponents
        exponents = re.findall(f"(?<={axis}\^)[0-9]+|{axis}(?!\^)", zkStr) or [0]
        exponents = np.array([1 if num == axis else int(num) for num in exponents])

        # Get the remainders mod 2
        remainder = exponents % 2

        # Check they all match
        allMatch = all(remainder) or all(~remainder)
        if not allMatch:
            raise ValueError(f"Noll index {j} has mixed parity")

        # Get the parity: +1 = even, -1 = odd
        parity.append(-2 * int(all(remainder)) + 1)

    return np.array(parity)
