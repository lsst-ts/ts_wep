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
    "zernikeFit",
    "getPsfGradPerZernike",
    "convertZernikesToPsfWidth",
    "getZernikeParity",
    "makeSparse",
    "makeDense",
    "checkNollIndices",
]

from typing import Optional

import galsim
import numpy as np


def createGalsimZernike(
    zkCoeff: np.ndarray,
    jmin: int = 4,
    obscuration: float = 0.612,
) -> galsim.zernike.Zernike:
    """Create a GalSim Zernike object with the given coefficients.

    Parameters
    ----------
    zkCoeff : np.ndarray
        Zernike coefficients in any units.
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    galsim.zernike.Zernike
        A GalSim Zernike object

    Raises
    ------
    ValueError
        If jmin is negative
    """
    # Check jmin
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")

    return galsim.zernike.Zernike(np.concatenate([np.zeros(jmin), zkCoeff]), R_inner=obscuration)


def createZernikeBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmin: int = 4,
    jmax: int = 22,
    obscuration: float = 0.612,
) -> np.ndarray:
    """Create a basis of Zernike polynomials.

    This function is evaluated on a grid of normalized pupil coordinates,
    where these coordinates are normalized pupil coordinates. Normalized
    pupil coordinates are defined such that u^2 + v^2 = 1 is the edge of
    the pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    jmax : int
        The maximum Noll index to fit, inclusive. Must be >= jmin.
        (the default is 22)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Zernike bases. The first axis indexes the Zernike polynomials.

    Raises
    ------
    ValueError
        If jmin is negative or jmax is less than jmin
    """
    # Check jmin and jmax
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")

    # Create the basis
    return galsim.zernike.zernikeBasis(jmax, u, v, R_inner=obscuration)[jmin:]


def createZernikeGradBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmin: int = 4,
    jmax: int = 22,
    obscuration: float = 0.612,
) -> np.ndarray:
    """Create a basis of Zernike gradient polynomials.

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
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    jmax : int
        The maximum Noll index to fit, inclusive. Must be >= jmin.
        (the default is 22)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Array of Zernike bases. First axis has length 2, corresponding to the
        u and v gradients. The second axis indexes the Zernike polynomials.

    Raises
    ------
    ValueError
        If jmin is negative or jmax is less than jmin
    """
    # Check jmin and jmax
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")

    gradBasis = galsim.zernike.zernikeGradBases(jmax, u, v, R_inner=obscuration)
    return gradBasis[:, jmin:, ...]


def zernikeEval(
    u: np.ndarray,
    v: np.ndarray,
    zkCoeff: np.ndarray,
    jmin: int = 4,
    obscuration: float = 0.612,
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
        Zernike coefficients in any units.
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.

    Raises
    ------
    ValueError
        If jmin is negative
    """
    # Check jmin
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")

    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, jmin, obscuration)

    # Evaluate on the grid
    return galsimZernike(u, v)


def zernikeGradEval(
    u: np.ndarray,
    v: np.ndarray,
    uOrder: int,
    vOrder: int,
    zkCoeff: np.ndarray,
    jmin: int = 4,
    obscuration: float = 0.612,
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
        Zernike coefficients in any units.
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.

    Raises
    ------
    ValueError
        If jmin is negative
    """
    # Check jmin
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")

    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, jmin, obscuration)

    # Apply derivatives
    for _ in range(uOrder):
        galsimZernike = galsimZernike.gradX
    for _ in range(vOrder):
        galsimZernike = galsimZernike.gradY

    # Evaluate on the grid
    return galsimZernike(u, v)


def zernikeFit(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    jmin: int = 4,
    jmax: int = 22,
    obscuration: float = 0.612,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit Zernike polynomials to the surface.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    z : np.ndarray
        The wavefront surface evaluated at the u, v points.
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    jmax : int
        The maximum Noll index to fit, inclusive. Must be >= jmin.
        (the default is 22)
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.612, corresponding to the Simonyi Survey Telescope.)
    mask : np.ndarray, optional
        A mask for the surface. The Zernikes are only fit to the unmasked
        points. (the default is None)

    Returns
    -------
    np.ndarray
        The best fit Zernike coefficients in the same units as z.

    Raises
    ------
    ValueError
        If jmin is negative or jmax is less than jmin
    """
    # Check jmin and jmax
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")

    mask = mask if mask is not None else np.full_like(u, True, dtype=bool)

    # Create a Zernike basis
    zkBasis = createZernikeBasis(u[mask], v[mask], jmin, jmax, obscuration)

    # Fit the Zernikes
    coeffs, *_ = np.linalg.lstsq(zkBasis.T, z[mask], rcond=-1)

    return coeffs


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
    diameter : float, optional
        The diameter of the telescope aperture, in meters.
        (the default, 8.36, corresponds to the LSST primary mirror)
    obscuration : float, optional
        Central obscuration of telescope aperture (i.e. R_outer / R_inner).
        (the default, 0.612, corresponds to the LSST primary mirror)
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    jmax : int, optional
        The max Zernike Noll index, inclusive. Must be >= jmin.
        (the default is 22.)

    Returns
    -------
    np.ndarray
        Gradient of the PSF FWHM with respect to the corresponding Zernike.
        Units are arcsec / micron.

    Raises
    ------
    ValueError
        If jmin is negative or jmax is less than jmin
    """
    # Check jmin and jmax
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")

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
        Central obscuration of telescope aperture (i.e. R_outer / R_inner).
        (the default, 0.612, corresponds to the LSST primary mirror)
    jmin : int
        The minimum Zernike Noll index, inclusive. Must be >= 0. The
        max Noll index is inferred from `jmin` and the length of `zernikes`.
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

    Raises
    ------
    ValueError
        If jmin is negative
    """
    # Check jmin
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")

    # Calculate jmax from jmin and the length of the zernike array
    jmax = jmin + np.array(zernikes).shape[-1] - 1

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


def getZernikeParity(jmin: int = 4, jmax: int = 22, axis: str = "x") -> np.ndarray:
    """Return the parity of the Zernike polynomials (Noll index >= 4).

    Parameters
    ----------
    jmin : int, optional
        The minimum Noll index, inclusive. Must be >= 0. (the default is 4)
    jmax : int, optional
        The maximum Noll index, inclusive. Must be >= jmin. (the default is 22)
    axis : str, optional
        The axis for which to return the parity. Can be "x" or "y".
        (the default is "x")

    Returns
    -------
    np.ndarray
        Array of parities, with +1 corresponding to even parity,
        and -1 corresponding to odd parity.

    Raises
    ------
    ValueError
        If axis is not one of "x" or "y"
    ValueError
        If jmin is negative or jmax is less than jmin
    """
    # Check jmin and jmax
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")

    parity = []
    for j in range(jmin, jmax + 1):
        n, m = galsim.zernike.noll_to_zern(j)
        if axis == "x":
            # if (-1)^n * m >= 0, x parity is even
            parity.append(2 * ((-1) ** n * m >= 0) - 1)
        elif axis == "y":
            # m >= 0, x parity is even
            parity.append(2 * (m >= 0) - 1)
        else:
            raise ValueError("axis must be either 'x' or 'y'.")

    return np.array(parity)


def makeSparse(values: np.ndarray, nollIndices: np.ndarray) -> np.ndarray:
    """Down-select the values array according to requested Noll indices.

    For example, if values=[1, -2, 3, -4], nollIndices=[4, 5, 7], this
    function returns [1, -2, -4].

    Parameters
    ----------
    values : np.ndarray
        Array of values to be down-selected. Selection is applied to
        the first axis, which must correspond to consecutive Noll
        indices, starting with Noll index 4.
    nollIndices : np.ndarray
        Array of Noll indices to select. Must contain only indices >= 4.
        Note these values must be unique, ascending, and >= 4.

    Returns
    -------
    np.ndarray
        Down-selected values
    """
    # Make sure these are arrays
    values, nollIndices = np.array(values), np.array(nollIndices)

    # Validate indices
    if np.any(nollIndices < 4):
        raise ValueError("nollIndices must be >= 4.")
    if not np.array_equal(nollIndices, np.sort(np.unique(nollIndices))):
        raise ValueError("Values in nollIndices must be unique and ascending.")

    return values[nollIndices - 4]


def makeDense(
    values: np.ndarray,
    nollIndices: np.ndarray,
    jmax: int | None = None,
) -> np.ndarray:
    """Inserts sparse values into dense array of Zeroes.

    For example, if values=[1, -2, -4], nollIndices=[4, 5, 7], this
    function returns [1, -2, 0, -4].

    Parameters
    ----------
    values : np.ndarray
        Array of sparse values that will be inserted into the dense array.
    nollIndices : np.ndarray
        Array of Noll indices to select. Must contain only indices >= 4.
        Note these values must be unique, ascending, and >= 4.
    jmax : int or None, optional
        The maximum Noll index of the dense array. If None, the max value
        of nollIndices is used. (the default is None)

    Returns
    -------
    np.ndarray
        A dense array of values, corresponding to Noll indices 4 - jmax.
        Missing values are zero.
    """
    # Make sure these are arrays
    values, nollIndices = np.array(values), np.array(nollIndices)

    # Validate indices and set jmax
    if np.any(nollIndices < 4):
        raise ValueError("nollIndices must be >= 4.")
    if not np.array_equal(nollIndices, np.sort(np.unique(nollIndices))):
        raise ValueError("Values in nollIndices must be unique and ascending.")
    jmax = nollIndices.max() if jmax is None else jmax
    if jmax < 4:
        raise ValueError("jmax must be >= 4.")

    # Remove indices above jmax.
    vals = values[nollIndices <= jmax]
    idx = nollIndices[nollIndices <= jmax] - 4

    # Create the dense array
    dense = np.zeros_like(np.arange(4, jmax + 1), dtype=values.dtype)

    # Insert values
    dense[idx] = vals

    return dense


def checkNollIndices(nollIndices: np.ndarray) -> None:
    """Check that Noll indices meet requirements.

    Parameters
    ----------
        nollIndices : np.ndarray
            Array of Noll indices.

    Raises
    ------
    ValueError
        If nollIndices contains values less than 4, if they're not ascending
        and unique, and if azimuthal pairs are not complete.
    """
    # Simple checks on values
    if any(nollIndices < 4):
        raise ValueError("nollIndices must be >= 4.")
    if not np.array_equal(nollIndices, np.sort(np.unique(nollIndices))):
        raise ValueError("Values in nollIndices must be unique and ascending.")

    # Now we will make sure azimuthal pairs are complete...

    # Create grid of Noll indices from 4 to jmax, as well as az. symm.
    # We select jmax that is greater than max value in Noll indices
    # and is also azimuthally symmetric. This is so that once we
    # downselect to indices without azimuthal symmetry, we are guaranteed
    # to have an even number of indices in our grid.
    grid = np.array([4])
    az = np.array([0])
    while grid[-1] < nollIndices.max() or az[-1] != 0:
        grid = np.append(grid, grid[-1] + 1)
        az = np.append(az, galsim.zernike.noll_to_zern(grid[-1])[1])

    # Remove azimuthally symmetric indices
    grid = grid[np.where(az != 0)]

    # Now consecutive Noll indices are azimuthal pairs
    # Create mapping between these
    paired_grid = grid.reshape(-1, 2)
    pairs = {i: j for i, j in paired_grid} | {j: i for i, j in paired_grid}

    # Check all pairs are complete
    for j in nollIndices:
        if j in pairs and pairs[j] not in nollIndices:
            raise ValueError(f"Noll index {j} is missing azimuthal pair, Noll index {pairs[j]}.")
