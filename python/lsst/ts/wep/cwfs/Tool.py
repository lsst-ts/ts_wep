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

import numpy as np

from lsst.ts.wep.cwfs import mathcwfs


def ZernikeAnnularEval(z, x, y, e, nMax=28):
    """Calculate the wavefront surface in the basis of annular Zernike
    polynomial.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of annular Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    e : float
        Obscuration value. It is 0.61 in LSST.
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 28.)

    Returns
    -------
    numpy.ndarray
        Wavefront surface.
    """

    # Check the preconditions
    z = _checkPrecondition(z, x, y, int(nMax))

    # Calculate the wavefront
    return mathcwfs.zernikeAnnularEval(z, x.flatten(), y.flatten(), e).reshape(x.shape)


def _checkPrecondition(z, x, y, nMax):
    """Check the preconditions before the evaluation related to Zernike
    polynomials.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    nMax : int
        Maximum number of Zernike terms.

    Returns
    -------
    numpy.ndarray
        Coefficient of Zernike polynomials.
    """

    # Check the dimensions of x and y are the same or not
    if x.shape != y.shape:
        print("x & y are not the same size")
        exit()

    # Check the number of terms of annular Zernike polynomials
    if len(z) > nMax:
        print(
            "Some Zernike related functions are not implemented with >%d terms." % nMax
        )
        return
    elif len(z) < nMax:
        # Put the higher order terms as zero to make sure nMax terms of
        # polynomials
        z = np.hstack((z, np.zeros(nMax - len(z))))

    return z


def ZernikeAnnularGrad(z, x, y, e, axis, nMax=22):
    """Evaluate the gradident of annular Zernike polynomials in a certain
    direction.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of annular Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    e : float
        Obscuration value. It is 0.61 in LSST.
    axis : str
        It can be "dx", "dy", "dx2", "dy2", or "dxy".
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 22.)

    Returns
    -------
    numpy.ndarray
        Integration elements of gradient part in pupul x and y directions.
    """

    # Check the preconditions
    z = _checkPrecondition(z, x, y, int(nMax))

    # Calculate the integration elements
    return mathcwfs.zernikeAnnularGrad(z, x.flatten(), y.flatten(), e, axis).reshape(
        x.shape
    )


def ZernikeAnnularJacobian(z, x, y, e, order, nMax=22):
    """Evaluate the Jacobian of annular Zernike polynomials in a certain order.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of annular Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    e : float
        Obscuration value. It is 0.61 in LSST.
    order : str
        Order of Jocobian Matrix. It can be "1st" or "2nd".
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 22.)

    Returns
    -------
    numpy.ndarray
        Jacobian elements in pupul x and y directions in a certain order.
    """

    # Check the preconditions
    z = _checkPrecondition(z, x, y, int(nMax))

    # Calculate the Jacobian
    return mathcwfs.zernikeAnnularJacobian(
        z, x.flatten(), y.flatten(), e, order
    ).reshape(x.shape)


def ZernikeAnnularFit(s, x, y, numTerms, e, nMax=28):
    """Get the coefficients of annular Zernike polynomials by fitting the
    wavefront surface.

    Parameters
    ----------
    s : numpy.ndarray
        Wavefront surface to be fitted.
    x : numpy.ndarray
        Normalized x coordinate between -1 and 1 (pupil coordinate).
    y : numpy.ndarray
        Normalized y coordinate between -1 and 1 (pupil coordinate).
    numTerms : int
        Number of annular Zernike terms used in the fit.
    e : float
        Obscuration ratio of annular Zernikes.
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 28.)

    Returns
    -------
    numpy.ndarray
        Coefficients of annular Zernike polynomials by the fitting.
    """

    # Check the dimensions of x and y are the same or not
    if x.shape != y.shape:
        print("x & y are not the same size")

    # Get the value that is finite
    sFinite = s[:].copy()
    xFinite = x[:].copy()
    yFinite = y[:].copy()

    finiteIndex = np.isfinite(sFinite + xFinite + yFinite)

    sFinite = sFinite[finiteIndex]
    xFinite = xFinite[finiteIndex]
    yFinite = yFinite[finiteIndex]

    # Do the fitting
    h = np.zeros([len(sFinite), numTerms])

    for ii in range(numTerms):
        z = np.zeros(numTerms)
        z[ii] = 1
        h[:, ii] = ZernikeAnnularEval(z, xFinite, yFinite, e, nMax=nMax)

    # Solve the equation: H*Z = S => Z = H^(-1)S
    z = np.linalg.lstsq(h, sFinite, rcond=None)[0]

    return z


def ZernikeGrad(z, x, y, axis, nMax=22):
    """Evaluate the gradident of Zernike polynomials in a certain axis.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    axis : str
        Integration direction. It can be "dx" or "dy".
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 22.)

    Returns
    -------
    numpy.ndarray
        Integration elements of gradient part in pupul x and y directions.
    """

    # Calculate the integration elements
    # Use obscuration (e) = 0 for standard Zernike polynomials
    return ZernikeAnnularGrad(z, x, y, 0, axis, nMax=nMax)


def ZernikeJacobian(z, x, y, order, nMax=22):
    """Evaluate the Jacobian of Zernike polynomials in a certain order.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    order : str
        Order of Jocobian Matrix. It can be "1st" or "2nd".
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 22.)

    Returns
    -------
    numpy.ndarray
        Jacobian elements in pupul x and y directions in a certain order.
    """

    # Calculate the Jacobian elements
    # Use obscuration (e) = 0 for standard Zernike polynomials
    return ZernikeAnnularJacobian(z, x, y, 0, order, nMax=nMax)


def ZernikeEval(z, x, y, nMax=28):
    """Calculate the wavefront surface in the basis of Zernike polynomial.

    Parameters
    ----------
    z : numpy.ndarray
        Coefficient of Zernike polynomials.
    x : numpy.ndarray
        X coordinate on pupil plane.
    y : numpy.ndarray
        Y coordinate on pupil plane.
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 28.)

    Returns
    -------
    numpy.ndarray
        Wavefront surface.
    """

    # Calculate the wavefront surface
    # Use obscuration (e) = 0 for standard Zernike polynomials
    return ZernikeAnnularEval(z, x, y, 0, nMax=nMax)


def ZernikeFit(s, x, y, numTerms, nMax=28):
    """Get the coefficients of Zernike polynomials by fitting the wavefront
    surface.

    Parameters
    ----------
    s : numpy.ndarray
        Wavefront surface to be fitted.
    x : numpy.ndarray
        Normalized x coordinate between -1 and 1 (pupil coordinate).
    y : numpy.ndarray
        Normalized y coordinate between -1 and 1 (pupil coordinate).
    numTerms : int
        Number of Zernike terms used in the fit.
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 28.)

    Returns
    -------
    numpy.ndarray
        Coefficients of Zernike polynomials by the fitting.
    """

    # Do the fitting to get coefficients of Zernike polynomials
    # Use obscuration (e) = 0 for standard Zernike polynomials
    return ZernikeAnnularFit(s, x, y, numTerms, 0, nMax=nMax)


def ZernikeMaskedFit(s, x, y, numTerms, mask, e, nMax=28):
    """Fit the wavefront surface on pupil (e.g. under the mask) to a linear
    combination of normal/ annular Zernike polynomials.

    Parameters
    ----------
    s : numpy.ndarray
        Wavefront surface to be fitted.
    x : numpy.ndarray
        Normalized x coordinate between -1 and 1 (pupil coordinate).
    y : numpy.ndarray
        Normalized y coordinate between -1 and 1 (pupil coordinate).
    numTerms : int
        Number of normal/ annular Zernike terms used in the fit.
    mask : numpy.ndarray[int]
        Mask used.
    e : float
         Obscuration ratio of annular Zernikes.
    nMax : int, optional
        Maximum number of Zernike terms. (the default is 28.)

    Returns
    -------
    numpy.ndarray
        Coefficients of normal/ annular Zernike polynomials by the fitting.
    """

    # Get S, x, y elements in mask
    j, i = np.nonzero(mask[:])
    s = s[i, j]
    x = x[i, j]
    y = y[i, j]

    # Calculate coefficients of normal/ spherical Zernike polynomials
    return ZernikeAnnularFit(s, x, y, numTerms, e, nMax=nMax)


def padArray(inArray, dim):
    """Extend the boundary of image.

    For example, the input image is 120x120 matrix. This function will create
    an image such as 140x140 (dim x dim) matrix and put the input image in the
    center of new image.

    Parameters
    ----------
    inArray : numpy.ndarray
        Input central image.
    dim : int
        Dimension of new extended image.

    Returns
    -------
    numpy.ndarray
        Extended image from the dimension of inArray to dim x dim.

    Raises
    ------
    Exception
        Check the dimension of inArray is n by n or not.
    Exception
        Check the extending dimension is bigger than the dimension of inArray
        or not.
    """

    # Check the conditions
    m, n = inArray.shape
    if m != n:
        raise Exception("padArray: array is not square.")

    if m > dim:
        raise Exception("padArray: array is larger than dimension.")

    # Extend the boundary of image by creating a bigger matrix and putting the
    # input image in the center
    out = np.zeros([dim, dim])
    ii = int(np.floor((dim - m) / 2))

    # Put the original image in the center of extended image
    out[ii : ii + m, ii : ii + m] = inArray

    return out


def extractArray(inArray, dim):
    """Extract the central image.

    For example, the input image is a 140x140 matrix. This function will
    extract the central matrix with the dimension of 120x120 (dim x dim).

    Parameters
    ----------
    inArray : numpy.ndarray
        Input image.
    dim : int
        Dimension of extracted image.

    Returns
    -------
    numpy.ndarray
        Extracted central image from the dimension of inArray to dim x dim.

    Raises
    ------
    Exception
        Check the dimension of inArray is n by n or not.
    Exception
        Check the extracted dimension is smaller than the dimension of inArray
        or not.
    """

    # Check the conditions
    m, n = inArray.shape
    if m != n:
        raise Exception("extractArray: array is not square.")

    if m < dim:
        raise Exception("extractArray: array is smaller than dimension")

    # Calculate the begining index to extract the central image
    ii = int(np.floor((m - dim) / 2))

    # Extract the cetral image
    out = inArray[ii : ii + dim, ii : ii + dim]

    return out
