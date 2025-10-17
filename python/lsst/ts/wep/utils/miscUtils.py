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
    "searchDonutPos",
    "rotMatrix",
    "padArray",
    "extractArray",
    "centerWithTemplate",
    "polygonContains",
    "conditionalSigmaClip",
    "binArray",
    "calcStampPowerSpectrum",
]

import numba
import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import center_of_mass
from scipy.signal import correlate
from scipy.stats import binned_statistic


def searchDonutPos(img: np.ndarray) -> tuple[float, float]:
    """Search the position of donut on image.

    Parameters
    ----------
    img : numpy.ndarray
         Donut image.

    Returns
    -------
    float
        X position of donut center in pixel.
    float
        Y position of donut center in pixel.
    """

    # Search the donut position by the center of mass
    # Need to update this method to the more robust one such as the convolution
    realcy, realcx = center_of_mass(img)

    return realcx, realcy


def rotMatrix(thetaDegrees: float) -> np.ndarray:
    """Create a 2-d rotation matrix for given angle.

    Parameters
    ----------
    thetaDegrees : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotation matrix for theta.
    """

    theta = np.radians(thetaDegrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def padArray(inArray: np.ndarray, dim: int) -> np.ndarray:
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


def extractArray(inArray: np.ndarray, dim: int) -> np.ndarray:
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

    # Extract the central image
    out = inArray[ii : ii + dim, ii : ii + dim]

    return out


def centerWithTemplate(
    image: np.ndarray,
    template: np.ndarray,
    rMax: float = 10,
) -> np.ndarray:
    """Center the image by correlating with the template.

    Parameters
    ----------
    image : np.ndarray
        The image to be centered
    template : np.ndarray
        The template to use for correlation
    rMax : float, optional
        The maximum distance the image can be shifted, in pixels.
        (the default is 10)

    Returns
    -------
    np.ndarray
        The centered image
    """
    # Replace any NaNs with zeros, because the NaNs screw up the correlation
    image = np.nan_to_num(image)

    # Correlate the template with the image
    corr = correlate(image, template, mode="same")

    # Mask within rMax of the center. We will not allow larger shifts.
    grid = np.arange(image.shape[0])
    rGrid = np.sqrt(np.sum(np.square(np.meshgrid(grid, grid) - grid.mean()), axis=0))
    mask = rGrid <= rMax

    # Get the index of maximum correlation
    idxMax = np.unravel_index(np.argmax(mask * corr), corr.shape)
    idxMax = np.array(idxMax)  # type: ignore

    # Find shift relative to the center
    centerShift = idxMax - image.shape[0] // 2  # type: ignore

    # Roll the image so that the pixel with max correlation is in the center
    image = np.roll(image, -centerShift, (0, 1))

    return image


@numba.jit(
    "boolean[:](float64, float64[:], float64[:,:])",
    nopython=True,
    fastmath=True,
)
def _polygonContainsRow(y: float, row: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return mask indicating if each point in the row is inside the polygon.

    Parameters
    ----------
    y : float
        The y value of every point in the row
    row : np.ndarray
        A 1D array of x values for every point in the row
    poly : np.ndarray
        An Nx2 array of points specifying the polygon vertices. It is assumed
        that an edge connects every pair of adjacent points, and that the final
        point is identical to the first point.

    Returns
    -------
    np.ndarray
        A 1D array with the same shape as row, indicating whether each point
        is inside the polygon
    """
    # If the row is totally outside the polygon, return False
    if row.max() < poly[:, 0].min() or row.min() > poly[:, 0].max():
        return np.full_like(row, False, dtype=np.bool_)

    # Determine which polygon edges cross our column
    dy0 = y - poly[:-1, 1]
    dy1 = y - poly[1:, 1]
    idx = np.where(dy0 * dy1 < 0)[0]

    # Solve for the x-values of these edges where they cross the row
    m = (poly[idx + 1, 0] - poly[idx, 0]) / (poly[idx + 1, 1] - poly[idx, 1])
    x = m * dy0[idx] + poly[idx, 0]

    # Count the number of edges to the right of each point in the row
    edgesToRight = np.sum(np.expand_dims(row, -1) < x, axis=-1)

    # The point is inside if the number of edges to the right is odd
    inside = edgesToRight % 2 == 1

    return inside


@numba.jit(
    "boolean[:,:](float64[:,:], float64[:,:], float64[:,:])",
    nopython=True,
    fastmath=True,
)
def _polygonContains(
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    poly: np.ndarray,
) -> np.ndarray:
    """Return mask indicating if each point in the grid is inside the polygon.

    This function works only in 2D, and assumes that xGrid and yGrid are
    regular grids like will be returned from np.meshgrid().

    This function is compiled with numba to increase speed. The explicit
    signature is provided so that the function is eagerly compiled on import,
    rather than on execution.

    Parameters
    ----------
    xGrid : np.ndarray
        A 2D array of x values for the grid points. Must be floats.
    yGrid : np.ndarray
        A 2D array of y values for the grid points. Must be floats.
    poly : np.ndarray
        An Nx2 array of points specifying the polygon vertices. It is assumed
        that an edge connects every pair of adjacent points, and that the final
        point is identical to the first point. Must be floats.

    Returns
    -------
    np.ndarray
        A 2D array with the same shape as xGrid and yGrid, indicating whether
        each point is inside the polygon
    """
    # Get the array of unique y values
    y = yGrid[:, 0]

    # Create an array full of False
    inside = np.full_like(xGrid, False, dtype=np.bool_)

    # Determine which rows have y-values that fall within polygon limits
    idx = np.where((y > poly[:, 1].min()) & (y < poly[:, 1].max()))[0]

    # If none do, we can return all False
    if len(idx) == 0:
        return inside

    # Add some tiny shifts to the y-values of the polygon vertices
    # This helps avoid problems with horizontal lines
    polyShifted = poly.copy()
    shifts = np.arange(len(poly) - 2) % 11 - 5
    polyShifted[1:-1, 1] += 1e-12 * shifts

    # Loop over rows inside polygon limits
    for i in range(len(idx)):
        inside[idx[i]] = _polygonContainsRow(
            y[idx[i]],
            xGrid[idx[i]],
            polyShifted,
        )

    return inside


def polygonContains(
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    poly: np.ndarray,
) -> np.ndarray:
    """Return mask indicating if each point in the grid is inside the polygon.

    Note this function works only in 2D, and assumes that xGrid and yGrid are
    regular grids like will be returned from np.meshgrid().

    Parameters
    ----------
    xGrid : np.ndarray
        A 2D array of x values for the grid points. Must be floats.
    yGrid : np.ndarray
        A 2D array of y values for the grid points. Must be floats.
    poly : np.ndarray
        An Nx2 array of points specifying the polygon vertices. It is assumed
        that an edge connects every pair of adjacent points, and that the final
        point is identical to the first point. Must be floats.

    Returns
    -------
    np.ndarray
        A 2D array with the same shape as xGrid and yGrid, indicating whether
        each point is inside the polygon

    Raises
    ------
    ValueError
        If xGrid and yGrid are not the same shape, if either xGrid or yGrid are
        not 2D, or if poly is not an Nx2 array
    """
    # Check shapes
    if not xGrid.shape == yGrid.shape:
        raise ValueError("xGrid and yGrid must have the same shape.")
    if not xGrid.ndim == 2:
        raise ValueError("xGrid and yGrid must be 2D.")
    if not poly.shape[1] == 2:
        raise ValueError("poly must be an Nx2 array.")

    # Cast everything to floats
    xGrid = xGrid.astype(float)
    yGrid = yGrid.astype(float)
    poly = poly.astype(float)

    # If the last point in poly isn't the same as the first,
    # append the first point to the end
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack((poly, poly[0]))

    return _polygonContains(xGrid, yGrid, poly)


# Function to apply conditional sigma clipping
def conditionalSigmaClip(array: np.ndarray, sigmaClipKwargs: dict, stdMin: float = 0.005) -> np.ndarray:
    """Apply conditional sigma clipping to an array.
    Note this function applies sigma clipping to each
    column of the input array, and replaces the column
    with the clipped data while maintaining masked
    elements as NaN. Sigma clipping is only applied
    if the standard deviation exceeds std_min. This function
    is used in the combineZernikesSigmaClipTask

    Parameters
    ----------
    array : np.ndarray
        Input array to be processed.
    sigma : float
        Number of standard deviations for the clipping limit.
    stdMin : float
        Minimum standard deviation to apply sigma clipping.
    stdFunc : str
        Function to calculate the standard deviation.
        Can be either "mad_std" or "std"

    Returns
    -------
    np.ndarray
        Processed array with conditional sigma clipping applied.

    Raises
    ------
    ValueError
        If stdFunc is not "mad_std" or "std"
    """

    if sigmaClipKwargs["stdfunc"] not in ["mad_std", "std"]:
        raise ValueError("stdfunc must be either 'mad_std' or 'std'")

    # Initialize an array to hold processed (clipped or original) data
    processedArray = np.copy(array)  # Use a copy to maintain original data

    for i in range(array.shape[1]):  # Iterate over columns
        column = array[:, i]
        # Apply sigma clipping if the standard deviation exceeds std_min
        if np.std(column) > stdMin:
            clipped = sigma_clip(column, **sigmaClipKwargs)
            # Replace processedArray column data with clipped data,
            # maintaining masked as NaN
            processedArray[:, i] = clipped.filled(np.nan)

    return processedArray


def binArray(array: np.ndarray, binning: int, method: str = "mean") -> np.ndarray:
    """Bin a 2d array by averaging over non-overlapping blocks.

    Parameters
    ----------
    array : np.ndarray
        The 2d array to be binned
    binning : int
        The binning factor
    method: str, optional
        A method used to average ('mean' or 'median').

    Returns
    -------
    np.ndarray
        The binned array

    Raises
    ------
    ValueError
        If method is not "mean" or "median"
    """
    # Ensure the array is divisible by the binning factor
    array = array[: array.shape[0] // binning * binning, : array.shape[1] // binning * binning]
    # Bin the array
    if method == "mean":
        binned = array.reshape(array.shape[0] // binning, binning, array.shape[1] // binning, binning).mean(
            axis=(1, 3)
        )
    elif method == "median":
        reshaped = array.reshape(array.shape[0] // binning, binning, array.shape[1] // binning, binning)
        binned = np.median(reshaped, axis=(1, 3))
    else:
        raise ValueError(f"{method} is not an accepted averaging method.")

    return binned


def calcStampPowerSpectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the power spectrum of the stamp image.

    Parameters
    ----------
    image : np.ndarray
        Array of stamp image

    Returns
    -------
    np.ndarray
        Array of Fourier modes k (technically |k|)
    np.ndarray
        Power at each mode k
    """
    # Get number of pixels
    npix = image.shape[0]

    # FFT image
    fourier_image = np.fft.fftn(image)

    # Get the square amplitudes
    fourier_amplitudes = np.abs(fourier_image) ** 2

    # Create grid of modes
    k = np.fft.fftfreq(npix) * npix
    k2D = np.meshgrid(k, k)
    knorm = np.sqrt(k2D[0] ** 2 + k2D[1] ** 2)

    # Flatten both arrays
    fourier_amplitudes = fourier_amplitudes.flatten()
    knorm = knorm.flatten()

    # Binned mean of spectrum
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    binned_power, _, _ = binned_statistic(
        knorm,
        fourier_amplitudes,
        statistic="mean",
        bins=kbins,
    )

    return kvals, binned_power
