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
    "getDefocalDisInMm",
    "createInstDictFromConfig",
    "rotMatrix",
    "padArray",
    "extractArray",
]

import numpy as np
from scipy.ndimage import center_of_mass


def searchDonutPos(img):
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


def getDefocalDisInMm(instName):
    """
    Get the defocal distance for the instrument

    Parameters
    ----------
    instName : str
        Instrument name, one of
        'lsst', 'lsstfam', 'comcam',
        'auxTel'

    Returns
    -------
    defocalDisInMm : float
        Defocal distance in mm.

    Raises
    ------
    ValueError
        Instrument name is not supported.
    """
    if instName in ["lsst", "lsstfam", "comcam"]:
        return 1.5
    elif instName == "auxTel":
        return 0.8
    else:
        raise ValueError(f"Instrument name ({instName}) is not supported.")


def createInstDictFromConfig(config):
    """Create configuration dictionary for the instrument.

    Parameters
    ----------
    config : lsst.pipe.base.PipelineTaskConfig
        Task configuration.

    Returns
    -------
    dict
        Instrument configuration parameters
    """

    return {
        "obscuration": config.instObscuration,
        "focalLength": config.instFocalLength,
        "apertureDiameter": config.instApertureDiameter,
        "offset": config.instDefocalOffset,
        "pixelSize": config.instPixelSize,
    }


def rotMatrix(thetaDegrees):
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

    # Extract the central image
    out = inArray[ii : ii + dim, ii : ii + dim]

    return out
