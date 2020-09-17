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

import os
import subprocess
import re
from scipy.ndimage.measurements import center_of_mass
from enum import IntEnum, auto

from lsst.utils import getPackageDir


class FilterType(IntEnum):
    U = 1
    G = auto()
    R = auto()
    I = auto()
    Z = auto()
    Y = auto()
    REF = auto()


class CamType(IntEnum):
    LsstCam = 1
    LsstFamCam = auto()
    ComCam = auto()
    AuxTel = auto()


class BscDbType(IntEnum):
    LocalDb = 1
    LocalDbForStarFile = auto()


class DefocalType(IntEnum):
    Intra = 1
    Extra = auto()


class ImageType(IntEnum):
    Amp = 1
    Eimg = auto()


class CentroidFindType(IntEnum):
    RandomWalk = 1
    Otsu = auto()


class DeblendDonutType(IntEnum):
    Adapt = 1


def getModulePath():
    """Get the path of module.

    Returns
    -------
    str
        Directory path of module.
    """

    return getPackageDir("ts_wep")


def getConfigDir():
    """Get the directory of configuration files.

    Returns
    -------
    str
        Directory of configuration files.
    """

    return os.path.join(getModulePath(), "policy")


def getObsLsstCmdTaskConfigDir():
    """Get the obs_lsst command line task configuration directory.

    Returns
    -------
    str
        obs_lsst command line task configuration directory.
    """

    return os.path.join(getPackageDir("obs_lsst"), "config")


def runProgram(command, binDir=None, argstring=None):
    """Run the program w/o arguments.

    Parameters
    ----------
    command : str
        Command of application.
    binDir : str, optional
        Directory of binary application. (the default is None.)
    argstring : str, optional
        Arguments of program. (the default is None.)

    Raises
    ------
    RuntimeError
        Error running of command.
    """

    # Directory of binary application
    if binDir is not None:
        command = os.path.join(binDir, command)

    # Arguments for the program
    if argstring is not None:
        command += " " + argstring

    # Call the program w/o arguments
    if subprocess.call(command, shell=True) != 0:
        raise RuntimeError("Error running: %s" % command)


def expandDetectorName(abbrevName):
    """Convert a detector name of the form Rxy_Sxy[_Ci] to canonical form:
    R:x,y S:x,y[,c] C0 -> A, C1 -> B.

    This is copied from lsst.obs.lsstSim:
    https://github.com/lsst/obs_lsstSim/blob/master/bin.src/
    makeLsstCameraRepository.py

    Parameters
    ----------
    abbrevName : str
        Abbreviated name.

    Returns
    -------
    str
        Detector canonical name.

    Raises
    ------
    ValueError
        Input does not match the abbreviated form of detector name.
    """

    m = re.match(r"R(\d)(\d)_S(\d)(\d)(?:_C([0,1]))?$", abbrevName)
    if m is None:
        raise ValueError("Cannot parse abbreviated name %r" % (abbrevName,))
    fullName = "R:%s,%s S:%s,%s" % tuple(m.groups()[0:4])
    subSensor = m.groups()[4]
    if subSensor is not None:
        fullName = fullName + "," + {"0": "A", "1": "B"}[subSensor]
    return fullName


def abbrevDectectorName(canonicalForm):
    """Convert a canonical name to abbreviate name (R:x,y S:x,y[,c] -->
    Rxy_Sxy[_Ci]).

    Parameters
    ----------
    canonicalForm : str
        Detector canonical name.

    Returns
    -------
    str
        Abbreviated name.

    Raises
    ------
    ValueError
        Input does not match the canonical form of detector name.
    """

    # Use the regular expression to analyze the input name
    m = re.match(r"R:(\d),(\d) S:(\d),(\d)(?:,([A,B]))?$", canonicalForm)

    # Raise error if the input does not match the form of regular expression
    if m is None:
        raise ValueError("Cannot parse canonical name %r" % (canonicalForm,))

    # Generate the abbreviated name
    abbrevName = "R%s%s_S%s%s" % tuple(m.groups()[0:4])

    # Check the sensor is wavefront sensor or not
    subSensor = m.groups()[4]
    if subSensor is not None:
        # Label the WFS sensor
        abbrevName = abbrevName + "_C" + {"A": "0", "B": "1"}[subSensor]

    # Return the abbreviate name
    return abbrevName


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


def writeFile(filePath, content):
    """Write the content to file.

    Parameters
    ----------
    filePath : str
        File path.
    content : str
        File content.
    """

    with open(filePath, "w") as file:
        file.write(content)


def readPhoSimSettingData(folderPath, fileName, atype):
    """Read the PhoSim setting data (segmentation or focal plane layout).

    Parameters
    ----------
    folderPath : str
        Path to folder.
    fileName : str
        File name ("segmentation.txt", "focalplanelayout.txt").
    atype : str
        Type of data to read ("readOutDim", "darkCurrent", "fieldCenter",
        "eulerRot").

    Returns
    -------
    dict
        Needed CCD data.

    Raises
    ------
    ValueError
        File can not be read.
    ValueError
        Type is not correct.
    """

    # Check the file name
    if fileName not in ("segmentation.txt", "focalplanelayout.txt"):
        raise ValueError("'%s' can not be read." % fileName)

    # Check the type
    if atype not in ("readOutDim", "darkCurrent", "fieldCenter", "eulerRot"):
        raise ValueError("'%s' can not be read." % atype)

    # Get the file path
    pathToFile = os.path.join(folderPath, fileName)

    # Amplifier list (only list the scientific ccd here)
    ampList = [
        "C00",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
    ]

    # Open the file to read
    ccdData = {}
    fid = open(pathToFile)
    for line in fid:
        line = line.strip()

        # Get each element
        lineElement = line.split()

        data = []
        # Analyze the sensor name to find the amplifier
        if fileName == "segmentation.txt":

            sensorNameStr = lineElement[0].split("_")
            if len(sensorNameStr) == 3:
                if sensorNameStr[2] in ampList:
                    # Get the segmentation in txt file
                    if atype == "readOutDim":
                        # parallel prescan, serial overscan, serial prescan,
                        # parallel overscan (pixel)
                        data = lineElement[15:19]
                    elif atype == "darkCurrent":
                        data = lineElement[13:15]

        elif fileName == "focalplanelayout.txt":

            # Analyze the sensor name to make sure this line of data is
            # needed
            sensorNameStr = lineElement[0].split("_")
            if len(sensorNameStr) == 2 or len(sensorNameStr) == 3:
                if atype == "fieldCenter":
                    # Collect the field center:
                    # x position (microns), y position (microns), pixel
                    # size (microns) number of x pixels, number of y pixels
                    data = lineElement[1:6]
                elif atype == "eulerRot":
                    # Collect the euler Rotation (degrees)
                    data = lineElement[10:13]

        # Collect the data
        if data:
            ccdData.update({lineElement[0]: data})

    # Close the file
    fid.close()

    return ccdData


def mapFilterRefToG(filterType):
    """Map the reference filter to the G filter.

    Parameters
    ----------
    filterType : enum 'FilterType'
        Filter type.

    Returns
    -------
    enum 'FilterType'
        Mapped filter type.
    """

    if filterType == FilterType.REF:
        return FilterType.G
    else:
        return filterType


def getBscDbType(bscDbType):
    """Get the bright star catalog (BSC) database type.

    Parameters
    ----------
    bscDbType : str
        BSC database type to use (localDb or file).

    Returns
    -------
    enum 'BscDbType'
        BSC database type.

    Raises
    ------
    ValueError
        The bscDb is not supported.
    """

    if bscDbType == "localDb":
        return BscDbType.LocalDb
    elif bscDbType == "file":
        return BscDbType.LocalDbForStarFile
    else:
        raise ValueError("The bscDb (%s) is not supported." % bscDbType)


def getImageType(imageType):
    """Get the image type.

    Parameters
    ----------
    imageType : str
        Image type to use (amp or eimage).

    Returns
    -------
    enum 'ImageType'
        ImageType enum.

    Raises
    ------
    ValueError
        The image type is not supported.
    """

    if imageType == "amp":
        return ImageType.Amp
    elif imageType == "eimage":
        return ImageType.Eimg
    else:
        raise ValueError("The %s is not supported." % imageType)


def getCentroidFindType(centroidFindType):
    """Get the centroid find type.

    Parameters
    ----------
    centroidFindType : str
        Centroid find algorithm to use (randomWalk or otsu).

    Returns
    -------
    enum 'CentroidFindType'
        Centroid find type algorithm.

    Raises
    ------
    ValueError
        The centroid find type is not supported.
    """

    if centroidFindType == "randomWalk":
        return CentroidFindType.RandomWalk
    elif centroidFindType == "otsu":
        return CentroidFindType.Otsu
    else:
        raise ValueError("The %s is not supported." % centroidFindType)


def getDeblendDonutType(deblendDonutType):
    """Get the deblend donut type.

    Parameters
    ----------
    deblendDonutType : str
        Deblend donut algorithm to use (adapt).

    Returns
    -------
    enum 'DeblendDonutType'
        Deblend donut type algorithm.

    Raises
    ------
    ValueError
        The deblend donut type is not supported.
    """

    if deblendDonutType == "adapt":
        return DeblendDonutType.Adapt
    else:
        raise ValueError("The %s is not supported." % deblendDonutType)


if __name__ == "__main__":
    pass
