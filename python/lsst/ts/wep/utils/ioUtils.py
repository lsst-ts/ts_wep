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
    "getModulePath",
    "getConfigDir",
    "getObsLsstCmdTaskConfigDir",
    "writeFile",
    "readPhoSimSettingData",
    "getAmpImagesFromDir",
]

import os
import re

from lsst.utils import getPackageDir


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
                    data = lineElement[12:15]

        # Collect the data
        if data:
            ccdData.update({lineElement[0]: data})

    # Close the file
    fid.close()

    return ccdData


def getAmpImagesFromDir(rawExpDir):
    """Apply regular expression to find
    repackaged amplifier image files.
    Use the negative lookahead to find those
    fits files that do not contain '_e' in their name.

    Parameters
    ----------
    rawExpDir : str
        path to the input directory with raw repackaged files

    Returns
    -------
    list [str]
        raw amplifier image files
    """
    return list(filter(re.compile(r"^((?!_e).)*fits$").match, os.listdir(rawExpDir)))
