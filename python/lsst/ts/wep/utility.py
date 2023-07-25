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
    "FilterType",
    "CamType",
    "BscDbType",
    "DefocalType",
    "ImageType",
    "CentroidFindType",
    "DonutTemplateType",
    "DeblendDonutType",
    "getModulePath",
    "getConfigDir",
    "getObsLsstCmdTaskConfigDir",
    "runProgram",
    "searchDonutPos",
    "writeFile",
    "readPhoSimSettingData",
    "mapFilterRefToG",
    "getBscDbType",
    "getImageType",
    "getCentroidFindType",
    "getDeblendDonutType",
    "getDonutTemplateType",
    "getAmpImagesFromDir",
    "writePipetaskCmd",
    "writeCleanUpRepoCmd",
    "getCamType",
    "getCamTypeFromButlerName",
    "getDefocalDisInMm",
]

import os
import re
import subprocess
from enum import IntEnum, auto

import numpy as np
from lsst.afw.cameraGeom import DetectorType
from lsst.utils import getPackageDir
from scipy.ndimage import center_of_mass


class FilterType(IntEnum):
    LSST_U = 1
    LSST_G = auto()
    LSST_R = auto()
    LSST_I = auto()
    LSST_Z = auto()
    LSST_Y = auto()
    REF = auto()


class CamType(IntEnum):
    LsstCam = 1
    LsstFamCam = auto()
    ComCam = auto()
    AuxTel = auto()
    AuxTelZWO = auto()


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
    ConvolveTemplate = auto()


class DonutTemplateType(IntEnum):
    Model = 1
    Phosim = auto()


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
                    data = lineElement[12:15]

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
        return FilterType.LSST_G
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
        Centroid find algorithm to use (randomWalk, otsu, or convolveTemplate).

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
    elif centroidFindType == "convolveTemplate":
        return CentroidFindType.ConvolveTemplate
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


def getDonutTemplateType(donutTemplateType):
    """Get the donut template type.

    Parameters
    ----------
    donutTemplateType : str
        Donut template type to use (model or phosim).

    Returns
    -------
    enum 'DonutTemplateType'
        Donut template type algorithm.

    Raises
    ------
    ValueError
        The donut template type is not supported.
    """

    if donutTemplateType == "model":
        return DonutTemplateType.Model
    elif donutTemplateType == "phosim":
        return DonutTemplateType.Phosim
    else:
        raise ValueError(f"The {donutTemplateType} is not supported.")


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


def writePipetaskCmd(
    repoDir, runName, instrument, collections, taskName=None, pipelineYaml=None
):
    """
    Format a command line call to run a Gen 3 pipeline task.
    Can also be a set of tasks if specified in a pipeline yaml file.

    Parameters
    ----------
    repoDir: str
        Location of Gen 3 repository.
    runName: str
        Name of collection for data produced by the task.
    instrument: str
        The instrument to use for the task.
    collections: str
        The data collections needed for the task.
    taskName: str, optional
        Full task function name in lsst namespace. One of taskName
        or pipelineYaml must be specified to run. (The default is None).
    pipelineYaml: str, optional
        Yaml file that specifies a pipeline configuration to run
        instead of a single task. (The default is None.)

    Returns
    -------
    str
        Pipetask run command.

    Raises
    ------
    ValueError
        Need to at least specify name of task or name of pipeline file.
    """
    if (taskName is None) and (pipelineYaml is None):
        raise ValueError("At least one of taskName or pipelineYaml must not be None")

    pipetaskCmd = "pipetask run "
    pipetaskCmd += f"-b {repoDir} "  # Specify repo
    pipetaskCmd += f"-i {collections} "  # Specify collections with data to use
    pipetaskCmd += f"--instrument {instrument} "
    pipetaskCmd += f"--register-dataset-types --output-run {runName}"
    if taskName is not None:
        pipetaskCmd += f" -t {taskName}"
    if pipelineYaml is not None:
        pipetaskCmd += f" -p {pipelineYaml}"

    return pipetaskCmd


def writeCleanUpRepoCmd(repoDir, runName):
    """
    Format a command line call to clean up the data created by a pipeline.

    Parameters
    ----------
    repoDir: str
        Location of Gen 3 repository.
    runName: str
        Name of collection for data produced by the task.

    Returns
    -------
    str
        Butler prune-collection command.
    """

    cleanUpCmd = "butler remove-runs "
    cleanUpCmd += f"{repoDir} {runName} --no-confirm"

    return cleanUpCmd


def getCamType(instName):
    """Get the camera type from instrument name.

    Parameters
    ----------
    instName : str
         Instrument name.

    Returns
    -------
    camType : enum 'CamType'
        Camera type.

    Raises
    ------
    ValueError
        Instrument name is not supported.
    """
    if instName == "lsst":
        return CamType.LsstCam
    elif instName == "lsstfam":
        return CamType.LsstFamCam
    elif instName == "comcam":
        return CamType.ComCam
    elif instName == "auxTel":
        return CamType.AuxTel
    else:
        raise ValueError(f"Instrument name ({instName}) is not supported.")


def getCamNameFromCamType(camType):
    """Get the camera name for policy files from CamType.

    Parameters
    ----------
    camType : enum 'CamType'
        Camera Type.

    Returns
    -------
    str
        Instrument Name.

    Raises
    ------
    ValueError
        Camera Type is not supported.
    """

    if camType == CamType.LsstCam:
        return "lsst"
    elif camType == CamType.LsstFamCam:
        return "lsstfam"
    elif camType == CamType.ComCam:
        return "comcam"
    elif camType == CamType.AuxTel:
        return "auxTel"
    elif camType == CamType.AuxTelZWO:
        return "auxTelZWO"
    else:
        raise ValueError(f"CamType ({camType}) is not supported.")


def getCamTypeFromButlerName(instName, detectorType):
    """Get the camera type from instrument name used by the LSST DM
    middleware for each instrument.

    Parameters
    ----------
    instName : str
        Instrument name.
    detectorType : lsst.afw.cameraGeom.DetectorType
        Type of CCD. "SCIENCE" or "WAVEFRONT".

    Returns
    -------
    camType : enum 'CamType'
        Camera type.

    Raises
    ------
    ValueError
        Combination of instrument name and detector type is not supported.
    ValueError
        Detector type is not supported.
    """
    if detectorType == DetectorType.WAVEFRONT:
        if instName == "LSSTCam":
            return CamType.LsstCam
        else:
            raise ValueError(
                f"Wavefront sensors for instrument name ({instName}) are not supported."
            )
    elif detectorType == DetectorType.SCIENCE:
        if instName == "LSSTCam":
            return CamType.LsstFamCam
        elif instName == "LSSTComCam":
            return CamType.ComCam
        elif instName == "LATISS":
            return CamType.AuxTel
        else:
            raise ValueError(
                f"Science sensors for instrument name ({instName}) are not supported."
            )
    else:
        raise ValueError(f"Detector Type ({detectorType.name}) is not supported.")


def getFilterTypeFromBandLabel(bandLabel):
    """Get the FilterType associated with the name of the bandpass
    accessed in an exposure using `exposure.filter.bandLabel`.

    Parameters
    ----------
    bandLabel : str
        Bandpass label of the exposure.

    Returns
    -------
    filterType : enum `FilterType`
        Filter type.
    """
    filterLabelDict = {}
    filterLabelDict["u"] = FilterType.LSST_U
    filterLabelDict["g"] = FilterType.LSST_G
    filterLabelDict["r"] = FilterType.LSST_R
    filterLabelDict["i"] = FilterType.LSST_I
    filterLabelDict["z"] = FilterType.LSST_Z
    filterLabelDict["y"] = FilterType.LSST_Y

    return filterLabelDict.get(bandLabel, FilterType.REF)


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
