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
    "resolveRelativeConfigPath",
    "readConfigYaml",
    "mergeConfigWithFile",
    "configClass",
    "getObsLsstCmdTaskConfigDir",
    "writeFile",
    "readPhoSimSettingData",
]

import inspect
import os
from functools import lru_cache
from typing import Any, Union

import numpy as np
import yaml
from lsst.utils import getPackageDir


def getModulePath() -> str:
    """Get the path of module.

    Returns
    -------
    str
        Directory path of module.
    """

    return getPackageDir("ts_wep")


def getConfigDir() -> str:
    """Get the directory of configuration files.

    Returns
    -------
    str
        Directory of configuration files.
    """

    return os.path.join(getModulePath(), "policy")


def resolveRelativeConfigPath(path: str) -> str:
    """Resolves policy paths.

    This does not check whether the absolute path actually points to a file.
    If the path does not start with "policy:", it is returned unchanged.

    Parameters
    ----------
    path : str
        Path to resolve. Can start with "policy:" or not.

    Returns
    -------
    str
        Resolved path.
    """
    if not path.startswith("policy:"):
        return path

    # Remove policy: prefix
    path = path.removeprefix("policy:")

    # Return the absolute policy path
    return os.path.join(getConfigDir(), path)


@lru_cache(10)
def readConfigYaml(path: str, recurseImports: bool = True) -> dict:
    """Read the config yaml file and return the corresponding dictionary.

    Parameters
    ----------
    path : str
        Path to the config yaml file. Can be an absolute or relative path, but
        if the path starts with "policy:", the path will be understood to be
        relative to the ts_wep policy directory.
    recurseImports : str, optional
        If True, and the config contains 'imports', open the file(s) provided
        under that keyword and merge the resulting configurations. In the case
        of overlapping keywords, each subsequent import overrides previous
        imports, and the imported configs are all overridden by the top level
        config. (the default is True)

    Returns
    -------
    dict
        Dictionary containing the configuration stored in the yaml file

    Raises
    ------
    ValueError
        If recurseImports is True and 'imports' doesn't map to a string
        or 1D list of strings
    """
    path = resolveRelativeConfigPath(path)

    # Read the parameter file into a dictionary
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    if not recurseImports:
        return config

    # Iteratively load imports and merge configs
    imports = np.atleast_1d(config.pop("imports", []))
    importedConfig: dict = dict()
    if imports.size > 0 and (imports.ndim != 1 or imports.dtype.kind not in ["U", "S"]):
        raise ValueError("'imports' must map to a string or list of strings")
    for path in imports:
        importedConfig = importedConfig | readConfigYaml(path)

    # Apply the overrides to the imported config
    config = importedConfig | config

    return config


def mergeConfigWithFile(configFile: Union[str, None], **kwargs: Any) -> dict:
    """Merge the passed keyword arguments with the values stored in the file.

    If configFile is not provided, the keyword arguments are returned verbatim.

    Parameters
    ----------
    configFile : str, optional
        Path to the config yaml file. Can be an absolute or relative path,
        but if the path starts with "policy:", the path will be understood
        to be relative to the ts_wep policy directory. This file can only
        contain keys that match keywords in kwargs, but it need not contain
        all the keywords.
    kwargs : Any
        Keyword arguments with which to replace the values from the file.
        Note that None values are always ignored in favor of the value in
        configFile.

    Returns
    -------
    dict
        Config dictionary

    Raises
    ------
    KeyError
        If configFile contains keywords not present in kwargs
    """
    # Get the default values from the file
    if configFile is None:
        fileConfig = dict()
    else:
        fileConfig = readConfigYaml(configFile)

    # Determine if the configFile contains keywords not present in kwargs
    extraKeys = set(fileConfig.keys()) - set(kwargs.keys())
    if len(extraKeys) > 0:
        raise KeyError(f"configFile contains unrecognized keys {extraKeys}")

    # Merge the two dictionaries
    mergedConfig = {}
    for key, val in kwargs.items():
        if val is None:
            mergedConfig[key] = fileConfig.get(key, None)
        else:
            mergedConfig[key] = val

    return mergedConfig


def configClass(config: Union[str, dict, None, Any], classObj: Any) -> Any:
    """Configure the class.

    This function is a generic wrapper around the process of passing a
    config to the __init__ function of a class, so that the passed config
    can take on a variety of types.

    If config is a string, it is assumed to be the path to a config
    file, and this path is passed to the configFile argument of the
    class constructor. If config is a dictionary, then the contents
    are passed as keyword arguments. If config is an instance of the
    class, the instance is returned unchanged. If config is None, then
    the class is instantiated using its defaults.

    Parameters
    ----------
    config : str, dict, None, or class instance
        The configuration for the class. See notes above.
    classObj : class
        A class that will be instantiated using the provided config.
        See notes above.

    Returns
    -------
    class instance
        An instance of the classObj class with the provided configuration

    Raises
    ------
    TypeError
        If classObj is not a class
    """
    # Check that classObj is a class
    if not inspect.isclass(classObj):
        raise TypeError("classObj must be a class.")

    # If config is an instance of this class, just return it
    if isinstance(config, classObj):
        return config

    # If config is a string, pass config as configFile
    if isinstance(config, str):
        return classObj(configFile=config)
    # If it's a dictionary, pass keyword arguments
    elif isinstance(config, dict):
        return classObj(**config)
    # If it's None, try instantiating with defaults
    elif config is None:
        return classObj()
    # If it's none of these, raise an error
    else:
        raise TypeError("config must be a string, dictionary, None, or an instance of classObj.")


def getObsLsstCmdTaskConfigDir() -> str:
    """Get the obs_lsst command line task configuration directory.

    Returns
    -------
    str
        obs_lsst command line task configuration directory.
    """

    return os.path.join(getPackageDir("obs_lsst"), "config")


def writeFile(filePath: str, content: str) -> None:
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


def readPhoSimSettingData(folderPath: str, fileName: str, atype: str) -> dict:
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
