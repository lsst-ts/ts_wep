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

__all__ = ["ParamReader"]

import os
import numpy as np
import yaml
import warnings


class ParamReader(object):
    """Initialization of parameter reader of yaml format class.

    Parameters
    ----------
    filePath : str, optional
        File path. (the default is None.)
    """

    def __init__(self, filePath=None):
        if filePath is None:
            self.filePath = ""
            self._content = dict()
        else:
            self.filePath = filePath
            self._content = self._readContent(self.filePath)

    def _readContent(self, filePath):
        """Read the content of file.

        Parameters
        ----------
        filePath : str
            File path.

        Returns
        -------
        dict
            Content of file.
        """

        try:
            with open(filePath, "r") as yamlFile:
                return yaml.safe_load(yamlFile)
        except IOError as err:
            warnings.warn(f"Cannot open {filePath}: {str(err)}.", category=UserWarning)
            return dict()

    def getFilePath(self):
        """Get the parameter file path.

        Returns
        -------
        str
            Get the file path.
        """

        return self.filePath

    def setFilePath(self, filePath):
        """Set the file path and load the setting.

        Parameters
        ----------
        filePath : str
            File path.
        """

        self.__init__(filePath=filePath)

    def getContent(self):
        """Get the content.

        Returns
        -------
        list or dict
            Content.
        """

        return self._content

    def getSetting(self, param):
        """Get the setting value.

        Parameters
        ----------
        param : str
            Parameter name.

        Returns
        -------
        int, float, list, or dict
            Parameter value.

        Raises
        ------
        ValueError
            The parameter does not exist.
        """

        try:
            return self._content[param]
        except KeyError:
            raise ValueError("The '%s' does not exist." % param)

    @staticmethod
    def writeMatToFile(matrix, filePath):
        """Write the matrix data to file.

        Parameters
        ----------
        matrix : numpy.ndarray
            Matrix data.
        filePath : str
            Yaml file path.
        """

        ParamReader._writeDataToFile(matrix.tolist(), filePath)

    @staticmethod
    def _writeDataToFile(data, filePath):
        """Write the data to file.

        Parameters
        ----------
        data : Python object
            Data.
        filePath : str
            Yaml file path.

        Raises
        ------
        ValueError
            The file name should end with ".yaml".
        """

        if filePath.endswith(".yaml"):
            with open(filePath, "w") as yamlFile:
                yaml.safe_dump(data, stream=yamlFile, default_flow_style=None)
        else:
            raise ValueError("The file name should end with '.yaml'.")

    def getMatContent(self):
        """Get the matrix content.

        Returns
        -------
        numpy.ndarray
            Matrix content.
        """

        if self._content == dict():
            mat = np.array([])
        else:
            mat = np.array(self._content)

        return mat

    def updateSettingSeries(self, settingSeries):
        """Update the settings based on a serious of setting.

        Parameters
        ----------
        settingSeries : dict
            A serious of setting to update.
        """

        for param, value in settingSeries.items():
            self.updateSetting(param, value)

    def updateSetting(self, param, value):
        """Update the setting.

        Parameters
        ----------
        param : str
            Parameter name.
        value : int, float, list, or dict
            Updated value.
        """

        origVal = self.getSetting(param)
        if not isinstance(value, type(origVal)):
            warnings.warn(
                "Update with the different type of value.", category=UserWarning
            )

        self._content[param] = value

    def saveSetting(self, filePath=None):
        """Save the setting.

        Parameters
        ----------
        filePath : str, optional
            File path. If None, the loaded file path will be used. (the
            default is None.)
        """

        if filePath is None:
            filePath = self.filePath
        else:
            self.filePath = filePath

        self._writeDataToFile(self._content, filePath)

    @staticmethod
    def getAbsPath(filePath, rootPath):
        """Get the absolute file path based on the root.

        Parameters
        ----------
        filePath : str
            File path.
        rootPath : str
            Root path.

        Returns
        -------
        str
            Absolute file path based on the root if the input file path is not
            absolute.

        Raises
        ------
        ValueError
            Input file does not exist.
        """

        filePathAbs = ""
        if os.path.isabs(filePath):
            filePathAbs = filePath
        else:
            filePathAbs = os.path.join(rootPath, filePath)

        if os.path.exists(filePathAbs):
            return filePathAbs

        raise ValueError("Input file (%s) does not exist." % filePath)
