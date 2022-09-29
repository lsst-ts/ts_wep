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

__all__ = ["Instrument"]

import os
import numpy as np

from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.Utility import CamType, getConfigDir, getCamNameFromCamType


class Instrument(object):
    """
    Instrument class to have the instrument information
    used in the Algorithm class to solve the TIE.
    """

    def __init__(self):

        # Set initial parameters
        self._dimOfDonutImg = 0

        # Keys that should be set in configuration file
        paramKeys = [
            "obscuration",
            "focalLength",
            "apertureDiameter",
            "offset",
            "pixelSize",
        ]
        self._instParams = {key: "" for key in paramKeys}
        self._instName = None

        self.maskParamReader = ParamReader()
        self._maskOffAxisCorr = []

        self.xSensor = np.array([])
        self.ySensor = np.array([])

        self.xoSensor = np.array([])
        self.yoSensor = np.array([])

    def configFromDict(
        self, configDict, dimOfDonutImgOnSensor, camType, maskConfigFile=None
    ):
        """Configure the instrument class from a dictionary.

        Parameters
        ----------
        configDict : dict
            Instrument parameter configuration dictionary. Keys needed are:
            "obscuration", "focalLength", "apertureDiameter",
            "offset", "pixelSize".
        dimOfDonutImgOnSensor : int
            Dimension of donut image on sensor in pixel.
        camType : enum 'CamType'
            Camera type.
        maskConfigFile : str or None, optional
            Mask migration (off-axis correction) file path.
            If None will load the default from policy/cwfs folder.
            (The default is None.)

        Raises
        ------
        AssertionError
            ConfigDict keys do not match required keys in self._instParams.
        ValueError
            Mask migrate file does not exist.
        """

        self._dimOfDonutImg = int(dimOfDonutImgOnSensor)

        # Check that dictionary keys match
        assert (
            self._instParams.keys() == configDict.keys()
        ), f"Config Dict Keys: {configDict.keys()} do not match required \
            instParamKeys: {self._instParams.keys()}"

        # Set configuration parameters
        for key in configDict.keys():
            self._instParams[key] = configDict[key]

        # Load mask configuration file for instrument
        if maskConfigFile is not None:
            if not os.path.exists(maskConfigFile):
                raise ValueError(
                    f"Mask migrate file at {maskConfigFile} does not exist."
                )
            self.maskParamReader.setFilePath(maskConfigFile)
            self._maskOffAxisCorr = self.maskParamReader.getMatContent()
        else:
            # Load default
            self.setDefaultMaskParams(camType)

        self._setSensorCoor()
        self._setSensorCoorAnnular()

    def configFromFile(
        self, dimOfDonutImgOnSensor, camType, instConfigFile=None, maskConfigFile=None
    ):
        """Configure the instrument class from a configuration file.

        Parameters
        ----------
        dimOfDonutImgOnSensor : int
            Dimension of donut image on sensor in pixel.
        camType : enum 'CamType'
            Camera type.
        instConfigFile : str or None, optional
            Instrument parameter configuration file path. If None will
            load the default from policy/cwfs folder. (The default is None.)
        maskConfigFile : str or None, optional
            Mask migration (off-axis correction) file path.
            If None will load the default from policy/cwfs folder.
            (The default is None.)

        Raises
        ------
        ValueError
            Instrument configuration file does not exist.
        ValueError
            Mask migrate file does not exist.
        """

        self._dimOfDonutImg = int(dimOfDonutImgOnSensor)
        self._instName = self._getInstName(camType)

        # Load instrument configuration file
        if instConfigFile is None:
            camName = getCamNameFromCamType(camType)
            instFileDir = os.path.join(getConfigDir(), "cwfs", "instData", camName)
            instParamFileName = "instParam.yaml"
            instConfigFilePath = os.path.join(instFileDir, instParamFileName)
        else:
            instConfigFilePath = instConfigFile

        if not os.path.exists(instConfigFilePath):
            raise ValueError(
                f"Instrument configuration file at {instConfigFilePath} does not exist."
            )
        instParamReader = ParamReader()
        instParamReader.setFilePath(instConfigFilePath)
        self._instParams = instParamReader.getContent()

        # Load mask configuration file
        if maskConfigFile is not None:
            if not os.path.exists(maskConfigFile):
                raise ValueError(
                    f"Mask migrate file at {maskConfigFile} does not exist."
                )
            self.maskParamReader.setFilePath(maskConfigFile)
            self._maskOffAxisCorr = self.maskParamReader.getMatContent()
        else:
            # Load default
            self.setDefaultMaskParams(camType)

        self._setSensorCoor()
        self._setSensorCoorAnnular()

    def setDefaultMaskParams(self, camType, maskParamFileName="maskMigrate.yaml"):
        """Load the default mask off-axis corrections. Note that there
        is no such file for auxiliary telescope.

        Parameters
        ----------
        camType : enum 'CamType'
            Camera type.
        maskParamFileName : str, optional
            Mask parameter file name in the policy/cwfs/instData/`instName`
            directory. (The default is "maskMigrate.yaml".)
        """

        # Path of mask off-axis correction file
        if camType not in [CamType.AuxTel, CamType.AuxTelZWO]:
            self._instName = self._getInstName(camType)
            instFileDir = self.getInstFileDir()
            maskParamFilePath = os.path.join(instFileDir, maskParamFileName)
            self.maskParamReader.setFilePath(maskParamFilePath)
            self._maskOffAxisCorr = self.maskParamReader.getMatContent()

    def _getInstName(self, camType):
        """Get the instrument name.

        Parameters
        ----------
        camType : enum 'CamType'
            Camera type.

        Returns
        -------
        str
            Instrument name.

        Raises
        ------
        ValueError
            Camera type is not supported.
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
            raise ValueError("Camera type (%s) is not supported." % camType)

    def getInstFileDir(self):
        """Get the instrument parameter file directory.

        Returns
        -------
        str
            Instrument parameter file directory.
        """

        return os.path.join(getConfigDir(), "cwfs", "instData", self._instName)

    def _setSensorCoor(self):
        """Set the sensor coordinate."""

        # 0.5 is the half of single pixel
        ySensorGrid, xSensorGrid = np.mgrid[
            -(self.dimOfDonutImg / 2 - 0.5) : (self.dimOfDonutImg / 2 + 0.5),
            -(self.dimOfDonutImg / 2 - 0.5) : (self.dimOfDonutImg / 2 + 0.5),
        ]

        sensorFactor = self.getSensorFactor()
        denominator = self.dimOfDonutImg / 2 / sensorFactor

        self.xSensor = xSensorGrid / denominator
        self.ySensor = ySensorGrid / denominator

    def _setSensorCoorAnnular(self):
        """Set the sensor coordinate with the annular aperature."""

        self.xoSensor = self.xSensor.copy()
        self.yoSensor = self.ySensor.copy()

        # Get the position index that is out of annular aperature range
        r2Sensor = self.xSensor**2 + self.ySensor**2
        idx = (r2Sensor > 1) | (r2Sensor < self.obscuration**2)

        # Define the value to be NaN if it is not in pupul
        self.xoSensor[idx] = np.nan
        self.yoSensor[idx] = np.nan

    @property
    def instParams(self):
        """Dictionary of the instrument configuration parameters."""
        return self._instParams

    @property
    def obscuration(self):
        """Obscuration (inner_radius / outer_radius of primary mirror)."""
        return self.instParams["obscuration"]

    @property
    def focalLength(self):
        """The focal length of telescope in meters."""
        return self.instParams["focalLength"]

    @property
    def apertureDiameter(self):
        """The aperture diameter in meters."""
        return self.instParams["apertureDiameter"]

    @property
    def defocalDisOffset(self):
        """The defocal distance offset in meters."""
        return self.instParams["offset"]

    @property
    def pixelSize(self):
        """The camera pixel size in meters."""
        return self.instParams["pixelSize"]

    @property
    def maskOffAxisCorr(self):
        """The mask off-axis correction."""
        return self._maskOffAxisCorr

    @property
    def dimOfDonutImg(self):
        """The dimension of the donut image size on the sensor in pixels."""
        return self._dimOfDonutImg

    def getMarginalFocalLength(self):
        """Get the marginal focal length in meter.

        Marginal_focal_length = sqrt(f^2 - (D/2)^2)

        Returns
        -------
        float
            Marginal focal length in meter.
        """

        marginalFL = np.sqrt(self.focalLength**2 - (self.apertureDiameter / 2) ** 2)

        return marginalFL

    def getSensorFactor(self):
        """Get the sensor factor.

        Returns
        -------
        float
            Sensor factor.
        """

        sensorFactor = self.dimOfDonutImg / (
            self.defocalDisOffset
            * self.apertureDiameter
            / self.focalLength
            / self.pixelSize
        )

        return sensorFactor

    def getSensorCoor(self):
        """Get the sensor coordinate.

        Returns
        -------
        numpy.ndarray
            X coordinate.
        numpy.ndarray
            Y coordinate.
        """
        # Set each time with current instParams
        self._setSensorCoor()
        return self.xSensor, self.ySensor

    def getSensorCoorAnnular(self):
        """Get the sensor coordinate with the annular aperature.

        Returns
        -------
        numpy.ndarray
            X coordinate.
        numpy.ndarray
            Y coordinate.
        """
        # Set each time with current instParams
        self._setSensorCoorAnnular()
        return self.xoSensor, self.yoSensor

    def calcSizeOfDonutExpected(self):
        """Calculate the size of expected donut (diameter).

        Returns
        -------
        float
            Size of expected donut (diameter) in pixel.
        """

        fNumber = self.focalLength / self.apertureDiameter

        return self.defocalDisOffset / fNumber / self.pixelSize
