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
from lsst.ts.wep.Utility import CamType


class Instrument(object):
    """Instrument class for wavefront estimation.

    Parameters
    ----------
    instDir : str
        Instrument configuration directory.
    """

    def __init__(self, instDir):

        self.instDir = instDir
        self.instName = ""
        self._dimOfDonutImg = 0
        self.announcedDefocalDisInMm = 0.0

        self._instParams = dict()
        self.instParamFile = ParamReader()
        self.maskParamFile = ParamReader()

        self.xSensor = np.array([])
        self.ySensor = np.array([])

        self.xoSensor = np.array([])
        self.yoSensor = np.array([])

    def config(
        self,
        camType,
        dimOfDonutImgOnSensor,
        announcedDefocalDisInMm=1.5,
        instParamFileName="instParam.yaml",
        maskMigrateFileName="maskMigrate.yaml",
    ):
        """Do the configuration of Instrument.

        Parameters
        ----------
        camType : enum 'CamType'
            Camera type.
        dimOfDonutImgOnSensor : int
            Dimension of donut image on sensor in pixel.
        announcedDefocalDisInMm : float
            Announced defocal distance in mm. It is noted that the defocal
            distance offset used in calculation might be different from this
            value. (the default is 1.5.)
        instParamFileName : str, optional
            Instrument parameter file name. (the default is "instParam.yaml".)
        maskMigrateFileName : str, optional
            Mask migration (off-axis correction) file name. (the default is
            "maskMigrate.yaml".)
        """

        self.instName = self._getInstName(camType)
        self._dimOfDonutImg = int(dimOfDonutImgOnSensor)
        self.announcedDefocalDisInMm = announcedDefocalDisInMm

        # Path of instrument param file
        instFileDir = self.getInstFileDir()
        instParamFilePath = os.path.join(instFileDir, instParamFileName)
        self.instParamFile.setFilePath(instParamFilePath)

        # Load instrument parameters
        self._instParams = self.instParamFile.getContent()

        # Path of mask off-axis correction file
        # There is no such file for auxiliary telescope
        maskParamFilePath = os.path.join(instFileDir, maskMigrateFileName)
        if os.path.exists(maskParamFilePath):
            self.maskParamFile.setFilePath(maskParamFilePath)

        self._setSensorCoor()
        self._setSensorCoorAnnular()

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

        return os.path.join(self.instDir, self.instName)

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

    def setAnnDefocalDisInMm(self, annDefocalDisInMm):
        """Set the announced defocal distance in mm.

        Parameters
        ----------
        annDefocalDisInMm : float
            Announced defocal distance in mm.
        """

        self.announcedDefocalDisInMm = annDefocalDisInMm

    def getAnnDefocalDisInMm(self):
        """Get the announced defocal distance in mm.

        Returns
        -------
        float
            Announced defocal distance in mm.
        """

        return self.announcedDefocalDisInMm

    def getInstFilePath(self):
        """Get the instrument parameter file path.

        Returns
        -------
        str
            Instrument parameter file path.
        """

        return self.instParamFile.getFilePath()

    @property
    def instParams(self):
        return self._instParams

    @property
    def obscuration(self):
        return self.instParams["obscuration"]

    @property
    def focalLength(self):
        return self.instParams["focalLength"]

    @property
    def apertureDiameter(self):
        return self.instParams["apertureDiameter"]

    @property
    def defocalDisOffset(self):
        offsetKey = "%.1fmm" % self.announcedDefocalDisInMm
        return self.instParams["offset"][offsetKey]

    @property
    def pixelSize(self):
        return self.instParams["pixelSize"]

    @property
    def maskOffAxisCorr(self):
        self._maskOffAxisCorr = self.maskParamFile.getMatContent()
        return self._maskOffAxisCorr

    @property
    def dimOfDonutImg(self):
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
