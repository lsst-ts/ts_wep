import os
import numpy as np

from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.Utility import CamType


class Instrument(object):

    def __init__(self, instDir):
        """Instrument class for wavefront estimation.

        Parameters
        ----------
        instDir : str
            Instrument configuration directory.
        """

        self.instDir = instDir
        self.instName = ""
        self.dimOfDonut = 0
        self.announcedDefocalDisInMm = 0.0

        self.instParamFile = ParamReader()
        self.maskParamFile = ParamReader()

        self.xSensor = np.array([])
        self.ySensor = np.array([])

        self.xoSensor = np.array([])
        self.yoSensor = np.array([])

    def config(self, camType, dimOfDonutOnSensor,
               announcedDefocalDisInMm=1.5,
               instParamFileName="instParam.yaml",
               maskMigrateFileName="maskMigrate.yaml"):
        """Do the configuration of Instrument.

        Parameters
        ----------
        camType : enum 'CamType'
            Camera type.
        dimOfDonutOnSensor : int
            Dimension of image on sensor in pixel.
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
        self.dimOfDonut = int(dimOfDonutOnSensor)
        self.announcedDefocalDisInMm = announcedDefocalDisInMm

        # Path of instrument param file
        instFileDir = self.getInstFileDir()
        instParamFilePath = os.path.join(instFileDir, instParamFileName)
        self.instParamFile.setFilePath(instParamFilePath)

        # Path of mask off-axis correction file
        maskParamFilePath = os.path.join(instFileDir, maskMigrateFileName)
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

        if (camType == CamType.LsstCam):
            return "lsst"
        elif (camType == CamType.ComCam):
            return "comcam"
        elif (camType == CamType.AuxTel):
            return "auxtel"
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

        ySensorGrid, xSensorGrid = np.mgrid[
            -(self.dimOfDonut/2-0.5):(self.dimOfDonut/2 + 0.5),
            -(self.dimOfDonut/2-0.5):(self.dimOfDonut/2 + 0.5)]

        sensorFactor = self.getSensorFactor()
        denominator = self.dimOfDonut / 2 / sensorFactor

        self.xSensor = xSensorGrid / denominator
        self.ySensor = ySensorGrid / denominator

    def _setSensorCoorAnnular(self):
        """Set the sensor coordinate with the annular aperature."""

        self.xoSensor = self.xSensor.copy()
        self.yoSensor = self.ySensor.copy()

        # Get the position index that is out of annular aperature range
        obscuration = self.getObscuration()
        r2Sensor = self.xSensor**2 + self.ySensor**2
        idx = (r2Sensor > 1) | (r2Sensor < obscuration**2)

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

    def getMaskOffAxisCorr(self):
        """Get the mask off-axis correction.

        Returns
        -------
        numpy.ndarray
            Mask off-axis correction.
        """

        return self.maskParamFile.getMatContent()

    def getDimOfDonutOnSensor(self):
        """Get the dimension of donut's size on sensor in pixel.

        Returns
        -------
        int
            Dimension of donut's size on sensor in pixel.
        """

        return self.dimOfDonut

    def getObscuration(self):
        """Get the obscuration.

        Returns
        -------
        float
            Obscuration.
        """

        return self.instParamFile.getSetting("obscuration")

    def getFocalLength(self):
        """Get the focal length of telescope in meter.

        Returns
        -------
        float
            Focal length of telescope in meter.
        """

        return self.instParamFile.getSetting("focalLength")

    def getApertureDiameter(self):
        """Get the aperture diameter in meter.

        Returns
        -------
        float
            Aperture diameter in meter.
        """

        return self.instParamFile.getSetting("apertureDiameter")

    def getDefocalDisOffset(self):
        """Get the defocal distance offset in meter.

        Returns
        -------
        float
            Defocal distance offset in meter.
        """

        offset = self.instParamFile.getSetting("offset")
        defocalDisInMm = "%.1fmm" % self.announcedDefocalDisInMm

        return offset[defocalDisInMm]

    def getCamPixelSize(self):
        """Get the camera pixel size in meter.

        Returns
        -------
        float
            Camera pixel size in meter.
        """

        return self.instParamFile.getSetting("pixelSize")

    def getMarginalFocalLength(self):
        """Get the marginal focal length in meter.

        Marginal_focal_length = sqrt(f^2 - (D/2)^2)

        Returns
        -------
        float
            Marginal focal length in meter.
        """

        focalLength = self.getFocalLength()
        apertureDiameter = self.getApertureDiameter()
        marginalFL = np.sqrt(focalLength**2 - (apertureDiameter/2)**2)

        return marginalFL

    def getSensorFactor(self):
        """Get the sensor factor.

        Returns
        -------
        float
            Sensor factor.
        """

        offset = self.getDefocalDisOffset()
        apertureDiameter = self.getApertureDiameter()
        focalLength = self.getFocalLength()
        pixelSize = self.getCamPixelSize()
        sensorFactor = self.dimOfDonut / (
            offset * apertureDiameter / focalLength / pixelSize)

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

        return self.xoSensor, self.yoSensor


if __name__ == "__main__":
    pass
