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

__all__ = ["DonutTemplateModel"]

import lsst.afw.cameraGeom as cameraGeom
import lsst.obs.lsst as obs_lsst
import numpy as np
from lsst.ts.wep.cwfs.compensableImage import CompensableImage
from lsst.ts.wep.cwfs.donutTemplateDefault import DonutTemplateDefault
from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.utils import CamType, FilterType, getConfigDir, readPhoSimSettingData


class DonutTemplateModel(DonutTemplateDefault):
    """Class to make the donut templates from the Instrument model."""

    def makeTemplate(
        self,
        sensorName,
        defocalType,
        imageSize,
        filterLabel=FilterType.REF,
        camType=CamType.LsstCam,
        opticalModel="offAxis",
        pixelScale=0.2,
        instParams=None,
    ):
        """Make the donut template image.

        Parameters
        ----------
        sensorName : str
            The camera detector for which we want to make a template. Should
            be in "Rxx_Sxx" format.
        defocalType : enum 'DefocalType'
            The defocal state of the sensor.
        imageSize : int
            Size of template in pixels. The template will be a square.
        filterLabel : enum `FilterType`, optional
            Filter for the exposure. (The default is FilterType.REF)
        camType : enum 'CamType', optional
            Camera type. (The default is CamType.LsstCam)
        opticalModel : str, optional
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
            (The default is "offAxis")
        pixelScale : float, optional
            The pixels to arcseconds conversion factor. (The default is 0.2)
        instParams : dict or None, optional
            Instrument parameter configuration dictionary. Keys needed are:
            "obscuration", "focalLength", "apertureDiameter",
            "offset", "pixelSize". If None, then it will default to configure
            the instrument from the default policy file for the camType.
            (The default is None)

        Returns
        -------
        numpy.ndarray [int]
            The donut template as a binary image.

        Raises
        ------
        ValueError
            Camera type is not supported.
        """

        configDir = getConfigDir()

        # Load Instrument parameters
        inst = Instrument()
        if instParams is None:
            inst.configFromFile(imageSize, camType)
        else:
            inst.configFromDict(instParams, imageSize, camType)
        pixelSizeInUm = inst.pixelSize * 1e6

        if camType in (CamType.LsstCam, CamType.LsstFamCam, CamType.ComCam):
            focalPlaneLayout = readPhoSimSettingData(
                configDir, "focalplanelayout.txt", "fieldCenter"
            )

            sensorXMicron, sensorYMicron = np.array(
                focalPlaneLayout[sensorName][:2], dtype=float
            )

        elif camType == CamType.AuxTel:
            # AuxTel only works with onAxis sources
            if opticalModel != "onAxis":
                raise ValueError(
                    str(
                        f"Optical Model {opticalModel} not supported with AuxTel. "
                        + "Must use 'onAxis'."
                    )
                )
            # Defocal distance for Latiss in mm
            # for LsstCam can use the default
            # hence only need to set here
            # load the info for auxTel
            pixelSizeInMeters = inst.pixelSize  # pixel size in meters.
            pixelSizeInUm = pixelSizeInMeters * 1e6

            camera = obs_lsst.Latiss.getCamera()
            sensorName = list(camera.getNameIter())[0]  # only one detector in latiss
            detector = camera.get(sensorName)
            xp, yp = detector.getCenter(cameraGeom.FOCAL_PLANE)  # center of CCD in mm

            # multiply by 1000 to for mm --> microns conversion
            sensorXMicron = yp * 1000
            sensorYMicron = xp * 1000

        else:
            raise ValueError("Camera type (%s) is not supported." % camType)

        # Create image for mask
        img = CompensableImage()

        # Convert pixel locations to degrees
        sensorXPixel = float(sensorXMicron) / pixelSizeInUm
        sensorYPixel = float(sensorYMicron) / pixelSizeInUm

        # Multiply by pixelScale then divide by 3600 for arcsec->deg conversion
        sensorXDeg = sensorXPixel * pixelScale / 3600
        sensorYDeg = sensorYPixel * pixelScale / 3600
        fieldXY = [sensorXDeg, sensorYDeg]

        # Define position of donut at center of current sensor in degrees
        boundaryT = 0
        maskScalingFactorLocal = 1
        img.setImg(
            fieldXY, defocalType, filterLabel, image=np.zeros((imageSize, imageSize))
        )
        img.makeMask(inst, opticalModel, boundaryT, maskScalingFactorLocal)

        return img.getNonPaddedMask()