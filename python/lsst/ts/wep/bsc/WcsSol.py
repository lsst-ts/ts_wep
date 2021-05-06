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

import numpy as np

import lsst.geom
import lsst.obs.lsst as obs_lsst
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS


class WcsSol(object):
    def __init__(self, camera=None):
        """Initialize the world coordinate system (WCS) solution class.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera, optional
            A collection of Detectors that also supports coordinate
            transformation. (the default is None.)
        """

        # Sky WCS solution that uses the functions in obs_lsst
        self.skyWcs = None

        if camera is None:
            self._camera = obs_lsst.lsstCamMapper.LsstCamMapper().camera
        else:
            self._camera = camera

    def setCamera(self, camera):
        """Set the camera object.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        """

        self._camera = camera

    def getCamera(self):
        """Get the camera object.

        Returns
        -------
        lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        """

        return self._camera

    def setObsMetaData(self, ra, dec, rotSkyPos, centerCcd="R22_S11"):
        """Set up the WCS by specifying the observation meta data.

        Parameters
        ----------
        ra : float
            Pointing ra in degree.
        dec : float
            Pointing decl in degree.
        rotSkyPos : float
            The orientation of the telescope in degrees.
        centerCcd: str, optional
            Center Ccd on the camera (the default is "R22_S11").
        """

        boresightPointing = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        self.centerCcd = centerCcd
        self.skyWcs = createInitialSkyWcsFromBoresight(
            boresightPointing,
            rotSkyPos * lsst.geom.degrees,
            self._camera[self.centerCcd],
            flipX=False,
        )

    def _formatCoordList(self, val1, val2, val1Name, val2Name):
        """
        Check if value entered is a single int or float rather than a list.
        If so, turn it into a list and return. Also make sure number of entered
        values are the same (i.e. number of ra values same as number
        of dec values).

        Parameters
        ----------
        val1: int, float or list
            Values of coordinate 1 (ra value(s) for example).
        val2: int, float or list
            Values of coordinate 2 (dec if val1 is ra).
        val1Name: str
            Name of the type of coordinate in val1 ("ra" for example).
        val2Name: str
            Name of the type of coordinate in val2 ("dec" is val1Name is "ra").

        Returns
        -------
        val1: list
            val1 as a list of values
        val2: list
            val2 as a list of values
        """

        if isinstance(val1, (int, float)) | isinstance(val2, (int, float)):
            val1 = [val1]
            val2 = [val2]
        assertMessage = "Size of %s not same as %s" % (val1Name, val2Name)
        assert len(val1) == len(val2), assertMessage

        return val1, val2

    def _raDecFromPixelCoords(self, xPix, yPix, chipNameList):
        """Convert pixel coordinates into RA, Dec.

        Parameters
        ----------
        xPix : list
            xPix is the x pixel coordinate.
        yPix : list
            yPix is the y pixel coordinate.
        chipNameList : list
            chipNameList is the name of the chip(s)
            on which the pixel coordinates are defined.

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the RA coordinate and
            the second row is the Dec coordinate (both in degrees; in the
            International Celestial Reference System).
        """

        raList = []
        decList = []
        centerChip = self._camera[self.centerCcd]

        for chipX, chipY, chipName in zip(xPix, yPix, chipNameList):

            cameraChip = self._camera[chipName]
            # Get x,y on specified detector in terms of mm from center of cam
            camXyMm = cameraChip.transform(
                lsst.geom.Point2D(chipX, chipY), PIXELS, FOCAL_PLANE
            )
            # Convert mm to pixels
            camPoint = centerChip.transform(camXyMm, FOCAL_PLANE, PIXELS)
            # Calculate correct ra, dec
            raPt, decPt = self.skyWcs.pixelToSky(camPoint)

            raList.append(raPt.asDegrees())
            decList.append(decPt.asDegrees())

        return np.array([raList, decList])

    def raDecFromPixelCoords(self, xPix, yPix, chipName):
        """Convert pixel coordinates into RA, Dec.

        Parameters
        ----------
        xPix : float or numpy.ndarray
            xPix is the x pixel coordinate.
        yPix : float or numpy.ndarray
            yPix is the y pixel coordinate.
        chipName : str or numpy.ndarray
            chipName is the name of the chip(s) on which the pixel coordinates
            are defined.  This can be an array (in which case there should be
            one chip name for each (xPix, yPix) coordinate pair), or a single
            value (in which case, all of the (xPix, yPix) points will be
            reckoned on that chip).

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the RA coordinate and
            the second row is the Dec coordinate (both in degrees; in the
            International Celestial Reference System).
        """

        xPix, yPix = self._formatCoordList(xPix, yPix, "xPix", "yPix")
        nPts = len(xPix)

        if isinstance(chipName, np.ndarray):
            chipNameList = chipName.tolist()
        else:
            chipNameList = [chipName] * nPts

        raDecArray = self._raDecFromPixelCoords(xPix, yPix, chipNameList)

        if nPts == 1:
            return raDecArray.flatten()
        else:
            return raDecArray

    def _focalPlaneCoordsFromRaDec(self, ra, dec):
        """Get the focal plane coordinates for all objects in the catalog.

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x focal plane
            coordinate and the second row is the y focal plane coordinate
            (both in millimeters).
        """

        xMmList = []
        yMmList = []

        for raPt, decPt in zip(ra, dec):

            cameraChip = self._camera[self.centerCcd]

            raDecPt = lsst.geom.SpherePoint(
                raPt * lsst.geom.degrees, decPt * lsst.geom.degrees
            )
            # Get xy in pixels
            xyPix = self.skyWcs.skyToPixel(raDecPt)
            xMm, yMm = cameraChip.transform(xyPix, PIXELS, FOCAL_PLANE)
            xMmList.append(xMm)
            yMmList.append(yMm)

        return np.array([xMmList, yMmList])

    def _pixelCoordsFromRaDec(self, ra, dec, chipNameList):
        """Get the pixel positions (or (-9999, -9999) if not on a chip)
        for objects based on their RA, and Dec (in degrees).

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.
        chipNameList : list
            chipName designates the names of the chips on which the pixel
            coordinates will be reckoned. If all entries are None,
            this method will calculate which chip each (RA, Dec) pair actually
            falls on, and return pixel coordinates for each (RA, Dec) pair on
            the appropriate chip. (the default is None).
            If the chipName is specified but the (RA, Dec) entry does not fall
            on the specified chip then the function will return (-9999, -9999)
            as the pixel coordinates.

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x pixel coordinate
            and the second row is the y pixel coordinate.
        """

        xMmList, yMmList = self._focalPlaneCoordsFromRaDec(ra, dec)

        xPixList = []
        yPixList = []
        detList = []

        for xMm, yMm, chipName in zip(xMmList, yMmList, chipNameList):

            # Look up detector corresponding to that position
            xyMm = lsst.geom.Point2D(xMm, yMm)
            try:
                det = self._camera.findDetectors(xyMm, FOCAL_PLANE)[0]
                detList.append(det.getName())
                # If detector is specified but pixels do not lie on that
                # detector then return -9999
                if (chipName is not None) and (det.getName() != chipName):
                    xPix, yPix = (-9999, -9999)
                else:
                    xPix, yPix = det.transform(xyMm, FOCAL_PLANE, PIXELS)
            except IndexError:
                xPix, yPix = (-9999, -9999)
            xPixList.append(xPix)
            yPixList.append(yPix)

        return np.array([xPixList, yPixList])

    def pixelCoordsFromRaDec(self, ra, dec, chipName=None):
        """Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in degrees).

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.
        chipName : numpy.ndarray, str, None
            chipName designates the names of the chips on which the pixel
            coordinates will be reckoned. If an array, there must be as many
            chipNames as there are (RA, Dec) pairs. If a single value, all of
            the pixel coordinates will be reckoned on the same chip. If None,
            this method will calculate which chip each(RA, Dec) pair actually
            falls on, and return pixel coordinates for each (RA, Dec) pair on
            the appropriate chip. (the default is None.)

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x pixel coordinate
            and the second row is the y pixel coordinate.

        Raises
        ------
        ValueError
            If chipName is not one of numpy.ndarray, str, or None.
        """

        ra, dec = self._formatCoordList(ra, dec, "ra", "dec")
        nPts = len(ra)

        if chipName is None:
            chipNameList = [None] * nPts
        elif isinstance(chipName, np.ndarray):
            chipNameList = chipName.tolist()
        elif isinstance(chipName, str):
            chipNameList = [chipName] * nPts
        else:
            raise ValueError(
                "chipName is an unallowed type. Can be None, string or array of strings."
            )

        pixArray = self._pixelCoordsFromRaDec(
            ra,
            dec,
            chipNameList,
        )

        if nPts == 1:
            return pixArray.flatten()
        else:
            return pixArray

    def focalPlaneCoordsFromRaDec(self, ra, dec):
        """Get the focal plane coordinates for all objects in the catalog.

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x focal plane
            coordinate and the second row is the y focal plane coordinate
            (both in millimeters).
        """

        ra, dec = self._formatCoordList(ra, dec, "ra", "dec")
        nPts = len(ra)

        focalPlaneArray = self._focalPlaneCoordsFromRaDec(ra, dec)

        if nPts == 1:
            return focalPlaneArray.flatten()
        else:
            return focalPlaneArray
