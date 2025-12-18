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

__all__ = ["DonutStamp"]

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.afw.geom import SkyWcs
from lsst.daf.base import PropertyList
from lsst.meas.algorithms.stamps import AbstractStamp
from lsst.ts.wep import Instrument
from lsst.ts.wep.image import Image
from lsst.ts.wep.imageMapper import ImageMapper
from lsst.ts.wep.utils import getCameraFromButlerName


@dataclass
class DonutStamp(AbstractStamp):
    """Single donut stamp

    Note all of the top-level stamp information is in the data
    visualization coordinate system (DVCS), while the information
    packaged in self.wep_im is in the global camera coordinate
    system (CCS). See https://sitcomtn-003.lsst.io and the Image
    docstring for more information.

    Parameters
    ----------
    stamp_im : `lsst.afw.image.MaskedImageF`
        The actual pixel values for the postage stamp
    sky_position : `lsst.geom.SpherePoint`
        Position of the center of the stamp. Note the user
        must keep track of the coordinate system
    centroid_position : `lsst.geom.Point2D`
        Position of the center of the stamp in pixels
    blend_centroid_positions : `numpy.ndarray`
        Positions of the centroids (in pixels) for sources
        blended with the central source
    defocal_type : `str`
        Defocal state of the stamp. "extra" or "intra" are
        allowed values.
    defocal_distance : `float`
        Defocal offset of the detector in mm. If the detector was not
        actually shifted, this should be the equivalent detector offset.
    detector_name : `str`
        CCD where the donut is found
    cam_name : `str`
        Camera name for the stamp image. "LSSTCam" or "LSSTComCam"
        are available camera names currently.
    bandpass : `str`
        The bandpass for the stamp image.
    donut_id: `str`
        The donut id defined at detection stage.
    archive_element : `afwTable.io.Persistable`, optional
        Archive element (e.g. Transform or WCS) associated with this stamp.
        (the default is None.)
    wep_im : `lsst.ts.wep.image.Image`
        ts.wep Image object used for mask creation, wavefront estimation, etc.
        The information contained in this object has been transformed to the
        camera coordinate system (CCS), with the CWFSs rotated to the same
        orientation as the science sensors. It is this object that will be used
        to interface with the wavefront estimator.

    """

    stamp_im: afwImage.MaskedImageF
    sky_position: lsst.geom.SpherePoint
    centroid_position: lsst.geom.Point2D
    blend_centroid_positions: np.ndarray
    defocal_type: str
    defocal_distance: float
    detector_name: str
    cam_name: str
    bandpass: str
    donut_id: str
    archive_element: Optional[afwTable.io.Persistable] = None
    wep_im: Image = field(init=False)

    def __post_init__(self) -> None:
        """
        This method sets up the WEP Image after initialization
        because we need to use the parameters set in the original `__init__`.
        """
        self._setWepImage()

    @classmethod
    def factory(
        cls,
        stamp_im: afwImage.MaskedImageF,
        metadata: PropertyList,
        index: int,
        archive_element: afwTable.io.Persistable | None = None,
    ) -> "DonutStamp":
        """This method is needed to service the FITS reader.
        We need a standard interface to construct objects like this.
        Parameters needed to construct this object are passed in via
        a metadata dictionary and then passed to the constructor of
        this class. They should each point to lists of values.

        Parameters
        ----------
        stamp_im : `lsst.afw.image.MaskedImage`
            Pixel data to pass to the constructor
        metadata : `lsst.daf.base.PropertyList`
            PropertyList containing the information
            needed by the constructor.
        index : `int`
            Index into the lists in ``metadata``
        archive_element : `afwTable.io.Persistable`, optional
            Archive element (e.g. Transform or WCS) associated with this stamp.
            (the default is None.)

        Returns
        -------
        DonutStamp
            An instance of this class
        """
        # Include blend centroid positions if they exist
        if metadata.get("BLEND_CX") is not None:
            blend_centroid_positions = np.array(
                [
                    metadata.getArray("BLEND_CX")[index].split(","),
                    metadata.getArray("BLEND_CY")[index].split(","),
                ],
                dtype=float,
            ).T
        else:
            blend_centroid_positions = np.array([["nan"], ["nan"]], dtype=float).T

        return cls(
            stamp_im=stamp_im,
            archive_element=archive_element,
            sky_position=lsst.geom.SpherePoint(
                lsst.geom.Angle(metadata.getArray("RA_DEG")[index], lsst.geom.degrees),
                lsst.geom.Angle(metadata.getArray("DEC_DEG")[index], lsst.geom.degrees),
            ),
            # Centroid position "CENT_X" and "CENT_Y" is in pixels
            centroid_position=lsst.geom.Point2D(
                metadata.getArray("CENT_X")[index], metadata.getArray("CENT_Y")[index]
            ),
            # Blend centroid positions "BLEND_CX" and "BLEND_CY" in pixels
            blend_centroid_positions=blend_centroid_positions,
            # "DET_NAME" stands for detector (CCD) name
            detector_name=metadata.getArray("DET_NAME")[index],
            # "CAM_NAME" stands for camera name
            cam_name=metadata.getArray("CAM_NAME")[index],
            # "DFC_TYPE" stands for defocal type in string form.
            # Need to convert to DefocalType
            defocal_type=metadata.getArray("DFC_TYPE")[index],
            # "DFC_DIST" stands for defocal distance
            # If this is an old version of the stamps without a defocal
            # distance set this to default value of 1.5 mm.
            defocal_distance=(
                metadata.getArray("DFC_DIST")[index] if metadata.get("DFC_DIST") is not None else 1.5
            ),
            # "BANDPASS" stands for the exposure bandpass
            # If this is an old version of the stamps without bandpass
            # information then an empty string ("") will be set as default.
            bandpass=(metadata.getArray("BANDPASS")[index] if metadata.get("BANDPASS") is not None else ""),
            # "DONUT_ID" is the donut identification number
            # created at detection stage. If this is an old
            # version of stamps without donut id,
            # an empty string will be set as default
            donut_id=(metadata.getArray("DONUT_ID")[index] if metadata.get("DONUT_ID") is not None else ""),
        )

    def getCamera(self) -> Camera:
        """
        Get the proper camera object for the donuts.

        Returns
        -------
        `lsst.afw.cameraGeom.Camera`
            Camera object for the exposures.

        Raises
        ------
        `ValueError`
            The camera is not supported.
        """

        return getCameraFromButlerName(self.cam_name)

    def calcFieldXY(self) -> tuple[float, float]:
        """
        Calculate the X, Y field position of the centroid in degrees.

        Note the coordinate system is DVCS!

        Returns
        -------
        `float`
            Field x position in degrees.
        `float`
            Field y position in degrees.
        """

        cam = self.getCamera()
        det = cam.get(self.detector_name)

        field_x, field_y = det.transform(self.centroid_position, PIXELS, FIELD_ANGLE)

        return np.degrees(field_x), np.degrees(field_y)

    def _setWepImage(self) -> None:
        """Return a ts.wep.image.Image object for the stamp.

        Note that the information from the butler is in the data visualization
        coordinate system (DVCS), but the WEP Image is in the global camera
        coordinate system (CCS). These coordinate systems are related by a
        transpose. See sitcomtn-003.lsst.io for more information.

        Furthermore, CWFS images that arrive from the butler are rotated with
        respect to the science sensors. The info in the WEP Images has been
        de-rotated so that everything aligns with the global coordinate system
        used by the science sensors.

        Returns
        -------
        ts.wep.image.Image

        Raises
        ------
        RuntimeError
            If the rotation angle of the detector with respect to the science
            sensors is not an integer multiple of 90 degrees.
        """
        # Get the camera and detector
        camera = self.getCamera()
        detector = camera.get(self.detector_name)

        # Get the rotation with respect to the science sensors
        eulerZ = detector.getOrientation().getYaw().asDegrees()
        nRot = int(eulerZ // 90)
        if not np.isclose(eulerZ % 90, 0):
            raise RuntimeError(
                f"The detector is rotated {eulerZ} deg with respect to the science "
                "sensors, but _setWepImage() only works for sensors whose rotations "
                "are an integer multiple of 90 deg."
            )

        # Rotate image to orientation of science sensors
        #   Note that np.rot90 performs right-handed rotations on arrays
        #   (i.e. rotates them counter-clockwise). However, because images
        #   are stored "upside down" in numpy (e.g. you plot them with
        #   plt.imshow(..., origin="lower")), np.rot90 actually performs
        #   left-handed rotations on images. We therefore need to use
        #   negative nRot to perform a regular right-handed rotation
        image = np.rot90(self.stamp_im.getImage().getArray(), -nRot)

        # Transpose the image (DVCS -> CCS)
        image = image.T

        # Get the field angle, and transpose (DVCS -> CCS)
        fieldAngle = self.calcFieldXY()
        fieldAngle = (fieldAngle[1], fieldAngle[0])

        # Determine the blend offsets
        if self.blend_centroid_positions.size > 0:
            # Get the offsets in the original pixel coordinates
            blendOffsets = self.blend_centroid_positions - self.centroid_position

            # Rotate the coordinates to match the science sensors
            rotMat = np.array([[0, -1], [1, 0]])
            rotMatMultiple = np.linalg.matrix_power(rotMat, nRot)
            blendOffsets = np.transpose(rotMatMultiple @ blendOffsets.T)

            # Transpose the coordinates (DVCS -> CCS)
            blendOffsets = blendOffsets[:, ::-1]

        else:
            blendOffsets = None

        # Package everything in an Image object
        wepImage = Image(
            image=image,
            fieldAngle=fieldAngle,
            defocalType=self.defocal_type,
            bandLabel=self.bandpass,
            blendOffsets=blendOffsets,
        )

        self.wep_im = wepImage

    def makeMask(
        self,
        instrument: Instrument,
        opticalModel: str = "offAxis",
        dilate: int = 0,
        dilateBlends: int = 0,
    ) -> None:
        """Create the mask for the image.

        Note that technically the image masks depend on the optical
        aberrations, but this function assumes the aberrations are zero.

        Parameters
        ----------
        instrument : Instrument
            Instrument to use.
        opticalModel : str, optional
            The optical model to use for mapping between the image and
            pupil planes. Can be "onAxis", or "offAxis". onAxis is an
            analytic model appropriate for donuts near the optical axis.
            It is valid for both slow and fast optical systems. The offAxis
            model is a numerically-fit model that is valid for fast optical
            systems at wide field angles. offAxis requires an accurate Batoid
            model.
        dilate : int, optional
            How many times to dilate the central mask. This adds a boundary
            of that many pixels to the mask. (the default is 0)
        dilateBlends : int, optional
            How many times to dilate the blend mask.
        """
        # Create the image mapper
        imageMapper = ImageMapper(instConfig=instrument, opticalModel=opticalModel)

        # Create the masks
        imageMapper.createImageMasks(
            self.wep_im,
            isBinary=True,
            dilate=dilate,
            dilateBlends=dilateBlends,
            doMaskBlends=False,
        )
        maskSource, maskBlends, maskBackground = self.wep_im.masks

        # Add mask planes for the source and blend
        afwImage.Mask.addMaskPlane("DONUT")
        afwImage.Mask.addMaskPlane("BLEND")

        # Create the stamp mask
        stampMask = maskSource * afwImage.Mask.getPlaneBitMask("DONUT")
        stampMask += maskBlends * afwImage.Mask.getPlaneBitMask("BLEND")
        stampMask = stampMask.astype(np.int32)

        # This mask is in the global CCS (see the docstring for
        # self._setWepImage()). We need to put it back in the
        # coordinate system of the info in the butler

        # Transpose the mask (CCS -> DVCS)
        stampMask = stampMask.T

        # Rotate to sensor orientation
        camera = self.getCamera()
        detector = camera.get(self.detector_name)
        eulerZ = -detector.getOrientation().getYaw().asDegrees()
        nRot = int(eulerZ // 90)
        stampMask = np.rot90(stampMask, -nRot)

        # First make sure the mask doesn't already have donut/blend bits
        # This is so if this function gets called multiple times, the donut
        # and blend bits don't get re-added.
        mask0 = self.stamp_im.mask.array.copy()
        bit = self.stamp_im.mask.getMaskPlaneDict()["DONUT"]
        mask0 &= ~(1 << bit)
        bit = self.stamp_im.mask.getMaskPlaneDict()["BLEND"]
        mask0 &= ~(1 << bit)

        # Add original mask to the new mask
        stampMask += mask0

        # Save mask
        self.stamp_im.setMask(afwImage.Mask(stampMask.copy()))

    def getLinearWCS(self) -> SkyWcs:
        """
        Get the linear WCS for the stamp.

        Returns
        -------
        `lsst.afw.geom.SkyWcs`
            Linear WCS for the stamp.
        """

        return self.archive_element
