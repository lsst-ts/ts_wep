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

import lsst.geom
import numpy as np
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.obs.lsst as obs_lsst
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.meas.algorithms.stamps import AbstractStamp
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.Utility import DefocalType


@dataclass
class DonutStamp(AbstractStamp):
    """Single donut stamp

    Parameters
    ----------
    stamp_im : `lsst.afw.image.MaskedImageF`
        The actual pixel values for the postage stamp
    sky_position : `lsst.geom.SpherePoint`
        Position of the center of the stamp. Note the user
        must keep track of the coordinate system
    centroid_position : `lsst.geom.Point2D`
        Position of the center of the stamp in pixels
    defocal_type : `str`
        Defocal state of the stamp. "extra" or "intra" are
        allowed values.
    defocal_distance : `float`
        Defocal offset of the instrument in mm.
    detector_name : `str`
        CCD where the donut is found
    cam_name : `str`
        Camera name for the stamp image. "LSSTCam" or "LSSTComCam"
        are available camera names currently.
    archive_element : `afwTable.io.Persistable`, optional
        Archive element (e.g. Transform or WCS) associated with this stamp.
        (the default is None.)
    comp_im : `CompensableImage`, init=False
        CompensableImage object to create masks for the stamp. This is
        initialized in the __post_init__ stage of the dataclass.
    mask_comp : `afwImage.Mask`, init=False
        Padded Mask for use at the offset planes. This is
        initialized in the __post_init__ stage of the dataclass.
    mask_pupil : `afwImage.Mask`, init=False
        Non-padded mask corresponding to aperture. This is
        initialized in the __post_init__ stage of the dataclass.
    """

    stamp_im: afwImage.maskedImage.MaskedImageF
    sky_position: lsst.geom.SpherePoint
    centroid_position: lsst.geom.Point2D
    defocal_type: str
    defocal_distance: float
    detector_name: str
    cam_name: str
    archive_element: Optional[afwTable.io.Persistable] = None
    comp_im: CompensableImage = field(default_factory=CompensableImage)
    mask_comp: afwImage.Mask = field(init=False)
    mask_pupil: afwImage.Mask = field(init=False)

    def __post_init__(self):
        """
        This method sets up the CompensableImage after initialization
        because we need to use the parameters set in the original `__init__`.
        """
        self._setCompensableImage()

    @classmethod
    def factory(cls, stamp_im, metadata, index, archive_element=None):
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
            defocal_distance=metadata.getArray("DFC_DIST")[index]
            if metadata.get("DFC_DIST") is not None
            else 1.5,
        )

    def getCamera(self):
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

        if self.cam_name == "LSSTCam":
            return obs_lsst.LsstCam().getCamera()
        elif self.cam_name == "LSSTComCam":
            return obs_lsst.LsstComCam().getCamera()
        elif self.cam_name == "LATISS":
            return obs_lsst.Latiss.getCamera()
        else:
            raise ValueError(f"Camera {self.cam_name} is not supported.")

    def calcFieldXY(self):
        """
        Calculate the X, Y field position of the centroid in degrees.

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

    def _setCompensableImage(self):
        """
        Set up the compensable image object in the dataclass.
        """

        field_xy = self.calcFieldXY()

        self.comp_im.setImg(
            field_xy,
            DefocalType(self.defocal_type),
            self.stamp_im.getImage().getArray(),
        )
        self.mask_pupil = afwImage.Mask()
        self.mask_comp = afwImage.Mask()

    def makeMasks(self, inst, model, boundaryT, maskScalingFactorLocal):
        """Get the binary mask which considers the obscuration and off-axis
        correction.

        Parameters
        ----------
        inst : `Instrument`
            Instrument to use.
        model : `str`
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        boundaryT : `int`
            Extended boundary in pixel. It defines how far the computation mask
            extends beyond the pupil mask. And, in fft, it is also the width of
            Neuman boundary where the derivative of the wavefront is set to
            zero.
        maskScalingFactorLocal : `float`
            Mask scaling factor (for fast beam) for local correction.

        Returns
        -------
        mask_pupil : `afwImage.Mask`
            Non-padded mask for use at the offset planes.
        mask_comp : `afwImage.Mask`
            Padded mask for use at the offset planes.
        """

        self.comp_im.makeMask(inst, model, boundaryT, maskScalingFactorLocal)

        # 0 flag in mask is part of image that is not donut
        # 1 flag in mask means it is part of the model donut
        maskDict = {"BKGRD": 0, "DONUT": 1}

        # Set masks
        self.mask_comp = afwImage.Mask(np.array(self.comp_im.mask_comp, dtype=np.int32))
        self.mask_comp.clearMaskPlaneDict()
        self.mask_comp.conformMaskPlanes(maskDict)
        # Will inherit conformed MaskPlaneDict
        self.mask_pupil = afwImage.Mask(
            np.array(self.comp_im.mask_pupil, dtype=np.int32)
        )
