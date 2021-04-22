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

from dataclasses import dataclass

import lsst.geom
import lsst.afw.image as afwImage
from lsst.meas.algorithms.stamps import AbstractStamp


@dataclass
class DonutStamp(AbstractStamp):
    """Single donut stamp
    Parameters
    ----------
    stamp_im : lsst.afw.image.MaskedImageF
        The actual pixel values for the postage stamp
    sky_position : lsst.geom.SpherePoint
        Position of the center of the stamp.  Note the user
        must keep track of the coordinate system
    centroid_position : lsst.geom.Point2I
        Position of the center of the stamp in pixels
    detector_name : str
        CCD where the donut is found
    """

    stamp_im: afwImage.maskedImage.MaskedImageF
    sky_position: lsst.geom.SpherePoint
    centroid_position: lsst.geom.Point2I
    detector_name: str

    @classmethod
    def factory(cls, stamp_im, metadata, index):
        """This method is needed to service the FITS reader.
        We need a standard interface to construct objects like this.
        Parameters needed to construct this object are passed in via
        a metadata dictionary and then passed to the constructor of
        this class. They should each point to lists of values.
        Parameters
        ----------
        stamp : lsst.afw.image.MaskedImage
            Pixel data to pass to the constructor
        metadata : lsst.daf.base.PropertyList
            PropertyList containing the information
            needed by the constructor.
        idx : int
            Index into the lists in ``metadata``
        Returns
        -------
        DonutStamp
            An instance of this class
        """
        return cls(
            stamp_im=stamp_im,
            sky_position=lsst.geom.SpherePoint(
                lsst.geom.Angle(metadata.getArray("RA_DEG")[index], lsst.geom.degrees),
                lsst.geom.Angle(metadata.getArray("DEC_DEG")[index], lsst.geom.degrees),
            ),
            centroid_position=lsst.geom.Point2I(
                metadata.getArray("CENT_X")[index], metadata.getArray("CENT_Y")[index]
            ),
            detector_name=metadata.getArray("DET_NAME")[index],
        )
