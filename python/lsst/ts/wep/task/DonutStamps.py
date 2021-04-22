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
from lsst.meas.algorithms.stamps import AbstractStamp, StampsBase, readFitsWithOptions


@dataclass
class DonutStamp(AbstractStamp):
    """Single donut stamp
    Parameters
    ----------
    stamp_im : `lsst.afw.image.MaskedImageF`
        The actual pixel values for the postage stamp
    sky_position : `lsst.geom.SpherePoint`
        Position of the center of the stamp.  Note the user
        must keep track of the coordinate system
    centroid_position : `lsst.geom.Point2I`
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
        this class.  If lists of values are passed with the following
        keys, they will be passed to the constructor, otherwise dummy
        values will be passed: RA_DEG, DEC_DEG.  They should
        each point to lists of values.
        Parameters
        ----------
        stamp : `lsst.afw.image.MaskedImage`
            Pixel data to pass to the constructor
        metadata : `dict`
            Dictionary containing the information
            needed by the constructor.
        idx : `int`
            Index into the lists in ``metadata``
        Returns
        -------
        stamp : `DonutStamp`
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


class DonutStamps(StampsBase):

    def _refresh_metadata(self):
        sky_positions = self.getSkyPositions()
        self._metadata["RA_DEG"] = [p.getRa().asDegrees() for p in sky_positions]
        self._metadata["DEC_DEG"] = [p.getDec().asDegrees() for p in sky_positions]
        xy0_positions = self.getXY0Positions()
        self._metadata["X0"] = [p.getX() for p in xy0_positions]
        self._metadata["Y0"] = [p.getY() for p in xy0_positions]
        centroid_positions = self.getCentroidPositions()
        self._metadata["CENT_X"] = [p.getX() for p in centroid_positions]
        self._metadata["CENT_Y"] = [p.getY() for p in centroid_positions]
        det_names = self.getDetectorNames()
        self._metadata["DET_NAME"] = [det for det in det_names]

    def getSkyPositions(self):
        return [s.sky_position for s in self._stamps]

    def getXY0Positions(self):
        return [s.stamp_im.getXY0() for s in self._stamps]

    def getCentroidPositions(self):
        return [s.centroid_position for s in self._stamps]

    def getDetectorNames(self):
        return [s.detector_name for s in self._stamps]

    def append(self, item):
        """Add an additional stamp.
        Parameters
        ----------
        item : `Stamp`
            Stamp object to append.
        """
        if not isinstance(item, DonutStamp):
            raise ValueError("Objects added must be a DonutStamp object.")
        self._stamps.append(item)
        return None

    def extend(self, stamp_list):
        """Extend DonutStamps instance by appending elements from another instance.
        Parameters
        ----------
        stamps_list : `list` [`DonutStamp`]
            List of DonutStamp object to append.
        """
        for s in stamp_list:
            if not isinstance(s, DonutStamp):
                raise ValueError("Can only extend with DonutStamp objects")
        self._stamps += stamp_list

    @classmethod
    def readFits(cls, filename):
        """Build an instance of this class from a file.
        Parameters
        ----------
        filename : `str`
            Name of the file to read
        Returns
        -------
        object : `DonutStamps`
            An instance of this class
        """
        return cls.readFitsWithOptions(filename, None)

    @classmethod
    def readFitsWithOptions(cls, filename, options):
        """Build an instance of this class with options.
        Parameters
        ----------
        filename : `str`
            Name of the file to read
        options : `PropertyList` or `dict`
            Collection of metadata parameters
        Returns
        -------
        object : `DonutStamps`
            An instance of this class
        """
        stamps, metadata = readFitsWithOptions(filename, DonutStamp.factory, options)
        return cls(
            stamps,
            metadata=metadata,
            use_mask=metadata["HAS_MASK"],
            use_variance=metadata["HAS_VARIANCE"],
        )
