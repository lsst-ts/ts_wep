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

from lsst.meas.algorithms.stamps import StampsBase, readFitsWithOptions
from lsst.ts.wep.task.DonutStamp import DonutStamp


class DonutStamps(StampsBase):
    """
    Holds a list of DonutStamp objects with helper functions.
    """

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
        """
        Get the ra, dec coordinates in degrees of the DonutStamps

        Returns
        -------
        list [lsst.geom.SpherePoint]
            Ra, Dec locations of donuts
        """
        return [s.sky_position for s in self._stamps]

    def getXY0Positions(self):
        """
        Get the corner position of the DonutStamp BBox locations.

        Returns
        -------
        list [lsst.geom.Point2I]
            Corner of BBox of DonutStamp lsst.afw.image.MaskedImage
        """
        return [s.stamp_im.getXY0() for s in self._stamps]

    def getCentroidPositions(self):
        """
        Get the centroid positions of the DonutStamps in the original image.

        Returns
        -------
        list [lsst.geom.Point2I]
            X,Y pixel locations of center of DonutStamp images
        """
        return [s.centroid_position for s in self._stamps]

    def getDetectorNames(self):
        """
        Get the detector name for each stamp.

        Returns
        -------
        list [str]
            Detector Names for each stamp
        """
        return [s.detector_name for s in self._stamps]

    def append(self, item):
        """Add an additional stamp.
        Parameters
        ----------
        item : DonutStamp
            DonutStamp object to append.
        """
        if not isinstance(item, DonutStamp):
            raise ValueError("Objects added must be a DonutStamp object.")
        self._stamps.append(item)
        return None

    def extend(self, stamp_list):
        """Extend DonutStamps instance by appending elements from another instance.
        Parameters
        ----------
        stamps_list : list [DonutStamp]
            List of DonutStamp object to append.
        """
        for s in stamp_list:
            if not isinstance(s, DonutStamp):
                raise ValueError("Can only extend with DonutStamp objects.")
        self._stamps += stamp_list

    @classmethod
    def readFits(cls, filename):
        """Build an instance of this class from a file.
        Parameters
        ----------
        filename : str
            Name of the file to read
        Returns
        -------
        DonutStamps
            An instance of this class
        """
        return cls.readFitsWithOptions(filename, None)

    @classmethod
    def readFitsWithOptions(cls, filename, options):
        """Build an instance of this class with options.
        Parameters
        ----------
        filename : str
            Name of the file to read
        options : PropertyList or dict
            Collection of metadata parameters
        Returns
        -------
        DonutStamps
            An instance of this class
        """
        stamps, metadata = readFitsWithOptions(filename, DonutStamp.factory, options)
        return cls(
            stamps,
            metadata=metadata,
            use_mask=metadata["HAS_MASK"],
            use_variance=metadata["HAS_VARIANCE"],
        )
