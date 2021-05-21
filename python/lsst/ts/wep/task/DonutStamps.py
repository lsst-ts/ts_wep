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
    Holds a list of DonutStamp objects with additional functions
    to get metadata as well as save as FITS files.
    This storage class is able to be used with the Gen 3 Butler
    and is storable and readable within a Gen 3 repository.
    """

    def _refresh_metadata(self):
        sky_positions = self.getSkyPositions()
        self.metadata["RA_DEG"] = [p.getRa().asDegrees() for p in sky_positions]
        self.metadata["DEC_DEG"] = [p.getDec().asDegrees() for p in sky_positions]
        xy0_positions = self.getXY0Positions()
        self.metadata["X0"] = [p.getX() for p in xy0_positions]
        self.metadata["Y0"] = [p.getY() for p in xy0_positions]
        centroid_positions = self.getCentroidPositions()
        self.metadata["CENT_X"] = [p.getX() for p in centroid_positions]
        self.metadata["CENT_Y"] = [p.getY() for p in centroid_positions]
        det_names = self.getDetectorNames()
        self.metadata["DET_NAME"] = [det for det in det_names]

    def getSkyPositions(self):
        """
        Get the ra, dec coordinates in degrees of the DonutStamps.

        Returns
        -------
        list [lsst.geom.SpherePoint]
            Ra, Dec locations of donuts.
        """
        return [stamp.sky_position for stamp in self]

    def getXY0Positions(self):
        """
        Get the corner position of the DonutStamp BBox locations.

        Returns
        -------
        list [lsst.geom.Point2I]
            Corner of BBox of DonutStamp lsst.afw.image.MaskedImage.
        """
        return [stamp.stamp_im.getXY0() for stamp in self]

    def getCentroidPositions(self):
        """
        Get the centroid positions of the DonutStamps in the original image.

        Returns
        -------
        list [lsst.geom.Point2I]
            X,Y pixel locations of center of DonutStamp images.
        """
        return [stamp.centroid_position for stamp in self]

    def getDetectorNames(self):
        """
        Get the detector name for each stamp.

        Returns
        -------
        list [str]
            Detector Names for each stamp.
        """
        return [stamp.detector_name for stamp in self]

    def append(self, newStamp):
        """Add an additional stamp.

        Parameters
        ----------
        newStamp : DonutStamp
            DonutStamp object to append.

        Raises
        ------
        ValueError
            newStamp must be a DonutStamp object.
        """
        if not isinstance(newStamp, DonutStamp):
            raise ValueError("Objects added must be a DonutStamp object.")
        # Follow same procedure as Stamps subclass
        self._stamps.append(newStamp)

    def extend(self, newStampList):
        """Extend DonutStamps instance by appending elements
        from another instance.

        Parameters
        ----------
        newStampList : list [DonutStamp]
            List of DonutStamp object to append.

        Raises
        ------
        ValueError
            newStampList must only contain DonutStamp objects.
        """
        for stamp in newStampList:
            if not isinstance(stamp, DonutStamp):
                raise ValueError("Can only extend with DonutStamp objects.")
        # Follow same procedure as Stamps subclass
        self._stamps += newStampList

    @classmethod
    def readFits(cls, filename):
        """Build an instance of this class from a file.

        Parameters
        ----------
        filename : str
            Name of the file to read.

        Returns
        -------
        DonutStamps
            An instance of this class.
        """
        return cls.readFitsWithOptions(filename, None)

    @classmethod
    def readFitsWithOptions(cls, filename, options):
        """Build an instance of this class with options.

        Parameters
        ----------
        filename : str
            Name of the file to read.
        options : lsst.daf.base.PropertyList or dict
            Collection of metadata parameters.

        Returns
        -------
        DonutStamps
            An instance of this class.
        """
        stamps, metadata = readFitsWithOptions(filename, DonutStamp.factory, options)
        return cls(
            stamps,
            metadata=metadata,
            use_mask=metadata["HAS_MASK"],
            use_variance=metadata["HAS_VARIANCE"],
        )
