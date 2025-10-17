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

__all__ = ["RefCatalogInterface"]

import lsst.geom
from lsst.meas.algorithms.htmIndexer import HtmIndexer
from lsst.obs.base import createInitialSkyWcsFromBoresight


class RefCatalogInterface(object):
    """
    Class to provide tools to interact with reference catalog
    in Butler repository and select pieces of the catalog
    that cover the sky area of a pointing.

    Parameters
    ----------
    boresightRa : float
        Boresight RA in degrees.
    boresightDec : float
        Boresight Dec in degrees.
    boresightRotAng : float
        Boresight rotation angle in degreees.
    """

    def __init__(
        self,
        boresightRa: float,
        boresightDec: float,
        boresightRotAng: float,
    ) -> None:
        # Set the pointing information
        self.boresightRa = boresightRa
        self.boresightDec = boresightDec
        self.boresightRotAng = boresightRotAng

    def getHtmIds(self, radius: float = 1.8) -> list:
        """
        Get the htmIds for the pieces of the reference catalog
        that overlap the circular area within `radius` (in degrees)
        of the boresight pointing. HtmIds are the spatial indices
        identifying hierarchical triangular mesh (HTM) shards that
        are used to store the reference catalogs in gen3 butler
        repositories in more easily accessible pieces.

        Parameters
        ----------
        radius : float, optional
            Radius in degrees of the pointing footprint.
            (the default is 1.8 degrees, enough to cover one LSST pointing.)

        Returns
        -------
        numpy.ndarray
            Array of htmIds in the butler for the pieces of the
            reference catalog that overlap the pointing.
        """
        # HTM depth specifies the resolution of HTM grid that covers the sky.
        # DM Gen3 ingests refernce catalogs with an HTM depth of 7.
        htmIdx = HtmIndexer(depth=7)
        centerPt = lsst.geom.SpherePoint(self.boresightRa, self.boresightDec, lsst.geom.degrees)
        htmIds = htmIdx.getShardIds(centerPt, lsst.geom.Angle(radius, lsst.geom.degrees))

        return htmIds[0]

    def getDataRefs(
        self,
        htmIds: list,
        butler: lsst.daf.butler.Butler,
        catalogName: str,
        collections: str | list[str],
    ) -> tuple[list, list]:
        """
        Get the butler references and dataIds
        for the reference catalog shards specified.

        Parameters
        ----------
        htmIds : array
            HtmIds for the shards of the reference catalog we need.
        butler : lsst.daf.butler.Butler
            Butler instance pointing at the repo with the reference catalog.
        catalogName : str
            Name of the reference catalog in the repository.
        collections : str or list of str
            Collections in the repository with the reference catalog.

        Returns
        -------
        list
            List of the deferred dataset references pointing to the pieces
            of the reference catalog we want in the butler.
        list
            List of the dataIds for the reference catalog shards.
        """

        registry = butler.registry
        deferredList = []
        dataIds = []
        for htmId in htmIds:
            # Shards of the reference catalog in the Gen3 butler
            # are identified with a dataId with the key labelled "htm7".
            htmDataId = {"htm7": htmId}
            dataRef = list(
                registry.queryDatasets(catalogName, dataId=htmDataId, collections=collections).expanded()
            )
            if len(dataRef) == 0:
                continue
            deferredList.append(butler.getDeferred(dataRef[0]))
            dataIds.append(dataRef[0].dataId)
        return deferredList, dataIds

    def getDetectorWcs(self, detector: lsst.afw.cameraGeom.Detector) -> lsst.afw.geom.SkyWcs:
        """
        Create a WCS for the detector with the initialized
        pointing information.

        Parameters
        ----------
        detector : lsst.afw.cameraGeom.Detector
            Detector for which we want to generate a source catalog.

        Returns
        -------
        lsst.afw.geom.SkyWcs
            Wcs object defining the pixel to sky (and inverse) transform for
            the supplied detector.
        """

        boresightPointing = lsst.geom.SpherePoint(self.boresightRa, self.boresightDec, lsst.geom.degrees)
        return createInitialSkyWcsFromBoresight(
            boresightPointing,
            self.boresightRotAng * lsst.geom.degrees,
            detector,
            flipX=False,
        )
