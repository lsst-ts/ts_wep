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

import lsst.geom
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.meas.algorithms.htmIndexer import HtmIndexer


class RefCatalogInterface(object):
    def __init__(self, boresightRa, boresightDec, boresightRotAng):

        self.boresightRa = boresightRa
        self.boresightDec = boresightDec
        self.boresightRotAng = boresightRotAng

    def getShardIds(self, radius=1.8):
        """
        Get the HtmIds for the pieces of the reference catalog
        that fall within radius (in degrees) of the boresight
        pointing.

        Parameters
        ----------
        radius : float
            Radius in degrees of the pointing footprint.

        Returns
        -------
        numpy.ndarray
            Array of htmIds in the butler for the pieces of the
            reference catalog that overlap the pointing.
        """
        htmIdx = HtmIndexer(depth=7)
        centerPt = lsst.geom.SpherePoint(
            self.boresightRa, self.boresightDec, lsst.geom.degrees
        )
        shardIds = htmIdx.getShardIds(
            centerPt, lsst.geom.Angle(radius, lsst.geom.degrees)
        )

        return shardIds[0]

    def getDataRefs(self, shardIds, butler, catalogName, collections):
        """
        Get the butler references and dataIds
        for the reference catalog shards specified.

        Parameters
        ----------
        shardIds : array
            HtmIds for the shards of the reference catalog we need.
        butler : lsst.daf.butler.Butler
            Butler instance pointing at the repo with the reference catalog.
        catalogName : str
            Name of the reference catalog in the repository.
        collections : str or list of str
            Collection in the repository with the reference catalog.

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
        for shardId in shardIds:
            shardDataId = {"htm7": shardId}
            dataRef = list(
                registry.queryDatasets(
                    catalogName, dataId=shardDataId, collections=collections
                ).expanded()
            )
            if len(dataRef) == 0:
                continue
            deferredList.append(butler.getDeferred(dataRef[0], collections=collections))
            dataIds.append(dataRef[0].dataId)
        return deferredList, dataIds

    def getDetectorWcs(self, detector):
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

        boresightPointing = lsst.geom.SpherePoint(
            self.boresightRa, self.boresightDec, lsst.geom.degrees
        )
        detWcs = createInitialSkyWcsFromBoresight(
            boresightPointing,
            self.boresightRotAng * lsst.geom.degrees,
            detector,
            flipX=False,
        )
        return detWcs
