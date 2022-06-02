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

__all__ = ["GenerateDonutCatalogOnlineTaskConfig", "GenerateDonutCatalogOnlineTask"]

import numpy as np
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from lsst.meas.algorithms import (
    LoadIndexedReferenceObjectsTask,
    ReferenceSourceSelectorTask,
    ReferenceObjectLoader,
)

from lsst.ts.wep.task.DonutSourceSelectorTask import DonutSourceSelectorTask


class GenerateDonutCatalogOnlineTaskConfig(pexConfig.Config):
    """Configuration for GenerateDonutCatalogOnlineTask."""

    refObjLoader = pexConfig.ConfigurableField(
        target=LoadIndexedReferenceObjectsTask,
        doc="Reference object loader for photometry",
    )
    referenceSelector = pexConfig.ConfigurableField(
        target=ReferenceSourceSelectorTask,
        doc="Selection of reference sources",
    )
    doReferenceSelection = pexConfig.Field(
        doc="Run the reference selector on the reference catalog?",
        dtype=bool,
        default=True,
    )
    filterName = pexConfig.Field(doc="Reference filter", dtype=str, default="g")
    donutSelector = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )


class GenerateDonutCatalogOnlineTask(pipeBase.Task):
    """
    Construct a source catalog from reference catalogs
    and pointing information.

    Parameters
    ----------
    dataIds : `list`
        List of the dataIds for the reference catalog shards.
    refCats : `list`
        List of the deferred dataset references pointing to the pieces
        of the reference catalog we want in the butler.
    **kwargs : dict[str, any]
        Dictionary of input argument: new value for that input argument.
    """

    ConfigClass = GenerateDonutCatalogOnlineTaskConfig
    _DefaultName = "generateDonutCatalogOnlineTask"

    def __init__(self, dataIds, refCats, **kwargs):

        super().__init__(**kwargs)
        refConfig = self.config.refObjLoader
        # refObjLoader handles the interaction with the butler repository
        # needed to get the pieces of the reference catalogs we need.
        self.refObjLoader = ReferenceObjectLoader(
            dataIds=dataIds, refCats=refCats, config=refConfig, log=self.log
        )

        if self.config.doReferenceSelection:
            self.makeSubtask("referenceSelector")

        self.filterName = self.config.filterName
        self.config.refObjLoader.pixelMargin = 0
        self.config.refObjLoader.anyFilterMapsToThis = self.filterName
        self.config.referenceSelector.magLimit.fluxField = f"{self.filterName}_flux"
        self.config.referenceSelector.signalToNoise.fluxField = (
            f"{self.filterName}_flux"
        )
        self.config.donutSelector.fluxField = f"{self.filterName}_flux"
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

    @timeMethod
    def run(self, detector, wcs):
        """Get the data from the reference catalog only from the
        shards of the reference catalogs that overlap our pointing.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.
        wcs : `lsst.afw.geom.SkyWcs`
            Wcs object defining the pixel to sky (and inverse) transform for
            the supplied ``bbox``.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - DonutCatalog: `pandas.DataFrame`
                    The final donut source catalog for the region.
        """
        bbox = detector.getBBox()
        # Get the refcatalog shard
        skyBox = self.refObjLoader.loadPixelBox(
            bbox, wcs, filterName=self.filterName, bboxToSpherePadding=0
        )

        if not skyBox.refCat.isContiguous():
            refCat = skyBox.refCat.copy(deep=True)
        else:
            refCat = skyBox.refCat

        donutCatalog = self._formatCatalog(refCat, detector)

        return pipeBase.Struct(donutCatalog=donutCatalog)

    def _formatCatalog(self, refCat, detector):
        """Format a reference afw table into the final format.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            Reference catalog in afw format.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.

        Returns
        -------
        refCat : `pandas.DataFrame`
            Reference catalog.
        """

        if self.config.doReferenceSelection:
            goodSources = self.referenceSelector.selectSources(refCat)
            refSelected = goodSources.selected
        else:
            refSelected = np.ones(len(refCat), dtype=bool)

        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            donutSelection = self.donutSelector.run(refCat, detector)
            donutSelected = donutSelection.selected
        else:
            donutSelected = np.ones(len(refCat), dtype=bool)

        selected = refSelected & donutSelected

        npRefCat = np.zeros(
            np.sum(selected),
            dtype=[
                ("coord_ra", "f8"),
                ("coord_dec", "f8"),
                ("centroid_x", "f8"),
                ("centroid_y", "f8"),
                ("source_flux", "f8"),
            ],
        )

        if npRefCat.size == 0:
            # Return an empty catalog if we don't have any selected sources.
            return npRefCat

        # Natively "coord_ra" and "coord_dec" are stored in radians.
        # Doing this as an array rather than by row with the coord access is
        # approximately 600x faster.
        npRefCat["coord_ra"] = refCat["coord_ra"][selected]
        npRefCat["coord_dec"] = refCat["coord_dec"][selected]

        npRefCat["centroid_x"] = refCat["centroid_x"][selected]
        npRefCat["centroid_y"] = refCat["centroid_y"][selected]

        fluxField = f"{self.filterName}_flux"

        # nan_to_num replaces nans with zeros, and this ensures that
        # we select fluxes that both filter out nans and are positive.
        (good,) = np.where(
            (np.nan_to_num(refCat[fluxField][selected]) > 0.0)
            & (np.nan_to_num(refCat[fluxField + "Err"][selected]) > 0.0)
        )
        npRefCat["source_flux"][good] = refCat[fluxField][selected][good]

        return pd.DataFrame.from_records(npRefCat)
