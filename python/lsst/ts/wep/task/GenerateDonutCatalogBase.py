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

import os
import numpy as np
import pandas as pd
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig


class GenerateDonutCatalogBaseConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument",)
):
    """
    Specify the minimum set of connections needed for
    a GenerateDonutCatalog...Task. For all such tasks we
    need the reference catalogs and will produce donut catalogs for
    a specified instrument.
    """

    refCatalogs = connectionTypes.PrerequisiteInput(
        doc="Reference catalog",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        multiple=True,
        deferLoad=True,
        name="cal_ref_cat",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=("instrument",),
        storageClass="DataFrame",
        name="donutCatalog",
    )


class GenerateDonutCatalogBaseConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogBaseConnections,
):
    """
    Configuration settings for a GenerateDonutCatalog...Task.
    Specifies filter and camera details.
    """

    filterName = pexConfig.Field(doc="Reference filter", dtype=str, default="g")
    cameraName = pexConfig.Field(doc="Camera Name", dtype=str, default="lsstCam")


class GenerateDonutCatalogBaseTask(pipeBase.PipelineTask):
    """
    Base class for donut catalog generation tasks.

    Subclasses must implement _DefaultName.
    """

    ConfigClass = GenerateDonutCatalogBaseConfig
    # _DefaultName implemented here in subclass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The filter in the reference catalog we want to use to find sources.
        self.filterName = self.config.filterName

        # TODO: Temporary until DM-24162 is closed at which point we
        # can remove this
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

    def filterResults(self, resultsDataFrame):
        """
        Run filtering on full set of sources on detector and return
        the dataframe with only sources that are acceptable for
        wavefront estimation.

        Parameters
        ----------
        resultsDataFrame : pandas DataFrame
            Full list of sources from reference catalog that appear
            on the detector.

        Returns
        -------
        pandas DataFrame
            Subset of resultsDataFrame sources that pass required filtering.
        """

        # TODO: Here is where we will set up specifications for the sources
        # we want to use (i.e., filter on magnitude, blended, etc.).
        # For now it just returns all sources.

        return resultsDataFrame

    def getRefObjLoader(self, refCatalogList):
        """
        Create a `ReferenceObjectLoader` from available reference catalogs
        in the repository.

        Parameters
        ----------
        refCatalogList : list
            List of deferred butler references for the reference catalogs.

        Returns
        -------
        lsst.meas.algorithms.ReferenceObjectsLoader
            Object to conduct spatial searches through the reference catalogs
        """

        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in refCatalogList],
            refCats=refCatalogList,
            config=LoadReferenceObjectsConfig(),
        )
        # This removes the padding around the border of detector BBox when
        # matching to reference catalog.
        # We remove this since we only want sources within detector.
        refObjLoader.config.pixelMargin = 0

        return refObjLoader

    def donutCatalogListToDataFrame(self, donutCatalogList, detectorList):
        """
        Reformat list of matched catalogs from each detector into a
        single pandas dataframe.

        Parameters
        ----------
        donutCatalogList : list
            List of lsst.afw.table.SimpleCatalog objects returned by the
            ReferenceObjectLoader search over the detector footprint.
        detectorList : list
            List of the names of the detectors associated with each
            donutCatalog in the donutCatalogList.

        Returns
        -------
        pandas DataFrame
            Complete catalog of reference sources in the pointing.
        """

        ra = []
        dec = []
        centroidX = []
        centroidY = []
        sourceFlux = []
        detNames = []

        for donutCatalog, detectorName in zip(donutCatalogList, detectorList):
            ra.append(donutCatalog["coord_ra"])
            dec.append(donutCatalog["coord_dec"])
            centroidX.append(donutCatalog["centroid_x"])
            centroidY.append(donutCatalog["centroid_y"])
            sourceFlux.append(donutCatalog[f"{self.filterName}_flux"])
            detNames.append([detectorName] * len(donutCatalog))

        fieldObjects = pd.DataFrame([])
        fieldObjects["coord_ra"] = np.hstack(ra).squeeze()
        fieldObjects["coord_dec"] = np.hstack(dec).squeeze()
        fieldObjects["centroid_x"] = np.hstack(centroidX).squeeze()
        fieldObjects["centroid_y"] = np.hstack(centroidY).squeeze()
        fieldObjects["source_flux"] = np.hstack(sourceFlux).squeeze()
        fieldObjects["detector"] = np.hstack(detNames).squeeze()

        return fieldObjects
