#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import typing
import os
import numpy as np
import pandas as pd
import lsst.geom
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.obs.lsst as obs_lsst
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig
from lsst.obs.base import createInitialSkyWcsFromBoresight


class GenerateDonutCatalogOnlineTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument",)
):
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


class GenerateDonutCatalogOnlineTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogOnlineTaskConnections,
):
    filterName = pexConfig.Field(doc="Reference filter", dtype=str, default="g")
    boresightRa = pexConfig.Field(
        doc="Boresight RA in degrees", dtype=float, default=0.0
    )
    boresightDec = pexConfig.Field(
        doc="Boresight Dec in degrees", dtype=float, default=0.0
    )
    boresightRotAng = pexConfig.Field(
        doc="Boresight Rotation Angle in degrees", dtype=float, default=0.0
    )
    cameraName = pexConfig.Field(doc="Camera Name", dtype=str, default="lsstCam")


class GenerateDonutCatalogOnlineTask(pipeBase.PipelineTask):
    """
    Create a WCS from boresight info and then use this
    with a reference catalog to select sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutCatalogOnlineTaskConfig
    _DefaultName = "GenerateDonutCatalogOnlineTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filterName = self.config.filterName
        self.boresightRa = self.config.boresightRa
        self.boresightDec = self.config.boresightDec
        self.boresightRotAng = self.config.boresightRotAng
        self.cameraName = self.config.cameraName

        # TODO: Temporary until DM-24162 is closed at which point we
        # can remove this
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

    def filterResults(self, resultsDataFrame):
        """
        Here is where we will set up specifications for the sources
        we want to use (i.e., filter on magnitude, blended, etc.).
        For now it just returns all sources.
        """

        return resultsDataFrame

    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
    ) -> pipeBase.Struct:

        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in refCatalogs],
            refCats=refCatalogs,
            config=LoadReferenceObjectsConfig(),
        )
        # This removes the padding around the border of detector BBox when
        # matching to reference catalog.
        # We remove this since we only want sources within detector.
        refObjLoader.config.pixelMargin = 0

        # Set up pandas dataframe
        fieldObjects = pd.DataFrame([])
        ra = []
        dec = []
        centroidX = []
        centroidY = []
        det_names = []

        # Get camera. Only 'lsstCam' for now.
        if self.cameraName == "lsstCam":
            lsst_cam = obs_lsst.LsstCam.getCamera()
        else:
            raise ValueError(f"{self.cameraName} is not a valid camera name.")

        for detector in lsst_cam:
            # Create WCS from boresight information
            # NOTE: 90.0 - boresightRotAng is a phosim artifact that is being fixed
            # TODO: Fix this when DM-29702 is merged
            detWcs = createInitialSkyWcsFromBoresight(
                lsst.geom.SpherePoint(
                    self.boresightRa, self.boresightDec, lsst.geom.degrees
                ),
                (90.0 - self.boresightRotAng) * lsst.geom.degrees,
                detector,
            )

            try:
                # Match detector layout to reference catalog
                donutCatalog = refObjLoader.loadPixelBox(
                    detector.getBBox(), detWcs, filterName=self.filterName
                ).refCat

                # Add matched information to list
                ra.append(donutCatalog["coord_ra"])
                dec.append(donutCatalog["coord_dec"])
                centroidX.append(donutCatalog["centroid_x"])
                centroidY.append(donutCatalog["centroid_y"])
                det_names.append([detector.getName()] * len(donutCatalog))

            except RuntimeError:
                continue

        # Flatten information from all detector lists and enter into dataframe
        fieldObjects["coord_ra"] = np.hstack(ra).squeeze()
        fieldObjects["coord_dec"] = np.hstack(dec).squeeze()
        fieldObjects["centroid_x"] = np.hstack(centroidX).squeeze()
        fieldObjects["centroid_y"] = np.hstack(centroidY).squeeze()
        fieldObjects["detector"] = np.hstack(det_names).squeeze()

        finalSources = self.filterResults(fieldObjects)

        return pipeBase.Struct(donutCatalog=finalSources)
