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

import typing
import os
import numpy as np
import pandas as pd
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.obs.lsst as obs_lsst
import lsst.geom
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig


class GenerateDonutCatalogOnlineTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument",)
):
    """
    Specify the connections needed for GenerateDonutCatalogOnlineTask.
    We need the reference catalogs and will produce donut catalogs for
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


class GenerateDonutCatalogOnlineTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogOnlineTaskConnections,
):
    """
    Configuration settings for GenerateDonutCatalogOnlineTask. Specifies
    pointing information, filter and camera details.
    """

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
    _DefaultName = "generateDonutCatalogOnlineTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The filter in the reference catalog we want to use to find sources.
        self.filterName = self.config.filterName

        # Pointing information to construct the WCS. All values in degrees.
        self.boresightRa = self.config.boresightRa
        self.boresightDec = self.config.boresightDec
        self.boresightRotAng = self.config.boresightRotAng

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
        resultsDataFrame: pandas DataFrame
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

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):

        # Get the instrument we are running the pipeline with
        cameraName = outputRefs.donutCatalog.dataId["instrument"]

        # Get the input reference catalogs for the task
        inputs = butlerQC.get(inputRefs)

        # Run task on specified instrument
        outputs = self.run(cameraName, **inputs)

        # Use butler to store output in repository
        butlerQC.put(outputs, outputRefs)

    def run(
        self, cameraName: str, refCatalogs: typing.List[afwTable.SimpleCatalog]
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

        # Get camera. Only 'LSSTCam' for now.
        if cameraName == "LSSTCam":
            camera = obs_lsst.LsstCam.getCamera()
        elif cameraName == "LSSTComCam":
            camera = obs_lsst.LsstComCam.getCamera()
        else:
            raise ValueError(f"{cameraName} is not a valid camera name.")

        boresightPointing = lsst.geom.SpherePoint(
            self.boresightRa, self.boresightDec, lsst.geom.degrees
        )

        for detector in camera:
            detWcs = createInitialSkyWcsFromBoresight(
                boresightPointing,
                self.boresightRotAng * lsst.geom.degrees,
                detector,
                flipX=False,
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

        # Return pandas DataFrame with sources in pointing
        # with ra, dec, filter flux, pixel XY information and detector name
        # for each source
        finalSources = self.filterResults(fieldObjects)

        return pipeBase.Struct(donutCatalog=finalSources)
