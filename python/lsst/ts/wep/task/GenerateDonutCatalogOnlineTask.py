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
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.geom
import lsst.obs.lsst as obs_lsst
import lsst.pipe.base as pipeBase
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.ts.wep.task.GenerateDonutCatalogBase import (
    GenerateDonutCatalogBaseConnections,
    GenerateDonutCatalogBaseConfig,
    GenerateDonutCatalogBaseTask,
)


class GenerateDonutCatalogOnlineTaskConfig(
    GenerateDonutCatalogBaseConfig,
    pipelineConnections=GenerateDonutCatalogBaseConnections,
):
    """
    Configuration settings for GenerateDonutCatalogOnlineTask. Specifies
    pointing information, filter and camera details.
    """

    boresightRa = pexConfig.Field(
        doc="Boresight RA in degrees", dtype=float, default=0.0
    )
    boresightDec = pexConfig.Field(
        doc="Boresight Dec in degrees", dtype=float, default=0.0
    )
    boresightRotAng = pexConfig.Field(
        doc="Boresight Rotation Angle in degrees", dtype=float, default=0.0
    )


class GenerateDonutCatalogOnlineTask(GenerateDonutCatalogBaseTask):
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

        self.boresightPointing = lsst.geom.SpherePoint(
            self.boresightRa, self.boresightDec, lsst.geom.degrees
        )

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We implement a runQuantum method to make sure our configured
        task runs with the instrument required by the pipeline.
        """

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

        # Get camera
        if cameraName == "LSSTCam":
            camera = obs_lsst.LsstCam.getCamera()
        elif cameraName == "LSSTComCam":
            camera = obs_lsst.LsstComCam.getCamera()
        else:
            raise ValueError(f"{cameraName} is not a valid camera name.")

        refObjLoader = self.getRefObjLoader(refCatalogs)

        detectorList = []
        donutCatalogList = []

        for detector in camera:

            detWcs = createInitialSkyWcsFromBoresight(
                self.boresightPointing,
                self.boresightRotAng * lsst.geom.degrees,
                detector,
                flipX=False,
            )

            try:
                # Match detector layout to reference catalog
                donutCatalog = refObjLoader.loadPixelBox(
                    detector.getBBox(), detWcs, filterName=self.filterName
                ).refCat

                detectorList.append(detector.getName())
                donutCatalogList.append(donutCatalog)

            # Except RuntimeError caused when no reference catalog
            # available for the region covered by detector
            except RuntimeError:
                continue

        fieldObjects = self.donutCatalogListToDataFrame(donutCatalogList, detectorList)

        # Return pandas DataFrame with sources in pointing
        # with ra, dec, filter flux, pixel XY information and detector name
        # for each source
        finalSources = self.filterResults(fieldObjects)

        return pipeBase.Struct(donutCatalog=finalSources)
