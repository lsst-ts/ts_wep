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


class GenerateDonutCatalogFromBoresightTaskConnections(
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


class GenerateDonutCatalogFromBoresightTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutCatalogFromBoresightTaskConnections,
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


class GenerateDonutCatalogFromBoresightTask(pipeBase.PipelineTask):

    ConfigClass = GenerateDonutCatalogFromBoresightTaskConfig
    _DefaultName = "GenerateDonutCatalogFromBoresightTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filterName = self.config.filterName
        self.boresightRa = self.config.boresightRa
        self.boresightDec = self.config.boresightDec
        self.boresightRotAng = self.config.boresightRotAng
        self.cameraName = self.config.cameraName

        # Temporary until DM-24162 is closed
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

    def run(
        self,
        refCatalogs: typing.List[afwTable.SimpleCatalog],
    ) -> pipeBase.Struct:

        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in refCatalogs],
            refCats=refCatalogs,
            config=LoadReferenceObjectsConfig(),
        )
        refObjLoader.config.pixelMargin = 0

        # Set up pandas dataframe
        field_objects = pd.DataFrame([])
        ra = []
        dec = []
        centroid_x = []
        centroid_y = []
        det_names = []

        # Get camera
        if self.cameraName == "lsstCam":
            lsst_cam = obs_lsst.LsstCam.getCamera()
        else:
            raise ValueError(f"{self.cameraName} is not a valid camera name.")

        for detector in lsst_cam:
            # Create WCS from boresight information
            detWcs = createInitialSkyWcsFromBoresight(
                lsst.geom.SpherePoint(
                    self.boresightRa, self.boresightDec, lsst.geom.degrees
                ),
                (90.0 - self.boresightRotAng) * lsst.geom.degrees,
                detector,
            )

            try:
                donutCatalog = refObjLoader.loadPixelBox(
                    detector.getBBox(), detWcs, filterName=self.filterName
                ).refCat

                ra.append(donutCatalog["coord_ra"])
                dec.append(donutCatalog["coord_dec"])
                centroid_x.append(donutCatalog["centroid_x"])
                centroid_y.append(donutCatalog["centroid_y"])
                det_names.append([detector.getName()] * len(donutCatalog))

            except RuntimeError:
                continue

        field_objects["coord_ra"] = np.hstack(ra).squeeze()
        field_objects["coord_dec"] = np.hstack(dec).squeeze()
        field_objects["centroid_x"] = np.hstack(centroid_x).squeeze()
        field_objects["centroid_y"] = np.hstack(centroid_y).squeeze()
        field_objects["detector"] = np.hstack(det_names).squeeze()

        return pipeBase.Struct(donutCatalog=field_objects)
