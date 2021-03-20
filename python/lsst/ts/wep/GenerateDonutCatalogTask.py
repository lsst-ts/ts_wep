import typing
import numpy as np
import lsst.geom
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig


class GenerateDonutCatalogTaskConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("exposure", "detector")):
    exposure = connectionTypes.Input(doc="Input exposure to make measurements "
                                          "on",
                                     dimensions=("exposure", "detector"),
                                     storageClass="Exposure",
                                     name="postISRCCD")
    refCatalogs = connectionTypes.PrerequisiteInput(doc="Reference catalog",
                                                    storageClass="SimpleCatalog",
                                                    dimensions=("htm7",),
                                                    multiple=True,
                                                    deferLoad=True,
                                                    name="cal_ref_cat")
    donutCatalog = connectionTypes.Output(doc="Donut Locations",
                                          dimensions=("exposure", "detector"),
                                          storageClass="SimpleCatalog",
                                          name="donutCatalog")


class GenerateDonutCatalogTaskConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=GenerateDonutCatalogTaskConnections):
    filterName = pexConfig.Field(doc="Reference filter", dtype=str, default='g')


class GenerateDonutCatalogTask(pipeBase.PipelineTask):

    ConfigClass = GenerateDonutCatalogTaskConfig
    _DefaultName = "GenerateDonutCatalogTask"

    def __init__(self, config: pexConfig.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, exposure: afwImage.Exposure,
            refCatalogs: typing.List[afwTable.SimpleCatalog]) -> pipeBase.Struct:

        expWcs = exposure.getWcs()
        expBBox = exposure.getBBox()

        refObjLoader = ReferenceObjectLoader(dataIds=[ref.dataId for ref in refCatalogs],
                                             refCats=refCatalogs, config=LoadReferenceObjectsConfig())
        donutCatalog = refObjLoader.loadPixelBox(expBBox, expWcs, filterName=self.config.filterName)

        return pipeBase.Struct(donutCatalog=donutCatalog.refCat)
