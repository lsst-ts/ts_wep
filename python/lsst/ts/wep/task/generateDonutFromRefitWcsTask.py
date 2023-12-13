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
from copy import copy

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import pandas as pd
from lsst.meas.algorithms import MagnitudeLimit, ReferenceObjectLoader
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.base.task import TaskError
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    donutCatalogToDataFrame,
    runSelection,
)
from lsst.ts.wep.task.generateDonutCatalogWcsTask import (
    GenerateDonutCatalogWcsTask,
    GenerateDonutCatalogWcsTaskConfig,
)
from lsst.utils.timer import timeMethod

__all__ = [
    "GenerateDonutFromRefitWcsTaskConnections",
    "GenerateDonutFromRefitWcsTaskConfig",
    "GenerateDonutFromRefitWcsTask",
]


class GenerateDonutFromRefitWcsTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
):
    """
    Specify the pipeline inputs and outputs needed
    for FitDonutWcsTask.
    """

    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="preFitPostISRCCD",
    )
    fitDonutCatalog = connectionTypes.Input(
        doc="Donut Locations From Direct Detection",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="DataFrame",
        name="directDetectDonutCatalog",
    )
    astromRefCat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for astrometry",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
    )
    photoRefCat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for donut selection",
        name="ps1_pv3_3pi_20170110",
        storageClass="SimpleCatalog",
        dimensions=("htm7",),
        deferLoad=True,
        multiple=True,
    )
    outputExposure = connectionTypes.Output(
        doc="Output exposure with new WCS",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="DataFrame",
        name="donutCatalog",
    )


class GenerateDonutFromRefitWcsTaskConfig(
    GenerateDonutCatalogWcsTaskConfig,
    pipelineConnections=GenerateDonutFromRefitWcsTaskConnections,
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    astromTask = pexConfig.ConfigurableField(
        target=AstrometryTask, doc="Task for WCS fitting."
    )
    maxFitScatter = pexConfig.Field(
        doc="Maximum allowed scatter for a successful fit (in arcseconds.)",
        dtype=float,
        default=1.0,
    )

    # Took these defaults from atmospec/centroiding which I used
    # as a template for implementing WCS fitting in a task.
    # https://github.com/lsst/atmospec/blob/main/python/lsst/atmospec/centroiding.py
    def setDefaults(self):
        super().setDefaults()
        self.astromTask.wcsFitter.retarget(FitAffineWcsTask)
        self.astromTask.doMagnitudeOutlierRejection = False
        self.astromTask.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 15
        self.astromTask.referenceSelector.magLimit = magLimit
        self.astromTask.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astromTask.matcher.maxRotationDeg = 5.99
        self.astromTask.matcher.maxOffsetPix = 3000
        self.astromTask.sourceSelector["science"].doRequirePrimary = False
        self.astromTask.sourceSelector["science"].doIsolated = False
        self.astromTask.sourceSelector["science"].doSignalToNoise = False


class GenerateDonutFromRefitWcsTask(GenerateDonutCatalogWcsTask):
    """
    Fit a new WCS to the image from a direct detect Donut
    Catalog and return the input exposure with the new
    WCS attached.
    """

    ConfigClass = GenerateDonutFromRefitWcsTaskConfig
    _DefaultName = "generateDonutFromRefitWcsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up the astrometry subtask for WCS fitting.
        self.makeSubtask("astromTask", refObjLoader=None)

    def formatDonutCatalog(self, catalog):
        """
        Create a minimal donut catalog in afwTable
        format from the input direct detect catalog.

        Parameters
        ----------
        catalog : `pandas.Dataframe`
            Catalog containing donut sources already detected
            on the exposure.

        Returns
        -------
        `lsst.afw.table.SimpleCatalog`
            Minimal catalog needed for astromeryTask to fit WCS.
        """

        sourceSchema = afwTable.SourceTable.makeMinimalSchema()
        measBase.SingleFrameMeasurementTask(schema=sourceSchema)  # expand the schema
        # add coord_raErr,  coord_decErr to the schema
        sourceSchema.addField(
            afwTable.Field["F"](
                name="coord_raErr", doc="position err in ra", units="rad"
            ),
        )
        sourceSchema.addField(
            afwTable.Field["F"](
                name="coord_decErr", doc="position err in dec", units="rad"
            ),
        )

        # create a catalog with that schema
        sourceCat = afwTable.SourceCatalog(sourceSchema)

        sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
        sourceIdKey = sourceSchema["id"].asKey()
        sourceRAKey = sourceSchema["coord_ra"].asKey()
        sourceDecKey = sourceSchema["coord_dec"].asKey()
        sourceInstFluxKey = sourceSchema["slot_ApFlux_instFlux"].asKey()
        sourceInstFluxErrKey = sourceSchema["slot_ApFlux_instFluxErr"].asKey()
        sourceRaErrKey = sourceSchema["coord_raErr"].asKey()
        sourceDecErrKey = sourceSchema["coord_decErr"].asKey()

        # decide if error needs to be computed based on the
        # value of coord_ra, coord_dec, or can we use
        # existing columns
        colnames = list(catalog.columns)
        make_new_raErr = True
        make_new_decErr = True
        if "coord_raErr" in colnames:
            make_new_raErr = False
        if "coord_decErr" in colnames:
            make_new_decErr = False

        Nrows = len(catalog)
        sourceCat.reserve(Nrows)

        for i in range(Nrows):
            src = sourceCat.addNew()
            src.set(sourceIdKey, i)

            # set ra,dec
            ra = lsst.geom.Angle(catalog["coord_ra"].iloc[i], lsst.geom.radians)
            src.set(sourceRAKey, ra)

            dec = lsst.geom.Angle(catalog["coord_dec"].iloc[i], lsst.geom.radians)
            src.set(sourceDecKey, dec)

            # set raErr, decErr
            if make_new_raErr:
                # set default 1% for raErr
                raErr = abs(ra) * 0.01
            else:
                # use the existing coord_raErr column
                raErr = lsst.geom.Angle(
                    catalog["coord_raErr"].iloc[i], lsst.geom.radians
                )
            src.set(sourceRaErrKey, raErr)

            if make_new_decErr:
                # set default 1% for raErr
                decErr = abs(dec) * 0.01
            else:
                # use the existing coord_decErr column
                decErr = lsst.geom.Angle(
                    catalog["coord_decErr"].iloc[i], lsst.geom.radians
                )
            src.set(sourceDecErrKey, decErr)

            # set the x,y centroid
            x = catalog["centroid_x"].iloc[i]
            y = catalog["centroid_y"].iloc[i]
            point = lsst.geom.Point2D(x, y)
            src.set(sourceCentroidKey, point)

            # set the flux and assume some small 1% flux error
            flux = catalog["source_flux"].iloc[i]
            src.set(sourceInstFluxKey, flux)

            fluxErr = abs(flux / 100.0)  # ensure positive error
            src.set(sourceInstFluxErrKey, fluxErr)

        return sourceCat

    @timeMethod
    def run(
        self,
        astromRefCat: typing.List[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
        fitDonutCatalog: pd.DataFrame,
        photoRefCat: typing.List[afwTable.SimpleCatalog],
    ) -> pipeBase.Struct:
        astromRefObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in astromRefCat],
            refCats=astromRefCat,
        )
        self.astromTask.setRefObjLoader(astromRefObjLoader)
        self.astromTask.refObjLoader.config.anyFilterMapsToThis = (
            self.config.anyFilterMapsToThis
        )
        afwCat = self.formatDonutCatalog(fitDonutCatalog)
        originalWcs = copy(exposure.wcs)

        successfulFit = False
        # Set a parameter in the metadata to
        # easily check whether the task ran WCS
        # fitting successfully or not. This will
        # give us information on our donut catalog output.
        self.metadata["wcsFitSuccess"] = False
        try:
            astromResult = self.astromTask.run(
                sourceCat=afwCat,
                exposure=exposure,
            )
            scatter = astromResult.scatterOnSky.asArcseconds()
            if scatter < self.config.maxFitScatter:
                successfulFit = True
                self.metadata["wcsFitSuccess"] = True
        except (RuntimeError, TaskError, IndexError, ValueError, AttributeError) as e:
            # IndexError raised for low source counts:
            # index 0 is out of bounds for axis 0 with size 0

            # ValueError: negative dimensions are not allowed
            # seen when refcat source count is low (but non-zero)

            # AttributeError: 'NoneType' object has no attribute 'asArcseconds'
            # when the result is a failure as the wcs is set to None on failure
            self.log.warning(f"Solving for WCS failed: {e}")
            # this is set to None when the fit fails, so restore it
            exposure.setWcs(originalWcs)
            donutCatalog = fitDonutCatalog
            self.log.warning(
                "Returning original exposure and WCS \
and direct detect catalog as output."
            )

        self.metadata["refCatalogSuccess"] = False
        if successfulFit:
            photoRefObjLoader = ReferenceObjectLoader(
                dataIds=[ref.dataId for ref in photoRefCat],
                refCats=photoRefCat,
            )
            detector = exposure.getDetector()
            filterName = exposure.filter.bandLabel

            try:
                # Match detector layout to reference catalog
                self.log.info("Running Donut Selector")
                donutSelectorTask = (
                    self.donutSelector if self.config.doDonutSelection is True else None
                )
                refSelection, blendCentersX, blendCentersY = runSelection(
                    photoRefObjLoader,
                    detector,
                    exposure.wcs,
                    filterName,
                    donutSelectorTask,
                )

                donutCatalog = donutCatalogToDataFrame(
                    refSelection, filterName, blendCentersX, blendCentersY
                )
                self.metadata["refCatalogSuccess"] = True

            # Except RuntimeError caused when no reference catalog
            # available for the region covered by detector
            except RuntimeError:
                self.log.warning("No catalogs cover this detector.")
                donutCatalog = fitDonutCatalog
                self.log.warning(
                    "Returning new WCS but original direct \
detect catalog as donutCatalog."
                )

        return pipeBase.Struct(outputExposure=exposure, donutCatalog=donutCatalog)
