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
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
from lsst.utils.timer import timeMethod
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.meas.algorithms import (
    LoadReferenceObjectsTask,
    MagnitudeLimit,
    ReferenceObjectLoader,
)


class FitWcsFromDetectedTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
):
    """
    Specify the pipeline inputs and outputs needed
    for FitDonutWcsTask.
    """

    inputExposure = connectionTypes.Input(
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
    outputExposure = connectionTypes.Output(
        doc="Output exposure with new WCS",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
    )


class FitWcsFromDetectedTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FitWcsFromDetectedTaskConnections,
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    astromRefObjLoader = pexConfig.ConfigurableField(
        target=LoadReferenceObjectsTask,
        doc="Reference object loader for astrometric calibration",
    )
    astromTask = pexConfig.ConfigurableField(
        target=AstrometryTask, doc="Task for WCS fitting."
    )

    # Took these defaults from atmospec/centroiding
    def setDefaults(self):
        super().setDefaults()
        self.astromRefObjLoader.pixelMargin = 1000

        self.astromTask.wcsFitter.retarget(FitAffineWcsTask)
        self.astromTask.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 15
        self.astromTask.referenceSelector.magLimit = magLimit
        self.astromTask.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astromTask.matcher.maxRotationDeg = 5.99
        self.astromTask.matcher.maxOffsetPix = 3000
        # self.astromTask.sourceSelector['matcher'].minSnr = 10


class FitWcsFromDetectedTask(pipeBase.PipelineTask):
    """
    Fit a new WCS to the image from a direct detect Donut
    Catalog and return the input exposure with the new
    WCS attached.
    """

    ConfigClass = FitWcsFromDetectedTaskConfig
    _DefaultName = "fitWcsFromDetectedTask"

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
        sourceCat = afwTable.SourceCatalog(sourceSchema)

        sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
        sourceIdKey = sourceSchema["id"].asKey()
        sourceRAKey = sourceSchema["coord_ra"].asKey()
        sourceDecKey = sourceSchema["coord_dec"].asKey()
        sourceInstFluxKey = sourceSchema["slot_ApFlux_instFlux"].asKey()
        sourceInstFluxErrKey = sourceSchema["slot_ApFlux_instFluxErr"].asKey()

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

            # set the x,y centroid
            x = catalog["centroid_x"].iloc[i]
            y = catalog["centroid_y"].iloc[i]
            point = lsst.geom.Point2D(x, y)
            src.set(sourceCentroidKey, point)

            # set the flux and assume some small 1% flux error
            flux = catalog["source_flux"].iloc[i]
            src.set(sourceInstFluxKey, flux)

            fluxErr = flux / 100.0
            src.set(sourceInstFluxErrKey, fluxErr)

        return sourceCat

    def fitWcs(self, catalog, exposure):
        """
        Perform the WCS fit and return the exposure
        with the new WCS attached.

        Parameters
        ----------
        catalog : `lsst.afw.table.SimpleCatalog`
            Catalog of donut sources directly detected
            on the exposure.
        exposure : `lsst.afw.image.Exposure`
            Exposure with the donut images and an
            included WCS.

        Returns
        -------
        `lsst.afw.image.Exposure`
            The same as the input exposure except with
            a newly fit WCS attached overwriting the old WCS.
        """

        afwCat = self.formatDonutCatalog(catalog)
        self.astromTask.run(
            sourceCat=afwCat,
            exposure=exposure,
        )

        return exposure

    @timeMethod
    def run(
        self,
        astromRefCat: typing.List[afwTable.SimpleCatalog],
        inputExposure: afwImage.Exposure,
        fitDonutCatalog: pd.DataFrame,
    ) -> pipeBase.Struct:
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in astromRefCat],
            refCats=astromRefCat,
        )
        self.astromTask.setRefObjLoader(refObjLoader)
        exposure = self.fitWcs(fitDonutCatalog, inputExposure)

        return pipeBase.Struct(outputExposure=exposure)
