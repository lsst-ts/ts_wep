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
from typing import Any

import numpy as np
from astropy.table import QTable

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.meas.algorithms import MagnitudeLimit, ReferenceObjectLoader
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.base.task import TaskError
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    addVisitInfoToCatTable,
    donutCatalogToAstropy,
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
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),  # type: ignore
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
        storageClass="AstropyQTable",
        name="directDetectDonutTable",
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
        name="post_isr_image",
    )
    donutCatalog = connectionTypes.Output(
        doc="Donut Locations",
        dimensions=(
            "visit",
            "detector",
            "instrument",
        ),
        storageClass="AstropyQTable",
        name="donutTable",
    )


class GenerateDonutFromRefitWcsTaskConfig(
    GenerateDonutCatalogWcsTaskConfig,
    pipelineConnections=GenerateDonutFromRefitWcsTaskConnections,  # type: ignore
):
    """
    Configuration settings for GenerateDonutCatalogWcsTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    astromTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=AstrometryTask, doc="Task for WCS fitting."
    )
    maxFitScatter: pexConfig.Field = pexConfig.Field(
        doc="Maximum allowed scatter for a successful fit (in arcseconds.)",
        dtype=float,
        default=1.0,
    )
    astromRefFilter: pexConfig.Field = pexConfig.Field(
        doc="Set filter to use in Astrometry task.",
        dtype=str,
        default="phot_g_mean",
    )
    photoRefFilter: pexConfig.Field = pexConfig.Field(
        doc="Set filter to use in Photometry catalog. "
        + "Cannot set both this and photoRefFilterPrefix. "
        + "If neither is set then will just try to use the name of the exposure filter.",
        dtype=str,
        optional=True,
    )
    photoRefFilterPrefix: pexConfig.Field = pexConfig.Field(
        doc="Set filter prefix to use. "
        + "Will then try to use exposure band label with given catalog prefix. "
        + "Cannot set both this and photoRefFilter. "
        + "If neither is set then will just try to use the name of the exposure filter.",
        dtype=str,
        optional=True,
    )
    catalogFilterList: pexConfig.ListField = pexConfig.ListField(
        dtype=str,
        doc="Filters from reference catalog to include in donut catalog.",
        default=["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"],
    )
    failTask: pexConfig.Field = pexConfig.Field(
        doc="Fail if error raised.",
        dtype=bool,
        default=False,
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Size of detector edge margin in pixels", dtype=int, default=80
    )

    # Took these defaults from atmospec/centroiding which I used
    # as a template for implementing WCS fitting in a task.
    # https://github.com/lsst/atmospec/blob/main/python/lsst/atmospec/centroiding.py
    def setDefaults(self) -> None:
        super().setDefaults()
        self.astromTask.wcsFitter.retarget(FitAffineWcsTask)
        self.astromTask.doMagnitudeOutlierRejection = False
        self.astromTask.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 15
        self.astromTask.referenceSelector.magLimit = magLimit
        self.astromTask.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astromTask.sourceSelector["science"].doRequirePrimary = False
        self.astromTask.sourceSelector["science"].doIsolated = False
        self.astromTask.sourceSelector["science"].doSignalToNoise = False
        self.astromTask.sourceSelector["science"].doCentroidErrorLimit = False


class GenerateDonutFromRefitWcsTask(GenerateDonutCatalogWcsTask):
    """
    Fit a new WCS to the image from a direct detect Donut
    Catalog and return the input exposure with the new
    WCS attached.
    """

    ConfigClass = GenerateDonutFromRefitWcsTaskConfig
    _DefaultName = "generateDonutFromRefitWcsTask"
    config: GenerateDonutFromRefitWcsTaskConfig
    astromTask: AstrometryTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set up the astrometry subtask for WCS fitting.
        self.makeSubtask("astromTask", refObjLoader=None)

    def formatDonutCatalog(self, catalog: QTable) -> afwTable.SimpleCatalog:
        """
        Create a minimal donut catalog in afwTable
        format from the input direct detect catalog.

        Parameters
        ----------
        catalog : `astropy.table.QTable`
            Catalog containing donut sources already detected
            on the exposure.

        Returns
        -------
        `lsst.afw.table.SimpleCatalog`
            Minimal catalog needed for astromeryTask to fit WCS.
        """

        sourceSchema = afwTable.SourceTable.makeMinimalSchema()
        measBase.SingleFrameMeasurementTask(schema=sourceSchema)  # expand the schema
        # add coord_raErr, coord_decErr to the schema
        # after 2025_22 these fields are present by default;
        # this is added for backwards compatibility
        for c in ["ra", "dec"]:
            name = f"coord_{c}Err"
            if name not in sourceSchema.getNames():
                sourceSchema.addField(
                    afwTable.Field["F"](name=name, doc=f"position err in {c}", units="rad"),
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
            ra = lsst.geom.Angle(catalog["coord_ra"][i].value, lsst.geom.radians)
            src.set(sourceRAKey, ra)

            dec = lsst.geom.Angle(catalog["coord_dec"][i].value, lsst.geom.radians)
            src.set(sourceDecKey, dec)

            # set raErr, decErr
            if make_new_raErr:
                # set default 1% for raErr
                raErr = abs(ra) * 0.01
            else:
                # use the existing coord_raErr column
                raErr = lsst.geom.Angle(catalog["coord_raErr"][i].value, lsst.geom.radians)
            src.set(sourceRaErrKey, raErr)

            if make_new_decErr:
                # set default 1% for raErr
                decErr = abs(dec) * 0.01
            else:
                # use the existing coord_decErr column
                decErr = lsst.geom.Angle(catalog["coord_decErr"][i].value, lsst.geom.radians)
            src.set(sourceDecErrKey, decErr)

            # set the x,y centroid
            x = catalog["centroid_x"][i]
            y = catalog["centroid_y"][i]
            point = lsst.geom.Point2D(x, y)
            src.set(sourceCentroidKey, point)

            # set the flux and assume some small 1% flux error
            flux = catalog["source_flux"][i].value
            src.set(sourceInstFluxKey, flux)

            fluxErr = abs(flux / 100.0)  # ensure positive error
            src.set(sourceInstFluxErrKey, fluxErr)

        return sourceCat

    def _clipDonutCatalogToDetector(
        self,
        donutCatalog: QTable,
        detector: lsst.afw.cameraGeom.Detector,
    ) -> QTable:
        """Clip donut catalog to only include sources within the
        detector bounding box, eroded by the configured edgeMargin.

        This acts as a final safety net to ensure no donuts outside
        the detector bounds make it into the output catalog,
        regardless of which code path produced the catalog.

        Parameters
        ----------
        donutCatalog : `astropy.table.QTable`
            Donut catalog to clip.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object from the camera.

        Returns
        -------
        clippedCatalog : `astropy.table.QTable`
            Catalog with only sources inside the eroded bounding box.
        """
        bbox = detector.getBBox()
        erodedBBox = bbox.erodedBy(self.config.edgeMargin)

        inBounds = np.array(
            [
                erodedBBox.contains(lsst.geom.Point2I(int(x), int(y)))
                for x, y in zip(donutCatalog["centroid_x"], donutCatalog["centroid_y"])
            ]
        )

        nRemoved = len(donutCatalog) - np.sum(inBounds)
        if nRemoved > 0:
            self.log.warning(
                "Removed %d donuts outside detector bounding box (eroded by %d px edge margin).",
                nRemoved,
                self.config.edgeMargin,
            )

        return donutCatalog[inBounds]

    @timeMethod
    def run(
        self,
        astromRefCat: typing.List[afwTable.SimpleCatalog],
        exposure: afwImage.Exposure,
        fitDonutCatalog: QTable,
        photoRefCat: typing.List[afwTable.SimpleCatalog],
    ) -> pipeBase.Struct:
        astromRefObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in astromRefCat],
            refCats=astromRefCat,
        )
        self.astromTask.setRefObjLoader(astromRefObjLoader)
        self.astromTask.refObjLoader.config.anyFilterMapsToThis = self.config.astromRefFilter
        afwCat = self.formatDonutCatalog(fitDonutCatalog)
        originalExposure = copy(exposure)
        bandLabel = originalExposure.filter.bandLabel

        # WCS has been fit but scatter is too high, so use original
        # catalog but make cuts according to desired final catalog
        # since we may allow more donuts for WCS fitting than we want
        # in the final catalog.
        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            fitDonutCatalog[f"{bandLabel}_flux"] = fitDonutCatalog["source_flux"]
            donutSelection = self.donutSelector.run(fitDonutCatalog, exposure.detector, bandLabel)
            trimmedDirectCatalog = fitDonutCatalog[donutSelection.selected]
            trimmedDirectCatalog.meta["blend_centroid_x"] = donutSelection.blendCentersX
            trimmedDirectCatalog.meta["blend_centroid_y"] = donutSelection.blendCentersY
        else:
            trimmedDirectCatalog = fitDonutCatalog

        successfulFit = False
        # Set a parameter in the metadata to
        # easily check whether the task ran WCS
        # fitting successfully or not. This will
        # give us information on our donut catalog output.
        self.metadata["wcsFitSuccess"] = False
        self.metadata["refCatalogSuccess"] = False
        try:
            astromResult = self.astromTask.run(
                sourceCat=afwCat,
                exposure=exposure,
            )
            scatter = astromResult.scatterOnSky.asArcseconds()
            if scatter < self.config.maxFitScatter:
                successfulFit = True
                self.metadata["wcsFitSuccess"] = True
            else:
                # WCS has been fit but scatter is
                # too high, so use original catalog
                donutCatalog = trimmedDirectCatalog
                self.log.warning("Returning original exposure and WCS and direct detect catalog as output.")

        except (RuntimeError, TaskError, IndexError, ValueError, AttributeError) as e:
            # IndexError raised for low source counts:
            # index 0 is out of bounds for axis 0 with size 0

            # ValueError: negative dimensions are not allowed
            # seen when refcat source count is low (but non-zero)

            # AttributeError: 'NoneType' object has no attribute 'asArcseconds'
            # when the result is a failure as the wcs is set to None on failure
            self.log.warning(f"Solving for WCS failed: {e}")
            if self.config.failTask:
                raise TaskError("Failing task due to wcs fit failure.")
            else:
                # this is set to None when the fit fails, so restore it
                donutCatalog = trimmedDirectCatalog
                self.log.warning("Returning original exposure and WCS and direct detect catalog as output.")

        if successfulFit:
            photoRefObjLoader = ReferenceObjectLoader(
                dataIds=[ref.dataId for ref in photoRefCat],
                refCats=photoRefCat,
            )
            detector = exposure.getDetector()
            filterName = exposure.filter.bandLabel
            catCreateErrorMsg = "Returning new WCS but original direct detect catalog as donutCatalog."

            # Check that there are catalogs
            if len(photoRefCat) == 0:
                self.log.warning("No catalogs cover this detector.")
                donutCatalog = trimmedDirectCatalog
                self.log.warning(catCreateErrorMsg)
                detectorName = detector.getName()
                donutCatalog["detector"] = np.array([detectorName] * len(donutCatalog), dtype=str)
                donutCatalog = addVisitInfoToCatTable(exposure, donutCatalog)
                donutCatalog = self._clipDonutCatalogToDetector(donutCatalog, detector)
                return pipeBase.Struct(outputExposure=exposure, donutCatalog=donutCatalog)

            # Check that specified filter exists in catalogs
            if self.config.photoRefFilter is not None and self.config.photoRefFilterPrefix is not None:
                raise ValueError("photoRefFilter and photoRefFilterConfig cannot both be set.")
            if self.config.photoRefFilter is not None:
                filterName = self.config.photoRefFilter
            elif self.config.photoRefFilterPrefix is not None:
                filterName = f"{self.config.photoRefFilterPrefix}_{exposure.filter.bandLabel}"

            # Test that given filter name exists in catalog.
            if f"{filterName}_flux" not in photoRefCat[0].get().schema:
                filterFailMsg = f"Photometric Reference Catalog does not contain photoRefFilter: {filterName}"
                self.log.warning(filterFailMsg)
                donutCatalog = trimmedDirectCatalog
                self.log.warning(catCreateErrorMsg)
                detectorName = detector.getName()
                donutCatalog["detector"] = np.array([detectorName] * len(donutCatalog), dtype=str)
                donutCatalog = addVisitInfoToCatTable(exposure, donutCatalog)
                donutCatalog = self._clipDonutCatalogToDetector(donutCatalog, detector)
                return pipeBase.Struct(outputExposure=exposure, donutCatalog=donutCatalog)

            try:
                # Match detector layout to reference catalog
                self.log.info("Running Donut Selector")
                donutSelectorTask = self.donutSelector if self.config.doDonutSelection is True else None
                edgeMargin = self.config.edgeMargin
                refSelection, blendCentersX, blendCentersY = runSelection(
                    photoRefObjLoader,
                    detector,
                    exposure.wcs,
                    filterName,
                    donutSelectorTask,
                    edgeMargin,
                )
                # Create list of filters to include in final catalog
                filterList = self.config.catalogFilterList
                availableRefFilters = [
                    col[:-5] for col in refSelection.schema.getNames() if col.endswith("_flux")
                ]

                # Validate all requested filters exist
                missing_filters = set(filterList) - set(availableRefFilters)
                if missing_filters:
                    raise TaskError(
                        f"Filter(s) {missing_filters} not in available columns"
                        " in reference catalog. Check catalogFilterList config "
                        f"(currently set as {filterList}). "
                        f"Available ref catalog filters are {availableRefFilters}."
                    )

                # Add photoRefFilter if not present and get its index
                if filterName not in filterList:
                    filterList.append(filterName)
                sortFilterIdx = filterList.index(filterName)

                donutCatalog = donutCatalogToAstropy(
                    refSelection,
                    filterList,
                    blendCentersX,
                    blendCentersY,
                    sortFilterIdx=sortFilterIdx,
                )
                self.metadata["refCatalogSuccess"] = True
            # Except RuntimeError caused when no reference catalog
            # available for the region covered by detector
            except RuntimeError:
                self.log.warning("No catalogs cover this detector.")
                donutCatalog = trimmedDirectCatalog
                self.log.warning(catCreateErrorMsg)

        detectorName = exposure.getDetector().getName()
        donutCatalog["detector"] = np.array([detectorName] * len(donutCatalog), dtype=str)
        donutCatalog = addVisitInfoToCatTable(exposure, donutCatalog)

        # Clip donut catalog to detector bounds as a final safety net.
        # This catches donuts outside the bounding box from all code paths:
        # fallback to fitDonutCatalog, WCS-projection edge effects, etc.
        donutCatalog = self._clipDonutCatalogToDetector(donutCatalog, exposure.getDetector())

        # We want the original image array with the new WCS
        # attached to it if the WCS fitting was successful.
        if self.metadata["wcsFitSuccess"] is True:
            originalExposure.setWcs(exposure.wcs)

        return pipeBase.Struct(outputExposure=originalExposure, donutCatalog=donutCatalog)
