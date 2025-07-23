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

__all__ = [
    "GenerateDonutDirectDetectTaskConnections",
    "GenerateDonutDirectDetectTaskConfig",
    "GenerateDonutDirectDetectTask",
]

from typing import Any

import astropy.units as u
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import numpy as np
from astropy.table import QTable
from lsst.afw.cameraGeom import Camera
from lsst.afw.image import Exposure
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.ts.wep.task.donutQuickMeasurementTask import DonutQuickMeasurementTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable
from lsst.ts.wep.utils import DefocalType, createTemplateForDetector, getTaskInstrument
from lsst.utils.timer import timeMethod


class GenerateDonutDirectDetectTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")  # type: ignore
):
    """
    Specify the pipeline connections needed for
    GenerateDonutDirectDetectTask. We
    need the defocal exposure, and will produce donut catalogs
    for a specified instrument.
    """

    exposure = connectionTypes.Input(
        doc="Input exposure to make measurements on",
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
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )


class GenerateDonutDirectDetectTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutDirectDetectTaskConnections,  # type: ignore
):
    """
    Configuration settings for GenerateDonutDirectDetectTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    measurementTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutQuickMeasurementTask,
        doc="How to run source detection and measurement.",
    )
    subtractBackground: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )
    opticalModel: pexConfig.Field = pexConfig.Field(
        doc="Specify the optical model (offAxis, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instConfigFile: pexConfig.Field = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded.",
        dtype=str,
        optional=True,
    )
    initialCutoutPadding: pexConfig.Field = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of the initial "
            + "donut diameter guess for template postage stamp size "
            + "and for bounding boxes used for estimating centroids."
        ),
        dtype=int,
        default=5,
    )
    donutSelector: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection: pexConfig.Field = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )
    edgeMargin: pexConfig.Field = pexConfig.Field(
        doc="Size of detector edge margin in pixels", dtype=int, default=80
    )


class GenerateDonutDirectDetectTask(pipeBase.PipelineTask):
    """
    Generate donut template and convolve with the defocal image to
    detect sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutDirectDetectTaskConfig
    _DefaultName = "generateDonutDirectDetectTask"
    config: GenerateDonutDirectDetectTaskConfig
    measurementTask: DonutQuickMeasurementTask
    donutSelector: DonutSourceSelectorTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Instantiate the quickFrameMeasurementTask
        self.makeSubtask("measurementTask")

        # Set up the donut selector task if we need it
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

        # Set up background subtraction task
        self.makeSubtask("subtractBackground")

        # Set which sensors are intra focal
        # to create correct template
        self.intraFocalNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

    def updateDonutCatalog(
        self, donutCat: QTable, exposure: afwImage.Exposure
    ) -> QTable:
        """
        Reorganize the content of donut catalog
        adding detector column, doing the transpose,
        and passing the exposure WCS boresight
        as coord_ra, coord_dec - these columns
        are required by EstimateZernikes, but not used explicitly
        downstream.


        Parameters
        ----------
        donutCat : astropy.table.QTable
            The donut catalog from running DonutDetector,
            contains columns 'y_center', 'x_center'
        exposure : lsst.afw.image.Exposure
            Exposure with the donut images.

        Returns
        -------
        donutCat : astropy.table.QTable
            Donut catalog with reorganized content.
        """
        # EstimateZernikes expects the following column names:
        # coord_ra; coord_dec; centroid_x; centroid_y;
        # source_flux; detector; mags

        # add a detector column
        donutCat["detector"] = exposure.detector.getName()

        # pass the boresight ra, dec
        wcs = exposure.getWcs()
        x = np.array(donutCat["centroid_x"])
        y = np.array(donutCat["centroid_y"])

        x = np.zeros(0)
        for row in donutCat["centroid_x"]:
            x = np.append(x, row)

        ra, dec = wcs.pixelToSkyArray(x, y, degrees=False)

        donutCat["coord_ra"] = ra * u.rad
        donutCat["coord_dec"] = dec * u.rad

        donutCatUpd = donutCat[
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "detector",
            ]
        ]
        donutCatUpd["source_flux"] = donutCat["source_flux"] * u.nJy
        fluxSort = np.argsort(donutCatUpd["source_flux"])[::-1]
        # It is possible for catalog to have multiple sources
        # detected, but with source selection turned off
        # the QTable metadata of `blend_centroid_x` and
        # `blend_centroid_y` will be empty.
        if self.config.doDonutSelection:
            donutCatUpd.meta["blend_centroid_x"] = [
                donutCat.meta["blend_centroid_x"][idx] for idx in fluxSort
            ]
            donutCatUpd.meta["blend_centroid_y"] = [
                donutCat.meta["blend_centroid_y"][idx] for idx in fluxSort
            ]

        donutCatUpd.sort("source_flux", reverse=True)

        return donutCatUpd

    def emptyTable(self) -> QTable:
        """Return empty donut table if no donuts got
        detected or selected.

        Returns
        -------
        astropy.table.QTable
            An empty donut table with correct columns.
        """
        donutColumns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
        ]
        donutTable = QTable(names=donutColumns)
        donutTable.meta["blend_centroid_x"] = ""
        donutTable.meta["blend_centroid_y"] = ""
        return donutTable

    @timeMethod
    def run(self, exposure: Exposure, camera: Camera) -> pipeBase.Struct:
        camName = camera.getName()
        detectorName = exposure.getDetector().getName()
        bandLabel = exposure.filter.bandLabel

        # We can pick an arbitrary defocalType because the templates
        # look the same for intra/extra images (note this is only
        # true in the rotated DVCS coordinate system)
        defocalType = DefocalType.Extra

        # Switch for the case of some corner detectors being in-focus, and the
        # other making giant donuts
        if detectorName in self.intraFocalNames:
            defocalType = DefocalType.Intra

        # Load the instrument
        instrument = getTaskInstrument(
            camName,
            detectorName,
            self.config.instConfigFile,
        )
        try:
            # Create the image template for the detector
            template = createTemplateForDetector(
                detector=exposure.detector,
                defocalType=defocalType,
                bandLabel=bandLabel,
                instrument=instrument,
                opticalModel=self.config.opticalModel,
                padding=self.config.initialCutoutPadding,
                isBinary=True,
            )
        except ValueError as e:
            err_msg = str(e)
            s = (
                f"Template creation error: {err_msg} \n\
That means that the provided exposure is very close to focus"
                if err_msg.startswith("negative dimensions")
                else err_msg
            )
            self.log.warning(f"Cannot create template: {s}")
            self.log.warning("Returning empty donut catalog")
            donutCatUpd = self.emptyTable()
            donutCatUpd = addVisitInfoToCatTable(exposure, donutCatUpd)
            return pipeBase.Struct(donutCatalog=donutCatUpd)

        # Run background subtraction
        self.subtractBackground.run(exposure=exposure)

        # Trim the exposure by the margin
        edgeMargin = self.config.edgeMargin
        bbox = exposure.getBBox()
        trimmedBBox = bbox.erodedBy(edgeMargin)
        exposureTrim = exposure[trimmedBBox].clone()

        # Run the measurement task
        objData = self.measurementTask.run(
            exposureTrim,
            template,
            donutDiameter=np.ceil(instrument.donutDiameter).astype(int),
            cutoutPadding=self.config.initialCutoutPadding,
        )
        if len(objData.detectedCatalog) > 0:
            donutTable = QTable(rows=list(objData.detectedCatalog.values()))
            # Use the aperture flux with a 70 pixel aperture
            donutTable[f"{bandLabel}_flux"] = donutTable["apFlux70"]

            # Set the required columns to be empty, unless
            # overwritten by donutSelector below
            donutTable.meta["blend_centroid_x"] = ""
            donutTable.meta["blend_centroid_y"] = ""

            # Run the donut selector task.
            if self.config.doDonutSelection:
                self.log.info("Running Donut Selector")
                donutSelection = self.donutSelector.run(
                    donutTable, exposure.detector, bandLabel
                )
                donutCatSelected = donutTable[donutSelection.selected]
                donutCatSelected.meta["blend_centroid_x"] = donutSelection.blendCentersX
                donutCatSelected.meta["blend_centroid_y"] = donutSelection.blendCentersY
            else:
                donutCatSelected = donutTable

            donutCatSelected.rename_column(f"{bandLabel}_flux", "source_flux")

            # If at least one donut got selected, update the column names
            # and content
            if len(donutCatSelected) > 0:
                donutCatUpd = self.updateDonutCatalog(donutCatSelected, exposure)
                donutCatUpd["detector"] = np.array(
                    [detectorName] * len(donutCatUpd), dtype=str
                )
            # If no donuts got selected, issue a warning and return an empty
            # donut table
            else:
                self.log.warning(
                    "No sources selected in the exposure. Returning an empty donut catalog."
                )
                donutCatUpd = self.emptyTable()
        else:

            self.log.warning(
                "No sources found in the exposure. Returning an empty donut catalog."
            )
            donutCatUpd = self.emptyTable()

        donutCatUpd = addVisitInfoToCatTable(exposure, donutCatUpd)

        return pipeBase.Struct(donutCatalog=donutCatUpd)
