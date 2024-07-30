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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import numpy as np
import pandas as pd
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.ts.wep.task.donutQuickMeasurementTask import DonutQuickMeasurementTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.utils import (
    DefocalType,
    createTemplateForDetector,
    getOffsetFromExposure,
    getTaskInstrument,
)
from lsst.utils.timer import timeMethod


class GenerateDonutDirectDetectTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
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
    pipelineConnections=GenerateDonutDirectDetectTaskConnections,
):
    """
    Configuration settings for GenerateDonutDirectDetectTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    measurementTask = pexConfig.ConfigurableField(
        target=DonutQuickMeasurementTask,
        doc="How to run source detection and measurement.",
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instConfigFile = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded, and the defocal offset will "
        + "be determined from the focusZ value in the exposure header.",
        dtype=str,
        optional=True,
    )
    initialCutoutPadding = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of the initial "
            + "donut diameter guess for template postage stamp size "
            + "and for bounding boxes used for estimating centroids."
        ),
        dtype=int,
        default=5,
    )
    donutSelector = pexConfig.ConfigurableField(
        target=DonutSourceSelectorTask, doc="How to select donut targets."
    )
    doDonutSelection = pexConfig.Field(
        doc="Whether or not to run donut selector.", dtype=bool, default=True
    )


class GenerateDonutDirectDetectTask(pipeBase.PipelineTask):
    """
    Generate donut template and convolve with the defocal image to
    detect sources on the detectors for AOS.
    """

    ConfigClass = GenerateDonutDirectDetectTaskConfig
    _DefaultName = "generateDonutDirectDetectTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Instantiate the quickFrameMeasurementTask
        self.makeSubtask("measurementTask")

        # Set up the donut selector task if we need it
        if self.config.doDonutSelection:
            self.makeSubtask("donutSelector")

    def updateDonutCatalog(self, donutCat, exposure):
        """
        Reorganize the content of donut catalog
        adding detector column, doing the transpose,
        and passing the exposure WCS boresight
        as coord_ra, coord_dec - these columns
        are required by EstimateZernikes, but not used explicitly
        downstream.


        Parameters
        ----------
        donutCat : pandas.DataFrame
            The donut catalog from running DonutDetector,
            contains columns 'y_center', 'x_center'
        exposure : lsst.afw.image.Exposure
            Exposure with the donut images.

        Returns
        -------
        donutCat : pandas.DataFrame
            Donut catalog with reorganized content.
        """
        # EstimateZernikes expects the following column names:
        # coord_ra; coord_dec; centroid_x; centroid_y;
        # source_flux; detector; mags

        # add a detector column
        donutCat["detector"] = exposure.detector.getName()

        # pass the boresight ra, dec
        wcs = exposure.getWcs()
        x = np.array(donutCat["centroid_x"].values)
        y = np.array(donutCat["centroid_y"].values)

        x = np.zeros(0)
        for row in donutCat["centroid_x"]:
            x = np.append(x, row)

        ra, dec = wcs.pixelToSkyArray(x, y, degrees=False)

        donutCat["coord_ra"] = ra
        donutCat["coord_dec"] = dec

        donutCatUpd = donutCat[
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "detector",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ]
        ]

        donutCatUpd = donutCatUpd.sort_values(
            "source_flux", ascending=False
        ).reset_index(drop=True)

        return donutCatUpd

    @timeMethod
    def run(self, exposure, camera):
        camName = camera.getName()
        detectorName = exposure.getDetector().getName()
        bandLabel = exposure.filter.bandLabel

        # We can pick an arbitrary defocalType because the templates
        # look the same for intra/extra images (note this is only
        # true in the rotated DVCS coordinate system)
        defocalType = DefocalType.Extra

        # Get the offset
        offset = getOffsetFromExposure(exposure, camName, defocalType)

        # Load the instrument
        instrument = getTaskInstrument(
            camName,
            detectorName,
            offset,
            self.config.instConfigFile,
        )

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

        # Run the measurement task
        objData = self.measurementTask.run(
            exposure,
            template,
            donutDiameter=np.ceil(instrument.donutDiameter).astype(int),
            cutoutPadding=self.config.initialCutoutPadding,
        )
        donutDf = pd.DataFrame.from_dict(objData.detectedCatalog, orient="index")
        # Use the aperture flux with a 70 pixel aperture
        donutDf[f"{bandLabel}_flux"] = donutDf["apFlux70"]

        # Run the donut selector task.
        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            donutSelection = self.donutSelector.run(
                donutDf, exposure.detector, bandLabel
            )
            donutCatSelected = donutDf[donutSelection.selected].reset_index(drop=True)
            donutCatSelected["blend_centroid_x"] = donutSelection.blendCentersX
            donutCatSelected["blend_centroid_y"] = donutSelection.blendCentersY
        else:
            # if donut selector was not run,
            # set the required columns to be empty
            donutDf["blend_centroid_x"] = ""
            donutDf["blend_centroid_y"] = ""
            donutCatSelected = donutDf

        donutCatSelected.rename(
            columns={f"{bandLabel}_flux": "source_flux"}, inplace=True
        )

        # update column names and content
        donutCatUpd = self.updateDonutCatalog(donutCatSelected, exposure)

        return pipeBase.Struct(donutCatalog=donutCatUpd)
