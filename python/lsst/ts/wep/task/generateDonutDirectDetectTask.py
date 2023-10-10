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
from lsst.ts.wep.cwfs.donutTemplateFactory import DonutTemplateFactory
from lsst.ts.wep.task.donutQuickMeasurementTask import DonutQuickMeasurementTask
from lsst.ts.wep.task.donutSourceSelectorTask import DonutSourceSelectorTask
from lsst.ts.wep.utils import (
    DefocalType,
    DonutTemplateType,
    createInstDictFromConfig,
    getCamTypeFromButlerName,
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
    donutDiameter = pexConfig.Field(
        dtype=int,
        doc="The expected diameter of donuts in a donut image, in pixels.",
        default=400,
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, paraxial, onAxis).",
        dtype=str,
        default="offAxis",
    )
    instObscuration = pexConfig.Field(
        doc="Obscuration (inner_radius / outer_radius of M1M3)",
        dtype=float,
        default=0.61,
    )
    instFocalLength = pexConfig.Field(
        doc="Instrument Focal Length in m", dtype=float, default=10.312
    )
    instApertureDiameter = pexConfig.Field(
        doc="Instrument Aperture Diameter in m", dtype=float, default=8.36
    )
    instDefocalOffset = pexConfig.Field(
        doc="Instrument defocal offset in mm. \
        If None then will get this from the focusZ value in exposure visitInfo. \
        (The default is None.)",
        dtype=float,
        default=None,
        optional=True,
    )
    instPixelSize = pexConfig.Field(
        doc="Instrument Pixel Size in m", dtype=float, default=10.0e-6
    )
    initialCutoutPadding = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of initial "
            + "`donutDiameter` guess for template postage stamp size "
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

        # Set up instrument configuration dict
        self.instParams = createInstDictFromConfig(self.config)

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
        detectorName = exposure.detector.getName()
        detectorType = exposure.detector.getType()
        filterName = exposure.filter.bandLabel
        defocalType = DefocalType.Extra  # we use one of the DefocalTypes,
        camType = getCamTypeFromButlerName(camera.getName(), detectorType)
        pixelScale = exposure.getWcs().getPixelScale().asArcseconds()

        # Get defocal distance from focusZ.
        if self.instParams["offset"] is None:
            self.instParams["offset"] = np.abs(exposure.visitInfo.focusZ)
        # LSST CWFS are offset +/- 1.5 mm when LSST camera defocus is at 0.
        if detectorType.name == "WAVEFRONT":
            if defocalType == DefocalType.Extra:
                self.instParams["offset"] = np.abs(self.instParams["offset"] - 1.5)
            elif defocalType == DefocalType.Intra:
                self.instParams["offset"] = np.abs(self.instParams["offset"] + 1.5)
            else:
                raise ValueError(f"Defocal Type {defocalType} not valid.")

        # create a donut template
        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )
        templateSize = int(self.config.donutDiameter + self.config.initialCutoutPadding)
        if templateSize % 2 == 1:
            templateSize += 1
        template = templateMaker.makeTemplate(
            detectorName,
            defocalType,
            templateSize,
            camType=camType,
            opticalModel=self.config.opticalModel,
            pixelScale=pixelScale,
            instParams=self.instParams,
        )

        objData = self.measurementTask.run(
            exposure,
            template,
            donutDiameter=self.config.donutDiameter,
            cutoutPadding=self.config.initialCutoutPadding,
        )
        donutDf = pd.DataFrame.from_dict(objData.detectedCatalog, orient="index")
        # Use the aperture flux with a 70 pixel aperture
        donutDf[f"{filterName}_flux"] = donutDf["apFlux70"]

        # Run the donut selector task.
        if self.config.doDonutSelection:
            self.log.info("Running Donut Selector")
            donutSelection = self.donutSelector.run(
                donutDf, exposure.detector, filterName
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
            columns={f"{filterName}_flux": "source_flux"}, inplace=True
        )

        # update column names and content
        donutCatUpd = self.updateDonutCatalog(donutCatSelected, exposure)

        return pipeBase.Struct(donutCatalog=donutCatUpd)
