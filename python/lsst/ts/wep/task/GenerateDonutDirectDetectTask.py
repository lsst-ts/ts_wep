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

import os
import numpy as np
from copy import copy
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.utils.timer import timeMethod
from lsst.ts.wep.DonutDetector import DonutDetector
from lsst.ts.wep.Utility import getCamType, DefocalType, DonutTemplateType
from lsst.ts.wep.cwfs.DonutTemplateFactory import DonutTemplateFactory


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


class GenerateDonutDirectDetectTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=GenerateDonutDirectDetectTaskConnections,
):
    """
    Configuration settings for GenerateDonutDirectDetectTask.
    Specifies filter and camera details as well as subtasks
    that run to do the source selection.
    """

    # Config setting for pipeline task with defaults
    donutTemplateSize = pexConfig.Field(
        doc="Size of Template in pixels", dtype=int, default=160
    )
    instName = pexConfig.Field(
        doc="Specify the instrument name (lsst, lsstfam, comcam, auxTel).",
        dtype=str,
        default="lsst",
    )
    opticalModel = pexConfig.Field(
        doc="Specify the optical model (offAxis, paraxial, onAxis).",
        dtype=str,
        default="offAxis",
    )
    removeBlends = pexConfig.Field(
        doc="Decide whether to remove blended objects from the donut catalog.",
        dtype=bool,
        default=True,
    )
    blendRadius = pexConfig.Field(
        doc="Specify the pixel radius within which an object is considered as blended.",
        dtype=int,
        default=200,
    )
    peakThreshold = pexConfig.Field(
        doc="Specify the fraction (between 0 and 1) of the highest pixel value in the convolved image.",
        dtype=float,
        default=0.99,
    )
    binaryChoice = pexConfig.Field(
        doc="Choose how donut detector should arrive at the binary image ('centroid' for centroidFinder,\
        'deblend' for adaptative image thresholding, 'exposure' to use the exposure image directly).",
        dtype=str,
        default="deblend",
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

        # TODO: Temporary until DM-24162 is closed at which point we
        # can remove this
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

        # Set size (in pixels) of donut template image used for
        # convolution with template
        self.donutTemplateSize = self.config.donutTemplateSize
        # specify instrument name
        self.instName = self.config.instName
        # specify optical model
        self.opticalModel = self.config.opticalModel
        # decide whether to remove blends from catalog
        self.removeBlends = self.config.removeBlends
        # specify blend radius
        self.blendRadius = self.config.blendRadius
        # specify peak threshold
        self.peakThreshold = self.config.peakThreshold
        # choose how to calculate binary image
        self.binaryChoice = self.config.binaryChoice

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
        donutCat["detector"] = exposure.getDetector().getName()

        # rename columns: transpose y --> x
        donutCat = donutCat.rename(
            columns={"y_center": "centroid_x", "x_center": "centroid_y"}
        )

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
        return donutCat

    @timeMethod
    def run(
        self,
        exposure: afwImage.Exposure,
    ) -> pipeBase.Struct:

        detectorName = exposure.getDetector().getName()
        defocalType = DefocalType.Extra  # we use one of the DefocalTypes,
        # TODO: perhaps could make that as another configurable
        camType = getCamType(self.instName)
        pixelScale = exposure.getWcs().getPixelScale().asArcseconds()

        # create a donut template
        templateMaker = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )

        template = templateMaker.makeTemplate(
            detectorName,
            defocalType,
            self.donutTemplateSize,
            camType=camType,
            opticalModel=self.opticalModel,
            pixelScale=pixelScale,
        )

        # given this template, detect donuts in one of the defocal images
        detector = DonutDetector()
        expArray = copy(exposure.getImage().getArray())
        donutDf = detector.detectDonuts(
            expArray,
            template,
            blendRadius=self.blendRadius,
            peakThreshold=self.peakThreshold,
            binaryChoice=self.binaryChoice,
        )

        # make a donut catalog :
        # 1) remove the blends in donut catalog
        if self.config.removeBlends:
            donutDfCopy = donutDf[~donutDf["blended"]].copy()
        else:
            donutDfCopy = donutDf

        # 2) update column names and content
        donutCatUpd = self.updateDonutCatalog(donutDfCopy, exposure)

        return pipeBase.Struct(donutCatalog=donutCatUpd)
