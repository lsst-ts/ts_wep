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

__all__ = ["DonutQuickMeasurementTaskConfig", "DonutQuickMeasurementTask"]

import numpy as np
from copy import copy
from scipy.signal import correlate

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.meas.base import MeasurementError
from lsst.pipe.tasks.quickFrameMeasurement import (
    QuickFrameMeasurementTaskConfig,
    QuickFrameMeasurementTask,
)


class DonutQuickMeasurementTaskConfig(QuickFrameMeasurementTaskConfig):

    initialCutoutPadding = pexConfig.Field(
        doc=str(
            "Additional padding in pixels on each side of initial "
            + "`donutDiameter` guess for template postage stamp size "
            + "and for bounding boxes used for estimating centroids."
        ),
        dtype=int,
        default=5,
    )


class DonutQuickMeasurementTask(QuickFrameMeasurementTask):

    ConfigClass = DonutQuickMeasurementTaskConfig
    _DefaultName = "donutQuickMeasurementTask"

    def run(self, exp, template, donutDiameter=None, cutoutPadding=None):
        """
        Run method for the task. This task runs a quick detection and
        measurement scheme to detect donuts in images based upon
        QuickFrameMeasurementTask.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Post-ISR image with donut sources.
        template: `numpy.ndarray`
            Donut template binary image.
        donutDiameter : `int` or `None`, optional
            Expected size of donut in pixels. If None
            then it will take this from the input
            configuration class. (The default is None.)
        cutOutPadding : `int` or `None`, optional
            Number of pixels to add in addition to
            `donutDiameter` when creating postage stamp
            for exact centroid measurement. If None it
            will take this from the input configuration
            class. (The default is None.)

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - detectedCatalog : `dict`
                    Dictionary of detected sources with location
                    and flux measurements.
        """
        if donutDiameter is None:
            donutDiameter = self.config.donutDiameter
        if cutoutPadding is None:
            cutoutPadding = self.config.initialCutoutPadding

        self.plateScale = exp.getWcs().getPixelScale().asArcseconds()
        median = np.nanmedian(exp.image.array)
        exp.image -= median  # is put back later
        expImageCopy = copy(exp.image.array)
        exp.image.array = np.array(
            correlate(exp.image.array, template, mode="same"),
            dtype=exp.image.array.dtype,
        )
        self.installPsf.run(exp)
        sources = self.detectObjectsInExp(
            exp,
            nSigma=self.config.nSigmaDetection,
            nPixMin=self.config.nPixMinDetection,
        )

        fpSet = sources.getFootprints()
        self.log.info("Found %d sources in exposure", len(fpSet))

        objData = {}
        nMeasured = 0

        exp.image.array = copy(expImageCopy)
        self.installPsf.run(exp)

        for srcNum, fp in enumerate(fpSet):
            try:
                src = self._measureFp(fp, exp)
                result = self._getDataFromSrcRecord(src)
            except MeasurementError:
                try:
                    # gets shape and centroid from footprint
                    result = self._getDataFromFootprintOnly(fp, exp)
                except MeasurementError as e:
                    self.log.info("Skipped measuring source %s: %s", srcNum, e)
                    continue
            objData[srcNum] = self._measurementResultToDict(result)
            nMeasured += 1

        self.log.info("Measured %d of %d sources in exposure", nMeasured, len(fpSet))

        exp.image += median  # put background back in

        boxSize = (
            donutDiameter + cutoutPadding
        )  # allow some slack, as cutting off side of donut is very bad
        for objNum in range(len(objData)):
            obj = objData[objNum]
            objCentroid = (obj["xCentroid"], obj["yCentroid"])
            centreOfMass = self._getCenterOfMass(exp, objCentroid, boxSize)
            obj["centroid_x"] = centreOfMass[0]
            obj["centroid_y"] = centreOfMass[1]

        return pipeBase.Struct(detectedCatalog=objData)
