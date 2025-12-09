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

import unittest

import numpy as np
import pandas as pd

from lsst.obs.lsst import LsstCam
from lsst.ts.wep.donutDetector import DonutDetector
from lsst.ts.wep.utils import createTemplateForDetector


class TestDonutDetector(unittest.TestCase):
    """Test the DonutDetector class."""

    def setUp(self) -> None:
        self.donutDetector = DonutDetector()

    def _makeData(self, imgSize: int, donutSep: int) -> tuple[np.ndarray, np.ndarray]:
        # Create the template
        camera = LsstCam().getCamera()
        detector = camera.get("R22_S11")
        template = createTemplateForDetector(detector, "extra")

        templateHalfWidth = int(len(template) / 2)

        blendedImg = np.zeros((imgSize, imgSize))
        center = int(imgSize / 2)
        leftCenter = int(center - donutSep / 2)
        rightCenter = int(center + donutSep / 2)

        # Place two template images to left and right along center line
        # separated by donutSep
        blendedImg[
            leftCenter - templateHalfWidth : leftCenter + templateHalfWidth,
            center - templateHalfWidth : center + templateHalfWidth,
        ] += template
        blendedImg[
            rightCenter - templateHalfWidth : rightCenter + templateHalfWidth,
            center - templateHalfWidth : center + templateHalfWidth,
        ] += template
        # Make binary image again after overlapping areas sum
        blendedImg[blendedImg > 1] = 1

        return template, blendedImg

    def testIdentifyBlendedDonuts(self) -> None:
        testDataFrame = pd.DataFrame()
        testDataFrame["x_center"] = [50.0, 100.0, 120.0]
        testDataFrame["y_center"] = [100.0, 100.0, 100.0]

        labeledDf = self.donutDetector.identifyBlendedDonuts(testDataFrame, 30.0)

        self.assertCountEqual(
            labeledDf.columns,
            [
                "x_center",
                "y_center",
                "blended",
                "blended_with",
                "num_blended_neighbors",
                "x_blend_center",
                "y_blend_center",
            ],
        )
        np.testing.assert_array_equal(labeledDf["blended"], [False, True, True])
        self.assertListEqual(labeledDf["blended_with"].values.tolist(), [None, [2], [1]])
        np.testing.assert_array_equal(labeledDf["num_blended_neighbors"], [0, 1, 1])
        self.assertListEqual(labeledDf["x_blend_center"].values.tolist(), [[], [120.0], [100.0]])
        self.assertListEqual(labeledDf["y_blend_center"].values.tolist(), [[], [100.0], [100.0]])

    def testDetectDonuts(self) -> None:
        template, testImg = self._makeData(480, 60)
        donutDf = self.donutDetector.detectDonuts(testImg, template, 126)

        self.assertCountEqual(
            donutDf.columns,
            [
                "x_center",
                "y_center",
                "blended",
                "blended_with",
                "num_blended_neighbors",
                "x_blend_center",
                "y_blend_center",
            ],
        )
        self.assertCountEqual(donutDf["x_center"], [270, 210])
        self.assertCountEqual(donutDf["y_center"], [240, 240])
        self.assertCountEqual(donutDf["blended"], [True, True])
        np.testing.assert_array_equal(list(donutDf["blended_with"]), [[1], [0]])
        self.assertCountEqual(donutDf["num_blended_neighbors"], [1, 1])
        np.testing.assert_array_equal(
            list(donutDf["x_blend_center"]),
            [[donutDf.iloc[1]["x_center"]], [donutDf.iloc[0]["x_center"]]],
        )
        np.testing.assert_array_equal(list(donutDf["y_blend_center"]), [[240.0], [240.0]])
