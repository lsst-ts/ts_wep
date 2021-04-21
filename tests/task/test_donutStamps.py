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

import tempfile
import numpy as np
from copy import copy

import lsst.geom
import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.daf.base import PropertyList
from lsst.ts.wep.task.DonutStamp import DonutStamp
from lsst.ts.wep.task.DonutStamps import DonutStamps


class TestDonutStamps(lsst.utils.tests.TestCase):
    def setUp(self):

        self.nStamps = 3
        self.stampSize = 100
        self.donutStamps = self._makeDonutStamps(self.nStamps, self.stampSize)

    def _makeDonutStamps(self, nStamps, stampSize):

        randState = np.random.RandomState(42)
        stampList = []

        for idx in range(nStamps):
            stamp = afwImage.maskedImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
            stamp.setXY0(idx + 10, idx + 15)
            stampList.append(stamp)

        ras = np.arange(nStamps)
        decs = np.arange(nStamps) + 5
        centX = np.arange(nStamps) + 20
        centY = np.arange(nStamps) + 25
        detectorNames = ["R22_S11"] * nStamps

        metadata = PropertyList()
        metadata["RA_DEG"] = ras
        metadata["DEC_DEG"] = decs
        metadata["CENT_X"] = centX
        metadata["CENT_Y"] = centY
        metadata["DET_NAME"] = detectorNames

        donutStampList = [
            DonutStamp.factory(stampList[idx], metadata, idx) for idx in range(nStamps)
        ]

        return DonutStamps(donutStampList, metadata=metadata)

    # Adapting some tests here from meas_algorithms/tests/test_stamps.py
    def _roundtrip(self, donutStamps):
        """Round trip a DonutStamps object to disk and check values"""
        with tempfile.NamedTemporaryFile() as f:
            donutStamps.writeFits(f.name)
            options = PropertyList()
            donutStamps2 = DonutStamps.readFitsWithOptions(f.name, options)
            self.assertEqual(len(donutStamps), len(donutStamps2))
            for stamp1, stamp2 in zip(donutStamps, donutStamps2):
                self.assertMaskedImagesAlmostEqual(stamp1.stamp_im, stamp2.stamp_im)
                self.assertAlmostEqual(
                    stamp1.sky_position.getRa().asDegrees(),
                    stamp2.sky_position.getRa().asDegrees(),
                )
                self.assertAlmostEqual(
                    stamp1.sky_position.getDec().asDegrees(),
                    stamp2.sky_position.getDec().asDegrees(),
                )
                self.assertAlmostEqual(
                    stamp1.centroid_position.getX(), stamp2.centroid_position.getX()
                )
                self.assertAlmostEqual(
                    stamp1.centroid_position.getY(), stamp2.centroid_position.getY()
                )
                self.assertEqual(stamp1.detector_name, stamp2.detector_name)

    def testGetSkyPositions(self):

        skyPos = self.donutStamps.getSkyPositions()
        for idx in range(self.nStamps):
            self.assertEqual(skyPos[idx].getRa().asDegrees(), idx)
            self.assertEqual(skyPos[idx].getDec().asDegrees(), idx + 5)

    def testGetXY0Positions(self):

        xyPos = self.donutStamps.getXY0Positions()
        for idx in range(self.nStamps):
            self.assertEqual(xyPos[idx].getX(), idx + 10)
            self.assertEqual(xyPos[idx].getY(), idx + 15)

    def testGetCentroidPositions(self):

        xyPos = self.donutStamps.getCentroidPositions()
        for idx in range(self.nStamps):
            self.assertEqual(xyPos[idx].getX(), idx + 20)
            self.assertEqual(xyPos[idx].getY(), idx + 25)

    def testGetDetectorNames(self):

        detNames = self.donutStamps.getDetectorNames()
        self.assertListEqual(detNames, ["R22_S11"] * self.nStamps)

    def testAppend(self):
        """Test ability to append to a Stamps object"""
        self.donutStamps.append(self.donutStamps[0])
        self._roundtrip(self.donutStamps)
        # check if appending something other than a DonutStamp raises
        with self.assertRaises(ValueError) as context:
            self.donutStamps.append("hello world")
        self.assertEqual(
            "Objects added must be a DonutStamp object.", str(context.exception)
        )

    def testExtend(self):
        donutStamps2 = copy(self.donutStamps)
        self.donutStamps.extend([stamp for stamp in donutStamps2])
        self._roundtrip(self.donutStamps)
        # check if extending with something other than a DonutStamps
        # object raises
        with self.assertRaises(ValueError) as context:
            self.donutStamps.extend(["hello", "world"])
        self.assertEqual(
            "Can only extend with DonutStamp objects.", str(context.exception)
        )

    def testIOsub(self):
        """
        Test the class' write and readFits when passing on a bounding box.
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(25, 25), lsst.geom.Extent2I(3, 3))
        with tempfile.NamedTemporaryFile() as f:
            self.donutStamps.writeFits(f.name)
            options = {"bbox": bbox}
            subStamps = DonutStamps.readFitsWithOptions(f.name, options)
            for stamp1, stamp2 in zip(self.donutStamps, subStamps):
                self.assertEqual(bbox.getDimensions(), stamp2.stamp_im.getDimensions())
                self.assertMaskedImagesAlmostEqual(
                    stamp1.stamp_im[bbox], stamp2.stamp_im
                )