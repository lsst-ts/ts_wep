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

        for i in range(nStamps):
            stamp = afwImage.maskedImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
            stamp.setXY0(i + 10, i + 15)
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
            DonutStamp.factory(stampList[i], metadata, i) for i in range(nStamps)
        ]

        donutStamps = DonutStamps(donutStampList, metadata=metadata)

        return donutStamps

    # Adapting some tests here from meas_algorithms/tests/test_stamps.py
    def _roundtrip(self, ss):
        """Round trip a Stamps object to disk and check values"""
        with tempfile.NamedTemporaryFile() as f:
            ss.writeFits(f.name)
            options = PropertyList()
            ss2 = DonutStamps.readFitsWithOptions(f.name, options)
            self.assertEqual(len(ss), len(ss2))
            for s1, s2 in zip(ss, ss2):
                self.assertMaskedImagesAlmostEqual(s1.stamp_im, s2.stamp_im)
                self.assertAlmostEqual(
                    s1.sky_position.getRa().asDegrees(),
                    s2.sky_position.getRa().asDegrees(),
                )
                self.assertAlmostEqual(
                    s1.sky_position.getDec().asDegrees(),
                    s2.sky_position.getDec().asDegrees(),
                )
                self.assertAlmostEqual(
                    s1.centroid_position.getX(), s2.centroid_position.getX()
                )
                self.assertAlmostEqual(
                    s1.centroid_position.getY(), s2.centroid_position.getY()
                )
                self.assertEqual(s1.detector_name, s2.detector_name)

    def testGetSkyPositions(self):

        skyPos = self.donutStamps.getSkyPositions()
        for i in range(self.nStamps):
            self.assertEqual(skyPos[i].getRa().asDegrees(), i)
            self.assertEqual(skyPos[i].getDec().asDegrees(), i + 5)

    def testGetXY0Positions(self):

        xyPos = self.donutStamps.getXY0Positions()
        for i in range(self.nStamps):
            self.assertEqual(xyPos[i].getX(), i + 10)
            self.assertEqual(xyPos[i].getY(), i + 15)

    def testGetCentroidPositions(self):

        xyPos = self.donutStamps.getCentroidPositions()
        for i in range(self.nStamps):
            self.assertEqual(xyPos[i].getX(), i + 20)
            self.assertEqual(xyPos[i].getY(), i + 25)

    def testGetDetectorNames(self):

        detNames = self.donutStamps.getDetectorNames()
        self.assertListEqual(detNames, ["R22_S11"] * self.nStamps)

    def testAppend(self):
        """Test ability to append to a Stamps object"""
        ss = copy(self.donutStamps)
        ss.append(self.donutStamps[0])
        self._roundtrip(ss)
        # check if appending something other than a DonutStamp raises
        with self.assertRaises(ValueError) as context:
            ss.append("hello world")
        self.assertEqual(
            "Objects added must be a DonutStamp object.", str(context.exception)
        )

    def testExtend(self):
        ss = copy(self.donutStamps)
        ss2 = copy(self.donutStamps)
        ss.extend([s for s in ss2])
        self._roundtrip(ss)
        # check if extending with something other than a DonutStamps
        # object raises
        with self.assertRaises(ValueError) as context:
            ss.extend(["hello", "world"])
        self.assertEqual(
            "Can only extend with DonutStamp objects.", str(context.exception)
        )

    def testIOsub(self):
        """
        Test the class' write and readFits when passing on a bounding box.
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(25, 25), lsst.geom.Extent2I(3, 3))
        ss = copy(self.donutStamps)
        with tempfile.NamedTemporaryFile() as f:
            ss.writeFits(f.name)
            options = {"bbox": bbox}
            subStamps = DonutStamps.readFitsWithOptions(f.name, options)
            for s1, s2 in zip(ss, subStamps):
                self.assertEqual(bbox.getDimensions(), s2.stamp_im.getDimensions())
                self.assertMaskedImagesAlmostEqual(s1.stamp_im[bbox], s2.stamp_im)
