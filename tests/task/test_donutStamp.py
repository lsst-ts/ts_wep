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

import lsst.afw.image as afwImage
from lsst.daf.base import PropertyList
from lsst.ts.wep.task.DonutStamp import DonutStamp


class TestDonutStamp(unittest.TestCase):
    def setUp(self):

        self.nStamps = 3
        self.stampSize = 25
        self.testStamps, self.testMetadata = self._makeStamps(
            self.nStamps, self.stampSize
        )

    def _makeStamps(self, nStamps, stampSize):

        randState = np.random.RandomState(42)
        stampList = []

        for i in range(nStamps):
            stamp = afwImage.maskedImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
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

        return stampList, metadata

    def testFactory(self):

        randState = np.random.RandomState(42)
        for i in range(self.nStamps):
            donutStamp = DonutStamp.factory(self.testStamps[i], self.testMetadata, i)
            np.testing.assert_array_almost_equal(
                donutStamp.stamp_im.image.array,
                randState.rand(self.stampSize, self.stampSize),
            )
            np.testing.assert_array_equal(
                donutStamp.stamp_im.mask.array,
                np.ones((self.stampSize, self.stampSize)) * 10,
            )
            np.testing.assert_array_equal(
                donutStamp.stamp_im.variance.array,
                np.ones((self.stampSize, self.stampSize)) * 100,
            )
            self.assertEqual(donutStamp.detector_name, "R22_S11")
            skyPos = donutStamp.sky_position
            self.assertEqual(skyPos.getRa().asDegrees(), i)
            self.assertEqual(skyPos.getDec().asDegrees(), i + 5)
            centroidPos = donutStamp.centroid_position
            self.assertEqual(centroidPos.getX(), i + 20)
            self.assertEqual(centroidPos.getY(), i + 25)
