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
from copy import copy

import lsst.afw.image as afwImage
import lsst.geom
import lsst.utils.tests
import numpy as np
from lsst.daf.base import PropertyList
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import DefocalType


class TestDonutStamps(lsst.utils.tests.TestCase):
    def setUp(self) -> None:
        self.nStamps = 3
        self.stampSize = 100
        self.donutStamps = self._makeDonutStamps(self.nStamps, self.stampSize)

    def _makeDonutStamps(
        self,
        nStamps: int,
        stampSize: int,
    ) -> DonutStamps:
        randState = np.random.RandomState(42)
        stampList = []

        for idx in range(nStamps):
            stamp = afwImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
            stamp.setXY0(idx + 10, idx + 15)
            stampList.append(stamp)

        ras = np.arange(nStamps)
        decs = np.arange(nStamps) + 5
        centX = np.arange(nStamps) + 20
        centY = np.arange(nStamps) + 25
        blendCentX = [f"{val}" for val in np.arange(30, 30 + nStamps)]
        blendCentY = [f"{val}" for val in np.arange(35, 35 + nStamps)]
        detectorNames = ["R22_S11"] * nStamps
        camNames = ["LSSTCam"] * nStamps
        dfcTypes = [DefocalType.Extra.value] * nStamps
        halfStampIdx = int(nStamps / 2)
        dfcTypes[:halfStampIdx] = [DefocalType.Intra.value] * halfStampIdx
        dfcDists = np.ones(nStamps) * 1.5
        bandpass = ["r"] * nStamps

        # Test mixture of donuts with blends and those without
        blendCentX[-1] = "nan"
        blendCentY[-1] = "nan"

        metadata = PropertyList()
        metadata["RA_DEG"] = ras
        metadata["DEC_DEG"] = decs
        metadata["CENT_X"] = centX
        metadata["CENT_Y"] = centY
        metadata["BLEND_CX"] = blendCentX
        metadata["BLEND_CY"] = blendCentY
        metadata["DET_NAME"] = detectorNames
        metadata["CAM_NAME"] = camNames
        metadata["DFC_TYPE"] = dfcTypes
        metadata["DFC_DIST"] = dfcDists
        metadata["BANDPASS"] = bandpass

        donutStampList = [DonutStamp.factory(stampList[idx], metadata, idx) for idx in range(nStamps)]

        return DonutStamps(donutStampList, metadata=metadata)

    # Adapting some tests here from meas_algorithms/tests/test_stamps.py
    def _roundtrip(self, donutStamps: DonutStamps) -> None:
        """Round trip a DonutStamps object to disk and check values"""
        with tempfile.NamedTemporaryFile() as f:
            donutStamps.writeFits(f.name)
            options = PropertyList()
            donutStamps2 = DonutStamps.readFitsWithOptions(f.name, options)
            self.assertEqual(len(donutStamps), len(donutStamps2))
            for stamp1, stamp2 in zip(donutStamps, donutStamps2):
                self.assertMaskedImagesAlmostEqual(stamp1.stamp_im, stamp2.stamp_im)  # type: ignore
                self.assertAlmostEqual(
                    stamp1.sky_position.getRa().asDegrees(),
                    stamp2.sky_position.getRa().asDegrees(),
                )
                self.assertAlmostEqual(
                    stamp1.sky_position.getDec().asDegrees(),
                    stamp2.sky_position.getDec().asDegrees(),
                )
                self.assertAlmostEqual(stamp1.centroid_position.getX(), stamp2.centroid_position.getX())
                self.assertAlmostEqual(stamp1.centroid_position.getY(), stamp2.centroid_position.getY())
                self.assertEqual(stamp1.detector_name, stamp2.detector_name)
                self.assertEqual(stamp1.cam_name, stamp2.cam_name)
                self.assertEqual(stamp1.defocal_type, stamp2.defocal_type)
                self.assertEqual(stamp1.defocal_distance, stamp2.defocal_distance)
                self.assertEqual(stamp1.bandpass, stamp2.bandpass)

    def testGetSkyPositions(self) -> None:
        skyPos = self.donutStamps.getSkyPositions()
        for idx in range(self.nStamps):
            self.assertEqual(skyPos[idx].getRa().asDegrees(), idx)
            self.assertEqual(skyPos[idx].getDec().asDegrees(), idx + 5)

    def testGetXY0Positions(self) -> None:
        xyPos = self.donutStamps.getXY0Positions()
        for idx in range(self.nStamps):
            self.assertEqual(xyPos[idx].getX(), idx + 10)
            self.assertEqual(xyPos[idx].getY(), idx + 15)

    def testGetCentroidPositions(self) -> None:
        xyPos = self.donutStamps.getCentroidPositions()
        for idx in range(self.nStamps):
            self.assertEqual(xyPos[idx].getX(), idx + 20)
            self.assertEqual(xyPos[idx].getY(), idx + 25)

    def testGetBlendCentroids(self) -> None:
        xPos, yPos = self.donutStamps.getBlendCentroids()
        for idx in range(self.nStamps - 1):
            self.assertEqual(xPos[idx], f"{idx + 30:.2f}")
            self.assertEqual(yPos[idx], f"{idx + 35:.2f}")
        self.assertEqual(xPos[-1], "nan")
        self.assertEqual(yPos[-1], "nan")

    def testGetDetectorNames(self) -> None:
        detNames = self.donutStamps.getDetectorNames()
        self.assertListEqual(detNames, ["R22_S11"] * self.nStamps)

    def testGetCameraNames(self) -> None:
        camNames = self.donutStamps.getCameraNames()
        self.assertListEqual(camNames, ["LSSTCam"] * self.nStamps)

    def testGetDefocalTypes(self) -> None:
        defocalTypes = self.donutStamps.getDefocalTypes()
        halfStampIdx = int(self.nStamps / 2)
        self.assertListEqual(
            defocalTypes[:halfStampIdx],
            [DefocalType.Intra.value] * halfStampIdx,
        )
        self.assertListEqual(
            defocalTypes[halfStampIdx:],
            [DefocalType.Extra.value] * int((self.nStamps - halfStampIdx)),
        )

    def testGetDefocalDistances(self) -> None:
        defocalDistances = self.donutStamps.getDefocalDistances()
        for idx in range(self.nStamps):
            self.assertEqual(defocalDistances[idx], 1.5)

    def testGetBandpass(self) -> None:
        bandpasses = self.donutStamps.getBandpasses()
        self.assertListEqual(bandpasses, ["r"] * self.nStamps)

    def testAppend(self) -> None:
        """Test ability to append to a Stamps object"""
        self.donutStamps.append(self.donutStamps[0])
        # Check that the length is 1 longer
        self.assertEqual(len(self.donutStamps), self.nStamps + 1)
        # Test roundtrip i/o
        self._roundtrip(self.donutStamps)
        # check if appending something other than a DonutStamp raises
        with self.assertRaises(ValueError) as context:
            self.donutStamps.append("hello world")
        self.assertEqual("Objects added must be a DonutStamp object.", str(context.exception))

    def testExtend(self) -> None:
        donutStamps2 = copy(self.donutStamps)
        self.donutStamps.extend([stamp for stamp in donutStamps2])
        # Check that the length is twice as long
        self.assertEqual(len(self.donutStamps), self.nStamps * 2)
        # Test roundtrip i/o
        self._roundtrip(self.donutStamps)
        # check if extending with something other than a DonutStamps
        # object raises
        with self.assertRaises(ValueError) as context:
            self.donutStamps.extend(["hello", "world"])
        self.assertEqual("Can only extend with DonutStamp objects.", str(context.exception))

    def testIOsub(self) -> None:
        """
        Test the class' write and readFits when passing on a bounding box.
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(25, 25), lsst.geom.Extent2I(4, 4))
        with tempfile.NamedTemporaryFile() as f:
            self.donutStamps.writeFits(f.name)
            options = {"bbox": bbox}
            subStamps = DonutStamps.readFitsWithOptions(f.name, options)
            for stamp1, stamp2 in zip(self.donutStamps, subStamps):
                self.assertEqual(bbox.getDimensions(), stamp2.stamp_im.getDimensions())
                self.assertMaskedImagesAlmostEqual(stamp1.stamp_im[bbox], stamp2.stamp_im)  # type: ignore
