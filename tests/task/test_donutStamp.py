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
from scipy.ndimage import rotate

import lsst.afw.image as afwImage
import lsst.geom
import lsst.obs.lsst as obs_lsst
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.daf.base import PropertyList
from lsst.ts.wep.image import Image
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.utils import BandLabel, DefocalType


class TestDonutStamp(unittest.TestCase):
    def setUp(self) -> None:
        self.nStamps = 3
        self.stampSize = 32
        self.testStamps, self.testMetadata = self._makeStamps(self.nStamps, self.stampSize)
        self.testDefaultStamps, self.testDefaultMetadata = self._makeStamps(
            self.nStamps, self.stampSize, testDefaults=True
        )

    def _makeStamps(self, nStamps: int, stampSize: int, testDefaults: bool = False) -> tuple[list, dict]:
        randState = np.random.RandomState(42)
        stampList = []

        for i in range(nStamps):
            stamp = afwImage.MaskedImageF(stampSize, stampSize)
            stamp.image.array += randState.rand(stampSize, stampSize)
            stamp.mask.array += 10
            stamp.variance.array += 100
            stampList.append(stamp)

        ras = np.arange(nStamps)
        decs = np.arange(nStamps) + 5
        centX = np.arange(nStamps) + 20
        centY = np.arange(nStamps) + 25
        blendCentX = np.array([f"{val}" for val in np.arange(30, 30 + nStamps)])
        blendCentY = np.array([f"{val}" for val in np.arange(35, 35 + nStamps)])
        detectorNames = ["R22_S11"] * nStamps
        camNames = ["LSSTCam"] * nStamps
        dfcTypes = [DefocalType.Extra.value] * nStamps
        halfStampIdx = int(nStamps / 2)
        dfcTypes[:halfStampIdx] = [DefocalType.Intra.value] * halfStampIdx
        dfcDists = np.ones(nStamps) * 1.25
        bandpass = ["r"] * nStamps
        donutId = np.arange(nStamps)

        metadata = PropertyList()
        metadata["RA_DEG"] = ras
        metadata["DEC_DEG"] = decs
        metadata["CENT_X"] = centX
        metadata["CENT_Y"] = centY
        metadata["DET_NAME"] = detectorNames
        metadata["CAM_NAME"] = camNames
        metadata["DFC_TYPE"] = dfcTypes
        metadata["DONUT_ID"] = donutId
        if testDefaults is False:
            metadata["DFC_DIST"] = dfcDists
            metadata["BLEND_CX"] = blendCentX
            metadata["BLEND_CY"] = blendCentY
            metadata["BANDPASS"] = bandpass

        return stampList, metadata

    def testFactory(self) -> None:
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
            blendCentroidPos = donutStamp.blend_centroid_positions
            np.testing.assert_array_equal(blendCentroidPos, np.array([[i + 30, i + 35]]))
            camName = donutStamp.cam_name
            self.assertEqual("LSSTCam", camName)
            defocalType = donutStamp.defocal_type
            if i < int(self.nStamps / 2):
                self.assertEqual(defocalType, DefocalType.Intra.value)
            else:
                self.assertEqual(defocalType, DefocalType.Extra.value)
            defocalDist = donutStamp.defocal_distance
            self.assertEqual(defocalDist, 1.25)
            bandpass = donutStamp.bandpass
            self.assertEqual(bandpass, "r")

            donutId = donutStamp.donut_id
            self.assertEqual(i, donutId)

            self.assertIsInstance(donutStamp.wep_im, Image)

    def testFactoryMetadataDefaults(self) -> None:
        """Some metadata values have been added since the original
        version of DonutStamp was created. When this occurs
        we need to set a default value to fill in the metadata
        to allow the butler to read old repositories.
        Here we test those values.
        """

        for i in range(self.nStamps):
            donutStamp = DonutStamp.factory(self.testDefaultStamps[i], self.testDefaultMetadata, i)
            defocalDist = donutStamp.defocal_distance
            # Test default metadata distance of 1.5 mm
            self.assertEqual(defocalDist, 1.5)
            # Test blend centroids arrays are nans
            np.testing.assert_array_equal(
                donutStamp.blend_centroid_positions,
                np.array([["nan"], ["nan"]], dtype=float).T,
            )
            # Test default bandpass value is empty string
            bandpass = donutStamp.bandpass
            self.assertEqual(bandpass, "")
            # Test default donutId value is empty string
            donutId = donutStamp.donut_id
            self.assertEqual(donutId, i)

    def testGetCamera(self) -> None:
        donutStamp = DonutStamp.factory(self.testStamps[0], self.testMetadata, 0)

        donutStamp.cam_name = "LSSTCam"
        self.assertEqual(
            donutStamp.getCamera(),
            obs_lsst.LsstCam().getCamera(),
        )
        donutStamp.cam_name = "LSSTComCam"
        self.assertEqual(
            donutStamp.getCamera(),
            obs_lsst.LsstComCam().getCamera(),
        )
        donutStamp.cam_name = "LATISS"
        self.assertEqual(
            donutStamp.getCamera(),
            obs_lsst.Latiss.getCamera(),
        )
        donutStamp.cam_name = "noCam"
        errMessage = "Camera noCam is not supported."
        with self.assertRaises(ValueError, msg=errMessage):
            donutStamp.getCamera()

    def testGetLinearWCS(self) -> None:
        wcs = lsst.afw.geom.makeSkyWcs(
            lsst.geom.Point2D(0.0, 0.0),
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            np.eye(2),
        )
        donutStamp = DonutStamp.factory(self.testStamps[0], self.testMetadata, 0, wcs)
        self.assertEqual(wcs, donutStamp.getLinearWCS())

    def testCalcFieldXY(self) -> None:
        donutStamp = DonutStamp(
            self.testStamps[0],
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Point2D(2047.5, 2001.5),
            np.array([[], []]).T,
            DefocalType.Extra.value,
            1.5e-3,
            "R22_S11",
            "LSSTCam",
            "r",
            "",
        )
        np.testing.assert_array_almost_equal(donutStamp.calcFieldXY(), (0, 0))

        # Test with locations of corner sensors.
        camera = obs_lsst.LsstCam().getCamera()
        for raftName in ["R00", "R04", "R40", "R44"]:
            for sensorName in ["SW0", "SW1"]:
                detName = f"{raftName}_{sensorName}"
                detector = camera.get(detName)
                detOrientation = detector.getOrientation()
                refPt = detOrientation.getReferencePoint()
                trueFieldPos = detOrientation.getFpPosition()
                # Convert to field angle
                trueFieldAngleX, trueFieldAngleY = detector.transform(trueFieldPos, FOCAL_PLANE, FIELD_ANGLE)
                donutStamp = DonutStamp(
                    self.testStamps[0],
                    lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
                    refPt,
                    np.array([[20], [20]]).T,
                    DefocalType.Extra.value,
                    0.0,
                    detName,
                    "LSSTCam",
                    "r",
                    "",
                )
                fieldAngle = donutStamp.calcFieldXY()
                self.assertEqual(fieldAngle[0], np.degrees(trueFieldAngleX))
                self.assertEqual(fieldAngle[1], np.degrees(trueFieldAngleY))

    def testMakeMask(self) -> None:
        donutStamp = DonutStamp(
            afwImage.MaskedImageF(160, 160),
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Point2D(2047.5, 2001.5),
            np.array([[], []]).T,
            DefocalType.Extra.value,
            1.5e-3,
            "R22_S11",
            "LSSTCam",
            "r",
            "",
        )

        # Check that mask is empty at start
        mask = donutStamp.stamp_im.mask
        self.assertEqual(mask.array.shape, (160, 160))
        self.assertEqual(mask.array.sum(), 0)

        # Check masks after creation
        donutStamp.makeMask(Instrument())
        mask = donutStamp.stamp_im.mask

        maskKeys = mask.getMaskPlaneDict().keys()
        self.assertTrue({"DONUT", "BLEND"} <= maskKeys)

        # Make sure not just an empty array
        self.assertGreater(mask.array.sum(), 0)

        # Donut at center of focal plane should be symmetric
        np.testing.assert_array_equal(mask.array[:159], mask.array[-159:][::-1])

    def testWepImage(self) -> None:
        # Goal: test the transformations that occur during WepImage creation
        rng = np.random.default_rng(0)
        camera = obs_lsst.LsstCam().getCamera()

        # Define a function that will test everything for the WEP Image
        def _testWepImage(
            raft: str,
            sensor: str,
        ) -> None:
            # Get the detector
            detName = f"{raft}_{sensor}"
            detector = camera.get(detName)

            # Create a random stamp image
            stamp = afwImage.MaskedImageF(160, 160)
            stamp.image.array += rng.random(size=stamp.image.array.shape)

            # Determine the center of the stamp
            center = detector.getCenter(PIXELS)

            # Determine the blend offsets
            offsets = np.arange(6).reshape(-1, 2)

            # Create the stamp
            donutStamp = DonutStamp(
                stamp,
                lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
                center,
                center + offsets,
                DefocalType.Extra.value,
                1.5e-3,
                detName,
                "LSSTCam",
                "r",
                "",
            )

            # Make the mask
            donutStamp.makeMask(Instrument())

            # Get the WEP Image
            wepImage = donutStamp.wep_im

            # Check that the rotation was applied to the image correctly
            # (Here we transform a WEP image into the coordinate system
            # in which data comes out of the butler, so this is the
            # inverse of the forwawrd transformation applied to the image
            # in _setWepImage())
            eulerZ = detector.getOrientation().getYaw().asDegrees()
            sameImage = np.allclose(
                rotate(wepImage.image.T, eulerZ),
                donutStamp.stamp_im.image.array,
            )
            self.assertTrue(sameImage)

            # Check that the field angle is correct
            center = np.rad2deg(detector.getCenter(FIELD_ANGLE))
            self.assertTrue(np.allclose(wepImage.fieldAngle, center[::-1]))

            # Check the blend offsets are correct
            theta = np.deg2rad(eulerZ)
            rotMat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )
            offsets = np.array([rotMat @ v for v in offsets])[:, ::-1]
            self.assertTrue(np.allclose(wepImage.blendOffsets, offsets))

            # Check the DefocalType and BandLabel
            self.assertEqual(wepImage.defocalType, DefocalType.Extra)
            self.assertEqual(wepImage.bandLabel, BandLabel.LSST_R)

        # Test all the CWFSs
        for raftName in ["R00", "R04", "R40", "R44"]:
            for sensorName in ["SW0", "SW1"]:
                _testWepImage(raftName, sensorName)

        # Test some science sensors
        _testWepImage("R22", "S22")
        _testWepImage("R21", "S11")
