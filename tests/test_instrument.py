# This file is part of ts-wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import inspect
import unittest
from pathlib import Path

import numpy as np
from batoid.optic import CompoundOptic

from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils import getConfigDir, readConfigYaml


class TestInstrument(unittest.TestCase):
    """Test the Instrument class."""

    def testCreateWithDefaults(self) -> None:
        Instrument()

    def testCreateFromAllPolicyFiles(self) -> None:
        instConfigPath = Path(getConfigDir()) / "instruments"
        paths = instConfigPath.glob("*.yaml")

        for path in paths:
            Instrument(str(path))

    def testBadDiameter(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(diameter=-1)

    def testBadObscuration(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(obscuration=-1)
        with self.assertRaises(ValueError):
            Instrument(obscuration=2)

    def testBadFocalLength(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(focalLength=-1)

    def testBadDefocalOffset(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(defocalOffset="bad")

    def testBadPixelSize(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(pixelSize=-1)

    def testBadWavelength(self) -> None:
        with self.assertRaises(TypeError):
            Instrument(wavelength="bad")
        with self.assertRaises(ValueError):
            Instrument(wavelength={"u": 500e-9})

    def testBadBatoidModelName(self) -> None:
        with self.assertRaises(TypeError):
            Instrument(batoidModelName=-1)

    def testBadRefBand(self) -> None:
        with self.assertRaises(ValueError):
            Instrument(refBand="bad")

    def testNoBatoidModel(self) -> None:
        inst = Instrument()
        inst.batoidModelName = None
        batoidModel = inst.getBatoidModel()
        self.assertIsNone(batoidModel)

    def testGetBatoidModel(self) -> None:
        batoidModel = Instrument().getBatoidModel()
        self.assertIsInstance(batoidModel, CompoundOptic)

    def testBadBatoidOffsetOptic(self) -> None:
        with self.assertRaises(RuntimeError):
            inst = Instrument()
            inst.batoidModelName = None
            inst.batoidOffsetOptic = "Detector"
        with self.assertRaises(TypeError):
            Instrument(batoidOffsetOptic=1)
        with self.assertRaises(ValueError):
            Instrument(batoidOffsetOptic="fake")
        # A bad element inside a list is also rejected
        with self.assertRaises(ValueError):
            Instrument(batoidOffsetOptic=["Detector", "fake"])
        with self.assertRaises(TypeError):
            Instrument(batoidOffsetOptic=["Detector", 1])

    def testBadBatoidOffsetValue(self) -> None:
        with self.assertRaises(RuntimeError):
            inst = Instrument()
            inst.batoidModelName = None
            inst.batoidOffsetValue = 1
        # Mismatched lengths between optics and values are rejected
        with self.assertRaises(ValueError):
            Instrument(
                batoidOffsetOptic=["LSSTCamera", "Detector"],
                batoidOffsetValue=[1.5e-3],
            )
        # One is set but the other is not. Use configFile=None so the missing
        # parameter is not filled in from a default config file.
        with self.assertRaises(ValueError):
            Instrument(
                configFile=None,
                diameter=8.36,
                obscuration=0.612,
                focalLength=10.312,
                pixelSize=10.0e-6,
                batoidModelName="LSST_r",
                batoidOffsetOptic="LSSTCamera",
            )
        # And vice versa
        with self.assertRaises(ValueError):
            Instrument(
                configFile=None,
                diameter=8.36,
                obscuration=0.612,
                focalLength=10.312,
                pixelSize=10.0e-6,
                batoidModelName="LSST_r",
                batoidOffsetValue=1.5e-3,
            )

    def testDefocalOffsetExclusiveWithBatoidOffsets(self) -> None:
        # defocalOffset and the batoid offset parameters are mutually
        # exclusive: setting both at once is an error.
        with self.assertRaises(ValueError):
            Instrument(
                configFile=None,
                diameter=8.36,
                obscuration=0.612,
                focalLength=10.312,
                pixelSize=10.0e-6,
                batoidModelName="LSST_r",
                defocalOffset=1.5e-3,
                batoidOffsetOptic="Detector",
                batoidOffsetValue=1.5e-3,
            )
        # An explicit defocalOffset may still coexist with a batoidModelName
        # (used only for the intrinsic Zernikes) as long as no batoid offsets
        # are set.
        inst = Instrument(
            configFile=None,
            diameter=8.36,
            obscuration=0.612,
            focalLength=10.312,
            pixelSize=10.0e-6,
            batoidModelName="LSST_r",
            defocalOffset=1.5e-3,
        )
        self.assertEqual(inst.defocalOffset, 1.5e-3)

        # The setter also enforces the exclusivity on an existing instance:
        # setting defocalOffset while batoid offsets are set is an error.
        offsetInst = Instrument("policy:instruments/LsstFamCam.yaml")
        with self.assertRaises(ValueError):
            offsetInst.defocalOffset = 2.5e-3
        # Clearing the batoid offsets first makes setting defocalOffset valid.
        offsetInst.batoidOffsetOptic = None
        offsetInst.batoidOffsetValue = None
        offsetInst.defocalOffset = 2.5e-3
        self.assertEqual(offsetInst.defocalOffset, 2.5e-3)

    def testHalfClearedBatoidOffsetNoRecursion(self) -> None:
        # Clearing only batoidOffsetOptic (leaving batoidOffsetValue set)
        # leaves an inconsistent state. Reading defocalOffset should raise a
        # clear error rather than recursing infinitely.
        inst = Instrument("policy:instruments/LsstCam.yaml")
        inst.batoidOffsetOptic = None
        with self.assertRaises(ValueError):
            inst.defocalOffset

    def testScalarBatoidOffsetReturnedAsList(self) -> None:
        # A scalar offset is normalized to a one-element list
        inst = Instrument(batoidOffsetOptic="LSSTCamera", batoidOffsetValue=1.5e-3)
        self.assertEqual(inst.batoidOffsetOptic, ["LSSTCamera"])
        self.assertEqual(inst.batoidOffsetValue, [1.5e-3])

    def testMultiOffset(self) -> None:
        # Full-array-mode wavefront geometry: camera + detector pistons
        famWf = Instrument("policy:instruments/LsstFamCamWavefront.yaml")
        self.assertEqual(famWf.batoidOffsetOptic, ["LSSTCamera", "Detector"])
        self.assertEqual(famWf.batoidOffsetValue, [1.5e-3, 1.5e-3])

        # defocalOffset is derived from the combined shift
        self.assertTrue(np.isfinite(famWf.defocalOffset))
        self.assertGreater(famWf.defocalOffset, 0)

        # Intrinsic Zernikes differ from a single 3 mm Detector offset
        # and from the single LSSTCamera offset of LsstFamCam
        detOnly = Instrument(batoidOffsetOptic="Detector", batoidOffsetValue=3.0e-3)
        famCam = Instrument("policy:instruments/LsstFamCam.yaml")
        zkMulti = famWf.getIntrinsicZernikes(0.3, 0.6, defocalType="extra")
        zkDet = detOnly.getIntrinsicZernikes(0.3, 0.6, defocalType="extra")
        zkFam = famCam.getIntrinsicZernikes(0.3, 0.6, defocalType="extra")
        self.assertFalse(np.allclose(zkMulti, zkDet, rtol=1e-2))
        self.assertFalse(np.allclose(zkMulti, zkFam, rtol=1e-2))

    def testGetIntrinsicZernikes(self) -> None:
        inst = Instrument()

        # First check the shape
        self.assertEqual(inst.getIntrinsicZernikes(0, 0, nollIndices=np.arange(4, 67)).shape, (63,))
        self.assertEqual(inst.getIntrinsicZernikes(1, 1.1, nollIndices=np.arange(4, 23)).shape, (19,))

        # Now check that in-place changes don't impact the cache
        intrZk = inst.getIntrinsicZernikes(1, 1)
        intrZk *= 3.14159
        close = np.isclose(inst.getIntrinsicZernikes(1, 1), intrZk, atol=0)
        self.assertFalse(np.any(close))

    def testGetOffAxisCoeff(self) -> None:
        inst = Instrument()

        # First check the shape
        self.assertEqual(
            inst.getOffAxisCoeff(0, 0, "intra", nollIndicesModel=np.arange(4, 67)).shape,
            (63,),
        )
        self.assertEqual(
            inst.getOffAxisCoeff(1, 1.1, "extra", nollIndicesModel=np.arange(4, 23)).shape,
            (19,),
        )

        # Now check that in-place changes don't impact the cache
        coeff = inst.getOffAxisCoeff(0, 0, "intra")
        coeff *= 3.14159
        close = np.isclose(inst.getOffAxisCoeff(0, 0, "intra"), coeff, atol=0)
        self.assertTrue(np.all(~close))

    def testBadMaskParams(self) -> None:
        with self.assertRaises(TypeError):
            Instrument(maskParams="bad")

    def testDefaultMaskParams(self) -> None:
        inst = Instrument()
        inst.maskParams = None
        self.assertEqual(inst.maskParams, dict())

    def testCreatePupilGrid(self) -> None:
        uImage, vImage = Instrument().createPupilGrid()
        self.assertEqual(uImage.shape, vImage.shape)
        self.assertTrue(np.allclose(uImage, vImage.T))

    def testCreateImageGrid(self) -> None:
        inst = Instrument()

        uImage, vImage = inst.createImageGrid(160)
        self.assertEqual(uImage.shape, vImage.shape)
        self.assertEqual(uImage.shape, (160, 160))

        uImage, vImage = inst.createImageGrid(221)
        self.assertEqual(uImage.shape, (221, 221))

        self.assertTrue(np.allclose(uImage, vImage.T))

    def testRadius(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.radius, 4.18, rtol=1e-3))

    def testArea(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.area, 34.33, rtol=1e-3))

    def testFocalRatio(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.focalRatio, 1.234, rtol=1e-3))

    def testPupilOffset(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.pupilOffset, 10.312**2 / 1.5e-3, rtol=1e-3))

    def testPixelScale(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.pixelScale, 0.2, rtol=1e-3))

    def testDonutRadius(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.donutRadius, 66.512, rtol=1e-3))

    def testDonutDiameter(self) -> None:
        inst = Instrument()
        self.assertTrue(np.isclose(inst.donutDiameter, 2 * 66.512, rtol=1e-3))

    def testPullFromBatoid(self) -> None:
        inst = Instrument(
            configFile=None,
            diameter=None,
            obscuration=None,
            focalLength=None,
            defocalOffset=None,
            pixelSize=10e-6,
            refBand="r",
            wavelength={"r": 622.3e-9},
            batoidModelName="LSST_r",
            batoidOffsetOptic="Detector",
            batoidOffsetValue=1.5e-3,
        )
        lsst = Instrument()

        # Test that the values from Batoid are all correct
        self.assertTrue(np.isclose(inst.diameter, lsst.diameter, rtol=1e-3))
        self.assertTrue(np.isclose(inst.obscuration, lsst.obscuration, rtol=1e-3))
        self.assertTrue(np.isclose(inst.focalLength, lsst.focalLength, rtol=1e-3))
        self.assertTrue(np.isclose(inst.defocalOffset, lsst.defocalOffset, rtol=1e-3))

    def testDefocalOffsetCalculation(self) -> None:
        inst = Instrument("policy:instruments/AuxTel.yaml")
        inst.batoidOffsetValue = 0.8e-3
        self.assertTrue(np.isclose(inst.defocalOffset, 34.94e-3, rtol=1e-3))

    def test_offsetToZ4Defocus(self) -> None:
        inst = Instrument("policy:instruments/LsstCam.yaml")
        self.assertAlmostEqual(inst.offsetToZ4Defocus(0.8e-3), 20.2943, places=4)

    def testImports(self) -> None:
        # Get LSST and ComCam instruments
        lsst = Instrument("policy:instruments/LsstCam.yaml")
        comcam = Instrument("policy:instruments/ComCam.yaml")

        # Get all the init arguments
        keys = list(inspect.signature(Instrument).parameters.keys())

        # Remove configFile
        keys.remove("configFile")

        # Remove keys that were present in the top level ComCam yaml
        # because these override values in LsstCam
        for key in readConfigYaml("policy:instruments/ComCam.yaml"):
            if key in keys:
                keys.remove(key)

        # defocalOffset is derived from batoidOffsetOptic, which ComCam
        # overrides, so it is transitively overridden and will not match.
        keys.remove("defocalOffset")

        # Iterate through the keys and make sure values are the same
        for key in keys:
            self.assertEqual(getattr(lsst, key), getattr(comcam, key))

    def test_intrinsicZernikesDefocused(self) -> None:
        inst = Instrument()
        z4_intra = inst.getIntrinsicZernikes(-0.3, 1.2, defocalType="intra", nollIndices=[4])
        z4_extra = inst.getIntrinsicZernikes(-0.3, 1.2, defocalType="extra", nollIndices=[4])
        self.assertTrue(np.isclose(-z4_extra, z4_intra, rtol=1e-2))

    def test_offsetCamera(self) -> None:
        """Just test shifting camera vs detector gives different intrinsics."""
        inst1 = Instrument()
        inst2 = Instrument(batoidOffsetOptic="LSSTCamera")
        for dftype in ["intra", "extra"]:
            zk_1 = inst1.getIntrinsicZernikes(0.3, 0.6, defocalType=dftype)
            zk_2 = inst2.getIntrinsicZernikes(0.3, 0.6, defocalType=dftype)
            self.assertFalse(np.allclose(zk_1, zk_2, rtol=1e-2))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
