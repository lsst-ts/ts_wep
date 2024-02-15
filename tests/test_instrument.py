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
from pathlib import Path

import numpy as np
from batoid.optic import CompoundOptic
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils import getConfigDir


class TestInstrument(unittest.TestCase):
    """Test the Instrument class."""

    def testCreateWithDefaults(self):
        Instrument()

    def testCreateFromAllPolicyFiles(self):
        instConfigPath = Path(getConfigDir()) / "instruments"
        paths = instConfigPath.glob("*.yaml")

        for path in paths:
            Instrument(str(path))

    def testBadDiameter(self):
        with self.assertRaises(ValueError):
            Instrument(diameter=-1)

    def testBadObscuration(self):
        with self.assertRaises(ValueError):
            Instrument(obscuration=-1)
        with self.assertRaises(ValueError):
            Instrument(obscuration=2)

    def testBadFocalLength(self):
        with self.assertRaises(ValueError):
            Instrument(focalLength=-1)

    def testBadDefocalOffset(self):
        with self.assertRaises(ValueError):
            Instrument(defocalOffset="bad")

    def testBadPixelSize(self):
        with self.assertRaises(ValueError):
            Instrument(pixelSize=-1)

    def testBadWavelength(self):
        with self.assertRaises(TypeError):
            Instrument(wavelength="bad")
        with self.assertRaises(ValueError):
            Instrument(wavelength={"u": 500e-9})

    def testBadBatoidModelName(self):
        with self.assertRaises(TypeError):
            Instrument(batoidModelName=-1)

    def testBadRefBand(self):
        with self.assertRaises(ValueError):
            Instrument(refBand="bad")

    def testNoBatoidModel(self):
        inst = Instrument()
        inst.batoidModelName = None
        batoidModel = inst.getBatoidModel()
        self.assertIsNone(batoidModel)

    def testGetBatoidModel(self):
        batoidModel = Instrument().getBatoidModel()
        self.assertIsInstance(batoidModel, CompoundOptic)

    def testBadBatoidOffsetOptic(self):
        with self.assertRaises(RuntimeError):
            inst = Instrument()
            inst.batoidModelName = None
            inst.batoidOffsetOptic = "Detector"
        with self.assertRaises(TypeError):
            Instrument(batoidOffsetOptic=1)
        with self.assertRaises(ValueError):
            Instrument(batoidOffsetOptic="fake")

    def testBadBatoidOffsetValue(self):
        with self.assertRaises(RuntimeError):
            inst = Instrument()
            inst.batoidModelName = None
            inst.batoidOffsetValue = 1

    def testGetIntrinsicZernikes(self):
        inst = Instrument()

        # First check the shape
        self.assertEqual(inst.getIntrinsicZernikes(0, 0, jmax=66).shape, (63,))
        self.assertEqual(inst.getIntrinsicZernikes(1, 2, jmax=22).shape, (19,))

        # Now check that in-place changes don't impact the cache
        intrZk = inst.getIntrinsicZernikes(1, 1)
        intrZk *= 3.14159
        close = np.isclose(inst.getIntrinsicZernikes(1, 1), intrZk, atol=0)
        self.assertFalse(np.any(close))

    def testGetOffAxisCoeff(self):
        inst = Instrument()

        # First check the shape
        self.assertEqual(inst.getOffAxisCoeff(0, 0, "intra", jmax=66).shape, (63,))
        self.assertEqual(inst.getOffAxisCoeff(1, 2, "extra", jmax=22).shape, (19,))

        # Now check that in-place changes don't impact the cache
        intrZk = inst.getOffAxisCoeff(0, 0, "intra")
        intrZk *= 3.14159
        close = np.isclose(inst.getOffAxisCoeff(0, 0, "intra"), intrZk, atol=0)
        self.assertTrue(np.all(~close))

    def testBadMaskParams(self):
        with self.assertRaises(TypeError):
            Instrument(maskParams="bad")

    def testDefaultMaskParams(self):
        inst = Instrument()
        inst.maskParams = None
        self.assertEqual(inst.maskParams, dict())

    def testCreatePupilGrid(self):
        uImage, vImage = Instrument().createPupilGrid()
        self.assertEqual(uImage.shape, vImage.shape)
        self.assertTrue(np.allclose(uImage, vImage.T))

    def testCreateImageGrid(self):
        inst = Instrument()

        uImage, vImage = inst.createImageGrid(160)
        self.assertEqual(uImage.shape, vImage.shape)
        self.assertEqual(uImage.shape, (160, 160))

        uImage, vImage = inst.createImageGrid(221)
        self.assertEqual(uImage.shape, (221, 221))

        self.assertTrue(np.allclose(uImage, vImage.T))

    def testRadius(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.radius, 4.18, rtol=1e-3))

    def testArea(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.area, 34.33, rtol=1e-3))

    def testFocalRatio(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.focalRatio, 1.234, rtol=1e-3))

    def testPupilOffset(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.pupilOffset, 10.312**2 / 1.5e-3, rtol=1e-3))

    def testPixelScale(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.pixelScale, 0.2, rtol=1e-3))

    def testDonutRadius(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.donutRadius, 66.512, rtol=1e-3))

    def testDonutDiameter(self):
        inst = Instrument()
        self.assertTrue(np.isclose(inst.donutDiameter, 2 * 66.512, rtol=1e-3))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
