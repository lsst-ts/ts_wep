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
from astropy.table import Table
from lsst.ts.wep import Instrument
from lsst.ts.wep.utils.modelUtils import forwardModelPair


class TestModelUtils(unittest.TestCase):
    """Test the model utility functions."""

    def testDefaultsRun(self) -> None:
        forwardModelPair()

    def testSuppliedZernikes(self) -> None:
        # Simulate then reuse Zernikes
        zk1, intra1, extra1 = forwardModelPair()
        zk2, intra2, extra2 = forwardModelPair(zkCoeff=zk1)

        # Make sure outputs are the same
        self.assertTrue(np.allclose(zk1, zk2))
        self.assertTrue(np.allclose(intra1.image, intra2.image))
        self.assertTrue(np.allclose(extra1.image, extra2.image))

    def testRandomSeed(self) -> None:
        # 3 models with 2 different seeds
        zk1, intra1, extra1 = forwardModelPair(seed=42)
        zk2, intra2, extra2 = forwardModelPair(seed=42)
        zk3, intra3, extra3 = forwardModelPair(seed=99)

        # Compare Zernikes
        self.assertTrue(np.allclose(zk1, zk2))
        self.assertFalse(np.allclose(zk1, zk3))

        # Compare Images
        self.assertTrue(np.allclose(intra1.image, intra2.image))
        self.assertTrue(np.allclose(extra1.image, extra2.image))
        self.assertFalse(np.allclose(intra1.image, intra3.image))
        self.assertFalse(np.allclose(extra1.image, extra3.image))

        # Compare angles
        self.assertTrue(np.allclose(intra1.fieldAngle, intra2.fieldAngle))
        self.assertTrue(np.allclose(extra1.fieldAngle, extra2.fieldAngle))
        self.assertFalse(np.allclose(intra1.fieldAngle, intra3.fieldAngle))
        self.assertFalse(np.allclose(extra1.fieldAngle, extra3.fieldAngle))

    def testAngles(self) -> None:
        # Specify single angles
        _, intra1, extra1 = forwardModelPair(fieldAngleIntra=(0.12, 0.34))
        _, intra2, extra2 = forwardModelPair(fieldAngleExtra=(0.12, 0.34))

        # Compare angles
        self.assertTrue(np.allclose(intra1.fieldAngle, intra2.fieldAngle))
        self.assertTrue(np.allclose(extra1.fieldAngle, extra2.fieldAngle))
        self.assertTrue(np.allclose(intra1.fieldAngle, extra2.fieldAngle))
        self.assertTrue(np.allclose(extra1.fieldAngle, intra2.fieldAngle))

    def testZernikes(self) -> None:
        # Zero Zernikes
        zk1, _, _ = forwardModelPair(zkNorm=0)
        zk2, _, _ = forwardModelPair(zkMax=0)
        self.assertTrue(np.allclose(0, zk1))
        self.assertTrue(np.allclose(0, zk2))

        # Divergent Zernikes
        _, intra, extra = forwardModelPair(zkNorm=1e9, zkMax=1e9)
        self.assertFalse(np.isfinite(intra.image).any())
        self.assertFalse(np.isfinite(extra.image).any())

    def testBlends(self) -> None:
        # Just going to test that blends get added by looking at flux
        _, intra1, extra1 = forwardModelPair()
        _, intra2, extra2 = forwardModelPair(
            blendOffsetsIntra=((40, 30),),
            blendOffsetsExtra=((40, 30),),
        )

        self.assertTrue(intra1.image.sum() < intra2.image.sum())
        self.assertTrue(extra1.image.sum() < extra2.image.sum())

        # Ensure that flux ratios work
        _, intra3, extra3 = forwardModelPair(
            blendOffsetsIntra=((40, 30),),
            blendOffsetsExtra=((40, 30),),
            blendRatiosIntra=[0.1],
            blendRatiosExtra=[2.0],
        )
        self.assertTrue(intra1.image.sum() < intra3.image.sum())
        self.assertTrue(intra3.image.sum() < intra2.image.sum())
        self.assertTrue(extra2.image.sum() < extra3.image.sum())

        # Make sure that blendOffsets is robust to 1D specification
        _, intra1, extra1 = forwardModelPair(blendOffsetsIntra=(40, 30))
        _, intra2, extra2 = forwardModelPair(blendOffsetsIntra=((40, 30),))
        self.assertTrue(np.allclose(intra1.image, intra2.image))

    def testSeeing(self) -> None:
        # No atmosphere should cover fewer pixels
        _, intra1, extra1 = forwardModelPair(seeing=0, skyLevel=0)
        _, intra2, extra2 = forwardModelPair(seeing=1, skyLevel=0)
        self.assertTrue(np.sum(intra1.image > 0) < np.sum(intra2.image > 0))
        self.assertTrue(np.sum(extra1.image > 0) < np.sum(extra2.image > 0))

    def testMiscenter(self) -> None:
        # Move extrafocal donut all the way out of frame
        _, intra, extra = forwardModelPair(skyLevel=0, miscenterExtra=(500, 0))
        self.assertTrue(intra.image.sum() > 0)
        self.assertTrue(np.isclose(extra.image.sum(), 0))

    def testHeightMap(self) -> None:
        # Simulate a pair of stamps at the center with no height offsets
        inst = Instrument()
        inst.heightMap = None
        _, intra, extra = forwardModelPair(
            instConfig=inst,
            skyLevel=0,
            seeing=0,
            zkCoeff=[0],
        )

        # Check that extrafocal isn't too much bigger than intrafocal
        self.assertTrue(np.sum(extra.image > 0) - np.sum(intra.image > 0) < 1000)

        # Now simulate a pair with a constant height offset of +0.5mm
        x = np.linspace(-320, +320, 1000)
        X, Y = np.meshgrid(x, x)
        Z = np.full_like(X, 0.5)
        heightTable = Table(
            rows=np.column_stack((X.ravel(), Y.ravel(), Z.ravel())),
            names=["x", "y", "z"],
            units=["mm", "mm", "mm"],
        )
        inst.heightMap = heightTable
        _, intra, extra = forwardModelPair(
            instConfig=inst,
            skyLevel=0,
            seeing=0,
            zkCoeff=[0],
        )

        # Check that extrafocal is much larger than intrafocal
        self.assertTrue(np.sum(extra.image > 0) - np.sum(intra.image > 0) > 1000)
