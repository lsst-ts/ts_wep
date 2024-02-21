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
from lsst.ts.wep import Image
from lsst.ts.wep.estimation import WfEstimator
from lsst.ts.wep.utils import WfAlgorithmName, convertZernikesToPsfWidth


class TestWfEstimator(unittest.TestCase):
    """Test the wavefront estimator class."""

    def testCreateWithDefaults(self):
        WfEstimator()

    def testBadAlgoName(self):
        with self.assertRaises(ValueError):
            WfEstimator(algoName="fake")

    def testBadAlgoConfig(self):
        with self.assertRaises(TypeError):
            WfEstimator(algoConfig=1)

    def testBadInstConfig(self):
        with self.assertRaises(TypeError):
            WfEstimator(instConfig=1)
        with self.assertRaises(FileNotFoundError):
            WfEstimator(instConfig="fake")

    def testBadJmax(self):
        with self.assertRaises(ValueError):
            WfEstimator(jmax=2)

    def testBadStartWithIntrinsic(self):
        with self.assertRaises(TypeError):
            WfEstimator(startWithIntrinsic="fake")

    def testBadReturnWfDev(self):
        with self.assertRaises(TypeError):
            WfEstimator(returnWfDev="fake")

    def testBadReturn4Up(self):
        with self.assertRaises(TypeError):
            WfEstimator(return4Up="fake")

    def testBadUnits(self):
        with self.assertRaises(ValueError):
            WfEstimator(units="parsecs")

    def testBadSaveHistory(self):
        with self.assertRaises(TypeError):
            WfEstimator(saveHistory="fake")

    def testDifferentJmax(self):
        # Create some dummy images
        intra = Image(
            np.zeros((180, 180)),
            (1.2, 0.3),
            "intra",
        )

        extra = Image(
            np.zeros((180, 180)),
            (0.9, -1),
            "extra",
        )

        # Test every wavefront algorithm
        for name in WfAlgorithmName:
            # Test two different values of jmax
            for jmax in [22, 28]:
                wfEst = WfEstimator(algoName=name, jmax=jmax)
                zk = wfEst.estimateZk(intra, extra)
                self.assertEqual(len(zk), len(np.arange(4, jmax + 1)))

    def testDifferentUnits(self):
        # Create some dummy images
        intra = Image(
            np.zeros((180, 180)),
            (1.2, 0.3),
            "intra",
        )

        extra = Image(
            np.zeros((180, 180)),
            (0.9, -1),
            "extra",
        )

        # Test every wavefront algorithm
        for name in WfAlgorithmName:
            zk = dict()
            # Test every available unit
            for units in ["m", "um", "nm", "arcsec"]:
                wfEst = WfEstimator(algoName=name, units=units)
                zk[units] = wfEst.estimateZk(intra, extra)

            self.assertTrue(np.allclose(zk["m"], zk["um"] / 1e6))
            self.assertTrue(np.allclose(zk["m"], zk["nm"] / 1e9))
            self.assertTrue(
                np.allclose(
                    convertZernikesToPsfWidth(zk["um"]),
                    zk["arcsec"],
                )
            )


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
