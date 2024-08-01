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
from lsst.ts.wep.estimation import DanishAlgorithm
from lsst.ts.wep.utils.modelUtils import forwardModelPair


class TestDanishAlgorithm(unittest.TestCase):
    """Test DanishAlgorithm."""

    def testBadLstsqKwargs(self):
        for kwarg in ["fun", "x0", "jac", "args"]:
            with self.assertRaises(KeyError):
                DanishAlgorithm(lstsqKwargs={kwarg: None})

    def testGoodLstsqKwargs(self):
        # Create estimator
        dan = DanishAlgorithm(lstsqKwargs={"max_nfev": 1})

        # Create some data
        zkTrue, intra, extra = forwardModelPair()

        # Estimate Zernikes
        dan.estimateZk(intra, extra, saveHistory=True)

        # Check that nfev in the algorithm history equals 1
        for key, hist in dan.history.items():
            if key != "zk":
                self.assertEqual(hist["lstsqResult"]["nfev"], 1)

    def testAccuracy(self):
        # Create estimator
        dan = DanishAlgorithm()

        # Try several different random seeds
        for seed in [12345, 23451, 34512, 45123, 51234]:
            # Get the test data
            zkTrue, intra, extra = forwardModelPair(seed=seed)

            # Test estimation with pairs and single donuts:
            for images in [[intra, extra], [intra], [extra]]:
                # Estimate Zernikes (in meters)
                zkEst = dan.estimateZk(*images)

                # Check that results are fairly accurate
                self.assertLess(np.sqrt(np.sum((zkEst - zkTrue) ** 2)), 0.35e-6)
