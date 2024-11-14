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
from lsst.ts.wep.estimation import WfEstimator
from lsst.ts.wep.utils import WfAlgorithmName, convertZernikesToPsfWidth
from lsst.ts.wep.utils.modelUtils import forwardModelPair


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

    def testBadNollIndices(self):
        with self.assertRaises(ValueError):
            WfEstimator(nollIndices=[3, 4, 5])
        with self.assertRaises(ValueError):
            WfEstimator(nollIndices=[4, 6, 5])
        with self.assertRaises(ValueError):
            WfEstimator(nollIndices=[4, 5, 5])
        with self.assertRaises(ValueError):
            WfEstimator(nollIndices=[4, 5])

    def testBadStartWithIntrinsic(self):
        with self.assertRaises(TypeError):
            WfEstimator(startWithIntrinsic="fake")

    def testBadReturnWfDev(self):
        with self.assertRaises(TypeError):
            WfEstimator(returnWfDev="fake")

    def testBadUnits(self):
        with self.assertRaises(ValueError):
            WfEstimator(units="parsecs")

    def testBadSaveHistory(self):
        with self.assertRaises(TypeError):
            WfEstimator(saveHistory="fake")

    def testDifferentNollIndices(self):
        # Get the test data
        zkTrue, intra, extra = forwardModelPair()

        # Test every wavefront algorithm
        for name in WfAlgorithmName:
            # Estimate [4, 5, 6]
            wfEst = WfEstimator(algoName=name, nollIndices=[4, 5, 6], units="m")
            zk0 = wfEst.estimateZk(intra, extra)
            self.assertEqual(len(zk0), 3)

            # Estimate with [4, 5, 6, 20, 21]
            wfEst = WfEstimator(algoName=name, nollIndices=[4, 5, 6, 20, 21], units="m")
            zk1 = wfEst.estimateZk(intra, extra)
            self.assertEqual(len(zk1), 5)

            #  Make sure results are pretty similar for [4, 5, 6]
            self.assertLess(np.sqrt(np.sum(np.square(zk1[:-2] - zk0))), 10e-9)

    def testStartWithIntrinsic(self):
        # Get the test data
        zkTrue, intra, extra = forwardModelPair()

        # Test every wavefront algorithm
        for name in WfAlgorithmName:
            # Estimate starting with intrinsics
            wfEst = WfEstimator(algoName=name, startWithIntrinsic=True, units="m")
            zk0 = wfEst.estimateZk(intra, extra)

            # Estimate starting with zeros
            wfEst = WfEstimator(algoName=name, startWithIntrinsic=False, units="m")
            zk1 = wfEst.estimateZk(intra, extra)

            # Make sure the results are pretty similar
            self.assertLess(np.sqrt(np.sum(np.square(zk1 - zk0))), 80e-9)

    def testReturnWfDev(self):
        # Get the test data
        zkTrue, intra, extra = forwardModelPair()

        # Test every wavefront algorithm
        for name in WfAlgorithmName:
            # Estimate OPD
            wfEst = WfEstimator(algoName=name, returnWfDev=False, units="m")
            opd = wfEst.estimateZk(intra, extra)

            # Estimate wavefront deviation
            wfEst = WfEstimator(algoName=name, returnWfDev=True, units="m")
            wfDev = wfEst.estimateZk(intra, extra)

            # Make sure that OPD = wf dev + intrinsics
            zkInt = wfEst.instrument.getIntrinsicZernikes(
                *intra.fieldAngle,
                nollIndices=np.arange(opd.size) + 4,
            )

            # Make sure the results are identical
            self.assertTrue(np.allclose(opd, wfDev + zkInt))

    def testUnits(self):
        # Get the test data
        zkTrue, intra, extra = forwardModelPair()

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
