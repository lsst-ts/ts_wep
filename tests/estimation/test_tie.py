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
from lsst.ts.wep import Image, ImageMapper
from lsst.ts.wep.estimation import TieAlgorithm


class TestTieAlgorithm(unittest.TestCase):
    """Test TieAlgorithm."""

    @staticmethod
    def _createData(seed: int = 1234):
        # Create some random Zernikes
        rng = np.random.default_rng(seed)
        zkTrue = rng.normal(0, 1e-5 / np.arange(1, 20) ** 2, size=19)
        zkTrue = np.clip(zkTrue, -1e-6, +1e-6)

        # Create a pair of images
        mapper = ImageMapper()

        intraStamp = mapper.mapPupilToImage(
            Image(
                np.zeros((180, 180)),
                (0, -1),
                "intra",
                "r",
                blendOffsets=[[70, 85]],
            ),
            zkTrue,
        )
        intraStamp.image *= 120
        intraStamp.image += rng.normal(scale=np.sqrt(intraStamp.image))
        intraStamp.image += rng.normal(scale=10, size=intraStamp.image.shape)

        extraStamp = mapper.mapPupilToImage(
            Image(
                np.zeros((180, 180)),
                (0, -1),
                "extra",
                "r",
            ),
            zkTrue,
        )
        extraStamp.image *= 60
        extraStamp.image += rng.normal(scale=np.sqrt(extraStamp.image))
        extraStamp.image += rng.normal(scale=15, size=extraStamp.image.shape)

        # Return the Zernikes and both images
        return zkTrue, intraStamp, extraStamp

    def testBadOpticalModel(self):
        with self.assertRaises(ValueError):
            TieAlgorithm(opticalModel="fake")

    def testBadMaxIter(self):
        with self.assertRaises(TypeError):
            TieAlgorithm(maxIter=10.2)
        with self.assertRaises(ValueError):
            TieAlgorithm(maxIter=-1)

    def testBadCompSequence(self):
        with self.assertRaises(ValueError):
            TieAlgorithm(compSequence=np.zeros((2, 3)))

    def testBadCompGain(self):
        with self.assertRaises(ValueError):
            TieAlgorithm(compGain=-1)

    def testBadCenterBinary(self):
        with self.assertRaises(TypeError):
            TieAlgorithm(centerBinary="fake")

    def testBadConvergeTol(self):
        with self.assertRaises(ValueError):
            TieAlgorithm(convergeTol=-1)

    def testBadMaskKwargs(self):
        with self.assertRaises(TypeError):
            TieAlgorithm(maskKwargs="fake")

    def testAccuracy(self):
        for seed in [12345, 23451, 34512, 45123, 51234]:
            # Get the test data
            zkTrue, intra, extra = self._createData(seed)

            # Create estimator
            tie = TieAlgorithm()

            # Estimate Zernikes (in meters)
            zkEst = tie.estimateZk(intra, extra)

            # Check that results are fairly accurate
            self.assertLess(np.sqrt(np.sum((zkEst - zkTrue) ** 2)), 0.35e-6)

    def testSaveHistory(self):
        # Run the algorithm
        zkTrue, intra, extra = self._createData()
        tie = TieAlgorithm()
        tie.estimateZk(intra, extra)

        # Check that no history was saved
        hist = tie.history
        self.assertEqual(len(hist), 0)

        # Estimate again while saving the history
        tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        self.assertIsInstance(hist, dict)

        # Check contents of the first entry
        iter0 = hist.pop(0)
        self.assertEqual(
            list(iter0.keys()),
            [
                "intraInit",
                "extraInit",
                "zkStartIntra",
                "zkStartExtra",
                "zkStartMean",
            ],
        )
        # All entries should all be arrays
        self.assertIsInstance(iter0["intraInit"], np.ndarray)
        self.assertIsInstance(iter0["extraInit"], np.ndarray)
        self.assertIsInstance(iter0["zkStartIntra"], np.ndarray)
        self.assertIsInstance(iter0["zkStartExtra"], np.ndarray)
        self.assertIsInstance(iter0["zkStartMean"], np.ndarray)

        # Check the subsequent iterations of the algorithm
        contents = {
            "recenter": bool,
            "intraCent": np.ndarray,
            "extraCent": np.ndarray,
            "intraComp": np.ndarray,
            "extraComp": np.ndarray,
            "mask": np.ndarray,
            "I0": np.ndarray,
            "dIdz": np.ndarray,
            "zkCompIntra": np.ndarray,
            "zkCompExtra": np.ndarray,
            "zkResid": np.ndarray,
            "zkBest": np.ndarray,
            "zkSum": np.ndarray,
            "converged": bool,
            "caustic": bool,
        }
        for iteration in hist.values():
            for key, val in contents.items():
                print(type(iteration[key]))
                self.assertIsInstance(iteration.pop(key), val)

            # Check that was all that was in the history
            self.assertEqual(len(iteration), 0)

    def testRecenter(self):
        # Run the algorithm with no recenter tolerance
        zkTrue, intra, extra = self._createData()
        tie = TieAlgorithm(centerTol=0)
        tie.estimateZk(intra, extra, saveHistory=True)

        # Check that every iteration recentered
        hist = tie.history
        for i in range(2, len(hist)):
            self.assertTrue(hist[i]["recenter"])
            self.assertFalse(np.allclose(hist[i]["intraCent"], hist[1]["intraCent"]))
            self.assertFalse(np.allclose(hist[i]["extraCent"], hist[1]["extraCent"]))

        # Run the algorithm with infinite recenter tolerance
        tie = TieAlgorithm(centerTol=np.inf)
        tie.estimateZk(intra, extra, saveHistory=True)

        # Check that every iteration recentered
        hist = tie.history
        for i in range(2, len(hist)):
            self.assertFalse(hist[i]["recenter"])
            self.assertTrue(np.allclose(hist[i]["intraCent"], hist[1]["intraCent"]))
            self.assertTrue(np.allclose(hist[i]["extraCent"], hist[1]["extraCent"]))

    def testConvergeTol(self):
        zkTrue, intra, extra = self._createData()

        # TIE with zero tolerance; check number of iterations matches maxIter
        tie = TieAlgorithm(convergeTol=0)
        tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        self.assertEqual(len(hist), tie.maxIter + 1)

        # And that the converge flag is False
        self.assertFalse(hist[max(hist)]["converged"])

        # TIE with infinite tolerance; check number of iterations
        # this is a little tricky because the number of iterations depends
        # on compSequence, because the algorithm can't converge until all
        # Zernikes are being compensated, so we will have to search where jmax
        # lies in the compSequence
        tie = TieAlgorithm(convergeTol=np.inf)
        zkEst = tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        jmax = len(zkEst) + 3
        compAll = np.where(tie.compSequence >= jmax)[0]
        if len(compAll) > 0:
            nIter = compAll[0] + 2
        else:
            nIter = len(tie.compSequence) + 2
        nIter = min(nIter, tie.maxIter)
        self.assertEqual(len(hist), nIter)

        # And that the converge flag is True
        self.assertTrue(hist[max(hist)]["converged"])

    def testCaustic(self):
        zkTrue, intra, extra = self._createData()

        # Use a huge gain to force a caustic
        tie = TieAlgorithm(compGain=10)
        tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        finalIter = hist[max(hist)]

        # Check the flag
        self.assertTrue(finalIter["caustic"])

        # Check the Zernike residuals are NaNs
        self.assertTrue(all(np.isnan(finalIter["zkResid"])))

        # Check final Zernikes are still finite and match previous iteration
        self.assertTrue(all(np.isfinite(finalIter["zkBest"])))
        self.assertTrue(np.allclose(finalIter["zkBest"], hist[max(hist) - 1]["zkBest"]))

    def testMaskBlends(self):
        zkTrue, intra, extra = self._createData()

        # Add a blend offset
        intra.blendOffsets = [[40, 50]]

        # Run TIE with no blend masking
        tie = TieAlgorithm()
        tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        mask1 = hist[max(hist)]["mask"]

        # Run TIE with with blend masking
        tie = TieAlgorithm(maskKwargs=dict(doMaskBlends=True))
        tie.estimateZk(intra, extra, saveHistory=True)
        hist = tie.history
        mask2 = hist[max(hist)]["mask"]

        self.assertGreater(mask1.sum(), mask2.sum())


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
