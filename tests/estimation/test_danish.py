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

from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()

# Then import libraries
import numpy as np  # noqa: E402
from lsst.ts.wep.estimation import DanishAlgorithm  # noqa: E402
from lsst.ts.wep.utils.modelUtils import forwardModelPair  # noqa: E402

# Directly configure NumPy if using version that supports it
try:  # noqa: E402
    np.config.threading.use_openmp = False  # noqa: E402
except (AttributeError, ImportError):  # noqa: E402
    pass  # noqa: E402


class TestDanishAlgorithm(unittest.TestCase):
    """Test DanishAlgorithm."""

    def testBadLstsqKwargs(self) -> None:
        for kwarg in ["fun", "x0", "jac", "args"]:
            with self.assertRaises(KeyError):
                DanishAlgorithm(lstsqKwargs={kwarg: None})

    def testGoodLstsqKwargs(self) -> None:
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

    def testAccuracyWithoutBinning(self) -> None:
        for jointFitPair in [True, False]:
            # Try several different random seeds
            for seed in [12345, 23451]:
                # Create estimator
                dan = DanishAlgorithm(
                    jointFitPair=jointFitPair,
                    lstsqKwargs={
                        "ftol": 1e-1,
                        "xtol": 1e-1,
                        "gtol": 1e-1,
                        "max_nfev": 10,
                        "verbose": 2,
                    },
                )
                # Get the test data
                zkTrue, intra, extra = forwardModelPair(seed=seed)

                # Test estimation with pairs and single donuts:
                for images in [[intra, extra], [intra], [extra]]:
                    # Estimate Zernikes (in meters)
                    zkEst, _ = dan.estimateZk(*images)

                    # Check that results are fairly accurate
                    self.assertLess(np.sqrt(np.sum((zkEst - zkTrue) ** 2)), 0.35e-6)

    def testAccuracyWithBinning(self) -> None:
        for jointFitPair in [True, False]:
            # Try several different random seeds
            for seed in [12345, 23451]:
                # Create estimator
                danBin = DanishAlgorithm(
                    jointFitPair=jointFitPair,
                    lstsqKwargs={
                        "ftol": 1e-1,
                        "xtol": 1e-1,
                        "gtol": 1e-1,
                        "max_nfev": 10,
                        "verbose": 2,
                    },
                    binning=2,
                )
                # Get the test data
                zkTrue, intra, extra = forwardModelPair(seed=seed)

                # Compute shape of binned images
                shapex, shapey = intra.image.shape
                binned_shapex = shapex // 2
                binned_shapey = shapey // 2

                # Ensure odd
                if binned_shapex % 2 == 0:
                    binned_shapex -= 1
                if binned_shapey % 2 == 0:
                    binned_shapey -= 1
                binned_shape = (binned_shapex, binned_shapey)

                # Test estimation with pairs and single donuts:
                for images in [[intra, extra], [intra], [extra]]:
                    # Estimate Zernikes (in meters)
                    zkEst, _ = danBin.estimateZk(*images, saveHistory=True)
                    self.assertLess(np.sqrt(np.sum((zkEst - zkTrue) ** 2)), 0.35e-6)

                    # Test that we binned the images.
                    if "intra" in danBin.history:
                        self.assertEqual(danBin.history["intra"]["image"].shape, binned_shape)
                    if "extra" in danBin.history:
                        self.assertEqual(danBin.history["extra"]["image"].shape, binned_shape)

    def testMetadata(self) -> None:
        zkTrue, intra, extra = forwardModelPair(seed=42)

        # Estimate with singles and pairs
        dan = DanishAlgorithm()
        zkPair, pairMeta = dan.estimateZk(intra, extra)
        zkIntra, intraMeta = dan.estimateZk(intra)
        zkExtra, extraMeta = dan.estimateZk(extra)

        # Check metadata
        for metaDict in pairMeta, intraMeta, extraMeta:
            self.assertEqual(["fwhm", "model_dx", "model_dy", "model_sky_level"], list(metaDict.keys()))
            self.assertAlmostEqual(metaDict["fwhm"], 1.034, delta=0.01)
        np.testing.assert_allclose(
            pairMeta["model_dx"], [intraMeta["model_dx"], extraMeta["model_dx"]], atol=0.01
        )
        np.testing.assert_allclose(
            pairMeta["model_dy"], [intraMeta["model_dy"], extraMeta["model_dy"]], atol=0.01
        )
        np.testing.assert_allclose(
            pairMeta["model_sky_level"],
            [intraMeta["model_sky_level"], extraMeta["model_sky_level"]],
            atol=0.01,
        )
