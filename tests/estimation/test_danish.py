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
from unittest.mock import patch

from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()

# Then import libraries
import danish as danish_pkg  # noqa: E402
from packaging.version import Version  # noqa: E402

_DANISH_V1_1 = Version(danish_pkg.__version__) >= Version("1.1")
_requires_danish_v1_1 = unittest.skipUnless(_DANISH_V1_1, "requires danish >= 1.1")
import numpy as np  # noqa: E402
from astropy.coordinates import Angle  # noqa: E402

from lsst.ts.wep.estimation import DanishAlgorithm, ObservingConditions  # noqa: E402
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
                        "ftol": 1e-3,
                        "xtol": 1e-3,
                        "gtol": 1e-3,
                        "max_nfev": 10,
                        "verbose": 2,
                        "x_scale": "jac",
                    },
                )
                # Get the test data
                zkTrue, intra, extra = forwardModelPair(seed=seed)

                # Test estimation with pairs and single donuts:
                for images in [[intra, extra], [intra], [extra]]:
                    # Estimate Zernikes (in meters)
                    zkEst, _ = dan.estimateZk(*images)  # type: ignore[arg-type]

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
                        "ftol": 1e-3,
                        "xtol": 1e-3,
                        "gtol": 1e-3,
                        "max_nfev": 10,
                        "verbose": 2,
                        "x_scale": "jac",
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
                    zkEst, _ = danBin.estimateZk(*images, saveHistory=True)  # type: ignore[arg-type, misc]
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
            self.assertEqual(
                [
                    "fwhm",
                    "model_dx",
                    "model_dy",
                    "chi_square",
                    "model_flux",
                    "model_bkg",
                    "exception_status",
                    "lstsq_cost",
                    "lstsq_optimality",
                    "lstsq_nfev",
                    "lstsq_njev",
                    "lstsq_status",
                    "lstsq_success",
                    "fit_success",
                ],
                list(metaDict.keys()),
            )
        np.testing.assert_allclose(pairMeta["fwhm"], [intraMeta["fwhm"], extraMeta["fwhm"]], atol=0.01)
        np.testing.assert_allclose(
            pairMeta["model_dx"], [intraMeta["model_dx"], extraMeta["model_dx"]], atol=0.01
        )
        np.testing.assert_allclose(
            pairMeta["model_dy"], [intraMeta["model_dy"], extraMeta["model_dy"]], atol=0.01
        )
        np.testing.assert_allclose(
            pairMeta["model_flux"], [intraMeta["model_flux"], extraMeta["model_flux"]], rtol=0.01
        )
        np.testing.assert_allclose(
            pairMeta["model_bkg"], [intraMeta["model_bkg"], extraMeta["model_bkg"]], atol=10.0
        )

    def testNegativeFluxSingleDonut(self) -> None:
        """Test that a single donut with negative flux returns NaN Zernikes
        instead of crashing with a ValueError in SVD."""
        zkTrue, intra, extra = forwardModelPair(seed=42)

        dan = DanishAlgorithm(
            lstsqKwargs={
                "ftol": 1e-3,
                "xtol": 1e-3,
                "gtol": 1e-3,
                "max_nfev": 10,
                "x_scale": "jac",
            },
        )

        # Corrupt the intra image to have negative total flux
        # (simulates a bad amplifier or saturation bleed)
        intra.image = -np.abs(intra.image) - 1000

        # Should not raise — should return NaN Zernikes
        zkEst, meta = dan.estimateZk(intra)

        self.assertTrue(np.all(np.isnan(zkEst)))
        self.assertIn("Non-positive flux", meta["exception_status"])
        self.assertFalse(meta["fit_success"])

    def testNegativeFluxSingleDonutWithHistory(self) -> None:
        """Test that saving history works for a negative-flux single donut."""
        zkTrue, intra, extra = forwardModelPair(seed=42)

        dan = DanishAlgorithm(
            lstsqKwargs={
                "ftol": 1e-3,
                "xtol": 1e-3,
                "gtol": 1e-3,
                "max_nfev": 10,
                "x_scale": "jac",
            },
        )

        intra.image = -np.abs(intra.image) - 1000

        zkEst, meta = dan.estimateZk(intra, saveHistory=True)

        self.assertTrue(np.all(np.isnan(zkEst)))

        # Check history was populated
        hist = dan.history
        self.assertIn("intra", hist)
        self.assertTrue(np.all(np.isnan(hist["intra"]["zkFit"])))
        self.assertTrue(np.all(np.isnan(hist["intra"]["model"])))
        self.assertTrue(hist["intra"]["GalSimFFTSizeError"])

    def testNegativeFluxPairFirstImage(self) -> None:
        """Test that a pair with a negative-flux first image returns NaN
        Zernikes instead of crashing."""
        zkTrue, intra, extra = forwardModelPair(seed=42)

        dan = DanishAlgorithm(
            lstsqKwargs={
                "ftol": 1e-3,
                "xtol": 1e-3,
                "gtol": 1e-3,
                "max_nfev": 10,
                "x_scale": "jac",
            },
        )

        # Corrupt the intra (first) image
        intra.image = -np.abs(intra.image) - 1000

        zkEst, meta = dan.estimateZk(intra, extra)

        self.assertTrue(np.all(np.isnan(zkEst)))
        self.assertIn("Non-positive flux", meta["exception_status"])
        self.assertFalse(meta["fit_success"])

    def testNegativeFluxPairSecondImage(self) -> None:
        """Test that a pair with a negative-flux second image returns NaN
        Zernikes instead of crashing."""
        zkTrue, intra, extra = forwardModelPair(seed=42)

        dan = DanishAlgorithm(
            lstsqKwargs={
                "ftol": 1e-3,
                "xtol": 1e-3,
                "gtol": 1e-3,
                "max_nfev": 10,
                "x_scale": "jac",
            },
        )

        # Corrupt the extra (second) image
        extra.image = -np.abs(extra.image) - 1000

        zkEst, meta = dan.estimateZk(intra, extra)

        self.assertTrue(np.all(np.isnan(zkEst)))
        self.assertIn("Non-positive flux", meta["exception_status"])
        self.assertFalse(meta["fit_success"])

    def testNegativeFluxPairWithHistory(self) -> None:
        """Test that saving history works for a negative-flux pair."""
        zkTrue, intra, extra = forwardModelPair(seed=42)

        dan = DanishAlgorithm(
            lstsqKwargs={
                "ftol": 1e-3,
                "xtol": 1e-3,
                "gtol": 1e-3,
                "max_nfev": 10,
                "x_scale": "jac",
            },
        )

        # Corrupt the extra image
        extra.image = -np.abs(extra.image) - 1000

        zkEst, meta = dan.estimateZk(intra, extra, saveHistory=True)

        self.assertTrue(np.all(np.isnan(zkEst)))

        hist = dan.history
        self.assertIn("intra", hist)
        self.assertIn("extra", hist)
        self.assertTrue(np.all(np.isnan(hist["extra"]["zkFit"])))
        self.assertTrue(np.all(np.isnan(hist["intra"]["zkFit"])))
        self.assertTrue(hist["extra"]["GalSimFFTSizeError"])

    @_requires_danish_v1_1
    def testSystematicLossAlpha(self) -> None:
        """Test that alpha is passed as loss_fn to SingleDonutModel and
        DZMultiDonutModel. Uses max_nfev=1 to keep runtime minimal."""
        _, intra, extra = forwardModelPair()

        # Single-donut path: verify loss_fn != chi2_loss for alpha=0.05
        dan = DanishAlgorithm(systematicLossAlpha=0.05, lstsqKwargs={"max_nfev": 1})
        with patch(
            "lsst.ts.wep.estimation.danish.danish.SingleDonutModel",
            wraps=danish_pkg.SingleDonutModel,
        ) as mock_model:
            dan.estimateZk(intra)

        loss_fn_nonzero = mock_model.call_args.kwargs["loss_fn"]
        self.assertIsNot(loss_fn_nonzero, danish_pkg.chi2_loss)

        # Pair path: verify DZMultiDonutModel also receives the loss_fn
        with patch(
            "lsst.ts.wep.estimation.danish.danish.DZMultiDonutModel",
            wraps=danish_pkg.DZMultiDonutModel,
        ) as mock_model:
            dan.estimateZk(intra, extra)

        self.assertIsNot(mock_model.call_args.kwargs["loss_fn"], danish_pkg.chi2_loss)

        # Default alpha=0: loss_fn should behave identically to chi2_loss
        dan_default = DanishAlgorithm(systematicLossAlpha=0.0, lstsqKwargs={"max_nfev": 1})
        with patch(
            "lsst.ts.wep.estimation.danish.danish.SingleDonutModel",
            wraps=danish_pkg.SingleDonutModel,
        ) as mock_model:
            dan_default.estimateZk(intra)

        loss_fn_zero = mock_model.call_args.kwargs.get("loss_fn", danish_pkg.chi2_loss)
        data, model_vals, var = np.ones(10), np.ones(10) * 2, np.ones(10)
        np.testing.assert_array_equal(
            loss_fn_zero(data, model_vals, var),
            danish_pkg.chi2_loss(data, model_vals, var),
        )

    @_requires_danish_v1_1
    def testDoAoiThroughput(self) -> None:
        """Test that doAoiThroughput passes correct bandpass_filter and airmass
        to DonutFactory. Uses max_nfev=1 to keep runtime minimal."""
        _, intra, extra = forwardModelPair()
        dan = DanishAlgorithm(doAoiThroughput=True, lstsqKwargs={"max_nfev": 1})

        # 45 deg altitude → raw airmass = 1/sin(45°) ≈ 1.414 → rounds to 1.4
        obs = ObservingConditions(altitude=Angle(np.pi / 4, "rad"))

        with patch(
            "lsst.ts.wep.estimation.danish.danish.DonutFactory",
            wraps=danish_pkg.DonutFactory,
        ) as mock_factory:
            dan.estimateZk(intra, extra, obs=obs)

        call_kwargs = mock_factory.call_args.kwargs
        self.assertEqual(call_kwargs["bandpass_filter"], "r")
        self.assertAlmostEqual(call_kwargs["airmass"], 1.4)

        # With no altitude, airmass falls back to the default (1.2)
        with patch(
            "lsst.ts.wep.estimation.danish.danish.DonutFactory",
            wraps=danish_pkg.DonutFactory,
        ) as mock_factory:
            dan.estimateZk(intra, extra, obs=ObservingConditions())

        call_kwargs = mock_factory.call_args.kwargs
        self.assertEqual(call_kwargs["bandpass_filter"], "r")
        self.assertAlmostEqual(call_kwargs["airmass"], 1.2)