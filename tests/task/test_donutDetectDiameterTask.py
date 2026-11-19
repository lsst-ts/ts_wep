# This file is part of ts_wep.
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

"""Unit tests for `lsst.ts.wep.task.donutDetectDiameterTask`."""

from __future__ import annotations

import unittest

import numpy as np
from astropy.table import QTable

import lsst.pex.config as pexConfig
from lsst.afw.image import ExposureF
from lsst.ts.wep.task.donutDetectDiameterTask import (
    DiskCorrelationBank,
    DonutDetectDiameterConfig,
    DonutDetectDiameterTask,
    _createDisk,
    _cropSame,
    _nOffsetsFor,
    _shiftSubtractNoise,
    makeDiameterLadder,
)


def makeDonutExposure(
    shape: tuple[int, int] = (512, 512),
    center: tuple[int, int] = (260, 250),
    diameter: float = 120.0,
    innerFrac: float = 0.61,
    amplitude: float = 50.0,
    noise: float = 1.0,
    seed: int = 42,
) -> ExposureF:
    """Build an exposure holding a single filled-annulus donut.

    Parameters
    ----------
    shape : tuple [int, int]
        Image shape ``(ny, nx)``.
    center : tuple [int, int]
        Donut center ``(y, x)`` in full-res pixels.
    diameter : float
        Outer donut diameter in full-res pixels.
    innerFrac : float
        Inner/outer diameter ratio of the annulus.
    amplitude : float
        Flux added to the annulus pixels above the noise floor.
    noise : float
        Standard deviation of the Gaussian background noise.
    seed : int
        Seed for the random background.

    Returns
    -------
    lsst.afw.image.ExposureF
        Exposure with the injected donut.
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, noise, (ny, nx)).astype(np.float32)
    cy, cx = center
    rOut = diameter / 2.0
    rIn = innerFrac * diameter / 2.0
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus = (r <= rOut) & (r >= rIn)
    image[annulus] += amplitude
    exposure = ExposureF(nx, ny)
    exposure.image.array[:] = image
    return exposure


class TestModuleHelpers(unittest.TestCase):
    """The free functions: ``_nOffsetsFor``, ``_createDisk``, ``_cropSame``,
    ``_shiftSubtractNoise`` and ``makeDiameterLadder``."""

    def testNOffsetsForClampsAndDecreases(self) -> None:
        # Clipped to [2, 10]: tiny disks hit the ceiling, huge ones the floor.
        self.assertEqual(_nOffsetsFor(0.1), 10)
        self.assertEqual(_nOffsetsFor(1000.0), 2)
        # 60 / 6 == 10, exactly on the ceiling.
        self.assertEqual(_nOffsetsFor(6.0), 10)
        # Monotonically non-increasing in diameter.
        vals = [_nOffsetsFor(d) for d in (4, 8, 16, 32, 64)]
        self.assertEqual(vals, sorted(vals, reverse=True))
        for v in vals:
            self.assertGreaterEqual(v, 2)
            self.assertLessEqual(v, 10)

    def testCreateDiskShapeSumAndRange(self) -> None:
        diameter = 10.0
        disk = _createDisk(diameter)
        # Grid is diameter + 5 per axis.
        self.assertEqual(disk.shape, (15, 15))
        self.assertEqual(disk.dtype, np.float64)
        # Area-coverage fractions in [0, 1].
        self.assertGreaterEqual(disk.min(), 0.0)
        self.assertLessEqual(disk.max(), 1.0)
        # Sum approximates the disk area pi * (d / 2)**2.
        self.assertAlmostEqual(disk.sum(), np.pi * (diameter / 2) ** 2, delta=2.0)

    def testCreateDiskAntiAliasedEdge(self) -> None:
        # Super-sampling must yield soft edge pixels strictly inside (0, 1),
        # not a hard binary mask.
        disk = _createDisk(12.0)
        soft = (disk > 0.0) & (disk < 1.0)
        self.assertTrue(np.any(soft))
        # The explicit nOffsets is honored: finer super-sampling resolves
        # intermediate edge levels that a coarse one quantizes away.
        coarse = _createDisk(12.0, nOffsets=1)
        fine = _createDisk(12.0, nOffsets=10)
        self.assertFalse(np.array_equal(coarse, fine))
        self.assertGreater(len(np.unique(fine)), len(np.unique(coarse)))

    def testCropSameShapeAndOrigin(self) -> None:
        imgShape = (5, 5)
        kerShape = (3, 3)
        full = np.arange(
            (imgShape[0] + kerShape[0] - 1) * (imgShape[1] + kerShape[1] - 1),
            dtype=float,
        ).reshape(imgShape[0] + kerShape[0] - 1, imgShape[1] + kerShape[1] - 1)
        out = _cropSame(full, imgShape, kerShape)
        self.assertEqual(out.shape, imgShape)
        # Crop origin uses the (k - 1) // 2 convention.
        r0 = (kerShape[0] - 1) // 2
        c0 = (kerShape[1] - 1) // 2
        np.testing.assert_array_equal(out, full[r0 : r0 + imgShape[0], c0 : c0 + imgShape[1]])

    def testShiftSubtractNoiseRecoversSigma(self) -> None:
        rng = np.random.default_rng(0)
        sigma = 4.0
        image = rng.normal(0.0, sigma, (400, 400))
        self.assertAlmostEqual(_shiftSubtractNoise(image), sigma, delta=0.2)

    def testShiftSubtractNoiseCancelsColumnOffset(self) -> None:
        # A fixed per-column offset is differenced away, so it must not
        # inflate the estimate.
        rng = np.random.default_rng(1)
        sigma = 2.0
        image = rng.normal(0.0, sigma, (400, 400))
        image += np.arange(400)[np.newaxis, :] * 10.0
        self.assertAlmostEqual(_shiftSubtractNoise(image), sigma, delta=0.2)

    def testShiftSubtractNoiseDegenerate(self) -> None:
        # A constant image has zero spread -> NaN.
        self.assertTrue(np.isnan(_shiftSubtractNoise(np.ones((10, 10)))))
        # A single row cannot be differenced -> NaN.
        self.assertTrue(np.isnan(_shiftSubtractNoise(np.ones((1, 10)))))

    def testMakeDiameterLadderMonotoneAndBounded(self) -> None:
        ladder = makeDiameterLadder(4.0, 60.0, innerFrac=0.61, k=3)
        self.assertTrue(np.all(np.diff(ladder) > 0))
        self.assertAlmostEqual(ladder[0], 4.0)
        self.assertLessEqual(ladder[-1], 60.0 + 1e-9)

    def testMakeDiameterLadderGeometricRatio(self) -> None:
        # Every k rungs the diameter grows by 1 / innerFrac.
        innerFrac, k = 0.61, 3
        ladder = makeDiameterLadder(4.0, 60.0, innerFrac=innerFrac, k=k)
        self.assertAlmostEqual(ladder[k] / ladder[0], 1.0 / innerFrac, places=6)
        ratio = (1.0 / innerFrac) ** (1.0 / k)
        np.testing.assert_allclose(ladder[1:] / ladder[:-1], ratio, rtol=1e-6)

    def testMakeDiameterLadderRejectsBadK(self) -> None:
        with self.assertRaises(ValueError):
            makeDiameterLadder(4.0, 60.0, k=0)


class TestDiskCorrelationBank(unittest.TestCase):
    """`DiskCorrelationBank` construction and correlation."""

    def setUp(self) -> None:
        self.diameters = makeDiameterLadder(4.0, 30.0, innerFrac=0.61, k=3)
        self.bank = DiskCorrelationBank((40, 50), self.diameters, innerFrac=0.61)

    def testConstructionAttributes(self) -> None:
        self.assertEqual(self.bank.binnedShape, (40, 50))
        # Diameters are sorted.
        np.testing.assert_array_equal(self.bank.diameters, np.sort(self.diameters))
        nDiam = len(self.bank.diameters)
        for arr in (
            self.bank.innerIndex,
            self.bank.outerIndex,
            self.bank.annulusNorm,
            self.bank.bgWeight,
            self.bank.supportArea,
        ):
            self.assertEqual(len(arr), nDiam)
        # Norms and support areas are never negative.
        self.assertTrue(np.all(self.bank.annulusNorm >= 0))
        self.assertTrue(np.all(self.bank.supportArea >= 0))

    def testInnerIndexLandsOnLadder(self) -> None:
        # Where an inner disk was found, its diameter is innerFrac * d_i.
        for i, ii in enumerate(self.bank.innerIndex):
            if ii >= 0:
                self.assertAlmostEqual(
                    self.bank.diameters[ii],
                    0.61 * self.bank.diameters[i],
                    delta=0.61 * self.bank.diameters[i] * 1e-2,
                )

    def testCorrelateShapeMismatchRaises(self) -> None:
        with self.assertRaises(ValueError):
            self.bank.correlate(np.zeros((10, 10)))

    def testCorrelateReturnsMapsPerDiameter(self) -> None:
        corr, corrSq = self.bank.correlate(np.zeros((40, 50)))
        self.assertEqual(len(corr), len(self.bank.diameters))
        self.assertEqual(len(corrSq), len(self.bank.diameters))
        for c in corr:
            self.assertEqual(c.shape, (40, 50))

    def testCorrelateRecoversTemplateSum(self) -> None:
        # Correlating a uniform unit image against a disk gives the template
        # sum at every interior pixel (each template pixel sees a 1).
        image = np.ones((40, 50))
        corr, corrSq = self.bank.correlate(image)
        i = len(self.bank.diameters) // 2
        expected = _createDisk(self.bank.diameters[i]).sum()
        self.assertAlmostEqual(corr[i][20, 25], expected, delta=expected * 1e-3)
        # image**2 == image for a unit image, so both correlations agree.
        self.assertAlmostEqual(corrSq[i][20, 25], expected, delta=expected * 1e-3)


class TestConfigValidation(unittest.TestCase):
    """`DonutDetectDiameterConfig.validate`."""

    def testDefaultsValidate(self) -> None:
        DonutDetectDiameterConfig().validate()

    def testRejectsBadRootOrder(self) -> None:
        config = DonutDetectDiameterConfig()
        config.rootOrder = 0
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

    def testRejectsNonIncreasingDiameterRange(self) -> None:
        config = DonutDetectDiameterConfig()
        config.dMinFull = config.dMaxFull
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

    def testRejectsBadBinning(self) -> None:
        config = DonutDetectDiameterConfig()
        config.detectionBinning = 0
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()


class TestTaskHelpers(unittest.TestCase):
    """Task methods exercised directly, without a full ``run``."""

    def setUp(self) -> None:
        self.task = DonutDetectDiameterTask(config=DonutDetectDiameterConfig())

    def testLadderBuiltOnConstruction(self) -> None:
        cfg = self.task.config
        self.assertAlmostEqual(self.task._diameters[0], cfg.dMinFull / cfg.detectionBinning)
        self.assertLessEqual(
            self.task._diameters[-1],
            cfg.dMaxFull / cfg.detectionBinning + 1e-9,
        )

    def testGetBankCachesAndRebuilds(self) -> None:
        bank1 = self.task._getBank((40, 40))
        bank2 = self.task._getBank((40, 40))
        self.assertIs(bank1, bank2)
        # A new shape forces a rebuild.
        bank3 = self.task._getBank((50, 40))
        self.assertIsNot(bank1, bank3)
        self.assertEqual(bank3.binnedShape, (50, 40))

    def testPrepImageBinsAndFillsBadPixels(self) -> None:
        exposure = ExposureF(64, 64)
        exposure.image.array[:] = 5.0
        # Flag a block of pixels bad; they must be zeroed before binning.
        exposure.mask.array[10:20, 10:20] = exposure.mask.getPlaneBitMask("SAT")
        binned, bbox = self.task._prepImage(exposure)
        binning = self.task.config.detectionBinning
        margin = self.task.config.edgeMargin
        # Edge-eroded then binned.
        expectedSide = (64 - 2 * margin) // binning
        self.assertEqual(binned.shape, (expectedSide, expectedSide))
        self.assertEqual(bbox, exposure.getBBox().erodedBy(margin))
        # The binned pixel covering the bad block is pulled below 5.
        self.assertLess(binned.min(), 5.0)

    def testPrepImageFullyMaskedIsZero(self) -> None:
        exposure = ExposureF(64, 64)
        exposure.image.array[:] = 7.0
        exposure.mask.array[:] = exposure.mask.getPlaneBitMask("BAD")
        binned, _ = self.task._prepImage(exposure)
        np.testing.assert_array_equal(binned, 0.0)

    def testPrepImageBinningOneKeepsResolution(self) -> None:
        config = DonutDetectDiameterConfig()
        config.detectionBinning = 1
        task = DonutDetectDiameterTask(config=config)
        exposure = ExposureF(32, 32)
        exposure.image.array[:] = 3.0
        binned, _ = task._prepImage(exposure)
        side = 32 - 2 * config.edgeMargin
        self.assertEqual(binned.shape, (side, side))

    def testPeakDiameterTooFewPoints(self) -> None:
        self.assertTrue(np.isnan(self.task._peakDiameter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))))

    def testPeakDiameterEdgeReturnsGridValue(self) -> None:
        ds = np.array([10.0, 20.0, 40.0])
        # Maximum at the first rung -> unrefinable, grid value returned.
        self.assertEqual(self.task._peakDiameter(ds, np.array([3.0, 2.0, 1.0])), 10.0)
        # Maximum at the last rung.
        self.assertEqual(self.task._peakDiameter(ds, np.array([1.0, 2.0, 3.0])), 40.0)

    def testPeakDiameterInteriorRefinement(self) -> None:
        ds = np.array([10.0, 20.0, 40.0])
        # Symmetric triplet peaks exactly on the middle rung.
        self.assertAlmostEqual(self.task._peakDiameter(ds, np.array([1.0, 3.0, 1.0])), 20.0, places=6)
        # An asymmetric interior peak lands between the rungs it leans toward.
        refined = self.task._peakDiameter(ds, np.array([1.0, 3.0, 2.0]))
        self.assertGreater(refined, 20.0)
        self.assertLess(refined, 40.0)

    def testPeakDiameterFlatTripletReturnsGridValue(self) -> None:
        ds = np.array([10.0, 20.0, 40.0])
        # A flat curve ties its argmax to the first rung (an edge), so the
        # grid value is returned with no sub-grid refinement or zero-division.
        self.assertEqual(self.task._peakDiameter(ds, np.array([2.0, 2.0, 2.0])), 10.0)


class TestTaskRun(unittest.TestCase):
    """End-to-end `DonutDetectDiameterTask.run` on a synthetic donut."""

    def testDetectsInjectedDonut(self) -> None:
        diameter = 120.0
        center = (260, 250)
        exposure = makeDonutExposure(diameter=diameter, center=center)
        task = DonutDetectDiameterTask(config=DonutDetectDiameterConfig())
        result = task.run(exposure)

        self.assertIsInstance(result.detections, QTable)
        self.assertGreaterEqual(len(result.detections), 1)
        for column in (
            "id",
            "centroid_x",
            "centroid_y",
            "diameter",
            "likeness",
            "snr",
            "peak_diameter",
            "curve_argmax_edge",
        ):
            self.assertIn(column, result.detections.colnames)

        # The exposure diameter matches the injection to within a bin.
        binning = task.config.detectionBinning
        self.assertAlmostEqual(result.diameter, diameter, delta=binning)

        # The brightest detection sits on the injected donut.
        row = result.detections[0]
        self.assertAlmostEqual(row["centroid_x"], center[1], delta=binning)
        self.assertAlmostEqual(row["centroid_y"], center[0], delta=binning)

        self.assertTrue(np.isfinite(result.pixelNoise))
        self.assertEqual(result.binning, binning)
        self.assertEqual(result.binnedImage.shape, result.likeness.shape)
        np.testing.assert_array_equal(result.diametersFull, result.diametersBinned * binning)

    def testNoDonutYieldsNoDetections(self) -> None:
        # Pure noise: the quality cuts should leave no surviving peaks and an
        # unmeasurable diameter, without raising.
        rng = np.random.default_rng(7)
        exposure = ExposureF(512, 512)
        exposure.image.array[:] = rng.normal(0.0, 1.0, (512, 512)).astype(np.float32)
        task = DonutDetectDiameterTask(config=DonutDetectDiameterConfig())
        result = task.run(exposure)
        self.assertEqual(len(result.detections), 0)
        self.assertTrue(np.isnan(result.diameter))
        self.assertTrue(np.isnan(result.diameterScatter))

    def testMultipleDonutsPooledScatter(self) -> None:
        # Two donuts of the same size give >= 2 detections and a finite
        # per-detection scatter.
        exposure = makeDonutExposure(diameter=120.0, center=(170, 160))
        yy, xx = np.indices((512, 512))
        r = np.sqrt((yy - 360) ** 2 + (xx - 350) ** 2)
        annulus = (r <= 60.0) & (r >= 0.61 * 60.0)
        exposure.image.array[annulus] += 50.0
        task = DonutDetectDiameterTask(config=DonutDetectDiameterConfig())
        result = task.run(exposure)
        self.assertGreaterEqual(len(result.detections), 2)
        self.assertTrue(np.isfinite(result.diameterScatter))


if __name__ == "__main__":
    unittest.main()
