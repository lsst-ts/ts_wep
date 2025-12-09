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
from scipy.ndimage import shift

from lsst.ts.wep.utils import (
    binArray,
    centerWithTemplate,
    conditionalSigmaClip,
    extractArray,
    padArray,
    polygonContains,
    rotMatrix,
)


class TestMiscUtils(unittest.TestCase):
    """Test the miscellaneous utility functions."""

    def testRotMatrix(self) -> None:
        # Test rotation with 0 degrees
        testTheta1 = 0
        rotMatrix1 = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(rotMatrix1, rotMatrix(testTheta1))

        # Test rotation with 90 degrees
        testTheta2 = 90
        rotMatrix2 = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(rotMatrix2, rotMatrix(testTheta2))

        # Test rotation with 45 degrees
        testTheta3 = 45
        rotMatrix3 = np.array([[0.707107, -0.707107], [0.707107, 0.707107]])
        np.testing.assert_array_almost_equal(rotMatrix3, rotMatrix(testTheta3))

    def testPadArray(self) -> None:
        imgDim = 10
        padPixelSize = 20

        img, imgPadded = self._padRandomImg(imgDim, padPixelSize)

        self.assertEqual(imgPadded.shape[0], imgDim + padPixelSize)

    def _padRandomImg(self, imgDim: int, padPixelSize: int) -> tuple[np.ndarray, np.ndarray]:
        img = np.random.rand(imgDim, imgDim)
        imgPadded = padArray(img, imgDim + padPixelSize)

        return img, imgPadded

    def testExtractArray(self) -> None:
        imgDim = 10
        padPixelSize = 20
        img, imgPadded = self._padRandomImg(imgDim, padPixelSize)

        imgExtracted = extractArray(imgPadded, imgDim)

        self.assertEqual(imgExtracted.shape[0], imgDim)

    def testCenterWithTemplate(self) -> None:
        # Create a template to use for correlating
        template = np.pad(np.ones((40, 40)), 5)

        # Expand template into a centered image
        image = np.pad(template, 55)

        # Roll the image to create a decentered image
        decentered = np.roll(image, (3, 4), (0, 1))

        # Recenter
        recentered = centerWithTemplate(decentered, template)

        # Compare the centers of mass
        grid = np.arange(len(image))
        x, y = np.meshgrid(grid, grid)
        dx = (x * image).sum() / image.sum() - (x * recentered).sum() / recentered.sum()
        dy = (y * image).sum() / image.sum() - (y * recentered).sum() / recentered.sum()
        self.assertTrue(dx == 0)
        self.assertTrue(dy == 0)

        # Now decenter using a sub-pixel shift
        decentered = shift(image, (-3.2, 4.1))

        # Recenter
        recentered = centerWithTemplate(decentered, template)

        # Compare the centers of mass
        # For this test, just require the final decenter is less than 0.5
        # in each dimension, since it is impossible to get 100% correct
        dx = (x * image).sum() / image.sum() - (x * recentered).sum() / recentered.sum()
        dy = (y * image).sum() / image.sum() - (y * recentered).sum() / recentered.sum()
        self.assertTrue(np.abs(dx) < 0.5)
        self.assertTrue(np.abs(dy) < 0.5)

    def testPolygonContains(self) -> None:
        # First a small test
        grid = np.arange(6).astype(float)
        x, y = np.meshgrid(grid, grid)
        poly = np.array([[0.9, 3.1, 3.1, 0.9, 0.9], [1.9, 1.9, 4.1, 4.1, 1.9]]).T
        poly = poly.astype(float)
        contains = polygonContains(x, y, poly)
        truth = np.full_like(contains, False)
        truth[2:5, 1:4] = True
        self.assertTrue(np.array_equal(contains, truth))

        # Now a bigger test
        # Uses ratio of points inside circle / inside square = pi / 4
        grid = np.linspace(-1, 1, 2000)
        x, y = np.meshgrid(grid, grid)
        theta = np.linspace(0, 2 * np.pi, 1000)
        poly = np.array([np.cos(theta), np.sin(theta)]).T
        contains = polygonContains(x, y, poly)
        self.assertTrue(np.isclose(contains.mean(), np.pi / 4, atol=1e-3))

        # Test bad shapes
        with self.assertRaises(ValueError):
            polygonContains(x, y[:-10], poly)
        with self.assertRaises(ValueError):
            polygonContains(x, y[..., None], poly)
        with self.assertRaises(ValueError):
            polygonContains(x, y, poly.T)

    def testConditionalSigmaClipping(self) -> None:
        # Create a sample array where:
        # - The first column has low variability
        # and should not be clipped.
        # - The second column has high variability
        # and should be clipped.
        sampleArray = np.array([[1.0, 100.0], [2.0, 200.0], [6.0, 600.0], [2.0, 200.0], [1.0, 100.0]])
        # Set sigma for sigma clipping
        # Set a std_min that will ensure the second column
        # is clipped but not the first
        sigmaClipKwargs = {"sigma": 1.5, "stdfunc": "mad_std"}
        stdMin = 50

        # Call the function with the sample array
        processedArray = conditionalSigmaClip(sampleArray, sigmaClipKwargs=sigmaClipKwargs, stdMin=stdMin)

        # Assert the first column remains unchanged
        np.testing.assert_array_equal(processedArray[:, 0], sampleArray[:, 0])

        # Assert the second column has NaNs due to clipping
        # This assumes the sigma clipping with std would indeed
        # clip values in the second column.
        # Checking for NaNs as a result of clipping
        assert np.isnan(processedArray[:, 1]).any(), "Expected NaNs in the second column after clipping"

    def testBinArray(self) -> None:
        # Create a 4x4 test array
        testArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        binning = 2

        # Test 'mean' method
        expectedMean = np.array([[3.5, 5.5], [11.5, 13.5]])
        resultMean = binArray(testArray, binning, method="mean")
        np.testing.assert_array_almost_equal(resultMean, expectedMean)

        # Test 'median' method (same result as mean in this symmetrical case)
        resultMedian = binArray(testArray, binning, method="median")
        np.testing.assert_array_almost_equal(resultMedian, expectedMean)

        # Simple 4x4 test array with hot pixel at (2,2)
        baseArray = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1]])
        binning = 2

        # Test 'mean' result
        resultMean = binArray(baseArray, binning, method="mean")
        expectedMeanTopLeft = 1.0
        expectedMeanBottomRight = (1000 + 1 + 1 + 1) / 4.0  # Hot pixel affects mean
        self.assertAlmostEqual(resultMean[0, 0], expectedMeanTopLeft)
        self.assertAlmostEqual(resultMean[1, 1], expectedMeanBottomRight)

        # Test 'median' result
        resultMedian = binArray(baseArray, binning, method="median")
        expectedMedianBottomRight = 1.0  # Median unaffected by single outlier
        self.assertAlmostEqual(resultMedian[0, 0], expectedMeanTopLeft)
        self.assertAlmostEqual(resultMedian[1, 1], expectedMedianBottomRight)

        # Verify that mean and median give *different* results due to outlier
        self.assertNotAlmostEqual(resultMean[1, 1], resultMedian[1, 1])

        # Test invalid method raises ValueError
        with self.assertRaises(ValueError):
            binArray(testArray, binning, method="sum")

        # Test non-divisible array is cropped correctly
        testArrayOdd = np.arange(25).reshape(5, 5)
        resultOdd = binArray(testArrayOdd, binning, method="mean")
        self.assertEqual(resultOdd.shape, (2, 2))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
