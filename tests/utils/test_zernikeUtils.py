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

import os
import unittest

import numpy as np
from lsst.ts.wep.utils import (
    calcAOSResid,
    checkNollIndices,
    convertZernikesToPsfWidth,
    createZernikeBasis,
    createZernikeGradBasis,
    getModulePath,
    getPsfGradPerZernike,
    getZernikeParity,
    makeDense,
    makeSparse,
    zernikeEval,
    zernikeFit,
    zernikeGradEval,
)
from lsst.utils.tests import TestCase


class TestZernikeUtils(TestCase):
    """Test the Zernike utility functions."""

    def testOrthonormality(self):
        # Create the Zernike basis
        grid = np.linspace(-1, 1, 1000)
        uGrid, vGrid = np.meshgrid(grid, grid)
        zkBasis = createZernikeBasis(uGrid, vGrid, obscuration=0.61)

        # Mask outside the pupil
        r = np.sqrt(uGrid**2 + vGrid**2)
        mask = (r >= 0.61) & (r <= 1)
        zkBasis *= mask

        # Calculate inner products
        innerProd = np.einsum("jab,kab->jk", zkBasis, zkBasis)

        # Calculate the expected norm of the Zernikes
        dA = np.square(uGrid[0, 1] - uGrid[0, 0])
        norm = np.pi * (1 - 0.61**2) / dA

        # Normalize the inner products
        innerProd /= norm

        # Check identity
        close = np.allclose(innerProd, np.identity(innerProd.shape[0]), atol=1e-2)
        self.assertTrue(close)

    def testZernikeEval(self):
        # Create a Zernike basis
        grid = np.linspace(-1, 1, 200)
        uGrid, vGrid = np.meshgrid(grid, grid)
        zkBasis = createZernikeBasis(uGrid, vGrid)

        # Evaluate each Zernike polynomial and compare to basis
        for i in range(len(zkBasis)):
            coeff = np.zeros(i + 1)
            coeff[-1] = 1
            np.allclose(zernikeEval(uGrid, vGrid, coeff), zkBasis[i])

    def testZernikeGradEval(self):
        # Create a Zernike gradient basis
        grid = np.linspace(-1, 1, 200)
        uGrid, vGrid = np.meshgrid(grid, grid)
        dzkdu, dzkdv = createZernikeGradBasis(uGrid, vGrid)

        # Evaluate each Zernike gradient and compare to bases
        for i in range(len(dzkdu)):
            coeff = np.zeros(i + 1)
            coeff[-1] = 1
            np.allclose(zernikeGradEval(uGrid, vGrid, 1, 0, coeff), dzkdu[i])
            np.allclose(zernikeGradEval(uGrid, vGrid, 0, 1, coeff), dzkdv[i])

    def testConvertZernikesToPsf(self):
        # Directory where expected values are stored
        testDataDir = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "psfGradientsPerZernike",
        )

        # LsstCam
        calculated = getPsfGradPerZernike(
            jmin=4,
            jmax=37,
            diameter=8.36,
            obscuration=0.612,
        )
        expected = np.genfromtxt(os.path.join(testDataDir, "lsstcam.txt"))
        self.assertTrue(np.allclose(calculated, expected, atol=1e-3))

        # AuxTel
        calculated = getPsfGradPerZernike(
            jmin=1,
            jmax=37,
            diameter=1.2,
            obscuration=0.3525,
        )
        expected = np.genfromtxt(os.path.join(testDataDir, "auxtel.txt"))
        self.assertTrue(np.allclose(calculated, expected, atol=1e-3))

        # Finally check converting array of 1's returns the coefficients
        coeffs = convertZernikesToPsfWidth(np.ones(19))
        self.assertTrue(np.allclose(coeffs, getPsfGradPerZernike()))

    def testCalcAOSResid(self):
        # Create array of 1's (units: micron)
        # This represents a very large wavefront error
        zk = np.ones(8)

        # Convert to PSF FWHM contribution
        zk_fwhm = convertZernikesToPsfWidth(zk)

        # Calculate AOS residual
        aos_resid = np.sqrt(np.sum(np.square(zk_fwhm)))

        # This should over-estimate the corrected value
        self.assertGreater(aos_resid, calcAOSResid(zk))

    def testGetZernikeParity(self):
        xParity = getZernikeParity()
        xTruth = [1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1]
        self.assertTrue(all(xParity == xTruth))

        yParity = getZernikeParity(axis="y")
        yTruth = [1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
        self.assertTrue(all(yParity == yTruth))

    def testJmin(self):
        # Create a pupil grid
        grid = np.linspace(-1, 1, 200)
        u, v = np.meshgrid(grid, grid)

        # And a set of Zernike coefficients
        zkCoeff = np.arange(4, 22)[::-1]
        zkCoeff = 1e-5 * zkCoeff[::-1] * (-1) ** zkCoeff

        # Zernike Bases
        zkB0 = createZernikeBasis(u, v, jmin=0)
        zkB4 = createZernikeBasis(u, v, jmin=4)
        self.assertTrue(np.allclose(zkB0[4:], zkB4))

        # Zernike grad Bases
        dzkB0 = createZernikeGradBasis(u, v, jmin=0)
        dzkB4 = createZernikeGradBasis(u, v, jmin=4)
        self.assertTrue(np.allclose(dzkB0[:, 4:, ...], dzkB4))

        # Evaluate Zernikes
        wf0 = zernikeEval(u, v, [0, 0, 0, 0] + list(zkCoeff), jmin=0)
        wf4 = zernikeEval(u, v, zkCoeff, jmin=4)
        self.assertTrue(np.allclose(wf0, wf4))

        # Evaluate Zernike gradients
        dwf0 = zernikeGradEval(u, v, 1, 1, [0, 0, 0, 0] + list(zkCoeff), jmin=0)
        dwf4 = zernikeGradEval(u, v, 1, 1, zkCoeff, jmin=4)
        self.assertTrue(np.allclose(dwf0, dwf4))

        # Fit Zernikes
        zkFit0 = zernikeFit(u, v, wf4, jmin=0)
        zkFit4 = zernikeFit(u, v, wf4, jmin=4)
        self.assertTrue(np.allclose(zkFit0[4:], zkFit4))

        # PSF conversion factors
        dpsf0 = getPsfGradPerZernike(jmin=0)
        dpsf4 = getPsfGradPerZernike(jmin=4)
        self.assertTrue(np.allclose(dpsf0[4:], dpsf4))

        # Convert Zernikes
        zkPsf0 = convertZernikesToPsfWidth([0, 0, 0, 0] + list(zkCoeff), jmin=0)
        zkPsf4 = convertZernikesToPsfWidth(zkCoeff, jmin=4)
        self.assertTrue(np.allclose(zkPsf0[4:], zkPsf4))

        # Zernike x parity
        xParity0 = getZernikeParity(jmin=0, axis="x")
        xParity4 = getZernikeParity(jmin=4, axis="x")
        self.assertTrue(np.allclose(xParity0[4:], xParity4))

        # Zernike y parity
        yParity0 = getZernikeParity(jmin=0, axis="y")
        yParity4 = getZernikeParity(jmin=4, axis="y")
        self.assertTrue(np.allclose(yParity0[4:], yParity4))

    def testSparseDense(self):
        vals = np.arange(4, 23)
        indices = np.array([4, 5, 6, 17])

        # Get sparse values
        sparse = makeSparse(vals, indices)
        np.testing.assert_array_equal(sparse, indices)

        # Make dense without explicit jmax
        dense = makeDense(sparse, indices)
        np.testing.assert_array_equal(dense[indices - 4], indices)
        self.assertEqual(dense[-1], indices[-1])

        # Make dense with explicit jmax
        dense = makeDense(sparse, indices, 22)
        np.testing.assert_array_equal(dense[indices - 4], vals[indices - 4])
        self.assertEqual(len(dense), len(vals))
        self.assertEqual(dense[-1], 0)

        # Test fully dense does nothing
        np.testing.assert_array_equal(vals, makeSparse(vals, vals))
        np.testing.assert_array_equal(vals, makeDense(vals, vals))

        # Test that these functions work with floats
        floats = np.full_like(vals, 1e-9, dtype=float)
        np.testing.assert_array_equal(floats, makeSparse(floats, vals))
        np.testing.assert_array_equal(floats, makeDense(floats, vals))

        # Test bad indices
        for func in [makeSparse, makeDense]:
            with self.assertRaises(ValueError):
                func([1, 2, 3], [3, 4, 5])
            with self.assertRaises(ValueError):
                func([1, 2, 3], [4, 6, 5])
            with self.assertRaises(ValueError):
                func([1, 2, 3], [4, 6, 6])

    def testCheckNollIndices(self):
        # These values should all pass
        checkNollIndices(np.array([4]))
        checkNollIndices(np.array([4, 5, 6]))
        checkNollIndices(np.array([20, 21]))
        checkNollIndices(np.array([11, 20, 21, 22]))

        # The rest should fail...

        # < 4
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([3, 4, 5, 6]))
        # not unique
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 5, 6, 6]))
        # not ascending
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 6, 5]))

        # missing azimuthal pairs
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 5]))
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 5, 11]))
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 5, 20, 21]))
        with self.assertRaises(ValueError):
            checkNollIndices(np.array([4, 5, 6, 20]))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
