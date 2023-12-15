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
    convertZernikesToPsfWidth,
    createZernikeBasis,
    createZernikeGradBasis,
    getModulePath,
    getPsfGradPerZernike,
    getZernikeParity,
    zernikeEval,
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

    def testGetZernikeParity(self):
        xParity = getZernikeParity(22)
        xTruth = [1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1]
        self.assertTrue(all(xParity == xTruth))

        yParity = getZernikeParity(22, axis="y")
        yTruth = [1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
        self.assertTrue(all(yParity == yTruth))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
