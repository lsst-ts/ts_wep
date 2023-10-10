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
from astropy.io import fits
from lsst.ts.wep.utils import (
    ZernikeAnnularEval,
    ZernikeAnnularFit,
    ZernikeAnnularGrad,
    ZernikeAnnularJacobian,
    ZernikeMaskedFit,
    getModulePath,
)
from lsst.utils.tests import TestCase
from scipy.integrate import nquad


class TestZernikeUtils(TestCase):
    """Test the Zernike utility functions."""

    def setUp(self):
        self.testDataDir = os.path.join(
            getModulePath(), "tests", "testData", "cwfsZernike"
        )

        # Generate the mesh of x, y-coordinate
        point = 400
        ratio = 0.9
        self.xx, self.yy = self._genGridXy(point, ratio)

        self.obscuration = 0.61

        numOfZk = 22
        self.zerCoef = np.arange(1, 1 + numOfZk) * 0.1

    def _genGridXy(self, point, ratio):
        yy, xx = np.mgrid[
            -(point / 2 - 0.5) : (point / 2 + 0.5),
            -(point / 2 - 0.5) : (point / 2 + 0.5),
        ]

        xx = xx / (point * ratio / 2)
        yy = yy / (point * ratio / 2)

        return xx, yy

    def testZernikeAnnularEval(self):
        surface = ZernikeAnnularEval(self.zerCoef, self.xx, self.yy, self.obscuration)

        self._checkAnsWithFile(surface, "annularZernikeEval.txt")

    def _checkAnsWithFile(self, value, ansFileName, rtol=1e-8, atol=1e-8):
        ansFilePath = os.path.join(self.testDataDir, ansFileName)
        ans = np.loadtxt(ansFilePath)

        self.assertFloatsAlmostEqual(value, ans, rtol=rtol, atol=atol)

    def testZernikeAnnularNormality(self):
        ansValue = np.pi * (1 - self.obscuration**2)
        for ii in range(28):
            z = np.zeros(28)
            z[ii] = 1

            normalization = nquad(
                self._genNormalizedFunc,
                [[self.obscuration, 1], [0, 2 * np.pi]],
                args=(z, self.obscuration),
            )[0]

            self.assertAlmostEqual(normalization, ansValue)

    def _genNormalizedFunc(self, r, theta, z, e):
        func = r * ZernikeAnnularEval(z, r * np.cos(theta), r * np.sin(theta), e) ** 2

        return func

    def testZernikeAnnularOrthogonality(self):
        # Check the orthogonality for Z1 - Z28
        for jj in range(28):
            z1 = np.zeros(28)
            z1[jj] = 1
            for ii in range(28):
                if ii != jj:
                    z2 = np.zeros(28)
                    z2[ii] = 1

                    orthogonality = nquad(
                        self._genOrthogonalFunc,
                        [[self.obscuration, 1], [0, 2 * np.pi]],
                        args=(z1, z2, self.obscuration),
                    )[0]

                    self.assertAlmostEqual(orthogonality, 0)

    def _genOrthogonalFunc(self, r, theta, z1, z2, e):
        func = (
            r
            * ZernikeAnnularEval(z1, r * np.cos(theta), r * np.sin(theta), e)
            * ZernikeAnnularEval(z2, r * np.cos(theta), r * np.sin(theta), e)
        )

        return func

    def testZernikeAnnularGradDx(self):
        surfGrad = ZernikeAnnularGrad(
            self.zerCoef, self.xx, self.yy, self.obscuration, "dx"
        )

        self._checkAnsWithFile(surfGrad, "annularZernikeGradDx.txt")

    def testZernikeAnnularGradDy(self):
        surfGrad = ZernikeAnnularGrad(
            self.zerCoef, self.xx, self.yy, self.obscuration, "dy"
        )

        self._checkAnsWithFile(surfGrad, "annularZernikeGradDy.txt")

    def testZernikeAnnularGradDx2(self):
        surfGrad = ZernikeAnnularGrad(
            self.zerCoef, self.xx, self.yy, self.obscuration, "dx2"
        )

        self._checkAnsWithFile(surfGrad, "annularZernikeGradDx2.txt")

    def testZernikeAnnularGradDy2(self):
        surfGrad = ZernikeAnnularGrad(
            self.zerCoef, self.xx, self.yy, self.obscuration, "dy2"
        )

        self._checkAnsWithFile(surfGrad, "annularZernikeGradDy2.txt")

    def testZernikeAnnularGradDxy(self):
        surfGrad = ZernikeAnnularGrad(
            self.zerCoef, self.xx, self.yy, self.obscuration, "dxy"
        )

        self._checkAnsWithFile(surfGrad, "annularZernikeGradDxy.txt")

    def testZernikeAnnularGradWrongAxis(self):
        with self.assertRaises(ValueError):
            ZernikeAnnularGrad(
                self.zerCoef, self.xx, self.yy, self.obscuration, "wrongAxis"
            )

    def testZernikeAnnularJacobian1st(self):
        annuZerJacobian = ZernikeAnnularJacobian(
            self.zerCoef, self.xx, self.yy, self.obscuration, "1st"
        )

        self._checkAnsWithFile(annuZerJacobian, "annularZernikeJaco1st.txt")

    def testZernikeAnnularJacobian2nd(self):
        annuZerJacobian = ZernikeAnnularJacobian(
            self.zerCoef, self.xx, self.yy, self.obscuration, "2nd"
        )

        self._checkAnsWithFile(annuZerJacobian, "annularZernikeJaco2nd.txt")

    def testZernikeAnnularJacobianWrongType(self):
        with self.assertRaises(ValueError):
            ZernikeAnnularJacobian(
                self.zerCoef, self.xx, self.yy, self.obscuration, "wrongType"
            )

    def testZernikeAnnularFit(self):
        opdFitsFile = os.path.join(self.testDataDir, "sim6_iter0_opd0.fits.gz")
        opd = fits.getdata(opdFitsFile)

        # x-, y-coordinate in the OPD image
        opdSize = opd.shape[0]
        opdGrid1d = np.linspace(-1, 1, opdSize)
        opdx, opdy = np.meshgrid(opdGrid1d, opdGrid1d)

        idx = opd != 0
        coef = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], 22, self.obscuration)

        ansOpdFileName = "sim6_iter0_opd.zer"
        ansOpdFilePath = os.path.join(self.testDataDir, ansOpdFileName)
        allOpdAns = np.loadtxt(ansOpdFilePath)
        self.assertLess(np.sum(np.abs(coef - allOpdAns[0, :])), 1e-10)

    def testZernikeMaskFit(self):
        e = 0.2
        nc = 6
        surface = ZernikeAnnularEval(self.zerCoef[0:nc], self.xx, self.yy, e)

        # mask data
        cut = -0.9
        r = np.sqrt(self.xx**2 + self.yy**2)
        idx = (r > 1) | (r < e) | (self.xx < cut)

        xx = self.xx[:].copy()
        yy = self.yy[:].copy()
        xx[idx] = np.nan
        yy[idx] = np.nan
        mask = ~np.isnan(xx)

        zr = ZernikeMaskedFit(surface, xx, yy, nc, mask, e)

        self.assertLess(np.sum(np.abs(zr - self.zerCoef[0:nc]) ** 2), 1e-10)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
