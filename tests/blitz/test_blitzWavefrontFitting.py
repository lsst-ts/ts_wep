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

"""Unit tests for `lsst.ts.wep.blitz.wavefrontFitting` helpers."""

from __future__ import annotations

import unittest

import numpy as np

from lsst.ts.wep.blitz.wavefrontFitting import _bkg_free_model


class _FakeMultiDonutModel:
    """Stands in for `danish.DZMultiDonutModel`.

    Returns a per-donut stack shaped like the real ``model``: an ``ndarray``
    of shape (nstar, npix, npix) whose slices are distinct, so the caller's
    donut selection is observable.
    """

    def __init__(self, nstar: int, npix: int, nbkg: int) -> None:
        self.nstar = nstar
        self.npix = npix
        self.nbkg = nbkg
        self.stack = np.arange(nstar * npix * npix, dtype=float).reshape(nstar, npix, npix)

    def model(self, **kwargs) -> np.ndarray:
        return self.stack


class TestBkgFreeModel(unittest.TestCase):
    """`_bkg_free_model`, the fitted-background removal helper."""

    def testHigherOrderSelectsSingleDonut(self) -> None:
        """``bkgOrder > 0`` must return just ``donut_idx``'s image.

        Danish's `DZMultiDonutModel.model` returns an ``ndarray`` of shape
        (nstar, npix, npix), not a ``list``, so a ``list`` membership test
        cannot be what picks the donut out of the stack.
        """
        nstar, npix, nbkg = 3, 4, 6
        donut_idx = 1
        danish_model = _FakeMultiDonutModel(nstar, npix, nbkg)
        model_img = np.ones((npix, npix), dtype=float)
        # "fluxes" present marks the multi-donut (is_multi) case.
        fit_params = {
            "fluxes": [10.0, 20.0, 30.0],
            "bkgs": [[1.0] * nbkg] * nstar,
        }

        out = _bkg_free_model(model_img, danish_model, fit_params, donut_idx, bkg_order=1)

        # The one donut's image, not the whole (nstar, npix, npix) stack.
        self.assertEqual(out.shape, (npix, npix))
        np.testing.assert_array_equal(out, danish_model.stack[donut_idx])


if __name__ == "__main__":
    unittest.main()
