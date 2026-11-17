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

"""NaN-centroid handling in the donut flux measurement.

A non-finite centroid -- a detection that never produced a position -- has no
stamp to measure, and ``round()`` raises on NaN, so one such row would crash
the measurement of every other donut.  The guard must emit NaNs for that row
and measure the rest.
"""

from __future__ import annotations

import unittest

import numpy as np
from astropy.table import QTable

import lsst.afw.image as afwImage
from lsst.ts.wep.blitz.measureDonutCandidates import (
    MeasureDonutCandidatesConfig,
    MeasureDonutCandidatesTask,
)

# Small radius so the half-window (round(radius * bkgAnnulusOuterFrac) = 14)
# fits well inside a tiny exposure around a centred donut.
_DONUT_RADIUS = 10.0


def _exposure(size: int = 100) -> afwImage.ExposureF:
    """A flat exposure; every in-bounds aperture reads a finite flux."""
    exposure = afwImage.ExposureF(size, size)
    exposure.image.array[:] = 1.0
    return exposure


class TestMeasureDonutCandidatesNaN(unittest.TestCase):
    """A single NaN centroid must not take down the whole measurement."""

    def setUp(self) -> None:
        self.task = MeasureDonutCandidatesTask(config=MeasureDonutCandidatesConfig())

    def _run(self, selections: QTable) -> QTable:
        return self.task.run(_exposure(), selections, donut_radius=_DONUT_RADIUS).measurements

    def testNonFiniteCentroidIsNaNWhileNeighboursSurvive(self) -> None:
        """The finite donut is measured; the non-finite ones are NaN."""
        selections = QTable(
            {
                "donut_id": [1, 2, 3],
                "centroid_x": [50.0, np.nan, 50.0],
                "centroid_y": [50.0, 50.0, np.nan],
            }
        )
        out = self._run(selections)
        # The finite, well-centred donut got a real measurement.
        self.assertTrue(np.isfinite(out["flux"][0]))
        # A NaN in either coordinate yields a NaN row rather than a crash.
        self.assertTrue(np.isnan(out["flux"][1]))
        self.assertTrue(np.isnan(out["flux"][2]))
        self.assertTrue(np.isnan(out["bkg"][1]))
        self.assertTrue(np.isnan(out["snr"][2]))

    def testAllNonFiniteCentroidsDoNotRaise(self) -> None:
        """The degenerate all-NaN input is exactly what the guard is for."""
        selections = QTable(
            {"donut_id": [1, 2], "centroid_x": [np.nan, np.nan], "centroid_y": [np.nan, np.nan]}
        )
        out = self._run(selections)
        self.assertTrue(np.all(np.isnan(out["flux"])))


if __name__ == "__main__":
    unittest.main()
