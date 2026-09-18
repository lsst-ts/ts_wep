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

"""`DonutSourceSelectorTask`'s Struct field name, which blitz does not own.

The cutout pipeline reads ``.sourceCat`` off the selector's result on both of
its selection paths.  That name is camelCase because it belongs to a non-blitz
task, and a snake-casing sweep over blitz once rewrote it to ``.source_cat`` --
which raised `AttributeError` inside the ``except Exception`` both call sites
wrap, so every detector logged "Donut selector failed" and returned zero
donuts.  A whole pipeline run looked successful and wrote an empty catalog.

These tests pin the attribute against the real task, on both call shapes the
cutout pipeline uses: the fluxless blitz-detection path and the refcat path.
Nothing here runs ISR, so it is the contract that is pinned, not the pipeline.
"""

import unittest

import numpy as np
from astropy.table import QTable

from lsst.obs.lsst import LsstCam
from lsst.ts.wep.blitz.utils import _REFCAT_COLUMNS
from lsst.ts.wep.task.donutSourceSelectorTask import (
    DonutSourceSelectorTask,
    DonutSourceSelectorTaskConfig,
)

# A real corner detector: the selector transforms centroids to field angles.
_DET_NAME = "R00_SW0"
# Both positions must clear two of the selector's own cuts, or the row counts
# below stop measuring the Struct field and start measuring selection.  The
# detector bbox is 4072x2000 and it is eroded by `unblendedSeparation` (160 px)
# to form the edge box, so stay well inside that margin; and the pair must be
# more than 160 px apart or the more central one claims the overlap and the
# other is dropped.
_XY = [(400.0, 400.0), (1800.0, 1600.0)]
_REF_FILTER = "phot_g_mean"


def _selector(allow_fluxless: bool) -> DonutSourceSelectorTask:
    """The selector configured the way both blitz tasks configure it.

    ``useCustomMagLimit`` matters on the refcat path and is not a convenience:
    both blitz `setDefaults` set it because the Monster refcat's filter names
    are full names (``phot_g_mean``), and the default policy lookup is keyed by
    band, so it raises `KeyError` on ``filterPHOT_G_MEAN``.
    """
    config = DonutSourceSelectorTaskConfig()
    config.allowFluxless = allow_fluxless
    config.useCustomMagLimit = True
    return DonutSourceSelectorTask(config=config)


def _blitz_detections() -> QTable:
    """What the blitz-detection path hands the selector.

    Mirrors `cutoutPipeline`: the `blitzDetect` columns, plus the
    refcat-provenance columns filled with NaN so a blitz-path donut is a row
    with NaN magnitudes rather than a row with a different schema.
    """
    table = QTable(
        {
            "donut_id": np.arange(1, len(_XY) + 1, dtype=np.int64),
            "centroid_x": np.array([x for x, _ in _XY]),
            "centroid_y": np.array([y for _, y in _XY]),
        }
    )
    for column in _REFCAT_COLUMNS:
        table[column] = np.full(len(table), np.nan)
    return table


def _refcat() -> QTable:
    """What the refcat path hands the selector: the same rows, with fluxes."""
    table = _blitz_detections()
    table[f"{_REF_FILTER}_flux"] = np.full(len(table), 1e-9)
    table["photo_mag"] = np.full(len(table), 14.0)
    table["astrom_mag"] = np.full(len(table), 14.0)
    return table


class TestSelectorResultFieldName(unittest.TestCase):
    """``.sourceCat`` is the selector's own spelling, so it stays camelCase."""

    def setUp(self) -> None:
        self.detector = LsstCam().getCamera()[_DET_NAME]

    def testBlitzPathResultHasSourceCat(self) -> None:
        """The fluxless call, which is the one that broke in production."""
        result = _selector(allow_fluxless=True).run(_blitz_detections(), self.detector, "")
        self.assertTrue(hasattr(result, "sourceCat"))
        self.assertEqual(len(result.sourceCat), len(_XY))

    def testRefcatPathResultHasSourceCat(self) -> None:
        result = _selector(allow_fluxless=False).run(_refcat(), self.detector, _REF_FILTER)
        self.assertTrue(hasattr(result, "sourceCat"))
        self.assertEqual(len(result.sourceCat), len(_XY))

    def testThereIsNoSnakeCaseAlias(self) -> None:
        """The bug was silent because the call sites catch `Exception`.

        Without this, a future sweep that rewrites ``.sourceCat`` again would
        not fail any test -- it would just produce empty catalogs.
        """
        result = _selector(allow_fluxless=True).run(_blitz_detections(), self.detector, "")
        self.assertFalse(hasattr(result, "source_cat"))


class TestCutoutPipelineReadsTheSelectorCorrectly(unittest.TestCase):
    """The attribute the cutout pipeline actually names, read off the source.

    Pins the call site rather than the task, so this fails if `cutoutPipeline`
    is edited to read a field the selector does not return -- which is the
    direction the regression came from.
    """

    def testEverySelectorAttributeReadExists(self) -> None:
        import inspect
        import re

        from lsst.ts.wep.blitz import cutoutPipeline

        source = inspect.getsource(cutoutPipeline)
        read_names = set(re.findall(r"select_task\.run\([^)]*\)\.([A-Za-z_]\w*)", source))
        self.assertTrue(read_names, "no select_task.run(...) attribute reads found")

        detector = LsstCam().getCamera()[_DET_NAME]
        result = _selector(allow_fluxless=True).run(_blitz_detections(), detector, "")
        for name in sorted(read_names):
            with self.subTest(attribute=name):
                self.assertTrue(
                    hasattr(result, name),
                    f"cutoutPipeline reads select_task.run(...).{name}, "
                    "which the selector's Struct does not carry",
                )


if __name__ == "__main__":
    unittest.main()
