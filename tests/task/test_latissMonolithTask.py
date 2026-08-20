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

"""Tests for LatissMonolithTask.

Split in two, because ``tests/testData/gen3TestRepo`` contains only LSSTCam
and LSSTComCam -- there is no LATISS data to run a self-contained pipeline
against:

* ``TestLatissMonolithTaskConfig`` needs no data at all. It checks the
  configuration, connections and table assembly, and therefore runs in CI.
* ``TestLatissMonolithTaskOnSky`` runs the real chain and is skipped unless
  ``/repo/main`` and a butler password are available, following
  ``test_calcZernikesTieTaskLatiss.py``.
"""

import os
import types
import unittest

import astropy.units as u
import numpy as np
import pytest

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.ts.wep.task.latissMonolithTask import (
    LatissMonolithTask,
    LatissMonolithTaskConfig,
    LatissMonolithTaskConnections,
)

# The pair used throughout: a BLOCK-T743 CWFS sequence on 20260625 with visit
# records defined. Note LATISS visits are not defined for the most recent
# nights even though the raws exist, so a pair must be chosen with care.
EXPOSURE_INTRA = 2026062500012
EXPOSURE_EXTRA = 2026062500013

NOLL_INDICES = list(range(4, 23))


class TestLatissMonolithTaskConfig(lsst.utils.tests.TestCase):
    """Configuration, connections and table assembly. Needs no butler."""

    def setUp(self) -> None:
        self.config = LatissMonolithTaskConfig()

    def testDefaultsAreAuxTelSpecific(self) -> None:
        """The AuxTel values that the fit depends on must be the defaults."""
        # onAxis is required: there is no off-axis batoid fit for AuxTel.
        self.assertEqual(self.config.opticalModel, "onAxis")
        # 228 px is what latiss_wep_align derives for dz=0.8. The ts_wep
        # default of 160 is LSSTCam-sized and clips a 194 px AuxTel donut.
        self.assertEqual(self.config.donutDiameter, 228)
        self.assertEqual(list(self.config.nollIndices), NOLL_INDICES)

    def testIsrDefaultsAreGainsAndOverscanOnly(self) -> None:
        """LATISS alignment sequences have no usable bias/dark/flat."""
        self.assertTrue(self.config.isrTask.doApplyGains)
        self.assertTrue(self.config.isrTask.doOverscan)
        self.assertEqual(self.config.isrTask.overscan.fitType, "MEDIAN_PER_ROW")
        for field in ("doBias", "doDark", "doFlat", "doDefect", "doLinearize", "doCrosstalk"):
            self.assertFalse(getattr(self.config.isrTask, field), field)

    def testBoresightToleranceIsInArcsecNotPixels(self) -> None:
        """Regression guard: the unit of maxDistanceFromBoresight.

        ``run_wep`` measures this distance with ``calculate_xy_offsets``, which
        applies the 0.09569 arcsec/px plate scale, so its default of 500 means
        500 *arcsec* (~5225 px). Interpreting it as pixels makes the cut ~10x
        too strict and rejects perfectly usable pairs -- two of 23 in the first
        multi-pair run, both of which ``run_wep`` would have accepted.
        """
        from lsst.ts.wep.task.latissMonolithTask import LATISS_PIXEL_SCALE

        self.assertAlmostEqual(LATISS_PIXEL_SCALE, 0.09569)
        self.assertEqual(self.config.maxDistanceFromBoresight, 500.0)
        # 500 arcsec must be most of the 4072 px detector, not a tight cut.
        self.assertGreater(self.config.maxDistanceFromBoresight / LATISS_PIXEL_SCALE, 4000.0)

    def testPairerSeparationIsInvertedForLatiss(self) -> None:
        """AuxTel moves M2, so extra-focal has the SMALLER focusZ.

        This is inverted relative to LSSTCam. pairTask hardcodes the same value
        for LATISS, but the task pins it so the pairing cannot drift.
        """
        self.assertTrue(self.config.pairer.doOverrideSeparation)
        self.assertEqual(self.config.pairer.overrideSeparation, -0.8)

    def testConnections(self) -> None:
        """Dimensions must be (instrument, detector), not visit.

        The task consumes two exposures and pairs them internally, so the
        quantum cannot be keyed on a single visit.
        """
        connections = LatissMonolithTaskConnections(config=self.config)
        self.assertEqual(set(connections.dimensions), {"instrument", "detector"})
        self.assertEqual(set(connections.inputs), {"raws"})
        self.assertEqual(set(connections.prerequisiteInputs), {"camera"})
        self.assertEqual(set(connections.outputs), {"zernikes", "donutStampsExtra", "donutStampsIntra"})
        # No intrinsicZernikes connection: LATISS has no such calibration.
        self.assertFalse(hasattr(connections, "intrinsicZernikes"))
        # And no refcat/astrometry: a LATISS exposure has one bright donut.
        self.assertFalse(hasattr(connections, "refCat"))

    def testDoSaveStampsRemovesStampOutputs(self) -> None:
        self.config.doSaveStamps = False
        connections = LatissMonolithTaskConnections(config=self.config)
        self.assertEqual(set(connections.outputs), {"zernikes"})

    def testStampSizePropagatesToCutoutSubtask(self) -> None:
        """donutDiameter is the single place the stamp size is set."""
        self.config.donutDiameter = 200
        task = LatissMonolithTask(config=self.config)
        self.assertEqual(task.cutOutDonuts.config.donutStampSize, 200)
        self.assertEqual(task.cutOutDonuts.config.opticalModel, "onAxis")
        self.assertEqual(task.cutOutDonuts.config.initialCutoutPadding, 40)

    def testZkTableSchemaAndFailedFit(self) -> None:
        """A failed fit becomes a NaN row that is excluded from the average."""
        task = LatissMonolithTask(config=self.config)
        nan = np.full(len(NOLL_INDICES), np.nan)
        failed = dict(
            zk_sum=nan,
            zk_fit=nan,
            zernikes_nm={j: np.nan for j in NOLL_INDICES},
            noll_indices=np.array(NOLL_INDICES),
            fwhm=np.nan,
            cost=np.nan,
            nfev=0,
            success=False,
        )
        good = dict(failed, zk_sum=np.full(len(NOLL_INDICES), 1e-7), cost=100.0, fwhm=1.5, success=True)

        stamps = _FakeStamps([_FakeStamp(), _FakeStamp()])
        table = task._makeZkTable([good, failed], stamps, stamps)

        self.assertEqual(len(table), 3)  # average + 2 pairs
        self.assertEqual(list(table["label"]), ["average", "pair1", "pair2"])
        self.assertEqual(list(table["used"]), [True, True, False])
        self.assertEqual(list(table["fit_success"]), [True, True, False])

        # The QA columns that CalcZernikesTask does not have.
        for column in ("cost", "fwhm", "nfev", "fit_success"):
            self.assertIn(column, table.colnames)

        # 1e-7 m == 100 nm, and the failed pair must not drag the average.
        self.assertAlmostEqual(table["Z4"][0].to_value(u.nm), 100.0, places=3)
        self.assertTrue(np.isnan(table["Z4"][2].to_value(u.nm)))

        # LATISS has no intrinsic Zernike calibration, so these are NaN by
        # design rather than by accident.
        self.assertTrue(np.all(np.isnan(table["Z4_intrinsic"].to_value(u.nm))))
        self.assertTrue(np.all(np.isnan(table["Z4_deviation"].to_value(u.nm))))

        self.assertEqual(table.meta["noll_indices"], NOLL_INDICES)
        self.assertEqual(table.meta["opd_columns"], [f"Z{j}" for j in NOLL_INDICES])
        self.assertTrue(table.meta["opd_zk_ref"])
        self.assertTrue(table.meta["peak_normalized_stamps"])

    def testMaxFitCostRejectsHighCostPairs(self) -> None:
        self.config.maxFitCost = 50.0
        task = LatissMonolithTask(config=self.config)
        result = dict(
            zk_sum=np.full(len(NOLL_INDICES), 1e-7),
            zk_fit=np.zeros(len(NOLL_INDICES)),
            zernikes_nm={j: 100.0 for j in NOLL_INDICES},
            noll_indices=np.array(NOLL_INDICES),
            fwhm=1.5,
            cost=100.0,  # above maxFitCost
            nfev=10,
            success=True,
        )
        stamps = _FakeStamps([_FakeStamp()])
        table = task._makeZkTable([result], stamps, stamps)
        # The fit succeeded but is rejected by the quality cut, so it is not
        # used and the average is NaN.
        self.assertTrue(table["fit_success"][1])
        self.assertFalse(table["used"][1])
        self.assertTrue(np.isnan(table["Z4"][0].to_value(u.nm)))


class _FakeStamp:
    """Minimal stand-in for DonutStamp, for table-assembly tests."""

    class _Point:
        x = 100.0
        y = 200.0

    centroid_position = _Point()
    detector_name = "RXX_S00"
    cam_name = "LATISS"

    def calcFieldXY(self) -> tuple[float, float]:
        return (0.0, 0.0)


class _FakeStamps(list):
    """A DonutStamps-like list carrying the metadata the table builder
    reads.
    """

    metadata: dict = {}


@pytest.mark.skipif(
    not os.path.exists("/sdf/data/rubin/repo/main"),
    reason="requires access to data in /repo/main",
)
@pytest.mark.skipif(not os.getenv("PGPASSFILE"), reason="requires access to butler db")
class TestLatissMonolithTaskOnSky(lsst.utils.tests.TestCase):
    """Run the real ISR -> QFM -> cutout -> danish chain on one on-sky pair."""

    repoDir = "/sdf/data/rubin/repo/main"

    @classmethod
    def setUpClass(cls) -> None:
        from lsst.obs.lsst import Latiss

        cls.butler = Butler.from_config(cls.repoDir, collections=["LATISS/defaults"])
        cls.camera = Latiss.getCamera()
        cls.rawIntra = cls.butler.get("raw", instrument="LATISS", exposure=EXPOSURE_INTRA, detector=0)
        cls.rawExtra = cls.butler.get("raw", instrument="LATISS", exposure=EXPOSURE_EXTRA, detector=0)

        task = LatissMonolithTask(config=LatissMonolithTaskConfig())
        cls.result = task.run(cls.rawExtra, cls.rawIntra, cls.camera)

    def testStampsAreTheConfiguredSize(self) -> None:
        for stamps in (self.result.donutStampsExtra, self.result.donutStampsIntra):
            self.assertEqual(len(stamps), 1)
            self.assertEqual(stamps[0].stamp_im.image.array.shape, (228, 228))

    def testFitSucceeds(self) -> None:
        """The whole point: the fit converges rather than returning NaN.

        Stock ts_wep 17.8.1 returns NaN on 11 of 12 LATISS pairs.
        """
        table = self.result.zernikes
        pair = table[table["label"] == "pair1"]
        self.assertTrue(bool(pair["fit_success"][0]))
        self.assertTrue(np.isfinite(pair["Z4"][0].to_value(u.nm)))
        # A converged fit takes many function evaluations; the degenerate
        # zkRef regression produced nfev=1.
        self.assertGreater(int(pair["nfev"][0]), 5)
        # And the fitted seeing must be physical.
        self.assertTrue(0.1 < pair["fwhm"][0].to_value(u.arcsec) < 5.0)

    def testZernikesAreOfPlausibleMagnitude(self) -> None:
        """Guards against the 43.5x zkRef regression returning silently.

        LATISS low-order aberrations are hundreds of nm, not tens of microns.
        """
        table = self.result.zernikes
        pair = table[table["label"] == "pair1"]
        for j in (4, 7, 8):
            self.assertLess(abs(pair[f"Z{j}"][0].to_value(u.nm)), 5000.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: types.ModuleType) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
