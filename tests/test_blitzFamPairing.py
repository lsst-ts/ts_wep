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

"""Full-array-mode donut association and work-unit grouping.

The pairing is the part of FAM with a failure mode that does not announce itself:
FAM sees the same star on the same detector twice, so a mis-association silently
fits two *different* stars as an intra/extra pair rather than raising. These tests
pin both paths -- exact refcat-id matching, and the spatial fallback -- and in
particular that the fallback still works at the edge of the field, where the
radial defocus shift is ~27 px and an uncorrected tolerance would quietly stop
pairing whole rafts.
"""

import time
import unittest
import unittest.mock

import batoid
import numpy as np

from lsst.pipe.base import NoWorkFound, UnprocessableDataError

from lsst.ts.wep.blitz import famPipeline

from lsst.ts.wep.blitz.dataStructures import Donut
from lsst.ts.wep.blitz.donutBlitzFamTask import DonutBlitzFamTaskConfig, DonutBlitzFamTask
from lsst.ts.wep.blitz.famPipeline import (
    _RAD_PER_PIXEL,
    _fam_group_donuts,
    _pair_donuts,
)
from lsst.ts.wep.blitz.utils import _CALIB_STORE, _defocal_radial_scale

_INTRA_OFFSETS = (0.0, -1.5e-3, 0.0)
_EXTRA_OFFSETS = (0.0, +1.5e-3, 0.0)
_DONUT_RADIUS = 65.5  # px, at 1.5 mm camera defocus


def _donut(donut_id, visit_id, offsets, thx=0.0, thy=0.0, snr=500.0):
    """A Donut carrying only the fields pairing and grouping look at."""
    return Donut(
        det_name="R01_S00",
        stamp=None,
        thx_ccs=thx,
        thy_ccs=thy,
        flux=1e5,
        band="r",
        det_id=0,
        visit_id=visit_id,
        x_det=0.0,
        y_det=0.0,
        donut_id=donut_id,
        inner_frac=0.0,
        outer_frac=0.0,
        outer_sector_minmax_frac=0.0,
        donut_radius=_DONUT_RADIUS,
        snr=snr,
        bkg=0.0,
        bkg_std=1.0,
        n_quarter=0,
        photo_mag=float("nan"),
        astrom_mag=float("nan"),
        nearby_photo=[],
        nearby_astrom=[],
        defocal_offsets=offsets,
    )


def _defocused_angles(thx, thy):
    """Where a star at in-focus ``(thx, thy)`` lands on each side of focus.

    This is the input the pairing actually receives: the measured field angle of
    the *donut*, not of the star. Built from the same radial scale the pairing
    divides out, so a wrong scale would cancel here rather than show up --
    `TestRadialScale` is what pins the scale itself, against its known px values.
    """
    intra_scale = _defocal_radial_scale(_INTRA_OFFSETS)
    extra_scale = _defocal_radial_scale(_EXTRA_OFFSETS)
    return (thx * intra_scale, thy * intra_scale), (thx * extra_scale, thy * extra_scale)


class FamPairingTestCase(unittest.TestCase):
    """Shared setup: the batoid telescope the radial scale is derived from."""

    @classmethod
    def setUpClass(cls) -> None:
        # `_defocal_radial_scale` traces chief rays through
        # ``_CALIB_STORE["telescope"]``, which the task populates before forking.
        _CALIB_STORE.clear()
        _CALIB_STORE["telescope"] = batoid.Optic.fromYaml("LSST_r.yaml")

    @classmethod
    def tearDownClass(cls) -> None:
        _CALIB_STORE.clear()


class TestRadialScale(FamPairingTestCase):
    """The correction the spatial fallback depends on."""

    def testScaleIsOppositeEitherSideOfFocus(self) -> None:
        intra = _defocal_radial_scale(_INTRA_OFFSETS)
        extra = _defocal_radial_scale(_EXTRA_OFFSETS)
        # One side stretches and the other compresses, by nearly the same amount
        # -- which is why the same star does not land at the same pixel twice.
        # A negative optic shift (intra) pushes a chief ray outward, a positive
        # one (extra) pulls it in: +/-13.6 px at 1.725 deg for a 1.5 mm camera
        # shift, hence the ~27 px total separation.
        self.assertGreater(intra, 1.0)
        self.assertLess(extra, 1.0)
        self.assertAlmostEqual(intra - 1.0, 1.0 - extra, places=4)

    def testNullDefocusIsUnity(self) -> None:
        self.assertAlmostEqual(_defocal_radial_scale((0.0, 0.0, 0.0)), 1.0)

    def testSeparationGrowsLinearlyToTheKnownEdgeValue(self) -> None:
        """~27 px between the two sides at 1.725 deg, and linear in field angle."""
        seps = {}
        for deg in (0.0, 1.0, 1.725):
            (ix, _), (ex, _) = _defocused_angles(np.deg2rad(deg), 0.0)
            seps[deg] = abs(ex - ix) / _RAD_PER_PIXEL
        self.assertAlmostEqual(seps[0.0], 0.0, places=6)
        self.assertAlmostEqual(seps[1.725], 27.2, delta=0.5)
        # Pure scale => the separation is proportional to field angle.
        self.assertAlmostEqual(seps[1.725] / seps[1.0], 1.725, places=3)


class TestPairDonuts(FamPairingTestCase):
    """`_pair_donuts` on both paths."""

    def testRefcatIdFastPath(self) -> None:
        """Same refcat id => same star, regardless of where the donuts landed."""
        # Deliberately give the two sides *different* positions and swapped SNR
        # order, so an id match is the only thing that could pair them correctly.
        intra = [
            _donut(101, 1, _INTRA_OFFSETS, thx=0.01, snr=100.0),
            _donut(102, 1, _INTRA_OFFSETS, thx=0.02, snr=900.0),
        ]
        extra = [
            _donut(102, 2, _EXTRA_OFFSETS, thx=0.02, snr=100.0),
            _donut(101, 2, _EXTRA_OFFSETS, thx=0.01, snr=900.0),
        ]
        pairs, unmatched, path = _pair_donuts(intra, extra, 0.25, "refcat", "refcat")
        self.assertEqual(path, "refcat_id")
        self.assertEqual(unmatched, [])
        # extra first, matching corner mode's paired group order.
        for e, i in pairs:
            self.assertEqual(e.donut_id, i.donut_id)
            self.assertEqual(e.visit_id, 2)
            self.assertEqual(i.visit_id, 1)
        self.assertEqual({e.donut_id for e, _ in pairs}, {101, 102})

    def testRefcatIdLeavesUnmatchedOnBothSides(self) -> None:
        intra = [_donut(1, 1, _INTRA_OFFSETS), _donut(2, 1, _INTRA_OFFSETS)]
        extra = [_donut(2, 2, _EXTRA_OFFSETS), _donut(3, 2, _EXTRA_OFFSETS)]
        pairs, unmatched, path = _pair_donuts(intra, extra, 0.25, "refcat", "refcat")
        self.assertEqual(path, "refcat_id")
        self.assertEqual(len(pairs), 1)
        self.assertEqual({d.donut_id for d in unmatched}, {1, 3})

    def testBlindPathFallsBackToSpatial(self) -> None:
        """Blind detection renumbers per exposure, so ids must not be trusted."""
        # Ids deliberately disagree with position: id 1 intra is at the same sky
        # position as id 2 extra. Only spatial matching gets this right.
        thetas = [(np.deg2rad(1.7), 0.0), (0.0, np.deg2rad(1.7))]
        intra, extra = [], []
        for k, (thx, thy) in enumerate(thetas):
            (ix, iy), (ex, ey) = _defocused_angles(thx, thy)
            intra.append(_donut(k + 1, 1, _INTRA_OFFSETS, thx=ix, thy=iy))
            extra.append(_donut(2 - k, 2, _EXTRA_OFFSETS, thx=ex, thy=ey))
        pairs, unmatched, path = _pair_donuts(
            intra, extra, 0.25, "blind_selected", "blind_selected"
        )
        self.assertEqual(path, "spatial")
        self.assertEqual(unmatched, [])
        self.assertEqual(len(pairs), 2)
        for e, i in pairs:
            # Paired by position, so their ids do *not* agree.
            self.assertNotEqual(e.donut_id, i.donut_id)
            np.testing.assert_allclose(
                (e.thx_ccs / _defocal_radial_scale(_EXTRA_OFFSETS)),
                (i.thx_ccs / _defocal_radial_scale(_INTRA_OFFSETS)),
                atol=1e-9,
            )

    def testSpatialPairsAtTheFieldEdgeWhereTheShiftIsLargest(self) -> None:
        """The case the radial correction exists for.

        At 1.725 deg the same star's two donuts are ~27 px apart -- well outside a
        16 px tolerance -- so without dividing the shift out the outer rafts would
        simply stop pairing, with nothing in the logs to say why.
        """
        tol_frac = 0.25  # 0.25 * 65.5 px = 16.4 px, smaller than the 27 px shift
        (ix, iy), (ex, ey) = _defocused_angles(np.deg2rad(1.725), 0.0)
        intra = [_donut(1, 1, _INTRA_OFFSETS, thx=ix, thy=iy)]
        extra = [_donut(1, 2, _EXTRA_OFFSETS, thx=ex, thy=ey)]

        pairs, unmatched, _ = _pair_donuts(intra, extra, tol_frac, "blind", "blind")
        self.assertEqual(len(pairs), 1, "corrected pairing must work at the edge")
        self.assertEqual(unmatched, [])

        # Same donuts, but with the correction defeated by claiming both sides sit
        # at the same (null) defocus: now the 27 px shift is not removed and the
        # pair is lost. This is the regression the correction guards against.
        intra[0].defocal_offsets = (0.0, 0.0, 0.0)
        extra[0].defocal_offsets = (0.0, 0.0, 0.0)
        pairs, unmatched, _ = _pair_donuts(intra, extra, tol_frac, "blind", "blind")
        self.assertEqual(pairs, [])
        self.assertEqual(len(unmatched), 2)

    def testSpatialRejectsBeyondTolerance(self) -> None:
        """A star present on one side only must not be paired with a neighbour."""
        (ix, iy), _ = _defocused_angles(np.deg2rad(1.0), 0.0)
        # Put the extra donut 40 px away in the common frame -- more than the
        # 16.4 px tolerance, and not explicable by the radial shift.
        _, (ex, ey) = _defocused_angles(np.deg2rad(1.0) + 40 * _RAD_PER_PIXEL, 0.0)
        intra = [_donut(1, 1, _INTRA_OFFSETS, thx=ix, thy=iy)]
        extra = [_donut(1, 2, _EXTRA_OFFSETS, thx=ex, thy=ey)]
        pairs, unmatched, path = _pair_donuts(intra, extra, 0.25, "blind", "blind")
        self.assertEqual(path, "spatial")
        self.assertEqual(pairs, [])
        self.assertEqual(len(unmatched), 2)

    def testSpatialRequiresMutualNearestNeighbours(self) -> None:
        """Two donuts on one side must not both claim the same partner."""
        (ix, iy), (ex, ey) = _defocused_angles(np.deg2rad(1.0), 0.0)
        intra = [_donut(1, 1, _INTRA_OFFSETS, thx=ix, thy=iy)]
        extra = [
            _donut(1, 2, _EXTRA_OFFSETS, thx=ex, thy=ey),
            # A second extra donut 5 px away: closest partner is the same intra
            # donut, but it is not that donut's closest, so it stays unmatched.
            _donut(2, 2, _EXTRA_OFFSETS, thx=ex + 5 * _RAD_PER_PIXEL, thy=ey),
        ]
        pairs, unmatched, _ = _pair_donuts(intra, extra, 0.25, "blind", "blind")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].donut_id, 1)
        self.assertEqual([d.donut_id for d in unmatched], [2])

    def testOneSidedInputIsAllUnmatched(self) -> None:
        intra = [_donut(1, 1, _INTRA_OFFSETS), _donut(2, 1, _INTRA_OFFSETS)]
        pairs, unmatched, path = _pair_donuts(intra, [], 0.25, "refcat", "refcat")
        self.assertEqual(path, "empty")
        self.assertEqual(pairs, [])
        self.assertEqual(len(unmatched), 2)

    def testMixedSelectionSourcesUseSpatial(self) -> None:
        """One side on the refcat and one blind cannot share an id space."""
        (ix, iy), (ex, ey) = _defocused_angles(np.deg2rad(1.0), 0.0)
        intra = [_donut(9999, 1, _INTRA_OFFSETS, thx=ix, thy=iy)]
        extra = [_donut(1, 2, _EXTRA_OFFSETS, thx=ex, thy=ey)]
        pairs, _, path = _pair_donuts(intra, extra, 0.25, "refcat", "blind_selected")
        self.assertEqual(path, "spatial")
        self.assertEqual(len(pairs), 1)


class TestFamGrouping(FamPairingTestCase):
    """`_fam_group_donuts` across the four dispatch modes."""

    def _sides(self, n_intra=3, n_extra=3):
        intra = [_donut(k, 1, _INTRA_OFFSETS, thx=0.001 * k) for k in range(n_intra)]
        extra = [_donut(k, 2, _EXTRA_OFFSETS, thx=0.001 * k) for k in range(n_extra)]
        return intra, extra

    def _group(self, mode, intra, extra):
        return _fam_group_donuts(
            mode=mode,
            det_name="R01_S00",
            intra=intra,
            extra=extra,
            tol_frac=0.25,
            intra_source="refcat",
            extra_source="refcat",
            band="r",
            rtp_deg=None,
            alt_rad=None,
        )

    def testPairedIsTheOnlyModeThatPairs(self) -> None:
        # One extra donut with no partner: only `paired` can notice.
        intra, extra = self._sides(n_intra=3, n_extra=4)
        groups, unmatched, path = self._group("paired", intra, extra)
        self.assertEqual(path, "refcat_id")
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(g.donuts) == 2 for g in groups))
        self.assertEqual([d.donut_id for d in unmatched], [3])

        for mode in ("unpaired", "full_detector", "full_detector_pair"):
            _, unmatched, path = self._group(mode, intra, extra)
            self.assertEqual(unmatched, [], msg=mode)
            self.assertEqual(path, "n/a", msg=mode)

    def testUnpairedIsOneGroupPerDonut(self) -> None:
        intra, extra = self._sides()
        groups, _, _ = self._group("unpaired", intra, extra)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(len(g.donuts) == 1 for g in groups))

    def testFullDetectorIsOneGroupPerSide(self) -> None:
        intra, extra = self._sides()
        groups, _, _ = self._group("full_detector", intra, extra)
        self.assertEqual(len(groups), 2)
        # Each group holds one side of focus and nothing from the other.
        for g in groups:
            self.assertEqual(len({d.visit_id for d in g.donuts}), 1)
            self.assertEqual(len(g.donuts), 3)

    def testFullDetectorPairIsOneJointGroupOverBothSides(self) -> None:
        """Unpaired by design: each donut's own offsets tell the fit its side."""
        intra, extra = self._sides(n_intra=3, n_extra=4)
        groups, unmatched, _ = self._group("full_detector_pair", intra, extra)
        self.assertEqual(len(groups), 1)
        # Every donut is in, including the one that has no partner.
        self.assertEqual(len(groups[0].donuts), 7)
        self.assertEqual(unmatched, [])
        self.assertEqual(
            {d.defocal_offsets for d in groups[0].donuts},
            {_INTRA_OFFSETS, _EXTRA_OFFSETS},
        )

    def testEmptySideIsSkippedNotEmitted(self) -> None:
        """An empty group would fit nothing but still report failure."""
        intra, _ = self._sides()
        for mode, expected in (
            ("paired", 0),
            ("unpaired", 3),
            ("full_detector", 1),
            ("full_detector_pair", 1),
        ):
            groups, _, _ = self._group(mode, intra, [])
            self.assertEqual(len(groups), expected, msg=mode)
        for mode in ("paired", "unpaired", "full_detector", "full_detector_pair"):
            groups, _, _ = self._group(mode, [], [])
            self.assertEqual(groups, [], msg=mode)

    def testGroupIdsAreUniqueAndCarryTheDetector(self) -> None:
        intra, extra = self._sides()
        for mode in ("paired", "unpaired", "full_detector", "full_detector_pair"):
            groups, _, _ = self._group(mode, intra, extra)
            ids = [g.group_id for g in groups]
            self.assertEqual(len(ids), len(set(ids)), msg=mode)
            self.assertTrue(all(i.startswith("R01_S00") for i in ids), msg=mode)

    def testUnknownModeRaises(self) -> None:
        intra, extra = self._sides()
        with self.assertRaises(ValueError):
            self._group("full_corner", intra, extra)


class TestFamOffsetSigns(unittest.TestCase):
    """The defocus triplets the task hands its donuts."""

    def testExtraIsPositiveAndIntraNegative(self) -> None:
        """`extra -> +offset`, matching corner mode; the likeliest sign error."""
        config = DonutBlitzFamTaskConfig()
        task = DonutBlitzFamTask(config=config)
        extra, intra = task._extraFocalOffsets, task._intraFocalOffsets
        # Camera offset by default, detector and M2 unused.
        self.assertEqual(extra, (0.0, config.cameraOffset, 0.0))
        self.assertEqual(intra, (0.0, -config.cameraOffset, 0.0))
        self.assertGreater(extra[1], 0.0)

    def testAllThreeComponentsAreSigned(self) -> None:
        config = DonutBlitzFamTaskConfig()
        config.detectorOffset = 1e-3
        config.cameraOffset = 2e-3
        config.m2Offset = 3e-3
        task = DonutBlitzFamTask(config=config)
        self.assertEqual(task._extraFocalOffsets, (1e-3, 2e-3, 3e-3))
        self.assertEqual(task._intraFocalOffsets, (-1e-3, -2e-3, -3e-3))


class TestWorkerNeverDies(unittest.TestCase):
    """A worker that dies instead of returning hangs the whole quantum.

    ``multiprocessing.Pool`` respawns a dead worker but never re-queues the task
    it was holding, so ``imap`` waits for a result that will never be produced --
    forever, with the pool looking perfectly healthy from outside. On top of
    that, the dying child's interpreter shutdown destroys the
    ``QuantumBackedButler`` datastore-records sqlite database that all the
    workers inherited across the fork, so every detector dispatched after it
    fails too.
    """

    def setUp(self):
        self._saved = dict(_CALIB_STORE)

    def tearDown(self):
        _CALIB_STORE.clear()
        _CALIB_STORE.update(self._saved)

    def _run_with_store_raising(self, exc):
        """Make the worker's very first ``_CALIB_STORE`` lookup raise ``exc``."""

        class Raiser(dict):
            def __getitem__(self, key):
                raise exc

        _CALIB_STORE.clear()
        _CALIB_STORE.update(Raiser())
        # Swap the module global itself: the worker reads it by name.
        with unittest.mock.patch.object(famPipeline, "_CALIB_STORE", Raiser()):
            return famPipeline._fam_detector_worker((42, time.time()))

    def testNoWorkFoundIsSkippedNotRaised(self):
        out = self._run_with_store_raising(
            UnprocessableDataError("Back-side bias voltage is turned off for R20_S20")
        )
        self.assertTrue(out["skipped"])
        self.assertIn("UnprocessableDataError", out["error"])
        self.assertEqual(out["det_id"], 42)

    def testOrdinaryExceptionIsAFailureNotASkip(self):
        out = self._run_with_store_raising(RuntimeError("boom"))
        self.assertFalse(out["skipped"])
        self.assertIn("RuntimeError", out["error"])

    def testBareBaseExceptionStillReturns(self):
        # Anything that is not KeyboardInterrupt/SystemExit must come back as a
        # dict rather than killing the child.
        out = self._run_with_store_raising(BaseException("naked"))
        self.assertFalse(out["skipped"])
        self.assertIn("BaseException", out["error"])

    def testKeyboardInterruptStillPropagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run_with_store_raising(KeyboardInterrupt())

    def testUnprocessableDataErrorIsNotAnException(self):
        # The whole trap: `except Exception` cannot catch this, which is why the
        # guard in the worker has to be written against BaseException.
        self.assertTrue(issubclass(UnprocessableDataError, NoWorkFound))
        self.assertFalse(issubclass(UnprocessableDataError, Exception))


if __name__ == "__main__":
    unittest.main()
