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

"""`DonutBlitzCornerTask` helpers that do not need the full pipeline.

These exercise methods of the corner task in isolation -- calling them unbound
against a lightweight stand-in -- so they stay cheap enough to run without
standing up every ISR and fitting subtask.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from lsst.ts.wep.blitz.dataStructures import Donut
from lsst.ts.wep.blitz.donutBlitzCorner import DonutBlitzCornerTask
from lsst.ts.wep.blitz.utils import _ZK_JMAX


def _minimalDonut(**overrides: Any) -> Donut:
    """A Donut carrying just enough for `_annotateDonuts` to read it."""
    kwargs: dict[str, Any] = dict(
        det_name="R00_SW0",
        stamp=np.ones((167, 167), dtype=float),
        thx_ccs=0.0,
        thy_ccs=0.0,
        flux=1.0,
        band="r",
        det_id=191,
        visit_id=1,
        x_det=0.0,
        y_det=0.0,
        donut_id=1,
        inner_frac=0.0,
        outer_frac=0.0,
        outer_asym=0.0,
        donut_radius=60.0,
        snr=100.0,
        bkg=0.0,
        bkg_std=1.0,
        n_quarter=0,
        photo_mag=float("nan"),
        astrom_mag=float("nan"),
        nearby_photo=[],
        nearby_astrom=[],
    )
    kwargs.update(overrides)
    return Donut(**kwargs)


class _RecordingCalib:
    """Stand-in `lsst.ip.isr.IntrinsicZernikes` recording its query points.

    `_annotateDonuts` reads the calibration only through
    `getIntrinsicZernikes`, so a recorder is enough to pin which field point
    and rotator angle the donut's intrinsics were evaluated at.  The signature
    mirrors the real calib exactly -- positional ``field_x``/``field_y`` in
    degrees, ``rotTelPos`` in degrees, and snake_case ``noll_indices`` -- so a
    call passing the wrong keyword fails here the way it would in production.
    """

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def getIntrinsicZernikes(
        self,
        field_x: float,
        field_y: float,
        rotTelPos: float = 0.0,
        noll_indices: Any = None,
    ) -> np.ndarray:
        self.calls.append(
            SimpleNamespace(
                field_x=field_x,
                field_y=field_y,
                rotTelPos=rotTelPos,
                noll_indices=noll_indices,
            )
        )
        return np.full((1, len(noll_indices)), 7.0)


class _FieldDependentNanCalib:
    """Intrinsics that extrapolate to NaN only off one side of the field.

    A real field-angle extrapolation failure strikes some field points and not
    others.  Here any donut at negative ``field_x`` comes back with NaN past
    the first four terms; the rest stay finite, so one detector can carry both
    a healthy and a poisoned donut.
    """

    def getIntrinsicZernikes(
        self,
        field_x: float,
        field_y: float,
        rotTelPos: float = 0.0,
        noll_indices: Any = None,
    ) -> np.ndarray:
        zk = np.full((1, len(noll_indices)), 7.0)
        if field_x < 0.0:
            zk[0, 4:] = np.nan
        return zk


class TestAnnotateDonutsIntrinsics(unittest.TestCase):
    """`_annotateDonuts` evaluates intrinsics at the rotated field point.

    The intrinsic Zernikes depend on the rotator angle, so a donut's field
    angle alone is not enough -- the rotTelPos has to be threaded through to
    `getIntrinsicZernikes`.  These pin that it is, for selected and rejected
    donuts alike, and that a detector with no calibration is left with no
    intrinsics rather than raising.
    """

    def _annotate(self, results: list[Any], rtp_deg: float, calib_by_name: dict) -> None:
        # `_annotateDonuts` never touches `self`, so an unbound call with a
        # dummy avoids standing up the whole task and its subtasks.
        DonutBlitzCornerTask._annotateDonuts(object(), results, rtp_deg, calib_by_name)

    def test_intrinsics_evaluated_at_the_rotated_field_point(self) -> None:
        rtp_deg = 23.0
        calib = _RecordingCalib()
        # Off axis in both axes, so a dropped or swapped angle would show.
        selected = _minimalDonut(thx_ccs=0.018, thy_ccs=0.009)
        rejected = _minimalDonut(donut_id=2, thx_ccs=-0.01, thy_ccs=0.02)
        results = [SimpleNamespace(det_name="R00_SW0", catalog=[selected], rejected_catalog=[rejected])]

        self._annotate(results, rtp_deg, {"R00_SW0": calib})

        # Both the selected and the rejected donut are evaluated.
        self.assertEqual(len(calib.calls), 2)
        for donut, call in zip((selected, rejected), calib.calls):
            self.assertAlmostEqual(call.rotTelPos, rtp_deg)
            self.assertAlmostEqual(call.field_x, np.degrees(donut.thx_ccs))
            self.assertAlmostEqual(call.field_y, np.degrees(donut.thy_ccs))
            np.testing.assert_array_equal(call.noll_indices, np.arange(4, _ZK_JMAX + 1))
            # The squeezed result lands on the donut.
            self.assertEqual(donut.intrinsic_zk.shape, (_ZK_JMAX + 1 - 4,))

    def test_no_calibration_leaves_intrinsics_unset(self) -> None:
        """A detector with no calib gets None intrinsics, not a crash."""
        donut = _minimalDonut()
        results = [SimpleNamespace(det_name="R00_SW0", catalog=[donut], rejected_catalog=[])]

        self._annotate(results, 23.0, {})

        self.assertIsNone(donut.intrinsic_zk)


class TestAnnotateDonutsRejectsNanIntrinsics(unittest.TestCase):
    """`_annotateDonuts` keeps NaN-intrinsics donuts out of the fit.

    Field-angle extrapolation can return intrinsics that are finite for the
    first terms and NaN beyond.  Such a donut must leave the accepted catalog
    -- otherwise its NaN rides into the joint model's reference wavefront and
    the fit returns at its initial guess (``fwhm=1.0``, zero deviation) while
    still reporting success -- but it must stay cataloged as a rejected row.
    """

    def _annotate(self, results: list[Any], rtp_deg: float, calib_by_name: dict) -> None:
        DonutBlitzCornerTask._annotateDonuts(object(), results, rtp_deg, calib_by_name)

    def test_nan_intrinsics_donut_moves_to_rejected(self) -> None:
        # `good` is at positive field_x (finite intrinsics); `bad` at negative
        # field_x (NaN tail), per `_FieldDependentNanCalib`.
        good = _minimalDonut(donut_id=1, thx_ccs=0.018, thy_ccs=0.009)
        bad = _minimalDonut(donut_id=2, thx_ccs=-0.01, thy_ccs=0.02)
        result = SimpleNamespace(det_name="R00_SW0", catalog=[good, bad], rejected_catalog=[])

        self._annotate([result], 23.0, {"R00_SW0": _FieldDependentNanCalib()})

        # `bad` left the accepted catalog and is flagged; `good` stayed clean.
        # Identity checks: a dataclass `==` over the donuts' array fields is
        # ambiguous.
        self.assertEqual(len(result.catalog), 1)
        self.assertIs(result.catalog[0], good)
        self.assertEqual(len(result.rejected_catalog), 1)
        self.assertIs(result.rejected_catalog[0], bad)
        self.assertTrue(bad.rejected_nan_intrinsics)
        self.assertTrue(bad.rejected)
        self.assertFalse(good.rejected_nan_intrinsics)


if __name__ == "__main__":
    unittest.main()
