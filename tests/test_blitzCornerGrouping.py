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

"""Corner-mode work-unit grouping, and the pairing path it reports.

Corner mode pairs by SNR rank and has no fallback, so unlike full-array mode's
`_pair_donuts` there is no algorithm choice to test.  What is worth pinning is
the provenance contract the two modes share: ``_build_wf_groups`` reports a
non-empty token for every mode, so an empty ``pair_path`` in a persisted catalog
means a bug rather than "not applicable".
"""

import unittest
from types import SimpleNamespace

from lsst.ts.wep.blitz.wavefrontFittingTask import _build_wf_groups

_NON_PAIRING_MODES = ("unpaired", "full_detector", "full_corner")


def _donut(det_name, donut_id, snr):
    """A stand-in for `Donut`: grouping reads only these three attributes."""
    return SimpleNamespace(det_name=det_name, donut_id=donut_id, snr=snr)


def _groups(mode, results_by_det):
    return _build_wf_groups(mode, results_by_det, "r", None, None)


class TestCornerGroupingPairPath(unittest.TestCase):
    """The ``pair_path`` token, per mode."""

    def setUp(self) -> None:
        # One corner, with a surplus SW0 donut so `paired` has something to leave
        # unmatched. SNRs are distinct so the rank zip is unambiguous.
        self.results_by_det = {
            "R00_SW0": [
                _donut("R00_SW0", 1, 900.0),
                _donut("R00_SW0", 2, 500.0),
                _donut("R00_SW0", 3, 100.0),
            ],
            "R00_SW1": [
                _donut("R00_SW1", 1, 800.0),
                _donut("R00_SW1", 2, 400.0),
            ],
        }

    def testPairedRecordsSnrRank(self) -> None:
        groups, unmatched, path = _groups("paired", self.results_by_det)
        self.assertEqual(path, "snr_rank")
        # Two pairs from the ranks that exist on both sides, and the faintest SW0
        # donut left over.
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(g.donuts) == 2 for g in groups))
        self.assertEqual([d.snr for d in unmatched], [100.0])

    def testNonPairingModesRecordNotApplicable(self) -> None:
        for mode in _NON_PAIRING_MODES:
            with self.subTest(mode=mode):
                _, unmatched, path = _groups(mode, self.results_by_det)
                self.assertEqual(path, "n/a")
                self.assertEqual(unmatched, [])

    def testEveryModeReportsANonEmptyPath(self) -> None:
        """The invariant the catalog builder relies on: never "" and never None."""
        for mode in ("paired",) + _NON_PAIRING_MODES:
            with self.subTest(mode=mode):
                path = _groups(mode, self.results_by_det)[2]
                self.assertIsInstance(path, str)
                self.assertTrue(path)

    def testUnknownModeRaises(self) -> None:
        # In particular full-array mode's `full_detector_pair`, whose name is
        # close enough to `full_corner` to be worth failing loudly on.
        for mode in ("full_detector_pair", "nonsense"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    _groups(mode, self.results_by_det)


if __name__ == "__main__":
    unittest.main()
