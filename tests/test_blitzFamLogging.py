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

"""The FAM per-detector log block and its aggregates.

This is the one part of a 189-detector quantum that is *only* observable through
the log, so the failure mode it guards against is a summary that either crashes or
lies at the end of a run that took an hour. Both are cheap to pin here: the
functions under test take plain worker-result dicts, so none of this needs a
butler, a fit, or a fork.

The cases that matter are the degenerate ones -- a detector that failed before it
timed anything, a quantum where every detector failed -- because those are exactly
when someone is reading the log, and `np.nanmean` of nothing warns and returns NaN
rather than raising.
"""

import unittest
import unittest.mock

import numpy as np

from lsst.ts.wep.blitz.donutBlitzFamTask import (
    _STAGE_KEYS,
    DonutBlitzFamTask,
    DonutBlitzFamTaskConfig,
    _detector_stage_times,
    _mean_std_max,
)
from lsst.ts.wep.blitz.wavefrontFittingTask import WavefrontFittingTaskConfig


def _cutout_result(visit_id, n_donuts=3, scatter=0.6, base=1.0):
    """One exposure's cutout result, holding only the keys the summary reads."""
    return {
        "visit_id": visit_id,
        "catalog": [object()] * n_donuts,
        "scatter_arcsec": scatter,
        "isr_run": base,
        "bkg_run": base / 2,
        "diam_run": base / 4,
        "blind_detect_run": base / 10,
        "wcs_refit_run": base / 5,
        "catalog_select_run": base / 20,
        "stamp_cut_run": base / 40,
    }


def _wf_result(success=True, group_size=2, nfev=6, elapsed=5.0):
    return {
        "group_size": group_size,
        "success": success,
        "fit_info": {"nfev": nfev, "elapsed": elapsed},
    }


def _worker_result(
    det_id=1,
    det_name="R01_S00",
    n_groups=2,
    base=1.0,
    error="",
    skipped=False,
    with_results=True,
):
    """A `_fam_detector_worker` return value, as the parent sees it."""
    results = (
        [_cutout_result(1000, base=base), _cutout_result(1001, base=base)]
        if with_results
        else []
    )
    return {
        "det_id": det_id,
        "det_name": det_name,
        "results": results,
        "wf_results": [_wf_result() for _ in range(n_groups)],
        "donuts": [],
        "unmatched_donuts": [],
        "pair_path": "refcat_id",
        "error": error,
        "skipped": skipped,
        "dispatch_to_arrival": 0.25,
        "io_run": 3.0 * base,
        "cutout_run": 2.0 * base,
        "fit_run": 40.0 * base,
        "worker_wall": 50.0 * base,
        "pid": 1234,
    }


class FamLoggingTestCase(unittest.TestCase):
    """Shared task instance; nothing here mutates it."""

    def setUp(self) -> None:
        self.task = DonutBlitzFamTask(config=DonutBlitzFamTaskConfig())
        # Deterministic output regardless of whether pytest is attached to a tty.
        self.task._colorLogEnabled = False

    def logLines(self, results):
        """`_logWorkerSummaries` output, as fully formatted strings."""
        with self.assertLogs(self.task.log.name, level="INFO") as cm:
            self.task._logWorkerSummaries(results)
        return [record.getMessage() for record in cm.records]

    def detectorLines(self, lines):
        """Just the indented per-detector lines."""
        return [line for line in lines if line.startswith("  ")]


class TestStageTimes(unittest.TestCase):
    """`_detector_stage_times` is where the two exposures get folded together."""

    def testCutoutStagesSumOverBothExposures(self) -> None:
        """The pair is summed, not sampled -- a stage total must cover both sides."""
        r = _worker_result(base=1.0)
        stages = _detector_stage_times(r)
        # isr_run is base per exposure, two exposures.
        self.assertAlmostEqual(stages["isr"], 2.0)
        self.assertAlmostEqual(stages["bkg"], 1.0)
        self.assertAlmostEqual(stages["select"], 0.1)

    def testWorkerLevelStagesArePassedThrough(self) -> None:
        """io/fit/wall are already per-detector; they must not be doubled."""
        stages = _detector_stage_times(_worker_result(base=1.0))
        self.assertAlmostEqual(stages["dispatch"], 0.25)
        self.assertAlmostEqual(stages["io"], 3.0)
        self.assertAlmostEqual(stages["fit"], 40.0)
        self.assertAlmostEqual(stages["wall"], 50.0)

    def testKeysAreExactlyTheStageKeysInOrder(self) -> None:
        """The per-detector line and the aggregate line iterate the same order."""
        self.assertEqual(tuple(_detector_stage_times(_worker_result())), _STAGE_KEYS)

    def testNoResultsGivesNaNNotZero(self) -> None:
        """A detector that never cut anything is absent, not instantaneous."""
        stages = _detector_stage_times(_worker_result(with_results=False))
        for key in ("isr", "bkg", "diam", "detect", "wcs", "select", "cut"):
            self.assertTrue(np.isnan(stages[key]), key)

    def testOneExposureMissingAStageIsNaN(self) -> None:
        """Half a pair is not a stage total, so NaN propagates deliberately."""
        r = _worker_result()
        del r["results"][1]["isr_run"]
        self.assertTrue(np.isnan(_detector_stage_times(r)["isr"]))


class TestMeanStdMax(unittest.TestCase):
    """The aggregate helper, whose whole job is not warning on absent data."""

    def testPlainValues(self) -> None:
        mean, std, maximum = _mean_std_max([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, np.std([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(maximum, 3.0)

    def testNaNsAreIgnoredNotPropagated(self) -> None:
        self.assertAlmostEqual(_mean_std_max([1.0, np.nan, 3.0])[0], 2.0)

    def testAllNaNAndEmptyAreNaNWithoutWarning(self) -> None:
        """np.nanmean of an all-NaN slice warns; this must not."""
        with np.errstate(all="raise"):
            for values in ([], [np.nan, np.nan]):
                mean, std, maximum = _mean_std_max(values)
                self.assertTrue(np.isnan(mean))
                self.assertTrue(np.isnan(std))
                self.assertTrue(np.isnan(maximum))

    def testNoneIsTreatedAsAbsent(self) -> None:
        """fit_info values can be None when a fit never produced them."""
        self.assertAlmostEqual(_mean_std_max([2.0, None])[0], 2.0)


class TestPerDetectorLines(FamLoggingTestCase):
    """One line per detector, whatever happened to it."""

    def testOneLinePerDetectorSortedByName(self) -> None:
        """`imap_unordered` scrambles the order; the log must not be scrambled."""
        results = [
            _worker_result(det_id=3, det_name="R13_S02"),
            _worker_result(det_id=1, det_name="R01_S00"),
            _worker_result(det_id=2, det_name="R02_S11"),
        ]
        lines = self.detectorLines(self.logLines(results))
        self.assertEqual(len(lines), 3)
        names = [line.split(":")[0].strip() for line in lines]
        self.assertEqual(names, ["R01_S00", "R02_S11", "R13_S02"])

    def testLineCarriesEveryStageAndTheFit(self) -> None:
        """The point of the change: one line holds isr..cut *and* danish."""
        (line,) = self.detectorLines(self.logLines([_worker_result()]))
        for key in _STAGE_KEYS:
            self.assertIn(f"{key}=", line)
        self.assertIn("scatter=0.60\"/0.60\"", line)
        self.assertIn("donuts=3+3", line)
        self.assertIn("pair=refcat_id", line)
        self.assertIn("2/2 ok", line)
        self.assertIn("nfev=6.0", line)

    def testFailedAndSkippedDetectorsStillGetALine(self) -> None:
        """A missing detector must be visibly missing, not absent from the block."""
        results = [
            _worker_result(det_id=1, det_name="R01_S00"),
            _worker_result(
                det_id=2,
                det_name="R02_S11",
                error="NoWorkFound: bad CCD",
                skipped=True,
                with_results=False,
            ),
            _worker_result(
                det_id=3, det_name="R13_S02", error="ValueError: boom", with_results=False
            ),
        ]
        lines = self.detectorLines(self.logLines(results))
        self.assertEqual(len(lines), 3)
        self.assertIn("SKIPPED", lines[1])
        self.assertIn("bad CCD", lines[1])
        self.assertIn("FAILED", lines[2])
        self.assertIn("boom", lines[2])

    def testUnnamedDetectorFallsBackToItsId(self) -> None:
        """A worker that failed before reading the raw has no det_name."""
        (line,) = self.detectorLines(
            self.logLines(
                [_worker_result(det_id=42, det_name="", error="boom", with_results=False)]
            )
        )
        self.assertIn("det42", line)

    def testMissingScatterIsNotAvailableNotZero(self) -> None:
        r = _worker_result()
        r["results"][0]["scatter_arcsec"] = None
        (line,) = self.detectorLines(self.logLines([r]))
        self.assertIn('scatter=N/A/0.60"', line)

    def testDetectorWithNoFitGroupsSaysSo(self) -> None:
        (line,) = self.detectorLines(self.logLines([_worker_result(n_groups=0)]))
        self.assertIn("no groups", line)

    def testGroupWithoutFitInfoOmitsNfev(self) -> None:
        """A timed-out group has an empty fit_info; that is not nfev=0."""
        r = _worker_result(n_groups=1)
        r["wf_results"][0] = {"group_size": 2, "success": False, "fit_info": {}}
        (line,) = self.detectorLines(self.logLines([r]))
        self.assertIn("0/1 ok", line)
        self.assertNotIn("nfev=", line)


class TestAggregates(FamLoggingTestCase):
    """The mean/std block, which is what makes 188 lines readable."""

    def testMeansMatchHandComputedValues(self) -> None:
        results = [
            _worker_result(det_id=1, det_name="R01_S00", base=1.0),
            _worker_result(det_id=2, det_name="R02_S11", base=3.0),
        ]
        (timing,) = [
            line for line in self.logLines(results) if line.startswith("Per-detector timing")
        ]
        self.assertIn("(n=2)", timing)
        # isr is 2*base per detector: mean of 2 and 6 is 4, std is 2.
        self.assertIn("isr=4.00+/-2.00s", timing)
        # io is passed straight through: mean of 3 and 9.
        self.assertIn("io=6.00+/-3.00s", timing)

    def testAggregatesCoverOnlyDetectorsThatRan(self) -> None:
        """A failed detector's NaN row would widen every std with an absence."""
        results = [
            _worker_result(det_id=1, det_name="R01_S00", base=1.0),
            _worker_result(det_id=2, det_name="R02_S11", base=1.0),
            _worker_result(
                det_id=3, det_name="R13_S02", error="boom", with_results=False
            ),
        ]
        lines = self.logLines(results)
        (timing,) = [line for line in lines if line.startswith("Per-detector timing")]
        self.assertIn("(n=2)", timing)
        # Two identical detectors: zero spread, and no NaN leaking in.
        self.assertIn("isr=2.00+/-0.00s", timing)
        self.assertNotIn("nan", timing)

    def testYieldLineIsPresentAndFinite(self) -> None:
        lines = self.logLines([_worker_result(), _worker_result(det_id=2, det_name="R02_S11")])
        (yields,) = [line for line in lines if line.startswith("Per-detector yield")]
        self.assertIn("donuts=6.0+/-0.0", yields)
        self.assertIn("groups=2.0+/-0.0", yields)
        self.assertIn('scatter=0.60+/-0.00"', yields)
        self.assertIn("per-group fit=5.0+/-0.0s", yields)

    def testSlowestDetectorsAreNamedWorstFirst(self) -> None:
        results = [
            _worker_result(det_id=1, det_name="R01_S00", base=1.0),
            _worker_result(det_id=2, det_name="R02_S11", base=5.0),
            _worker_result(det_id=3, det_name="R13_S02", base=3.0),
        ]
        (slowest,) = [line for line in self.logLines(results) if line.startswith("Slowest")]
        self.assertLess(slowest.index("R02_S11"), slowest.index("R13_S02"))
        self.assertLess(slowest.index("R13_S02"), slowest.index("R01_S00"))

    def testNoAggregatesWhenEveryDetectorFailed(self) -> None:
        """The all-failed case: per-detector lines, no NaN-only statistics block."""
        results = [
            _worker_result(det_id=i, det_name=f"R0{i}_S00", error="boom", with_results=False)
            for i in (1, 2)
        ]
        lines = self.logLines(results)
        self.assertEqual(len(self.detectorLines(lines)), 2)
        self.assertFalse([line for line in lines if line.startswith("Per-detector timing")])

    def testEmptyResultsLogNothing(self) -> None:
        """assertLogs fails on no output, so check the log call count directly."""
        with unittest.mock.patch.object(self.task.log, "info") as info:
            self.task._logWorkerSummaries([])
        info.assert_not_called()


class TestPerGroupLogGating(unittest.TestCase):
    """FAM turns off the ~10k per-group WF lines; corner mode keeps them."""

    def testDefaultIsOnSoCornerModeIsUnchanged(self) -> None:
        self.assertTrue(WavefrontFittingTaskConfig().logPerGroup)

    def testFamTurnsItOff(self) -> None:
        config = DonutBlitzFamTaskConfig()
        config.validate()
        self.assertFalse(config.wfFittingTask.logPerGroup)


if __name__ == "__main__":
    unittest.main()
