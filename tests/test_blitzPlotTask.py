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

"""The diagnostic plot task, driven from a real `_build_donut_catalog` table.

The plot task is the only in-tree consumer of the output catalog, so it is what
breaks when a column is renamed.  These tests build the catalog through the
builder rather than by hand, so the two stay pinned to the same schema.
"""

import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from lsst.ts.wep.blitz.catalogBuilder import (
    _build_donut_catalog,
    _CatalogOptions,
    _CatalogTimings,
)
from lsst.ts.wep.blitz.dataStructures import (
    CutoutResult,
    Donut,
    WfDonutResult,
    WfGroupResult,
)
from lsst.ts.wep.blitz.donutBlitzPlot import (
    _PANEL_ERROR_CHARS,
    DonutBlitzPlotConfig,
    DonutBlitzPlotTask,
    _corner_of,
    _detector_stats_lines,
    _donut_annotation,
    _donut_rows_by_detector,
    _RowHalf,
    _wf_groups_from_catalog,
    _wf_row_pairs,
)
from lsst.ts.wep.blitz.utils import _CUTOUT_STAGE_KEYS, _ZK_JMAX, CORNER_PAIRS

_VISIT_ID = 2026070900036
_STAMP_SIZE = 167
_BINNING = 2
_WF_IMG_SIZE = 83  # _STAMP_SIZE // _BINNING, forced odd
# Ordered as _OFFSET_OPTICS: (detector, camera, m2). Corner mode shifts the
# detector plane, intra being the negation of extra.
_EXTRA_OFFSETS = (+1.5e-3, 0.0, 0.0)
_INTRA_OFFSETS = (-1.5e-3, 0.0, 0.0)


def _donut(det_name, donut_id, **overrides):
    kwargs = dict(
        det_name=det_name,
        stamp=np.random.default_rng(donut_id)
        .normal(100.0, 5.0, (_STAMP_SIZE, _STAMP_SIZE))
        .astype(np.float32),
        thx_ccs=0.01,
        thy_ccs=0.02,
        flux=1e5,
        band="r",
        det_id=191,
        visit_id=_VISIT_ID,
        x_det=100.0,
        y_det=200.0,
        donut_id=donut_id,
        inner_frac=0.01,
        outer_frac=0.02,
        outer_sector_minmax_frac=0.03,
        donut_radius=60.0,
        snr=500.0,
        bkg=10.0,
        bkg_std=3.0,
        n_quarter=1,
        photo_mag=14.0,
        astrom_mag=14.2,
        nearby_photo=[(1.0, 2.0, 15.0)],
        nearby_astrom=[(3.0, -4.0, 16.0)],
    )
    kwargs.update(overrides)
    return Donut(**kwargs)


def _result(det_name, rejected=()):
    return CutoutResult(
        det_name=det_name,
        catalog=[],
        rejected_catalog=list(rejected),
        isr_run=0.0,
        bkg_run=0.0,
        diam_run=0.0,
        detect_run=0.0,
        wcs_refit_run=0.0,
        catalog_select_run=0.0,
        stamp_cut_run=0.0,
        scatter_arcsec=0.3,
        wcs_refit_error="",
        cat_select_error="",
        selection_source="refcat",
        # Matches the Donut's own n_quarter: the stamps were rotated by it, and
        # the plot reads it back from det_meta to place the refcat overlays.
        n_quarter=1,
        pair_path="snr_rank",
        wcs=None,
    )


def _wf_result(donuts, group_id):
    """One joint fit over ``donuts``, with images the plot task can draw."""
    rng = np.random.default_rng(0)
    out = WfGroupResult.empty(group_id, n_zk=_ZK_JMAX + 1)
    out.group_size = len(donuts)
    out.success = True
    out.det_names = [d.det_name for d in donuts]
    out.donut_results = [
        WfDonutResult(
            donut_id=d.donut_id,
            det_name=d.det_name,
            visit_id=d.visit_id,
            zk_dev=np.full(_ZK_JMAX + 1, 1e-7),
            zk_intrinsic=np.zeros(_ZK_JMAX + 1),
            img=rng.normal(100.0, 5.0, (_WF_IMG_SIZE, _WF_IMG_SIZE)),
            model_img=rng.normal(100.0, 5.0, (_WF_IMG_SIZE, _WF_IMG_SIZE)),
            fit_success=True,
            fit_elapsed=12.5,
            setup_elapsed=0.75,
            fit_nfev=40,
            fit_cost=3.5,
            fit_optimality=2.5e-9,
            fit_njev=38,
            fit_outcome="ok",
            fit_dx=0.1 * i,
            fit_dy=0.2 * i,
            fit_flux=1e5,
            fit_fwhm=0.9,
            blend_frac=0.01 * i,
            group_id=group_id,
            group_size=len(donuts),
        )
        for i, d in enumerate(donuts)
    ]
    return out


def _options():
    return _CatalogOptions(
        stamp_size=_STAMP_SIZE,
        binning=_BINNING,
        noll_indices=tuple(range(4, 23)),
        aperture_margin_frac=0.05,
        bkg_inner_disc_frac=0.67,
        bkg_annulus_inner_frac=1.25,
        bkg_annulus_outer_frac=1.4,
        max_donuts=8,
        wf_mode="paired",
    )


def _catalog():
    """A catalog covering every row class the plot task branches on.

    One fitted pair across the two detectors of a corner, one candidate no fit
    claimed (paired-mode surplus), and one donut rejected on SNR.

    The timings are non-zero so the plot titles take their populated
    branches: both the ``butler=`` suffix and the per-dataset ``butler.get``
    breakdown are skipped entirely at zero, and they are where the elapsed
    ``meta`` Quantities get formatted.
    """
    # Corner mode's sides: SW0 extra-focal (+), SW1 intra-focal (-).
    extra = _donut("R00_SW0", 1, defocal_offsets=_EXTRA_OFFSETS)
    intra = _donut("R00_SW1", 2, defocal_offsets=_INTRA_OFFSETS)
    surplus = _donut("R00_SW0", 3, defocal_offsets=_EXTRA_OFFSETS)
    rejected = _donut("R00_SW0", 4, rejected=True, rejected_snr=True, defocal_offsets=_EXTRA_OFFSETS)
    return _build_donut_catalog(
        [_result("R00_SW0", rejected=[rejected]), _result("R00_SW1")],
        [_wf_result([extra, intra], "R00_1_2")],
        [extra, intra, surplus],
        [surplus],
        _VISIT_ID,
        _options(),
        rtp_rad=0.25,
        timings=_CatalogTimings(
            run_elapsed=30.0,
            refcat_elapsed=1.5,
            butler_elapsed=4.0,
            butler_times={"raw": 3.0, "bias": 1.0},
            cutout_elapsed=6.0,
            danish_elapsed=12.0,
        ),
    )


class TestDonutBlitzPlotTask(unittest.TestCase):
    """The plot task reads a builder-produced catalog end to end."""

    def testPlotsAreWrittenFromABuilderCatalog(self) -> None:
        catalog = _catalog()
        task = DonutBlitzPlotTask(config=DonutBlitzPlotConfig())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                task.run(catalog)
                written = sorted(os.listdir(tmp))
            finally:
                os.chdir(cwd)
        self.assertEqual(
            written,
            [f"donut_diag_{_VISIT_ID}.png", f"wf_diag_{_VISIT_ID}.png"],
        )

    def testEmptyCatalogWritesNothing(self) -> None:
        task = DonutBlitzPlotTask(config=DonutBlitzPlotConfig())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                task.run(_build_donut_catalog([], [], [], [], _VISIT_ID, _options()))
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)


class TestDonutRowsByDetector(unittest.TestCase):
    """The per-detector accepted/rejected split the donut plot lays out."""

    def testSplitsOnCandidateNotOnWhetherAFitUsedTheDonut(self) -> None:
        """A candidate no fit claimed still belongs in the accepted panel.

        `_catalog` gives R00_SW0 two fit-consumed donuts' worth of rows plus a
        paired-mode surplus candidate and one SNR-rejected donut, so this pins
        the distinction that makes the split "candidate" rather than "fitted".
        """
        by_det = _donut_rows_by_detector(_catalog())
        self.assertEqual([name for name, _, _ in by_det], ["R00_SW0", "R00_SW1"])
        accepted = {name: acc for name, acc, _ in by_det}
        rejected = {name: rej for name, _, rej in by_det}
        # SW0 carries donut 1 (fitted) and donut 3 (surplus, never fitted).
        self.assertEqual(sorted(accepted["R00_SW0"]["donut_id"].tolist()), [1, 3])
        self.assertEqual(rejected["R00_SW0"]["donut_id"].tolist(), [4])
        self.assertEqual(rejected["R00_SW1"]["donut_id"].tolist(), [])

    def testDetectorsAreSortedAndEmptyOnesAbsent(self) -> None:
        by_det = _donut_rows_by_detector(_catalog())
        names = [name for name, _, _ in by_det]
        self.assertEqual(names, sorted(names))
        for _, acc, rej in by_det:
            self.assertGreater(len(acc) + len(rej), 0)


class TestDetectorStatsLines(unittest.TestCase):
    """The monospace stats block for one detector's panel."""

    def _stats(self):
        return _catalog().meta["det_meta"][f"R00_SW0_{_VISIT_ID}"]

    def testEveryCutoutStageGetsALineAndScatterHangsOffAstrom(self) -> None:
        """The panel is driven off `_CUTOUT_STAGE_KEYS`, so it cannot drift.

        The scatter annotation is the one stage-specific detail, and nothing
        renders it but this function, so it is worth pinning here rather than
        trusting a PNG.
        """
        lines = _detector_stats_lines("R00_SW0", 191, 8, self._stats())
        self.assertEqual(lines[0], "R00_SW0 (191)")
        self.assertEqual(lines[1], "donuts: 8")
        stage_lines = lines[2:]
        self.assertEqual(len(stage_lines), len(_CUTOUT_STAGE_KEYS))
        for label, line in zip(_CUTOUT_STAGE_KEYS, stage_lines):
            self.assertTrue(line.startswith(f"{label}:"), line)
            # One scatter annotation, on astrom and nowhere else.
            self.assertEqual('"' in line, label == "astrom")

    def testAbsentDetectorReportsNaNRatherThanRaising(self) -> None:
        """A detector with no det_meta entry still gets a full block.

        The plot draws the 2x2 corner grid regardless of which detectors ran,
        so an empty dict has to be survivable -- and has to read as unknown,
        not as a fast zero.
        """
        lines = _detector_stats_lines("R44_SW1", 204, 0, {})
        self.assertEqual(lines[0], "R44_SW1 (204)")
        self.assertIn("N/A", " ".join(lines))
        self.assertIn("nan", " ".join(lines))
        # No error lines, since an absent entry has no error strings.
        self.assertFalse([ln for ln in lines if "ERR" in ln])

    def testStageErrorsAreTruncatedToTheColumnWidth(self) -> None:
        stats = dict(self._stats())
        stats["wcs_refit_error"] = "E" * 200
        stats["cat_select_error"] = "C" * 200
        lines = _detector_stats_lines("R00_SW0", 191, 8, stats)
        wcs_line = next(ln for ln in lines if ln.startswith("WCS ERR:"))
        cat_line = next(ln for ln in lines if ln.startswith("CAT ERR:"))
        self.assertEqual(wcs_line, f"WCS ERR: {'E' * _PANEL_ERROR_CHARS}")
        self.assertEqual(cat_line, f"CAT ERR: {'C' * _PANEL_ERROR_CHARS}")


class TestDonutAnnotation(unittest.TestCase):
    """The three-line caption above one donut stamp."""

    def _row(self, det_name="R00_SW0", donut_id=1):
        catalog = _catalog()
        names = np.asarray(catalog["det_name"], dtype=str)
        rows = catalog[(names == det_name) & (catalog["donut_id"] == donut_id)]
        return rows[0]

    def testFiniteMeasurementsAreFormattedToTheirPlaces(self) -> None:
        text = _donut_annotation(self._row(), rejected=False)
        snr_line, frac_line, id_line = text.split("\n")
        # _donut() builds snr=500, inner=0.01, outer=0.02, osm=0.03.
        self.assertEqual(snr_line.strip(), "snr=500")
        self.assertEqual(frac_line.split(), ["if=0.010", "of=0.020", "osm=0.030"])
        self.assertEqual(id_line, "id=1")

    def testNonFiniteReadsAsQuestionMarkNotZero(self) -> None:
        """`?` distinguishes "never measured" from a real zero.

        This is the whole reason the formatting is conditional, so it is the
        part worth a test.
        """
        row = self._row()
        row["snr"] = float("nan")
        row["inner_frac"] = float("nan")
        text = _donut_annotation(row, rejected=False)
        self.assertIn("snr=?", text)
        self.assertIn("if=?", text)
        self.assertNotIn("snr=0", text)

    def testRejectionFlagsAreListedInOrder(self) -> None:
        row = self._row(donut_id=4)
        # _catalog's donut 4 is rejected on SNR alone.
        self.assertIn("[snr]", _donut_annotation(row, rejected=True))
        row["rejected_sat"] = True
        row["rejected_outer_frac"] = True
        self.assertIn("[sat|outer|snr]", _donut_annotation(row, rejected=True))

    def testDonutIdZeroIsOmitted(self) -> None:
        """id=0 means "no id assigned", so the line is left blank."""
        row = self._row()
        row["donut_id"] = 0
        self.assertEqual(_donut_annotation(row, rejected=False).split("\n")[2], "")


class TestWfRowPairs(unittest.TestCase):
    """Corner assignment and row padding for the WF plot's 2x2 grid.

    Pure layout, and the one part of that plot where a real bug could hide, so
    it is pinned directly rather than through a rendered figure.
    """

    def testEveryCornerGetsTheSameRowCount(self) -> None:
        """Equal heights are what make two modes' plots blinkable.

        Padding to a config-derived height rather than to the tallest corner is
        the property: figure dimensions and axes positions then depend only on
        config, not on how many donuts a mode happened to fit.
        """
        plottable, unfitted = _wf_groups_from_catalog(_catalog())
        row_pairs = _wf_row_pairs(plottable, unfitted, "paired", max_donuts=8)
        self.assertEqual(sorted(row_pairs), sorted(CORNER_PAIRS))
        self.assertEqual({len(v) for v in row_pairs.values()}, {8})

    def testACornerExceedingMaxDonutsGrowsTheLayout(self) -> None:
        """More fits than maxDonuts must grow the grid, not lose rows."""
        plottable, unfitted = _wf_groups_from_catalog(_catalog())
        # One real group in R00; ask for fewer rows than that.
        row_pairs = _wf_row_pairs(plottable, unfitted, "paired", max_donuts=0)
        # max_donuts=0 floors at 1, and R00 has a fitted row plus a surplus.
        self.assertEqual({len(v) for v in row_pairs.values()}, {2})
        populated = [p for p in row_pairs["R00"] if p != (None, None)]
        self.assertEqual(len(populated), 2)

    def testPairedModeGivesOneGroupBothHalvesOfARow(self) -> None:
        plottable, _ = _wf_groups_from_catalog(_catalog())
        row_pairs = _wf_row_pairs(plottable, [], "paired", max_donuts=1)
        intra, extra = row_pairs["R00"][0]
        self.assertIs(intra, extra)

    def testUnpairedModeExplodesAndPairsByPosition(self) -> None:
        """A non-paired mode's groups are split per donut, then laid out.

        The pairing is cosmetic in that case, so the two halves must be
        *different* single-donut records -- the opposite of paired mode above.
        """
        plottable, _ = _wf_groups_from_catalog(_catalog())
        row_pairs = _wf_row_pairs(plottable, [], "unpaired", max_donuts=1)
        intra, extra = row_pairs["R00"][0]
        self.assertIsNot(intra, extra)
        self.assertEqual(len(intra.donuts), 1)
        self.assertEqual(len(extra.donuts), 1)
        self.assertEqual(intra.donuts[0].defocal, "intra")
        self.assertEqual(extra.donuts[0].defocal, "extra")

    def testUnrecognizedDetectorFallsBackRatherThanRaising(self) -> None:
        """The grid has to be drawn whatever the catalog names.

        `_corner_of` falling back is deliberate: a group whose detectors are
        all unrecognized is a catalog problem the plot should survive.
        """
        plottable, _ = _wf_groups_from_catalog(_catalog())
        stray = replace(plottable[0], det_names=["NOT_A_CORNER"])
        row_pairs = _wf_row_pairs([stray], [], "paired", max_donuts=1)
        self.assertEqual(_corner_of(stray), next(iter(CORNER_PAIRS)))
        self.assertNotEqual(row_pairs[next(iter(CORNER_PAIRS))][0], (None, None))


class TestRowHalf(unittest.TestCase):
    """The per-side flattening the WF drawing code consumes."""

    def _group(self):
        plottable, _ = _wf_groups_from_catalog(_catalog())
        return plottable[0]

    def testBlankHalfIsARecordNotNone(self) -> None:
        """A padded row still yields a `_RowHalf`, with img None.

        That is what lets the drawing code ask one question instead of
        threading None checks through every field.
        """
        half = _RowHalf.from_group(None, "intra", det_hdr="hdr")
        self.assertIsNone(half.img)
        self.assertIsNone(half.model)
        self.assertIsNone(half.donut_id)
        self.assertEqual(half.det_hdr, "hdr")
        self.assertEqual(len(half.zk_dev), 0)
        # nfev=0 on a never-fitted half, so the label reads x0.
        self.assertIn("x0", half.bar_label)

    def testAGroupWithNoDonutOfThatSideIsAlsoBlank(self) -> None:
        """An exploded or unfitted group matches only its own side."""
        single = replace(self._group(), donuts=[self._group().donuts[0]])
        wrong_side = "extra" if single.donuts[0].defocal == "intra" else "intra"
        self.assertIsNone(_RowHalf.from_group(single, wrong_side, "").img)
        self.assertIsNotNone(_RowHalf.from_group(single, single.donuts[0].defocal, "").img)

    def testBarLabelReportsStatusAndNfev(self) -> None:
        group = self._group()
        half = _RowHalf.from_group(group, "intra", "")
        self.assertRegex(half.bar_label, r"^t=[\d.]+s (ok|fail|x0) nfev=\d+$")
        # A fit that ran but failed says so, rather than reading as x0.
        failed = replace(group, success=False)
        self.assertIn("fail", _RowHalf.from_group(failed, "intra", "").bar_label)


class TestWfGroupsFromCatalog(unittest.TestCase):
    """The catalog -> per-fit inversion the WF plot draws from.

    Pinned directly rather than through a rendered figure: a PNG proves the
    records were usable, not that they carried the right values.
    """

    def testFittedGroupCarriesItsFitScalars(self) -> None:
        plottable, _ = _wf_groups_from_catalog(_catalog())
        self.assertEqual(len(plottable), 1)
        group = plottable[0]
        # Both detectors of the corner, in catalog order.
        self.assertEqual([str(n) for n in group.det_names], ["R00_SW0", "R00_SW1"])
        self.assertTrue(group.success)
        # The values `_wf_result` sets, stripped of their units.
        self.assertEqual(group.fit_info.nfev, 40)
        self.assertAlmostEqual(group.fit_info.fwhm, 0.9)
        self.assertAlmostEqual(group.fit_info.elapsed, 12.5)
        self.assertEqual(len(group.donuts), 2)
        self.assertEqual(
            sorted(d.defocal for d in group.donuts),
            ["extra", "intra"],
        )
        for donut in group.donuts:
            self.assertIsNotNone(donut.model_img)

    def testUnfittedSurplusDonutIsDataOnly(self) -> None:
        """The surplus candidate: a stamp to draw, but nothing fit it."""
        _, unfitted = _wf_groups_from_catalog(_catalog())
        self.assertEqual(len(unfitted), 1)
        group = unfitted[0]
        self.assertFalse(group.success)
        self.assertIsNone(group.donuts[0].model_img)
        self.assertEqual(len(group.zk_dev), 0)
        # nfev=0 is what makes the bar label read "x0" instead of "fail".
        self.assertEqual(group.fit_info.nfev, 0)
        self.assertTrue(np.isnan(group.fit_info.elapsed))
        self.assertTrue(np.isnan(group.fit_info.fwhm))

    def testRejectedDonutsAreNotDrawn(self) -> None:
        """A donut rejected on SNR is neither fitted nor a candidate."""
        plottable, unfitted = _wf_groups_from_catalog(_catalog())
        drawn = {int(d.donut_id) for g in plottable + unfitted for d in g.donuts}
        self.assertEqual(drawn, {1, 2, 3})

    def testExplodedKeepsTheGroupScalarsPerDonut(self) -> None:
        """The non-paired layout path, which the fixture's mode does not use.

        ``_saveWfDiagnosticPlot`` calls this for every mode whose groups do not
        pair intra with extra, so it is worth pinning even though a paired
        fixture never reaches it.
        """
        group = _wf_groups_from_catalog(_catalog())[0][0]
        singles = group.exploded()
        self.assertEqual(len(singles), len(group.donuts))
        for single, donut in zip(singles, group.donuts):
            self.assertEqual(len(single.donuts), 1)
            self.assertEqual(single.donuts[0].donut_id, donut.donut_id)
            # The scalars are the group's, shared by each copy.
            self.assertEqual(single.fit_info, group.fit_info)
            self.assertEqual(single.success, group.success)
            np.testing.assert_array_equal(single.zk_dev, group.zk_dev)
        # Exploding must not disturb the group it came from.
        self.assertEqual(len(group.donuts), 2)

    def testEmptyCatalogYieldsNoGroups(self) -> None:
        catalog = _build_donut_catalog([], [], [], [], _VISIT_ID, _options())
        self.assertEqual(_wf_groups_from_catalog(catalog), ([], []))


if __name__ == "__main__":
    unittest.main()
