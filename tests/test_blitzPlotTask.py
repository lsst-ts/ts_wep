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

import astropy.units as u
import numpy as np
from astropy.table import QTable

from lsst.ts.wep.blitz.catalogBuilder import (
    _build_detector_image_table,
    _build_donut_catalog,
    _build_overlay_table,
    _CatalogOptions,
    _CatalogTimings,
)
from lsst.ts.wep.blitz.dataStructures import (
    CutoutResult,
    DetectorView,
    Donut,
    SourceSet,
    WfDonutResult,
    WfGroupResult,
)
from lsst.ts.wep.blitz.donutBlitzPlot import (
    _FP_LAYOUT,
    _FP_MOSAIC,
    _FP_PANEL_ASPECT,
    _PANEL_ERROR_CHARS,
    DonutBlitzPlotConfig,
    DonutBlitzPlotTask,
    _binned_coord,
    _corner_of,
    _detector_stats_lines,
    _donut_annotation,
    _donut_rows_by_detector,
    _focal_plane_axes_rects,
    _pair_links,
    _rot90_display,
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


_VIEW_BINNING = 4
# Un-binned corner-sensor geometry, scaled down so the fixtures stay cheap
# while roughly keeping the 4072x2000 aspect.
_VIEW_BBOX = (200, 408)


def _source_set(n, seed, width, height, with_mag=True):
    rng = np.random.default_rng(seed)
    return SourceSet(
        x_det=rng.uniform(0, width, n),
        y_det=rng.uniform(0, height, n),
        donut_id=np.arange(1, n + 1),
        mag=(rng.uniform(13.0, 20.0, n) if with_mag else np.full(n, np.nan)).astype(np.float32),
    )


def _view(det_name, n_quarter, refcat=True, field_dist=None):
    """A `DetectorView` shaped like a real corner sensor's, cheap to draw.

    ``field_dist`` defaults to a plane rising with x, which is not the real arc
    but is monotonic and so gives every panel an unambiguous most- and
    least-vignetted end -- which is what the orientation check needs.
    """
    height, width = _VIEW_BBOX
    hb, wb = height // _VIEW_BINNING, width // _VIEW_BINNING
    rng = np.random.default_rng(abs(hash(det_name)) % 2**32)
    if field_dist is None:
        field_dist = np.tile(np.linspace(1.0, 2.0, wb, dtype=np.float32), (hb, 1))
    return DetectorView(
        image=rng.normal(12.0, 2.0, (hb, wb)).astype(np.float32),
        binning=_VIEW_BINNING,
        bbox_min=(0, 0),
        bbox_shape=(height, width),
        refcat=_source_set(12, 1, width, height) if refcat else None,
        detections=_source_set(5, 2, width, height, with_mag=False),
        selections=_source_set(4, 3, width, height),
        field_dist=field_dist,
        max_field_dist_deg=1.5,
    )


def _result_with_view(det_name, n_quarter, **kwargs):
    result = replace(_result(det_name), n_quarter=n_quarter)
    result.view = _view(det_name, n_quarter, **kwargs)
    return result


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


class TestRot90Display(unittest.TestCase):
    """The whole-detector rotation, against numpy's own answer.

    `_rot90_display` maps *indices* through ``np.rot90(arr, -n).T``, so numpy
    applying that transform to an array is the independent check: rotate an
    array of unique values, then confirm the value numpy put at the mapped
    position is the one that started at the original index.
    """

    def testEveryRotationMatchesNumpy(self) -> None:
        # Non-square and coprime sides, so a transposed or mirrored mapping
        # cannot coincidentally agree.
        height, width = 5, 7
        arr = np.arange(height * width).reshape(height, width)
        for n_quarter in range(4):
            display = np.rot90(arr, -n_quarter).T
            for r in range(height):
                for c in range(width):
                    x, y = _rot90_display(r, c, height, width, n_quarter)
                    # The return is *plot* coordinates, which is how the panel
                    # uses them: x runs along the displayed array's columns and
                    # y down its rows, so reading the array back needs [y, x].
                    self.assertEqual(
                        display[int(y), int(x)],
                        arr[r, c],
                        f"n_quarter={n_quarter} at ({r}, {c})",
                    )

    def testQuarterTurnsAreTakenModFour(self) -> None:
        """Corner sensors report n_quarter > 4, so the wrap is load-bearing."""
        for n_quarter in range(4):
            self.assertEqual(
                _rot90_display(2, 3, 5, 7, n_quarter),
                _rot90_display(2, 3, 5, 7, n_quarter + 4),
            )


class TestBinnedCoord(unittest.TestCase):
    """The detector-to-binned mapping, and its half-pixel term."""

    def testASyntheticBrightPixelLandsUnderItsOwnMarker(self) -> None:
        """The regression test for the ``(b - 1) / 2`` correction.

        Put a single bright un-binned pixel at a known detector coordinate, bin
        the image the way the worker does, and confirm `_binned_coord` rounds
        to the binned pixel that actually holds the flux.  Sweeps a whole block
        so the naive ``u / b`` mapping, which is off by 0.375 binned pixels at
        ``b = 4``, fails rather than passes by luck.
        """
        binning = 4
        size = 64
        bbox_min_x, bbox_min_y = 12, 20
        row = {"binning": binning, "bbox_min_x": bbox_min_x, "bbox_min_y": bbox_min_y}
        wrong = 0
        for step in range(2 * binning):
            x_det = bbox_min_x + 24 + step
            y_det = bbox_min_y + 36 + step
            image = np.zeros((size, size), dtype=np.float32)
            image[y_det - bbox_min_y, x_det - bbox_min_x] = 1.0
            # Block mean, which is what afwMath.binImage does.
            binned = image.reshape(size // binning, binning, size // binning, binning).mean(axis=(1, 3))
            hot = np.unravel_index(int(np.argmax(binned)), binned.shape)

            r, c = _binned_coord(x_det, y_det, row)
            self.assertEqual((int(round(float(r))), int(round(float(c)))), hot)

            naive_r = (y_det - bbox_min_y) / binning
            naive_c = (x_det - bbox_min_x) / binning
            if (int(round(naive_r)), int(round(naive_c))) != hot:
                wrong += 1
        # The naive mapping must actually be wrong somewhere, or this test
        # would pass just as happily against the bug it guards.
        self.assertGreater(wrong, 0)

    def testBinningAndOriginAreReadFromTheRow(self) -> None:
        """Never assumed, so an older run's plot cannot mis-scale."""
        r, c = _binned_coord(
            100.0, 200.0, {"binning": 2, "bbox_min_x": 10, "bbox_min_y": 20}
        )
        self.assertAlmostEqual(float(c), (100.0 - 10 - 0.5) / 2)
        self.assertAlmostEqual(float(r), (200.0 - 20 - 0.5) / 2)


class TestFocalPlaneAxesRects(unittest.TestCase):
    """The explicit panel geometry that replaced ``subplot_mosaic``.

    The point of placing rectangles by hand is that the intra-corner and
    inter-corner gaps become independently settable, which a uniform grid
    cannot do: matplotlib's ``wspace``/``hspace`` are fractions of *average*
    axes size, so mixing tall and wide panels yielded 0.34-0.46 in horizontal
    seams against 0.11 in vertical ones.  These tests pin the invariants that
    fix buys.
    """

    def setUp(self) -> None:
        # Real corner orientations: R00/R44 display tall, R04/R40 wide.
        self.n_quarter = {
            "R00_SW0": 2,
            "R00_SW1": 4,
            "R04_SW0": 3,
            "R04_SW1": 5,
            "R40_SW0": 1,
            "R40_SW1": 3,
            "R44_SW0": 0,
            "R44_SW1": 2,
        }
        self.layout = _FP_LAYOUT
        self.rects = _focal_plane_axes_rects(self.n_quarter, self.layout)

    def _inches(self):
        """Rects as ``(x0, x1, y0, y1)`` in inches, which is what gaps mean."""
        side = self.layout.fig_side
        return {
            name: (x * side, (x + w) * side, y * side, (y + h) * side)
            for name, (x, y, w, h) in self.rects.items()
        }

    @staticmethod
    def _gap(a, b):
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return max(bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1)

    def testEveryCornerSensorGetsARect(self) -> None:
        self.assertEqual(sorted(self.rects), sorted(self.n_quarter))

    def testIntraCornerSeamsAllMatchTheConfiguredGap(self) -> None:
        """The seam the vignetting arc has to read continuously across.

        Asserted against the literal as well as the config, so this fails on a
        bad *value* and not only on a broken solver -- comparing solely against
        `_FpLayout` would follow the config wherever it went.
        """
        self.assertAlmostEqual(self.layout.gap_intra, 0.44, places=6)
        ext = self._inches()
        for corner, _, _, first, second in _FP_MOSAIC:
            with self.subTest(corner=corner):
                self.assertAlmostEqual(
                    self._gap(ext[first], ext[second]), self.layout.gap_intra, places=6
                )

    def testInterCornerGapsAreEqualBothDirections(self) -> None:
        """Horizontal and vertical, which single ``wspace`` could not do."""
        self.assertAlmostEqual(self.layout.gap_inter, 0.30, places=6)
        ext = self._inches()
        for a, b in (
            ("R00_SW0", "R40_SW1"),
            ("R04_SW1", "R44_SW0"),
            ("R00_SW1", "R04_SW0"),
            ("R40_SW0", "R44_SW0"),
        ):
            with self.subTest(pair=(a, b)):
                self.assertAlmostEqual(
                    self._gap(ext[a], ext[b]), self.layout.gap_inter, places=6
                )

    def testPanelsAreExactlyTheForcedAspect(self) -> None:
        """2:1, not the sensors' true 2.036:1.

        The 1.8% stretch is deliberate: exact 2:1 is what lets the four corner
        blocks tile a square with nothing left over, and the leftover is what
        used to land in the seams.
        """
        for name, (_, _, w, h) in self.rects.items():
            with self.subTest(detector=name):
                self.assertAlmostEqual(max(w, h) / min(w, h), _FP_PANEL_ASPECT, places=6)

    def testContentBoxIsSquareAndPanelsDoNotOverlap(self) -> None:
        ext = self._inches()
        width = max(e[1] for e in ext.values()) - min(e[0] for e in ext.values())
        height = max(e[3] for e in ext.values()) - min(e[2] for e in ext.values())
        self.assertAlmostEqual(width, height, places=6)
        names = sorted(ext)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                with self.subTest(pair=(a, b)):
                    # Two rectangles miss each other exactly when they are
                    # separated along at least one axis, which is what the max
                    # over the four separations being non-negative says.
                    self.assertGreaterEqual(self._gap(ext[a], ext[b]), -1e-9, "panels overlap")

    def testTallAndWideCornersSplitTheirQuadrantDifferently(self) -> None:
        """Which way a pair stacks follows from data, not hardcoded table."""
        ext = self._inches()
        # R00 displays tall (even n_quarter): side by side, so they share rows.
        r00 = (ext["R00_SW1"], ext["R00_SW0"])
        self.assertAlmostEqual(r00[0][2], r00[1][2], places=6)
        self.assertLess(r00[0][1], r00[1][0])
        # R40 displays wide (odd): stacked, so they share columns.
        r40 = (ext["R40_SW1"], ext["R40_SW0"])
        self.assertAlmostEqual(r40[0][0], r40[1][0], places=6)
        self.assertLess(r40[1][3], r40[0][2])

    def testMissingDetectorsStillGetTheirRects(self) -> None:
        """The grid is always the whole focal plane, so gap reads as a gap."""
        rects = _focal_plane_axes_rects({"R00_SW0": 2}, self.layout)
        self.assertEqual(sorted(rects), sorted(self.n_quarter))


class TestPairLinks(unittest.TestCase):
    """Which modes can name a pair, and which only group in bulk."""

    def testPairedModeGivesOneLinkPerPair(self) -> None:
        catalog = _catalog()
        links = _pair_links(catalog)
        self.assertEqual(len(links), 1)
        (det_a, _, _), (det_b, _, _) = links[0]
        # A paired group is exactly two donuts on the two sensors of one
        # corner.
        self.assertEqual({det_a, det_b}, {"R00_SW0", "R00_SW1"})

    def testLinksAreNotFilteredOnFitSuccess(self) -> None:
        """Pairing precedes fitting, so a failed pair was still a pair."""
        catalog = _catalog()
        catalog["fit_success"] = False
        self.assertEqual(len(_pair_links(catalog)), 1)

    def testBulkGroupingModesYieldNoLinks(self) -> None:
        """``full_corner``/``full_detector`` share one group_id across donuts.

        "The pair" is then not something the id identifies, so these return
        nothing rather than a combinatorial tangle.
        """
        for mode in ("unpaired", "full_corner", "full_detector"):
            with self.subTest(mode=mode):
                catalog = _catalog()
                catalog.meta["wf_mode"] = mode
                self.assertEqual(_pair_links(catalog), [])


class TestFocalPlanePlot(unittest.TestCase):
    """Focal-plane selection plot, driven through the real table builders."""

    def _tables(self, results):
        return (
            _build_detector_image_table(results, _VISIT_ID, "LSSTCam"),
            _build_overlay_table(results, _VISIT_ID, "LSSTCam"),
        )

    def _write(self, catalog, images, overlays):
        task = DonutBlitzPlotTask(config=DonutBlitzPlotConfig())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                task.run(catalog, detector_images=images, selection_overlays=overlays)
                return sorted(os.listdir(tmp))
            finally:
                os.chdir(cwd)

    def testCatalogAloneStillWritesTheOtherTwoPlots(self) -> None:
        """The standalone-degradation contract.

        Run without ``doSelectionOutput`` and this task gets the catalog only;
        the focal-plane plot is skipped and the other two are unaffected.
        """
        self.assertEqual(
            self._write(_catalog(), None, None),
            [f"donut_diag_{_VISIT_ID}.png", f"wf_diag_{_VISIT_ID}.png"],
        )

    def testSelectionPlotIsWrittenWhenTheTablesArePresent(self) -> None:
        results = [_result_with_view("R00_SW0", 2), _result_with_view("R00_SW1", 4)]
        images, overlays = self._tables(results)
        self.assertEqual(
            self._write(_catalog(), images, overlays),
            [
                f"donut_diag_{_VISIT_ID}.png",
                f"selection_diag_{_VISIT_ID}.png",
                f"wf_diag_{_VISIT_ID}.png",
            ],
        )

    def testADetectorMissingFromTheImageTableLeavesItsCellBlank(self) -> None:
        """Only R00, but the figure still spans the whole focal plane."""
        images, overlays = self._tables([_result_with_view("R00_SW0", 2)])
        self.assertEqual(images["det_name"].tolist(), ["R00_SW0"])
        written = self._write(_catalog(), images, overlays)
        self.assertIn(f"selection_diag_{_VISIT_ID}.png", written)

    def testOverlaysAreOptional(self) -> None:
        """Images without overlays draw pixels and donuts, not nothing."""
        results = [_result_with_view("R00_SW0", 2), _result_with_view("R00_SW1", 4)]
        images, _ = self._tables(results)
        self.assertIn(f"selection_diag_{_VISIT_ID}.png", self._write(_catalog(), images, None))

    def testNoRefcatStillProducesOverlayRowsForTheOtherStages(self) -> None:
        results = [_result_with_view("R00_SW0", 2, refcat=False)]
        images, overlays = self._tables(results)
        self.assertFalse(bool(images["has_refcat"][0]))
        self.assertNotIn("refcat", set(overlays["kind"].tolist()))
        self.assertEqual({"detection", "selection"}, set(overlays["kind"].tolist()))


class TestFocalPlaneOrientation(unittest.TestCase):
    """The regression guard for the ``origin="upper"`` bug.

    The display transform is ``np.rot90(arr, -n_quarter).T``, and that
    transpose is a *reflection*: it inverts handedness.  Drawing the result
    with y increasing upward leaves the inversion in and mirrors all eight
    panels, which put the most-vignetted corner of each sensor toward the
    focal-plane center instead of away from it.  Row-downward display supplies
    the compensating flip.

    The ``field_dist`` grid comes from the real camera rather than a synthetic
    ramp, because that is the whole content of the check: a made-up gradient
    has an arbitrary direction, so it cannot know which side of a sensor faces
    the focal-plane center, and asserting against it only pins the invention.
    With real geometry the assertion is the one that failed under
    ``origin="lower"``: a panel's most-vignetted pixel must land farther from
    the figure center than its least-vignetted one.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
        from lsst.obs.lsst import LsstCam

        camera = LsstCam.getCamera()
        cls.panels = {}
        for _, _, _, first, second in _FP_MOSAIC:
            for det_name in (first, second):
                detector = camera[det_name]
                bbox = detector.getBBox()
                height, width = bbox.getHeight(), bbox.getWidth()
                # Coarse: only the extreme cells matter, and a full-resolution
                # grid would make this the slowest test in the file.
                step = 8
                binning = _VIEW_BINNING * step
                hb, wb = height // binning, width // binning
                offset = (binning - 1) / 2
                xs = np.arange(wb) * binning + offset + bbox.getMinX()
                ys = np.arange(hb) * binning + offset + bbox.getMinY()
                gx, gy = np.meshgrid(xs, ys)
                mapping = detector.getTransform(PIXELS, FIELD_ANGLE).getMapping()
                angles = mapping.applyForward(np.vstack([gx.ravel(), gy.ravel()]))
                field = np.degrees(np.hypot(angles[0], angles[1])).reshape(hb, wb)
                cls.panels[det_name] = (
                    detector.getOrientation().getNQuarter(),
                    field,
                )

    def _panel_extremes(self, det_name):
        """Figure-space positions of a panel's min- and max-field pixels.

        Rendered by the real `_drawFocalPlanePanel` and read back through the
        axes' own transform, rather than recomputed here.  That distinction is
        the test: deriving the y convention locally would only check this
        method's arithmetic and would pass no matter what the panel did.

        What actually orients the panel is the descending ``set_ylim``.
        Mutating it makes this test fail; mutating ``origin`` alone does not,
        because `imshow` sets ``ylim`` from ``origin`` and the panel then
        overrides it -- so the explicit limit is what survives.  The two agree,
        and the comment in `_drawFocalPlanePanel` explains why both are spelled
        out, but only one of them is load-bearing and it is the ``ylim``.
        """
        n_quarter, field = self.panels[det_name]
        hb, wb = field.shape
        rng = np.random.default_rng(0)
        row = {
            "det_name": det_name,
            "det_id": -1,
            "n_quarter": n_quarter,
            "binning": _VIEW_BINNING,
            "bbox_min_x": 0,
            "bbox_min_y": 0,
            "image": rng.normal(12.0, 2.0, (hb, wb)).astype(np.float32),
            "field_dist": field * u.deg,
            "max_field_dist": 1.725 * u.deg,
            "has_refcat": True,
            "selection_source": "refcat",
        }

        from matplotlib.figure import Figure

        rects = _focal_plane_axes_rects({name: nq for name, (nq, _) in self.panels.items()}, _FP_LAYOUT)
        fig = Figure(figsize=(_FP_LAYOUT.fig_side, _FP_LAYOUT.fig_side))
        ax = fig.add_axes(rects[det_name])
        task = DonutBlitzPlotTask(config=DonutBlitzPlotConfig())
        task._drawFocalPlanePanel(ax, row, QTable(), {}, _FP_LAYOUT)

        display = np.rot90(field, -n_quarter).T
        dw = display.shape[1]

        def to_figure(flat):
            y, x = divmod(int(flat), dw)
            # Array index straight through the axes' own transform: the y
            # direction is whatever the panel's `set_ylim` made it, so nothing
            # about the orientation convention is re-derived here.
            px, py = ax.transData.transform((x, y))
            inv = fig.transFigure.inverted().transform((px, py))
            return float(inv[0]), float(inv[1])

        return to_figure(np.argmin(display)), to_figure(np.argmax(display))

    def testMostVignettedEndLandsFartherFromTheFigureCenter(self) -> None:
        """All eight corner sensors, against real camera geometry.

        Under ``origin="lower"`` the tall panels (R00, R44) fail this.  The
        wide ones survive *this metric* -- their max-field cell is displaced
        mostly horizontally, and a y-flip does not reverse that comparison --
        which is worth knowing, because a 4-of-8 measurement is exactly what
        once led to blaming `_FP_MOSAIC`'s within-corner detector order instead
        of the origin. The pair order was not the bug; do not "fix" it on such
        a number.
        """
        for det_name in sorted(self.panels):
            with self.subTest(detector=det_name):
                lo, hi = self._panel_extremes(det_name)
                self.assertGreater(
                    np.hypot(hi[0] - 0.5, hi[1] - 0.5),
                    np.hypot(lo[0] - 0.5, lo[1] - 0.5),
                    f"{det_name}: max-field end is not farther from the figure center",
                )


if __name__ == "__main__":
    unittest.main()
