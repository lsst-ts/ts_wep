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

"""The output catalog builder, shared by corner and full-array mode.
"""

import unittest

import numpy as np

from lsst.ts.wep.blitz.catalogBuilder import CatalogOptions, build_donut_catalog
from lsst.ts.wep.blitz.dataStructures import Donut, WfResult
from lsst.ts.wep.blitz.donutBlitzMonolithTask import (
    DonutBlitzMonolithTask,
    DonutBlitzMonolithTaskConfig,
)
from lsst.ts.wep.blitz.utils import _ZK_JMAX

_IMAGE_COLUMNS = ("stamp", "wf_img", "model_img")


def _donut(det_name="R00_SW0", det_id=191, donut_id=1, **overrides):
    """A Donut with plausible scalars and a small stamp."""
    kwargs = dict(
        det_name=det_name,
        stamp=np.ones((167, 167), dtype=np.float32),
        thx_ccs=0.01,
        thy_ccs=0.02,
        flux=1e5,
        band="r",
        det_id=det_id,
        visit_id=2026070900036,
        centroid_x_raw=100.0,
        centroid_y_raw=200.0,
        id=donut_id,
        inner_frac=0.01,
        outer_frac=0.02,
        outer_sector_minmax_frac=0.03,
        field_dist_deg=1.5,
        donut_radius=60.0,
        obscuration=0.612,
        snr=500.0,
        bkg=10.0,
        bkg_std=3.0,
        nearest_neighbor_dist_px=250.0,
        n_neighbors_in_stamp=0,
        catalog_centroid_offset_px=0.5,
        n_quarter=0,
        nearby_photo=[(1.0, 2.0, 15.0)],
        nearby_astrom=[],
    )
    kwargs.update(overrides)
    return Donut(**kwargs)


def _result(det_name="R00_SW0", rejected=()):
    """A per-detector cutout result dict, as the workers return."""
    return {
        "det_name": det_name,
        "scatter_arcsec": 0.3,
        "wcs_refit_error": "",
        "cat_select_error": "",
        "rejected_catalog": list(rejected),
        # Both provenance fields are unconditional in the real results: the cutout
        # pipeline always sets a selection_source, and it seeds pair_path with
        # "n/a" for the grouping stage to overwrite.
        "selection_source": "refcat",
        "pair_path": "n/a",
    }


def _options(**overrides) -> CatalogOptions:
    kwargs = dict(
        stamp_size=167,
        binning=2,
        noll_indices=tuple(range(4, 20)),
        aperture_margin_frac=0.05,
        bkg_inner_disc_frac=0.67,
        bkg_annulus_inner_frac=1.25,
        bkg_annulus_outer_frac=1.4,
        max_donuts=8,
        wf_mode="paired",
    )
    kwargs.update(overrides)
    return CatalogOptions(**kwargs)


class TestCatalogOptions(unittest.TestCase):
    """The options object and its derived quantities."""

    def testDerivedSizes(self) -> None:
        # 167 // 2 = 83, already odd.
        self.assertEqual(_options().wf_img_size, 83)
        # 168 // 2 = 84, forced odd downwards -- the case the fitter's
        # _bin_stamp_odd handles by trimming.
        self.assertEqual(_options(stamp_size=168).wf_img_size, 83)
        self.assertEqual(_options(noll_indices=(4, 11, 22)).zk_dev_jmax, 22)

    def testMonolithWiresEveryFieldFromConfig(self) -> None:
        """Corner mode's options come from the config fields they claim to.

        The extraction replaced ~10 ``self.<subtask>.config.<field>`` reads with
        an options object; a crossed pair of floats here would silently change
        the annulus geometry recorded in ``meta`` and drawn on the plots.
        """
        config = DonutBlitzMonolithTaskConfig()
        config.cutStampsTask.stampSize = 215
        config.cutStampsTask.maxDonuts = 11
        config.wfFittingTask.binning = 3
        # Must be pair-complete: the config rejects a truncated +/-m doublet
        # because the coefficients could not then be rotated between frames.
        config.wfFittingTask.nollIndices = [4, 5, 6, 7, 8]
        config.measureCandidatesTask.apertureMarginFrac = 0.11
        config.measureCandidatesTask.bkgInnerDiscFrac = 0.22
        config.measureCandidatesTask.bkgAnnulusInnerFrac = 1.33
        config.measureCandidatesTask.bkgAnnulusOuterFrac = 1.44
        config.wfEstimationMode = "unpaired"
        config.saveStamps = False

        options = DonutBlitzMonolithTask(config=config)._catalogOptions()

        self.assertEqual(options.stamp_size, 215)
        self.assertEqual(options.max_donuts, 11)
        self.assertEqual(options.binning, 3)
        self.assertEqual(options.noll_indices, (4, 5, 6, 7, 8))
        self.assertAlmostEqual(options.aperture_margin_frac, 0.11)
        self.assertAlmostEqual(options.bkg_inner_disc_frac, 0.22)
        self.assertAlmostEqual(options.bkg_annulus_inner_frac, 1.33)
        self.assertAlmostEqual(options.bkg_annulus_outer_frac, 1.44)
        self.assertEqual(options.wf_mode, "unpaired")
        self.assertFalse(options.save_stamps)
        # Corner mode's tables are small, so the WF images always come along.
        self.assertTrue(options.save_wf_images)


class TestBuildDonutCatalog(unittest.TestCase):
    """Row content and the optional image columns."""

    def testEmptyInputGivesEmptyTable(self) -> None:
        table = build_donut_catalog([], [], [], [], 1, _options())
        self.assertEqual(len(table), 0)

    def testOneRowPerDonutIncludingRejected(self) -> None:
        accepted = _donut(donut_id=1)
        rejected = _donut(donut_id=2, rejected=True, rejected_snr=True)
        table = build_donut_catalog(
            [_result(rejected=[rejected])], [], [accepted], [], 42, _options()
        )
        self.assertEqual(len(table), 2)
        by_id = {int(r["id"]): r for r in table}
        self.assertTrue(bool(by_id[1]["candidate"]))
        self.assertFalse(bool(by_id[2]["candidate"]))
        # Nothing was fitted, so nothing is "used" and the deviations are NaN.
        self.assertFalse(any(bool(r["used"]) for r in table))
        self.assertTrue(np.all(np.isnan(table["zk_dev_ccs"].value)))

    def testDedupeKeepsSurplusDonutsOnce(self) -> None:
        """A donut in both `donuts` and `unmatched_donuts` yields one row."""
        d = _donut(donut_id=7)
        table = build_donut_catalog([_result()], [], [d], [d], 42, _options())
        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table[0]["candidate"]))

    def testSameStarOnBothSidesOfFocusKeepsBothRows(self) -> None:
        """The full-array case that ``(det_name, id)`` alone would have collapsed.

        FAM cuts the same star on the same detector once per side of focus, so the
        two donuts share an ``id`` -- the same refcat source, or the same ``1..N``
        blind-detection slot. Only ``visit_id`` separates them, in the row key and
        in the wavefront-result lookup.
        """
        intra_visit, extra_visit = 2026070900036, 2026070900037
        intra = _donut(donut_id=7, visit_id=intra_visit)
        extra = _donut(donut_id=7, visit_id=extra_visit)
        # Distinguishable fits, one per side, so a mixed-up lookup is visible.
        wf_results = [
            {
                "donuts": [
                    WfResult(
                        donut_id=7,
                        det_name="R00_SW0",
                        visit_id=visit,
                        zk_dev=np.full(_ZK_JMAX + 1, value),
                        zk_intrinsic=np.zeros(_ZK_JMAX + 1),
                        img=None,
                        model_img=None,
                        fit_success=True,
                        fit_elapsed=1.0,
                        setup_elapsed=0.1,
                        fit_nfev=10,
                        fit_cost=1.0,
                        fit_dx=0.0,
                        fit_dy=0.0,
                        fit_flux=1e5,
                        fit_fwhm=1.0,
                        blend_frac=0.0,
                        group_id="g",
                        group_size=2,
                        fit_mode="paired",
                    )
                ]
            }
            for visit, value in ((intra_visit, 1e-6), (extra_visit, 2e-6))
        ]
        results = [
            {**_result(), "visit_id": visit} for visit in (intra_visit, extra_visit)
        ]
        table = build_donut_catalog(
            results, wf_results, [intra, extra], [], extra_visit, _options()
        )

        self.assertEqual(len(table), 2)
        by_visit = {int(r["visit_id"]): r for r in table}
        self.assertEqual(set(by_visit), {intra_visit, extra_visit})
        # Each row picked up its own side's fit, in um.
        self.assertAlmostEqual(by_visit[intra_visit]["zk_dev_ccs"][4].value, 1.0)
        self.assertAlmostEqual(by_visit[extra_visit]["zk_dev_ccs"][4].value, 2.0)
        # Per-detector metadata is likewise kept per exposure, not overwritten.
        self.assertEqual(
            set(table.meta["det_meta"]),
            {f"R00_SW0_{intra_visit}", f"R00_SW0_{extra_visit}"},
        )

    def testImageColumnsAreOptionalAndIndependent(self) -> None:
        args = ([_result()], [], [_donut()], [], 42)
        cases = {
            (True, True): ("stamp", "wf_img", "model_img"),
            (True, False): ("stamp",),
            (False, True): ("wf_img", "model_img"),
            (False, False): (),
        }
        for (save_stamps, save_wf), expected in cases.items():
            options = _options(save_stamps=save_stamps, save_wf_images=save_wf)
            table = build_donut_catalog(*args, options)
            present = tuple(c for c in _IMAGE_COLUMNS if c in table.colnames)
            self.assertEqual(
                present,
                expected,
                msg=f"save_stamps={save_stamps} save_wf_images={save_wf}",
            )
            # Dropping images must not disturb anything else.
            self.assertIn("zk_dev_ccs", table.colnames)
            self.assertIn("snr", table.colnames)
            self.assertEqual(len(table), 1)

    def testMetaCarriesOptionsAndGeometry(self) -> None:
        options = _options(binning=4, max_donuts=3, wf_mode="full_detector")
        table = build_donut_catalog(
            [_result()], [], [_donut()], [], 99, options, rtp_rad=0.25
        )
        self.assertEqual(table.meta["visit_id"], 99)
        self.assertEqual(table.meta["binning"], 4)
        self.assertEqual(table.meta["max_donuts"], 3)
        self.assertEqual(table.meta["wf_mode"], "full_detector")
        self.assertEqual(table.meta["zk_dev_jmax"], max(options.noll_indices))
        self.assertAlmostEqual(table.meta["rot_tel_pos"], np.degrees(0.25))
        self.assertAlmostEqual(
            table.meta["bkg_annulus_outer_frac"], options.bkg_annulus_outer_frac
        )
        self.assertAlmostEqual(
            table.meta["bkg_inner_disc_frac"], options.bkg_inner_disc_frac
        )
        # Per-detector metadata survives for the plots.  Corner mode supplies no
        # per-result visit_id, so the key falls back to this table's visit.
        self.assertIn("R00_SW0_99", table.meta["det_meta"])
        self.assertAlmostEqual(
            table.meta["det_meta"]["R00_SW0_99"]["scatter_arcsec"], 0.3
        )

    def testDetMetaCarriesIdProvenance(self) -> None:
        """det_meta records where the donut ids came from, and how they paired.

        Without this a consumer cannot tell a Monster source id from a blind
        detection's 1..N slot, and a FAM run that quietly fell back to spatial
        pairing is invisible once the run's logs are gone.
        """
        results = [
            {
                **_result(),
                "visit_id": 1,
                "selection_source": "refcat",
                "pair_path": "refcat_id",
            },
            {
                **_result(),
                "visit_id": 2,
                "selection_source": "blind_selected",
                "pair_path": "spatial",
            },
        ]
        table = build_donut_catalog(results, [], [_donut()], [], 42, _options())
        det_meta = table.meta["det_meta"]
        self.assertEqual(det_meta["R00_SW0_1"]["selection_source"], "refcat")
        self.assertEqual(det_meta["R00_SW0_1"]["pair_path"], "refcat_id")
        self.assertEqual(det_meta["R00_SW0_2"]["selection_source"], "blind_selected")
        self.assertEqual(det_meta["R00_SW0_2"]["pair_path"], "spatial")

    def testDetMetaProvenanceRecordsNotApplicable(self) -> None:
        """"Nothing to report" is a named token, not an empty string.

        The producers cover every case -- corner mode's ``snr_rank``/``n/a``, the
        no-detections early return's ``no_detections`` -- so the builder copies
        both fields verbatim with no default. An empty string reaching ``det_meta``
        would therefore be a bug, not "not applicable", and these are the values
        that used to arrive as ``None`` or absent.
        """
        results = [
            {**_result(), "selection_source": "no_detections", "pair_path": "n/a"}
        ]
        table = build_donut_catalog(results, [], [_donut()], [], 42, _options())
        entry = table.meta["det_meta"]["R00_SW0_42"]
        self.assertEqual(entry["selection_source"], "no_detections")
        self.assertEqual(entry["pair_path"], "n/a")


if __name__ == "__main__":
    unittest.main()
