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

import astropy.units as u
import numpy as np
from astropy.table import Table

import lsst.geom as geom
from lsst.afw.image import VisitInfo
from lsst.daf.base import DateTime
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, astropy_to_arrow

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
        n_quarter=0,
        photo_mag=14.0,
        astrom_mag=14.2,
        nearby_photo=[(1.0, 2.0, 15.0)],
        nearby_astrom=[],
        # Radians on the Donut, degrees in the catalog column.
        coord_ra=np.radians(30.0),
        coord_dec=np.radians(-20.0),
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
        "n_quarter": 0,
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
        self.assertEqual(_options(noll_indices=(4, 11, 22)).zk_deviation_jmax, 22)

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
        by_id = {int(r["donut_id"]): r for r in table}
        self.assertTrue(bool(by_id[1]["candidate"]))
        self.assertFalse(bool(by_id[2]["candidate"]))
        # No fit claimed either donut, so both group_ids are empty and the
        # deviations are NaN.
        self.assertEqual({str(r["group_id"]) for r in table}, {""})
        self.assertFalse(any(bool(r["group_fit_success"]) for r in table))
        self.assertTrue(np.all(np.isnan(table["zk_deviation_ccs"].value)))
        # `rejected` is not a column: the rejected_* reasons carry it, and their
        # OR is exactly ~candidate.
        self.assertNotIn("rejected", table.colnames)
        self.assertTrue(bool(by_id[2]["rejected_snr"]))

    def testDedupeKeepsSurplusDonutsOnce(self) -> None:
        """A donut in both `donuts` and `unmatched_donuts` yields one row."""
        d = _donut(donut_id=7)
        table = build_donut_catalog([_result()], [], [d], [d], 42, _options())
        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table[0]["candidate"]))

    def testSameStarOnBothSidesOfFocusKeepsBothRows(self) -> None:
        """The full-array case that ``(det_name, donut_id)`` alone would have collapsed.

        FAM cuts the same star on the same detector once per side of focus, so the
        two donuts share a ``donut_id`` -- the same refcat source, or the same ``1..N``
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
                        fit_optimality=1e-8,
                        fit_njev=9,
                        fit_outcome="ok",
                        fit_dx=0.0,
                        fit_dy=0.0,
                        fit_flux=1e5,
                        fit_fwhm=1.0,
                        blend_frac=0.0,
                        group_id="g",
                        group_size=2,
                    )
                ]
            }
            for visit, value in ((intra_visit, 1e-6), (extra_visit, 2e-6))
        ]
        results = [
            {**_result(), "visit_id": visit} for visit in (intra_visit, extra_visit)
        ]
        table = build_donut_catalog(
            results,
            wf_results,
            [intra, extra],
            [],
            extra_visit,
            _options(),
            intra_visit_id=intra_visit,
            extra_visit_id=extra_visit,
        )

        self.assertEqual(len(table), 2)
        # Which visit was which side, so a consumer need not infer it. Set inside
        # the builder for both modes, so meta stays schema-identical.
        self.assertEqual(table.meta["intra_visit_id"], intra_visit)
        self.assertEqual(table.meta["extra_visit_id"], extra_visit)
        self.assertEqual(table.meta["ref_visit_id"], extra_visit)
        by_visit = {int(r["visit_id"]): r for r in table}
        self.assertEqual(set(by_visit), {intra_visit, extra_visit})
        # Each row picked up its own side's fit, in um.
        self.assertAlmostEqual(by_visit[intra_visit]["zk_deviation_ccs"][4].value, 1.0)
        self.assertAlmostEqual(by_visit[extra_visit]["zk_deviation_ccs"][4].value, 2.0)
        # Per-detector metadata is likewise kept per exposure, not overwritten.
        self.assertEqual(
            set(table.meta["det_meta"]),
            {f"R00_SW0_{intra_visit}", f"R00_SW0_{extra_visit}"},
        )

    def testGroupIdIsCarriedAndGroupColumnsAreReplicated(self) -> None:
        """``group_id`` labels the fit; ``group_*`` repeat across its rows.

        The label comes straight off the `WfResult`, so a consumer can collapse
        the replicated group-level values to one measurement per fit -- which an
        array index into a list that no longer exists could not support.  A
        donut no fit claimed gets the empty string rather than a sentinel int.
        """
        paired = [_donut(donut_id=1), _donut(donut_id=2)]
        surplus = _donut(donut_id=3)
        gid = "R00_SW0_6222323257218103680_6219321315596159872"
        wf_results = [
            {
                "donuts": [
                    WfResult(
                        donut_id=d.donut_id,
                        det_name=d.det_name,
                        visit_id=d.visit_id,
                        zk_dev=np.full(_ZK_JMAX + 1, 1e-6),
                        zk_intrinsic=np.zeros(_ZK_JMAX + 1),
                        img=None,
                        model_img=None,
                        fit_success=True,
                        fit_elapsed=12.5,
                        setup_elapsed=0.75,
                        fit_nfev=40,
                        fit_cost=3.5,
                        fit_optimality=2.5e-9,
                        fit_njev=38,
                        fit_outcome="ok",
                        # Per-donut even inside a joint fit, so these differ.
                        fit_dx=0.1 * i,
                        fit_dy=0.2 * i,
                        fit_flux=1e5,
                        fit_fwhm=0.9,
                        blend_frac=0.01 * i,
                        group_id=gid,
                        group_size=2,
                    )
                    for i, d in enumerate(paired)
                ]
            }
        ]
        table = build_donut_catalog(
            [_result()],
            wf_results,
            paired + [surplus],
            [surplus],
            42,
            _options(),
        )

        by_id = {int(r["donut_id"]): r for r in table}
        self.assertEqual({str(by_id[i]["group_id"]) for i in (1, 2)}, {gid})
        # No fit claimed the surplus donut: empty label, and nothing to succeed.
        self.assertEqual(str(by_id[3]["group_id"]), "")
        self.assertFalse(bool(by_id[3]["group_fit_success"]))
        # "" is the outcome reserved for "no group claimed it", distinct from
        # every failure mode a claimed donut could report.
        self.assertEqual(str(by_id[3]["group_fit_outcome"]), "")
        # Group-level values are identical on both rows of the group.
        for col, value in (
            ("group_size", 2),
            ("group_fit_success", True),
            ("group_fit_elapsed", 12.5 * u.s),
            ("group_setup_elapsed", 0.75 * u.s),
            ("group_fit_nfev", 40),
            ("group_fit_njev", 38),
            ("group_fit_cost", 3.5),
            ("group_fit_optimality", 2.5e-9),
            ("group_fit_outcome", "ok"),
            ("group_fwhm", 0.9 * u.arcsec),
        ):
            self.assertEqual(by_id[1][col], value, msg=col)
            self.assertEqual(by_id[2][col], value, msg=col)
        # Per-donut values are not.
        self.assertNotEqual(by_id[1]["fit_dx"], by_id[2]["fit_dx"])
        self.assertNotEqual(by_id[1]["blend_frac"], by_id[2]["blend_frac"])
        # The pre-rename names are gone, so a stale consumer fails loudly.
        for gone in ("group", "used", "rejected", "fit_success", "fit_elapsed"):
            self.assertNotIn(gone, table.colnames)

    def testDefocalOffsetsCarryTheSideOfFocus(self) -> None:
        """The triplet is the defocal state, and its sign is the side.

        Both modes negate the extra-focal triplet to get the intra-focal one, so
        a consumer reads the side off the sign without needing to know that
        SW0/SW1 straddle focus (corner) or which visit was intra (FAM).
        """
        extra = _donut(donut_id=1, defocal_offsets=(+1.5e-3, 0.0, 0.0))
        intra = _donut(donut_id=2, defocal_offsets=(-1.5e-3, 0.0, 0.0))
        table = build_donut_catalog(
            [_result()], [], [extra, intra], [], 42, _options()
        )
        by_id = {int(r["donut_id"]): r for r in table}
        self.assertEqual(table["defocal_offsets"].unit, u.m)
        np.testing.assert_allclose(
            by_id[1]["defocal_offsets"].to_value(u.m), [1.5e-3, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            by_id[2]["defocal_offsets"].to_value(u.m), [-1.5e-3, 0.0, 0.0]
        )

    def testDefocalOffsetsAreNaNWhenUnannotated(self) -> None:
        """A donut that never reached the fitter still gets a well-shaped row."""
        table = build_donut_catalog(
            [_result()], [], [_donut(defocal_offsets=None)], [], 42, _options()
        )
        offsets = table["defocal_offsets"].to_value(u.m)
        self.assertEqual(offsets.shape, (1, 3))
        self.assertTrue(np.all(np.isnan(offsets)))

    def testSkyPositionIsCarriedInDegrees(self) -> None:
        """The refcat position, converted from the Donut's afw-native radians.

        This is what any cross-visit or cross-detector match needs; the field
        angle plus the boresight meta only gets there approximately.
        """
        table = build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
        self.assertEqual(table["coord_ra"].unit, u.deg)
        self.assertEqual(table["coord_dec"].unit, u.deg)
        self.assertAlmostEqual(table["coord_ra"][0].to_value(u.deg), 30.0)
        self.assertAlmostEqual(table["coord_dec"][0].to_value(u.deg), -20.0)

    def testSkyPositionIsNaNOffTheRefcatPath(self) -> None:
        """NaN, not a WCS projection of the centroid.

        Mixing refcat truth and a projected centroid into one column would
        reproduce the mixed provenance that `donut_id` already carries.
        """
        table = build_donut_catalog(
            [_result()],
            [],
            [_donut(coord_ra=float("nan"), coord_dec=float("nan"))],
            [],
            42,
            _options(),
        )
        self.assertTrue(np.isnan(table["coord_ra"][0].to_value(u.deg)))
        self.assertTrue(np.isnan(table["coord_dec"][0].to_value(u.deg)))

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
            self.assertIn("zk_deviation_ccs", table.colnames)
            self.assertIn("snr", table.colnames)
            self.assertEqual(len(table), 1)

    def testMetaCarriesOptionsAndGeometry(self) -> None:
        options = _options(binning=4, max_donuts=3, wf_mode="full_detector")
        table = build_donut_catalog(
            [_result()], [], [_donut()], [], 99, options, rtp_rad=0.25
        )
        self.assertEqual(table.meta["ref_visit_id"], 99)
        # Corner mode has one exposure holding both sides of focus, so the two
        # side keys default to it rather than being left out for FAM to add.
        self.assertEqual(table.meta["intra_visit_id"], 99)
        self.assertEqual(table.meta["extra_visit_id"], 99)
        self.assertEqual(table.meta["binning"], 4)
        self.assertEqual(table.meta["max_donuts"], 3)
        self.assertEqual(table.meta["wf_mode"], "full_detector")
        # Both jmax keys name which Zernike set they bound; neither is bare.
        self.assertEqual(table.meta["zk_deviation_jmax"], max(options.noll_indices))
        self.assertEqual(table.meta["zk_intrinsic_jmax"], _ZK_JMAX)
        self.assertAlmostEqual(
            table.meta["rot_tel_pos"].to_value(u.deg), np.degrees(0.25)
        )
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
            table.meta["det_meta"]["R00_SW0_99"]["astrom_scatter"].to_value(u.arcsec),
            0.3,
        )

    def testIntrinsicsSurviveAnUnfittedRow(self) -> None:
        """Intrinsics come off the donut when no fit claimed the row.

        Intrinsic Zernikes are a function of field position, not of the fit,
        so a row no fit consumed still has them -- taking them from
        `_NULL_WF` made them all-NaN even where the calibration was known.
        Deviations *are* a measurement, so they stay NaN.
        """
        d = _donut(donut_id=1)
        d.intrinsic_zk = np.full(_ZK_JMAX + 1 - 4, 2.0)  # µm, Noll 4.._ZK_JMAX
        table = build_donut_catalog([_result()], [], [d], [], 42, _options())

        self.assertEqual(str(table["group_id"][0]), "")  # nothing fitted it
        intrinsic = table["zk_intrinsic_ccs"][0].to_value(u.micron)
        self.assertFalse(np.any(np.isnan(intrinsic)))
        self.assertAlmostEqual(intrinsic[4], 2.0)
        # Noll 0..3 are carried for indexing only and are always 0.0.
        np.testing.assert_array_equal(intrinsic[:4], np.zeros(4))
        # The fit result, by contrast, is genuinely absent.
        self.assertTrue(np.all(np.isnan(table["zk_deviation_ccs"][0].to_value(u.micron))))

    def testIntrinsicsAreZeroNotNaNWithoutCalibration(self) -> None:
        """No calibration is 0.0, which is distinct from "no fit" being NaN."""
        d = _donut(donut_id=1)
        d.intrinsic_zk = None
        table = build_donut_catalog([_result()], [], [d], [], 42, _options())
        intrinsic = table["zk_intrinsic_ccs"][0].to_value(u.micron)
        self.assertFalse(np.any(np.isnan(intrinsic)))
        np.testing.assert_array_equal(intrinsic, np.zeros_like(intrinsic))

    def testObservationMetaIsPresentWithoutAVisitInfo(self) -> None:
        """The meta schema must not depend on what the caller passed.

        A consumer reading ``meta["date"]`` should find an empty value, not a
        KeyError, when no `VisitInfo` reached the builder -- otherwise every
        reader needs a `.get()` and the two modes could drift apart silently.

        The empty value is None for ``date`` alone, because a `Time` cannot
        hold NaN and a masked one will not serialize; the rest are NaN
        Quantities that still declare what they would have measured.
        """
        table = build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
        self.assertIn("date", table.meta)
        self.assertIsNone(table.meta["date"])
        for key, physical_type in (
            ("exposure_time", "time"),
            ("boresight_alt", "angle"),
            ("boresight_az", "angle"),
            ("boresight_rot_angle", "angle"),
            ("boresight_par_angle", "angle"),
        ):
            self.assertIn(key, table.meta)
            self.assertTrue(np.isnan(table.meta[key].value), msg=key)
            self.assertEqual(table.meta[key].unit.physical_type, physical_type, msg=key)
        self.assertEqual(table.meta["mode"], "")
        self.assertEqual(table.meta["instrument"], "")
        # Versions are read off the installed packages, so they are always set.
        for key in ("ts_wep_version", "danish_version", "batoid_version"):
            self.assertIn(key, table.meta)
            self.assertIsInstance(table.meta[key], str)

    def testObservationMetaIsReadFromAVisitInfo(self) -> None:
        """And the three rotation angles stay distinguishable.

        ``rot_tel_pos`` is *derived* from the other two and is the one the
        ``_ocs`` columns were rotated by, so all three are recorded under
        names that cannot be confused for one another.
        """
        visit_info = VisitInfo(
            exposureTime=15.0,
            date=DateTime(61293.229166, DateTime.MJD, DateTime.TAI),
            boresightAzAlt=geom.SpherePoint(120.0 * geom.degrees, 65.0 * geom.degrees),
            boresightRotAngle=30.0 * geom.degrees,
        )
        table = build_donut_catalog(
            [_result()], [], [_donut()], [], 42, _options(),
            mode="fam", visit_info=visit_info, instrument="LSSTCam",
            rtp_rad=np.deg2rad(12.5),
        )
        self.assertEqual(table.meta["mode"], "fam")
        self.assertEqual(table.meta["instrument"], "LSSTCam")
        # The date carries its own scale, so reading it back in TAI is explicit
        # rather than conventional.
        self.assertAlmostEqual(table.meta["date"].tai.mjd, 61293.229166)
        self.assertAlmostEqual(table.meta["exposure_time"].to_value(u.s), 15.0)
        self.assertAlmostEqual(table.meta["boresight_alt"].to_value(u.deg), 65.0)
        self.assertAlmostEqual(table.meta["boresight_az"].to_value(u.deg), 120.0)
        self.assertAlmostEqual(table.meta["boresight_rot_angle"].to_value(u.deg), 30.0)
        # The derived angle is not either input.
        self.assertAlmostEqual(table.meta["rot_tel_pos"].to_value(u.deg), 12.5)

    def testMetaSurvivesTheArrowRoundTrip(self) -> None:
        """Units in ``meta`` must survive the way the catalog is persisted.

        The table goes to the butler as ``ArrowAstropy``, which yaml-dumps
        ``meta`` into the parquet schema.  Not every object survives that -- a
        masked `Time`, for one, raises on the dump -- and a value that does not
        would fail at ``butler.put``, inside a pipeline run, rather than here.
        Nested dicts are covered too: ``det_meta`` and ``butler_times`` hold
        Quantities one level down.

        The `Table` wrapper is what both tasks put (the columns' units move
        into the schema rather than the values), so the round trip starts
        where the real one does.
        """
        visit_info = VisitInfo(
            exposureTime=15.0,
            date=DateTime(61293.229166, DateTime.MJD, DateTime.TAI),
            boresightAzAlt=geom.SpherePoint(120.0 * geom.degrees, 65.0 * geom.degrees),
            boresightRotAngle=30.0 * geom.degrees,
        )
        table = build_donut_catalog(
            [{**_result(), "isr_run": 1.25}], [], [_donut()], [], 42, _options(),
            visit_info=visit_info, run_elapsed=12.5,
            butler_times={"raw": 2.0}, rtp_rad=np.deg2rad(12.5),
        )

        meta = arrow_to_astropy(astropy_to_arrow(Table(table))).meta

        self.assertAlmostEqual(meta["date"].tai.mjd, 61293.229166)
        self.assertAlmostEqual(meta["boresight_alt"].to_value(u.deg), 65.0)
        self.assertAlmostEqual(meta["exposure_time"].to_value(u.s), 15.0)
        self.assertAlmostEqual(meta["rot_tel_pos"].to_value(u.deg), 12.5)
        self.assertAlmostEqual(meta["run_elapsed"].to_value(u.s), 12.5)
        self.assertAlmostEqual(meta["butler_times"]["raw"].to_value(u.s), 2.0)
        det_entry = meta["det_meta"]["R00_SW0_42"]
        self.assertAlmostEqual(det_entry["astrom_scatter"].to_value(u.arcsec), 0.3)
        self.assertAlmostEqual(det_entry["isr_run"].to_value(u.s), 1.25)
        # The notes ride along, so a catalog read back cold still explains the
        # keys whose units cannot.
        self.assertIn("date", meta["notes"])

    def testMetaRoundTripsWithoutAVisitInfo(self) -> None:
        """The None date is the other half of the serialization contract.

        `Time` has no NaN, so an absent date is None -- which has to survive
        the yaml dump as well, or the empty case would be the one that breaks
        in production.
        """
        table = build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
        meta = arrow_to_astropy(astropy_to_arrow(Table(table))).meta
        self.assertIsNone(meta["date"])
        self.assertTrue(np.isnan(meta["boresight_alt"].to_value(u.deg)))

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

    def testDetectorOrientationIsPerDetectorNotPerRow(self) -> None:
        """n_quarter lives in det_meta, keyed by detector *and* visit.

        It is a per-detector constant, and only useful next to x_det/y_det -- so
        it is recorded for provenance (undoing the CCS stamp rotation without
        loading the camera model) without being replicated onto every row.
        """
        results = [
            {**_result(), "visit_id": 1, "n_quarter": 1},
            {**_result(), "visit_id": 2, "n_quarter": 3},
        ]
        table = build_donut_catalog(results, [], [_donut()], [], 42, _options())
        self.assertNotIn("n_quarter", table.colnames)
        det_meta = table.meta["det_meta"]
        self.assertEqual(det_meta["R00_SW0_1"]["n_quarter"], 1)
        self.assertEqual(det_meta["R00_SW0_2"]["n_quarter"], 3)

    def testDetMetaProvenanceRecordsNotApplicable(self) -> None:
        """"Nothing to report" is a named token, not an empty string.

        The producers cover every case -- corner mode's ``snr_rank``/``n/a``, the
        no-detections early return's ``no_detections`` -- so the builder copies
        both fields verbatim with no default. An empty string, ``None`` or a
        missing key reaching ``det_meta`` would therefore be a bug, not "not
        applicable".
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
