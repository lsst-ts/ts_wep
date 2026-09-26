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

"""The output catalog builder, shared by corner and full-array mode."""

import unittest

import astropy.units as u
import numpy as np
from astropy.table import Table

import lsst.geom as geom
from lsst.afw.image import VisitInfo
from lsst.daf.base import DateTime
from lsst.daf.butler.formatters.parquet import arrow_to_astropy, astropy_to_arrow
from lsst.ts.wep.blitz.catalogBuilder import (
    _build_detector_image_table,
    _build_donut_catalog,
    _build_overlay_table,
    _CatalogOptions,
    _CatalogTimings,
    _rotate_zk_to_eb,
)
from lsst.ts.wep.blitz.dataStructures import (
    CutoutResult,
    DetectorView,
    Donut,
    SourceSet,
    WfDonutResult,
    WfGroupResult,
)
from lsst.ts.wep.blitz.donutBlitzCorner import (
    DonutBlitzCornerConfig,
    DonutBlitzCornerTask,
)
from lsst.ts.wep.blitz.lsstCam import _LSSTCAM
from lsst.ts.wep.blitz.utils import _OFFSET_OPTICS, _ZK_JMAX
from lsst.ts.wep.blitz.wavefrontFitting import WavefrontFittingConfig

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


def _result(det_name="R00_SW0", rejected=(), **overrides):
    """A per-detector cutout result, as the workers return.

    Every stage timing is 0.0 rather than NaN so that a test overriding one
    reads a plain number back; the NaN conventions are `CutoutResult`'s own
    classmethods' business, not this fixture's.
    """
    kwargs = dict(
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
        # Both provenance fields are unconditional in the real results: the
        # cutout pipeline always sets a selection_source, and it seeds
        # pair_path with "n/a" for the grouping stage to overwrite.
        selection_source="refcat",
        n_quarter=0,
        pair_path="n/a",
        wcs=None,
    )
    kwargs.update(overrides)
    return CutoutResult(**kwargs)


def _wf_donut(donut, group_id="g", **overrides):
    """A successful `WfDonutResult` for ``donut``, keyed to match it."""
    kwargs = dict(
        donut_id=donut.donut_id,
        det_name=donut.det_name,
        visit_id=donut.visit_id,
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
        fit_dx=0.1,
        fit_dy=0.2,
        fit_flux=1e5,
        fit_fwhm=0.9,
        blend_frac=0.01,
        group_id=group_id,
        group_size=1,
    )
    kwargs.update(overrides)
    return WfDonutResult(**kwargs)


def _wf_group(donuts, group_id="g", success=True):
    """A `WfGroupResult` carrying ``donuts``.

    Only ``donut_results`` is read by `_build_donut_catalog` -- the group's own
    scalars reach the table through each `WfDonutResult`'s replicated copy --
    so the rest take their no-fit values from `WfGroupResult.empty`.
    """
    out = WfGroupResult.empty(group_id, n_zk=len(_options().noll_indices))
    out.group_size = len(donuts)
    out.donut_results = list(donuts)
    out.det_names = [d.det_name for d in donuts]
    out.success = success
    return out


def _options(**overrides) -> _CatalogOptions:
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
    return _CatalogOptions(**kwargs)


class TestCatalogOptions(unittest.TestCase):
    """The options object and its derived quantities."""

    def testDerivedSizes(self) -> None:
        # 167 // 2 = 83, already odd.
        self.assertEqual(_options().wf_img_size, 83)
        # 168 // 2 = 84, forced odd downwards -- the case the fitter's
        # _bin_stamp_odd handles by trimming.
        self.assertEqual(_options(stamp_size=168).wf_img_size, 83)
        self.assertEqual(_options(noll_indices=(4, 11, 22)).zk_deviation_jmax, 22)

    def testBkgWidthFollowsDanishsPolynomial(self) -> None:
        """nbkg is danish's (order+1)(order+2)/2, clamped to 0 below zero.

        The 0 is what drops the column: a zero-width array column writes to
        parquet and then cannot be read back, so it must never be emitted.
        """
        self.assertEqual(_options(bkg_order=-1).nbkg, 0)
        self.assertEqual(_options(bkg_order=0).nbkg, 1)
        self.assertEqual(_options(bkg_order=1).nbkg, 3)
        self.assertEqual(_options(bkg_order=2).nbkg, 6)
        # Matches the config's own default of a constant background.
        self.assertEqual(_options().nbkg, 1)

    def testCornerWiresEveryFieldFromConfig(self) -> None:
        """Corner mode's options come from the config fields they claim to.

        The extraction replaced ~10 ``self.<subtask>.config.<field>`` reads
        with an options object; a crossed pair of floats here would silently
        change the annulus geometry recorded in ``meta`` and drawn on the
        plots.
        """
        config = DonutBlitzCornerConfig()
        config.cutStamps.stampSize = 215
        config.cutStamps.maxDonuts = 11
        config.wavefrontFit.binning = 3
        # Must be pair-complete: the config rejects a truncated +/-m doublet
        # because the coefficients could not then be rotated between frames.
        config.wavefrontFit.nollIndices = [4, 5, 6, 7, 8]
        config.measureCandidates.apertureMarginFrac = 0.11
        config.measureCandidates.bkgInnerDiscFrac = 0.22
        config.measureCandidates.bkgAnnulusInnerFrac = 1.33
        config.measureCandidates.bkgAnnulusOuterFrac = 1.44
        config.wfEstimationMode = "unpaired"
        config.saveStamps = False
        config.wavefrontFit.bkgOrder = 1

        options = DonutBlitzCornerTask(config=config)._catalogOptions()

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
        self.assertEqual(options.bkg_order, 1)


class TestRotateZkToEb(unittest.TestCase):
    """The E/B (aligned/cross) rotation applied to the catalog's Zernikes.

    Noll 1..11, so `getNollPairs` yields the doublets (2, 3), (6, 5),
    (8, 7), (10, 9) and the m == 0 singles 1, 4, 11. Note slots 5 and 9
    hold the *cosine* member: which of a doublet's two consecutive indices
    carries the positive m alternates.
    """

    def _zk(self, **slots) -> np.ndarray:
        """One row over Noll 0..11, zero except the given slots."""
        row = np.zeros((1, 12))
        for j, value in slots.items():
            row[0, int(j)] = value
        return row

    def testZeroAngleIsIdentity(self) -> None:
        # phi = atan2(0, 1) = 0, so every doublet rotates by 0.
        zk = self._zk(**{"2": 3.0, "3": -4.0, "6": 5.0, "4": 9.0})
        (out,) = _rotate_zk_to_eb([zk], np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(out, zk)

    def testDoubletRotatesByMPhi(self) -> None:
        """Power moves into E at the angle that aligns the doublet.

        The rotation angle is m * phi, not phi, so the |m| = 2 doublet
        aligns at half the phi the |m| = 1 doublet needs. Getting the
        factor of m wrong would leave residual B power here.
        """
        # |m| = 1 doublet (j_cos=2, j_sin=3): all power in the sine slot,
        # phi = pi/2, so a = pi/2 and it rotates entirely into cosine.
        zk = self._zk(**{"3": 1.0})
        (out,) = _rotate_zk_to_eb([zk], np.array([0.0]), np.array([1.0]))
        self.assertAlmostEqual(out[0, 2], 1.0)
        self.assertAlmostEqual(out[0, 3], 0.0)

        # |m| = 2 doublet (j_cos=6, j_sin=5): same alignment at phi = pi/4.
        zk = self._zk(**{"5": 1.0})
        (out,) = _rotate_zk_to_eb([zk], np.array([1.0]), np.array([1.0]))
        self.assertAlmostEqual(out[0, 6], 1.0)
        self.assertAlmostEqual(out[0, 5], 0.0)

    def testEbAmplitudeIsRotationInvariant(self) -> None:
        """A doublet's E/B split does not depend on the field azimuth.

        This is the property the ``_eb`` columns exist for: unlike the
        ``_ccs`` coefficients, they are comparable across field positions.
        """
        zk = self._zk(**{"2": 0.3, "3": -0.4})
        for phi in (0.0, 0.7, 1.9, -2.5):
            (out,) = _rotate_zk_to_eb([zk], np.array([np.cos(phi)]), np.array([np.sin(phi)]))
            self.assertAlmostEqual(np.hypot(out[0, 2], out[0, 3]), 0.5)

    def testMZeroSlotsPassThrough(self) -> None:
        # Singles have no partner to mix with, so they are copied verbatim --
        # NaN included, which is how an unfit donut's rows stay NaN.
        zk = self._zk(**{"4": 7.0, "11": -2.0})
        zk[0, 1] = np.nan
        (out,) = _rotate_zk_to_eb([zk], np.array([0.3]), np.array([0.9]))
        self.assertAlmostEqual(out[0, 4], 7.0)
        self.assertAlmostEqual(out[0, 11], -2.0)
        self.assertTrue(np.isnan(out[0, 1]))

    def testHalfNaNDoubletGivesTwoNaNs(self) -> None:
        """One non-finite member NaNs *both* slots, not just its own.

        E and B are each a combination of both members, so a doublet with
        one member missing has no defined E/B split; carrying the finite
        member through would silently pass off a ``_ccs`` value as an
        ``_eb`` one.
        """
        zk = self._zk(**{"2": np.nan, "3": 1.0, "6": 2.0, "5": np.inf})
        (out,) = _rotate_zk_to_eb([zk], np.array([0.3]), np.array([0.9]))
        self.assertTrue(np.isnan(out[0, 2]))
        self.assertTrue(np.isnan(out[0, 3]))
        self.assertTrue(np.isnan(out[0, 6]))
        self.assertTrue(np.isnan(out[0, 5]))

    def testUndefinedAngleNaNsEveryDoublet(self) -> None:
        # phi is meaningless at the field center, so no doublet is defined,
        # but the m == 0 terms do not depend on it and survive.
        zk = self._zk(**{"2": 1.0, "3": 2.0, "6": 3.0, "5": 4.0, "4": 7.0})
        (out,) = _rotate_zk_to_eb([zk], np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.all(np.isnan(out[0, [2, 3, 5, 6]])))
        self.assertAlmostEqual(out[0, 4], 7.0)

    def testPerRowAngles(self) -> None:
        # thx/thy are per-row: each donut rotates by its own field azimuth.
        zk = np.vstack([self._zk(**{"3": 1.0}), self._zk(**{"3": 1.0})])
        (out,) = _rotate_zk_to_eb([zk], np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        # Row 0 at phi = 0 is untouched; row 1 at phi = pi/2 rotates.
        self.assertAlmostEqual(out[0, 3], 1.0)
        self.assertAlmostEqual(out[1, 2], 1.0)

    def testUnitsPreservedOnlyWhereGiven(self) -> None:
        # The catalog hands in Quantities and expects Quantities back; the
        # helper is also called on bare arrays in tests like these.
        zk = self._zk(**{"2": 1.0})
        quantity, bare = _rotate_zk_to_eb([zk * u.micron, zk], np.array([1.0]), np.array([0.0]))
        self.assertEqual(quantity.unit, u.micron)
        self.assertFalse(isinstance(bare, u.Quantity))

    def testRejectsNon2DInput(self) -> None:
        # A single donut's coefficient vector is a common thing to pass by
        # mistake, and would broadcast into nonsense rather than raise.
        with self.assertRaises(ValueError):
            _rotate_zk_to_eb([np.zeros(12)], np.array([1.0]), np.array([0.0]))


class TestBuildDonutCatalog(unittest.TestCase):
    """Row content and the optional image columns."""

    def testEmptyInputGivesEmptyTable(self) -> None:
        table = _build_donut_catalog([], [], [], [], 1, _options())
        self.assertEqual(len(table), 0)

    def testOneRowPerDonutIncludingRejected(self) -> None:
        accepted = _donut(donut_id=1)
        rejected = _donut(donut_id=2, rejected=True, rejected_snr=True)
        table = _build_donut_catalog([_result(rejected=[rejected])], [], [accepted], [], 42, _options())
        self.assertEqual(len(table), 2)
        by_id = {int(r["donut_id"]): r for r in table}
        self.assertTrue(bool(by_id[1]["candidate"]))
        self.assertFalse(bool(by_id[2]["candidate"]))
        # No fit claimed either donut, so both group_ids are empty and the
        # deviations are NaN.
        self.assertEqual({str(r["group_id"]) for r in table}, {""})
        self.assertFalse(any(bool(r["group_fit_success"]) for r in table))
        self.assertTrue(np.all(np.isnan(table["zk_deviation_ccs"].value)))
        # `rejected` is not a column: the rejected_* reasons carry it, and
        # their OR is exactly ~candidate.
        self.assertNotIn("rejected", table.colnames)
        self.assertTrue(bool(by_id[2]["rejected_snr"]))

    def testDedupeKeepsSurplusDonutsOnce(self) -> None:
        """A donut in both `donuts` and `unmatched_donuts` yields one row."""
        d = _donut(donut_id=7)
        table = _build_donut_catalog([_result()], [], [d], [d], 42, _options())
        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table[0]["candidate"]))

    def testSameStarOnBothSidesOfFocusKeepsBothRows(self) -> None:
        """The full-array case ``(det_name, donut_id)`` alone would collapse.

        FAM cuts the same star on the same detector once per side of focus, so
        the two donuts share a ``donut_id`` -- the same refcat source, or the
        same ``1..N`` blitz-detection slot. Only ``visit_id`` separates them,
        in the row key and in the wavefront-result lookup.
        """
        intra_visit, extra_visit = 2026070900036, 2026070900037
        intra = _donut(donut_id=7, visit_id=intra_visit)
        extra = _donut(donut_id=7, visit_id=extra_visit)
        # Distinguishable fits, one per side, so a mixed-up lookup is visible.
        wf_results = [
            _wf_group(
                [
                    WfDonutResult(
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
            )
            for visit, value in ((intra_visit, 1e-6), (extra_visit, 2e-6))
        ]
        results = [_result(visit_id=visit) for visit in (intra_visit, extra_visit)]
        table = _build_donut_catalog(
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
        # Which visit was which side, so a consumer need not infer it. Set
        # inside the builder for both modes, so meta stays schema-identical.
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

        The label comes straight off the `WfDonutResult`, so a consumer can
        collapse the replicated group-level values to one measurement per fit
        -- which an array index into a list that no longer exists could not
        support.  A donut no fit claimed gets the empty string rather than a
        sentinel int.
        """
        paired = [_donut(donut_id=1), _donut(donut_id=2)]
        surplus = _donut(donut_id=3)
        gid = "R00_SW0_6222323257218103680_6219321315596159872"
        wf_results = [
            _wf_group(
                [
                    WfDonutResult(
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
                ],
                group_id=gid,
            )
        ]
        table = _build_donut_catalog(
            [_result()],
            wf_results,
            paired + [surplus],
            [surplus],
            42,
            _options(),
        )

        by_id = {int(r["donut_id"]): r for r in table}
        self.assertEqual({str(by_id[i]["group_id"]) for i in (1, 2)}, {gid})
        # No fit claimed the surplus donut: empty label, and nothing to
        # succeed.
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

        Both modes negate the extra-focal triplet to get the intra-focal one,
        so a consumer reads the side off the sign without needing to know that
        SW0/SW1 straddle focus (corner) or which visit was intra (FAM).
        """
        extra = _donut(donut_id=1, defocal_offsets=(+1.5e-3, 0.0, 0.0))
        intra = _donut(donut_id=2, defocal_offsets=(-1.5e-3, 0.0, 0.0))
        table = _build_donut_catalog([_result()], [], [extra, intra], [], 42, _options())
        by_id = {int(r["donut_id"]): r for r in table}
        self.assertEqual(table["defocal_offsets"].unit, u.m)
        np.testing.assert_allclose(by_id[1]["defocal_offsets"].to_value(u.m), [1.5e-3, 0.0, 0.0])
        np.testing.assert_allclose(by_id[2]["defocal_offsets"].to_value(u.m), [-1.5e-3, 0.0, 0.0])

    def testDefocalOffsetsAxisIsLabelledInMeta(self) -> None:
        """The column is a bare length-3 array, so meta must name the axis.

        Nothing in the column itself says which optic each slot moves, and an
        astropy column description cannot carry it: the ArrowAstropy round trip
        drops descriptions on numeric columns, so it would read back as None.
        `offset_optics` is the batoid optic names, `notes` the sign convention.
        """
        table = _build_donut_catalog([_result()], [], [_donut()], [], 42, _options())

        meta = arrow_to_astropy(astropy_to_arrow(Table(table))).meta

        self.assertEqual(meta["offset_optics"], list(_OFFSET_OPTICS))
        self.assertEqual(len(meta["offset_optics"]), table["defocal_offsets"].shape[1])
        # The sign is the side of focus, which no unit or label can express.
        self.assertIn("extra-focal", meta["notes"]["defocal_offsets"])

    def testNollIndicesNoteDisownsTheDenseLayout(self) -> None:
        """`noll_indices` is the fitted set, not the zk_* column layout.

        The dense arrays are indexed by Noll j directly, so a consumer that
        reads `noll_indices` as the axis labelling silently mis-indexes -- the
        default set skips 20 and 21, leaving those slots NaN inside the array
        rather than absent from it.  Asserted on the real default rather than
        this module's contiguous `_options`, since the gap is the whole point.
        """
        noll_indices = WavefrontFittingConfig().nollIndices
        self.assertNotIn(20, noll_indices)
        self.assertNotIn(21, noll_indices)

        table = _build_donut_catalog(
            [_result()], [], [_donut()], [], 42, _options(noll_indices=tuple(noll_indices))
        )

        # Slot 20 exists in the array despite never being fitted, which is what
        # makes the distinction observable.
        self.assertGreater(table["zk_deviation_ccs"].shape[1], 21)
        self.assertEqual(table.meta["noll_indices"], list(noll_indices))
        self.assertIn("distinct", table.meta["notes"]["noll_indices"])

    def testDefocalOffsetsAreNaNWhenUnannotated(self) -> None:
        """A donut that never reached the fitter gets a well-shaped row."""
        table = _build_donut_catalog([_result()], [], [_donut(defocal_offsets=None)], [], 42, _options())
        offsets = table["defocal_offsets"].to_value(u.m)
        self.assertEqual(offsets.shape, (1, 3))
        self.assertTrue(np.all(np.isnan(offsets)))

    def testSkyPositionIsCarriedInDegrees(self) -> None:
        """The refcat position, converted from the Donut's afw-native radians.

        This is what any cross-visit or cross-detector match needs; the field
        angle plus the boresight meta only gets there approximately.
        """
        table = _build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
        self.assertEqual(table["coord_ra"].unit, u.deg)
        self.assertEqual(table["coord_dec"].unit, u.deg)
        self.assertAlmostEqual(table["coord_ra"][0].to_value(u.deg), 30.0)
        self.assertAlmostEqual(table["coord_dec"][0].to_value(u.deg), -20.0)

    def testSkyPositionIsNaNOffTheRefcatPath(self) -> None:
        """NaN, not a WCS projection of the centroid.

        Mixing refcat truth and a projected centroid into one column would
        reproduce the mixed provenance that `donut_id` already carries.
        """
        table = _build_donut_catalog(
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
            table = _build_donut_catalog(*args, options)
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

    def testFitBkgCarriesDanishsFittedBackground(self) -> None:
        """The fitted background reaches the table at its configured width.

        Every other free parameter of the danish fit is reported (fit_dx,
        fit_dy, fit_flux, group_fwhm, the Zernikes); this is the last one, and
        the only quantity the table's own `bkg` column can be checked against.
        """
        for bkg_order, width in ((0, 1), (1, 3), (2, 6)):
            fitted = np.arange(width, dtype=float) + 10.0
            donut = _donut(donut_id=1)
            wf_results = [_wf_group([_wf_donut(donut, fit_bkg=fitted)])]
            table = _build_donut_catalog(
                [_result()],
                wf_results,
                [donut],
                [],
                42,
                _options(bkg_order=bkg_order),
            )
            self.assertEqual(table["fit_bkg"].shape, (1, width), msg=f"bkgOrder={bkg_order}")
            np.testing.assert_allclose(table["fit_bkg"][0], fitted)
            # Unitless, like the `bkg` column it is meant to be compared with.
            self.assertIsNone(table["fit_bkg"].unit)

    def testFitBkgIsAbsentWhenDanishModelsNoBackground(self) -> None:
        """bkgOrder=-1 drops the column rather than emitting a zero-width one.

        A ``(n, 0)`` array column writes to parquet happily and then raises on
        read, which would break ``butler.get`` for the *whole* results table --
        and that table is the plot task's input.  So the empty case must be no
        column at all.
        """
        table = _build_donut_catalog([_result()], [], [_donut()], [], 42, _options(bkg_order=-1))
        self.assertNotIn("fit_bkg", table.colnames)
        # Dropping it must not disturb the neighboring fit columns.
        self.assertIn("fit_flux", table.colnames)
        self.assertIn("bkg", table.colnames)

    def testFitBkgIsNaNFilledToWidthWhenNoFitRan(self) -> None:
        """A donut no fit claimed still needs a row of the common width.

        Rows come from a list of dicts, so a short vector on one row would make
        the column ragged and object-dtype -- unpersistable.  The width comes
        from config, not from the data, because a run whose every fit failed
        has no fitted vector to take it from.
        """
        fitted = _donut(donut_id=1)
        surplus = _donut(donut_id=2)
        wf_results = [_wf_group([_wf_donut(fitted, fit_bkg=np.array([7.0, 8.0, 9.0]))])]
        table = _build_donut_catalog(
            [_result()],
            wf_results,
            [fitted, surplus],
            [surplus],
            42,
            _options(bkg_order=1),
        )
        self.assertEqual(table["fit_bkg"].shape, (2, 3))
        by_id = {int(r["donut_id"]): r for r in table}
        np.testing.assert_allclose(by_id[1]["fit_bkg"], [7.0, 8.0, 9.0])
        self.assertTrue(np.all(np.isnan(by_id[2]["fit_bkg"])))

    def testFitBkgWidthIsTakenFromConfigNotTheFit(self) -> None:
        """A stale vector of the wrong width is discarded, not emitted ragged.

        `bkg_order` and the fitter's `bkgOrder` are wired from the same config
        field, so a mismatch means something is already inconsistent; falling
        back to NaN keeps the column well-shaped rather than unpersistable.
        """
        donut = _donut(donut_id=1)
        wf_results = [_wf_group([_wf_donut(donut, fit_bkg=np.arange(6, dtype=float))])]
        table = _build_donut_catalog([_result()], wf_results, [donut], [], 42, _options(bkg_order=0))
        self.assertEqual(table["fit_bkg"].shape, (1, 1))
        self.assertTrue(np.all(np.isnan(table["fit_bkg"][0])))

    def testFitBkgSurvivesTheArrowRoundTrip(self) -> None:
        """The column has to survive the way the catalog is persisted.

        The zero-width case proved that an array column can pass every
        in-memory check and still fail on read, so each width is round-tripped
        here rather than trusted.
        """
        for bkg_order, width in ((0, 1), (1, 3), (2, 6)):
            donut = _donut(donut_id=1)
            fitted = np.arange(width, dtype=float) + 3.0
            wf_results = [_wf_group([_wf_donut(donut, fit_bkg=fitted)])]
            table = _build_donut_catalog(
                [_result()], wf_results, [donut], [], 42, _options(bkg_order=bkg_order)
            )
            back = arrow_to_astropy(astropy_to_arrow(Table(table)))
            self.assertEqual(back["fit_bkg"].shape, (1, width), msg=f"bkgOrder={bkg_order}")
            np.testing.assert_allclose(back["fit_bkg"][0], fitted)

    def testMetaCarriesOptionsAndGeometry(self) -> None:
        options = _options(binning=4, max_donuts=3, wf_mode="full_detector")
        table = _build_donut_catalog([_result()], [], [_donut()], [], 99, options, rtp_rad=0.25)
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
        self.assertAlmostEqual(table.meta["rot_tel_pos"].to_value(u.deg), np.degrees(0.25))
        self.assertAlmostEqual(table.meta["bkg_annulus_outer_frac"], options.bkg_annulus_outer_frac)
        self.assertAlmostEqual(table.meta["bkg_inner_disc_frac"], options.bkg_inner_disc_frac)
        # Per-detector metadata survives for the plots.  Corner mode supplies
        # no per-result visit_id, so the key falls back to this table's visit.
        self.assertIn("R00_SW0_99", table.meta["det_meta"])
        self.assertAlmostEqual(
            table.meta["det_meta"]["R00_SW0_99"]["astrom_scatter"].to_value(u.arcsec),
            0.3,
        )

    def testMetaRecordsTheZernikeNormalizationAnnulus(self) -> None:
        """The zk_* columns are meaningless without the radii they sit on.

        A coefficient normalized on a 4.18 m annulus and one normalized on
        4.165 m are different numbers for the same wavefront, so a reader
        comparing catalogs -- across optics models, or against an external
        wavefront -- needs the domain recorded rather than assumed. Carries
        units, since the values are lengths and everything else in meta does.
        """
        table = _build_donut_catalog([_result()], [], [_donut()], [], 99, _options())
        self.assertEqual(table.meta["zk_r_outer"].to_value(u.m), _LSSTCAM.zk_r_outer)
        self.assertEqual(table.meta["zk_r_inner"].to_value(u.m), _LSSTCAM.zk_r_inner)
        # The inner radius must be the obscuration applied to the outer one,
        # not an independent number that could disagree with it.
        self.assertAlmostEqual(
            table.meta["zk_r_inner"] / table.meta["zk_r_outer"],
            table.meta["obscuration"],
        )
        self.assertIn("zk_r_outer", table.meta["notes"])

    def testIntrinsicsSurviveAnUnfittedRow(self) -> None:
        """Intrinsics come off the donut when no fit claimed the row.

        Intrinsic Zernikes are a function of field position, not of the fit,
        so a row no fit consumed still has them -- taking them from
        `_NULL_WF_DONUT` made them all-NaN even where the calibration was
        known.
        Deviations *are* a measurement, so they stay NaN.
        """
        d = _donut(donut_id=1)
        d.intrinsic_zk = np.full(_ZK_JMAX + 1 - 4, 2.0)  # µm, Noll 4.._ZK_JMAX
        table = _build_donut_catalog([_result()], [], [d], [], 42, _options())

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
        table = _build_donut_catalog([_result()], [], [d], [], 42, _options())
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
        table = _build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
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
        table = _build_donut_catalog(
            [_result()],
            [],
            [_donut()],
            [],
            42,
            _options(),
            mode="fam",
            visit_info=visit_info,
            instrument="LSSTCam",
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
        table = _build_donut_catalog(
            [_result(isr_run=1.25)],
            [],
            [_donut()],
            [],
            42,
            _options(),
            visit_info=visit_info,
            timings=_CatalogTimings(run_elapsed=12.5, butler_times={"raw": 2.0}),
            rtp_rad=np.deg2rad(12.5),
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
        # The Zernike domain is only useful on a catalog read back cold, which
        # is exactly the path that would drop it.
        self.assertAlmostEqual(meta["zk_r_outer"].to_value(u.m), _LSSTCAM.zk_r_outer)
        self.assertAlmostEqual(meta["zk_r_inner"].to_value(u.m), _LSSTCAM.zk_r_inner)
        # The notes ride along, so a catalog read back cold still explains the
        # keys whose units cannot.
        self.assertIn("date", meta["notes"])

    def testMetaRoundTripsWithoutAVisitInfo(self) -> None:
        """The None date is the other half of the serialization contract.

        `Time` has no NaN, so an absent date is None -- which has to survive
        the yaml dump as well, or the empty case would be the one that breaks
        in production.
        """
        table = _build_donut_catalog([_result()], [], [_donut()], [], 42, _options())
        meta = arrow_to_astropy(astropy_to_arrow(Table(table))).meta
        self.assertIsNone(meta["date"])
        self.assertTrue(np.isnan(meta["boresight_alt"].to_value(u.deg)))

    def testDetMetaCarriesIdProvenance(self) -> None:
        """det_meta records where the donut ids came from, and how they paired.

        Without this a consumer cannot tell a Monster source id from a blitz
        detection's 1..N slot, and a FAM run that quietly fell back to spatial
        pairing is invisible once the run's logs are gone.
        """
        results = [
            _result(visit_id=1, selection_source="refcat", pair_path="refcat_id"),
            _result(visit_id=2, selection_source="blitz_selected", pair_path="spatial"),
        ]
        table = _build_donut_catalog(results, [], [_donut()], [], 42, _options())
        det_meta = table.meta["det_meta"]
        self.assertEqual(det_meta["R00_SW0_1"]["selection_source"], "refcat")
        self.assertEqual(det_meta["R00_SW0_1"]["pair_path"], "refcat_id")
        self.assertEqual(det_meta["R00_SW0_2"]["selection_source"], "blitz_selected")
        self.assertEqual(det_meta["R00_SW0_2"]["pair_path"], "spatial")

    def testDetectorOrientationIsPerDetectorNotPerRow(self) -> None:
        """n_quarter lives in det_meta, keyed by detector *and* visit.

        It is a per-detector constant, and only useful next to x_det/y_det --
        so it is recorded for provenance (undoing the CCS stamp rotation
        without loading the camera model) without being replicated onto every
        row.
        """
        results = [
            _result(visit_id=1, n_quarter=1),
            _result(visit_id=2, n_quarter=3),
        ]
        table = _build_donut_catalog(results, [], [_donut()], [], 42, _options())
        self.assertNotIn("n_quarter", table.colnames)
        det_meta = table.meta["det_meta"]
        self.assertEqual(det_meta["R00_SW0_1"]["n_quarter"], 1)
        self.assertEqual(det_meta["R00_SW0_2"]["n_quarter"], 3)

    def testDetMetaProvenanceRecordsNotApplicable(self) -> None:
        """ "Nothing to report" is a named token, not an empty string.

        The producers cover every case -- corner mode's ``snr_rank``/``n/a``,
        the no-detections early return's ``no_detections`` -- so the builder
        copies both fields verbatim with no default. An empty string, ``None``
        or a missing key reaching ``det_meta`` would therefore be a bug, not
        "not applicable".
        """
        results = [_result(selection_source="no_detections", pair_path="n/a")]
        table = _build_donut_catalog(results, [], [_donut()], [], 42, _options())
        entry = table.meta["det_meta"]["R00_SW0_42"]
        self.assertEqual(entry["selection_source"], "no_detections")
        self.assertEqual(entry["pair_path"], "n/a")


_VIEW_BINNING = 4
# Corner-sensor proportions, small enough to keep the fixtures cheap.
_VIEW_SHAPE = (200, 408)


def _source_set(n, seed, with_mag=True):
    rng = np.random.default_rng(seed)
    height, width = _VIEW_SHAPE
    return SourceSet(
        x_det=rng.uniform(0, width, n),
        y_det=rng.uniform(0, height, n),
        donut_id=np.arange(1, n + 1, dtype=np.int64),
        mag=(rng.uniform(13.0, 20.0, n) if with_mag else np.full(n, np.nan)).astype(np.float32),
    )


def _view(shape=None, refcat=_source_set(6, 1), detections=None, selections=None):
    height, width = shape or _VIEW_SHAPE
    hb, wb = height // _VIEW_BINNING, width // _VIEW_BINNING
    return DetectorView(
        image=np.random.default_rng(0).normal(12.0, 2.0, (hb, wb)).astype(np.float32),
        binning=_VIEW_BINNING,
        bbox_min=(0, 0),
        bbox_shape=(height, width),
        refcat=refcat,
        detections=_source_set(4, 2, with_mag=False) if detections is None else detections,
        selections=_source_set(3, 3) if selections is None else selections,
        field_dist=np.linspace(1.0, 2.0, hb * wb, dtype=np.float32).reshape(hb, wb),
        max_field_dist_deg=1.725,
    )


def _result_with_view(det_name="R00_SW0", view=None, **overrides):
    result = _result(det_name=det_name, **overrides)
    result.view = _view() if view is None else view
    return result


class TestBuildDetectorImageTable(unittest.TestCase):
    """The per-detector binned-image table the focal-plane plot reads."""

    def testNoViewsGivesAnEmptyTable(self) -> None:
        """What the plot task tests before drawing."""
        table = _build_detector_image_table([_result()], 42, "LSSTCam")
        self.assertEqual(len(table), 0)
        self.assertEqual(list(table.columns), [])

    def testDetectorsWithoutAViewAreAbsentNotPlaceheld(self) -> None:
        """A dead worker's detector has no image, and no row either."""
        table = _build_detector_image_table([_result_with_view("R00_SW0"), _result("R00_SW1")], 42, "LSSTCam")
        self.assertEqual(table["det_name"].tolist(), ["R00_SW0"])

    def testRoundTripsThroughArrowWithDtypesIntact(self) -> None:
        """Fixed-shape 2-D columns survive where variable-length ones do not.

        The reason the overlay sources went into a long-form table instead of
        array columns, so it is worth pinning that these two *do* survive.
        """
        table = _build_detector_image_table([_result_with_view()], 42, "LSSTCam")
        back = arrow_to_astropy(astropy_to_arrow(Table(table)))
        hb, wb = (d // _VIEW_BINNING for d in _VIEW_SHAPE)
        self.assertEqual(back["image"].shape, (1, hb, wb))
        self.assertEqual(back["image"].dtype, np.float32)
        self.assertEqual(back["field_dist"].shape, (1, hb, wb))
        np.testing.assert_allclose(back["image"][0], table["image"][0])

    def testHasRefcatDistinguishesNoRefcatFromAnEmptyOne(self) -> None:
        """Overlay table renders both as zero rows, so column carries it."""
        table = _build_detector_image_table(
            [
                _result_with_view("R00_SW0", view=_view(refcat=None)),
                _result_with_view("R00_SW1", view=_view(refcat=_source_set(0, 4))),
            ],
            42,
            "LSSTCam",
        )
        by_det = dict(zip(table["det_name"].tolist(), table["has_refcat"].tolist()))
        self.assertFalse(by_det["R00_SW0"])
        self.assertTrue(by_det["R00_SW1"])

    def testMismatchedImageShapesWarnRatherThanRaise(self) -> None:
        """A fixed-shape column cannot hold both, so the minority is dropped.

        Raising deep inside astropy would lose the whole plot over one odd
        detector; keeping the majority and saying what went is the better
        failure.
        """
        results = [
            _result_with_view("R00_SW0"),
            _result_with_view("R00_SW1"),
            _result_with_view("R04_SW0", view=_view(shape=(200, 200))),
        ]
        with self.assertLogs(level="WARNING") as captured:
            table = _build_detector_image_table(results, 42, "LSSTCam")
        self.assertEqual(sorted(table["det_name"].tolist()), ["R00_SW0", "R00_SW1"])
        self.assertIn("R04_SW0", "\n".join(captured.output))


class TestBuildOverlayTable(unittest.TestCase):
    """The long-form selection-source table."""

    def testOneRowPerSourceTaggedByStage(self) -> None:
        table = _build_overlay_table([_result_with_view()], 42, "LSSTCam")
        kinds = table["kind"].tolist()
        self.assertEqual(kinds.count("refcat"), 6)
        self.assertEqual(kinds.count("detection"), 4)
        self.assertEqual(kinds.count("selection"), 3)
        self.assertEqual(len(table), 13)

    def testNoRefcatEmitsNoRefcatRows(self) -> None:
        table = _build_overlay_table([_result_with_view(view=_view(refcat=None))], 42, "LSSTCam")
        self.assertNotIn("refcat", set(table["kind"].tolist()))
        self.assertEqual({"detection", "selection"}, set(table["kind"].tolist()))

    def testAcceptedAndRejectedDonutsAreDeliberatelyAbsent(self) -> None:
        """They are donut-catalog rows; duplicating would let the two drift."""
        table = _build_overlay_table([_result_with_view()], 42, "LSSTCam")
        self.assertEqual(set(table["kind"].tolist()), {"refcat", "detection", "selection"})
        self.assertNotIn("candidate", table.columns)
        self.assertNotIn("snr", table.columns)

    def testRoundTripsThroughArrowWithUnitsAndValuesIntact(self) -> None:
        """Long form is what makes this survivable at all.

        The three stages have unrelated lengths, and variable-length array
        columns do not round trip -- one row per source sidesteps that
        entirely. The builder widens `SourceSet.mag` from float32 to float64 on
        the way in, so the units and values are what there is to pin here, not
        the dtype.
        """
        table = _build_overlay_table([_result_with_view()], 42, "LSSTCam")
        back = arrow_to_astropy(astropy_to_arrow(Table(table)))
        self.assertEqual(len(back), len(table))
        self.assertEqual(back["x_det"].unit, u.pix)
        self.assertEqual(back["mag"].unit, u.mag)
        self.assertEqual(back["kind"].tolist(), table["kind"].tolist())
        np.testing.assert_allclose(
            np.asarray(back["x_det"], dtype=float), np.asarray(table["x_det"], dtype=float)
        )

    def testNoSourcesAtAllGivesAnEmptyTable(self) -> None:
        empty = _view(refcat=None, detections=_source_set(0, 5), selections=_source_set(0, 6))
        table = _build_overlay_table([_result_with_view(view=empty)], 42, "LSSTCam")
        self.assertEqual(len(table), 0)


if __name__ == "__main__":
    unittest.main()
