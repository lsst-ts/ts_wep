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

"""The per-corner `zernikes` compatibility table."""

import os
import tempfile
import unittest

import astropy.units as u
import numpy as np
from astropy.table import QTable

from lsst.ts.wep.blitz.catalogBuilder import _build_donut_catalog, _CatalogOptions
from lsst.ts.wep.blitz.dataStructures import Donut, WfDonutResult, WfGroupResult
from lsst.ts.wep.blitz.donutBlitzCorner import DonutBlitzCornerConfig, DonutBlitzCornerConnections
from lsst.ts.wep.blitz.utils import _ZK_JMAX
from lsst.ts.wep.blitz.zernikesTable import build_zernikes_tables
from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask

from test_blitzCatalogBuilder import _result

# Deliberately short, and both halves of every +/-m pair, as the fitter
# requires.
_NOLL = (4, 5, 6, 7, 8, 9, 10, 11)
_VISIT = 2026070900036

# The worked example from the design: an uneven corner, which is where the four
# dispatch modes diverge most. R00_SW0 (det 191) is extra-focal, R00_SW1 (192)
# intra-focal.
_EXTRA_SNR = {1: 900.0, 2: 500.0, 3: 100.0}
_INTRA_SNR = {11: 800.0, 12: 400.0}


def _donut(det_name, det_id, donut_id, snr, **overrides):
    """One donut of the R00 fixture.

    Field angles and centroid vary with ``donut_id`` so that a test can tell
    which donut a row was populated from -- a fixture where every donut looked
    alike could not catch an intra/extra transposition.
    """
    kwargs = dict(
        det_name=det_name,
        stamp=np.ones((167, 167), dtype=np.float32),
        thx_ccs=0.01 * donut_id,
        thy_ccs=0.02 * donut_id,
        flux=1e5,
        band="r",
        det_id=det_id,
        visit_id=_VISIT,
        x_det=100.0 + donut_id,
        y_det=200.0 + donut_id,
        donut_id=donut_id,
        inner_frac=0.01,
        outer_frac=0.02,
        outer_sector_minmax_frac=0.03,
        donut_radius=60.0,
        snr=snr,
        bkg=10.0,
        bkg_std=3.0,
        n_quarter=0,
        photo_mag=14.0 + donut_id / 100.0,
        astrom_mag=14.2,
        nearby_photo=[],
        nearby_astrom=[],
        coord_ra=np.radians(30.0),
        coord_dec=np.radians(-20.0),
    )
    kwargs.update(overrides)
    return Donut(**kwargs)


def _fixture_donuts():
    """The R00 fixture: extra-focal A,B,C and intra-focal P,Q."""
    extra = [_donut("R00_SW0", 191, i, snr) for i, snr in _EXTRA_SNR.items()]
    intra = [_donut("R00_SW1", 192, i, snr) for i, snr in _INTRA_SNR.items()]
    return extra, intra


def _options(**overrides) -> _CatalogOptions:
    kwargs = dict(
        stamp_size=167,
        binning=2,
        noll_indices=_NOLL,
        aperture_margin_frac=0.05,
        bkg_inner_disc_frac=0.67,
        bkg_annulus_inner_frac=1.25,
        bkg_annulus_outer_frac=1.4,
        max_donuts=8,
        wf_mode="paired",
        save_stamps=False,
        save_wf_images=False,
    )
    kwargs.update(overrides)
    return _CatalogOptions(**kwargs)


def _catalog(groups, unmatched=(), mode="paired", zk_um=1.0, fwhm=None, success=None):
    """A per-donut catalog with ``groups`` mapping group_id -> member donuts.

    ``zk_um`` may be a scalar (all deviations that value) or a dict keyed by
    group_id, so a test can give each group a distinguishable Zernike vector.
    """
    extra, intra = _fixture_donuts()
    all_donuts = extra + intra
    fwhm = fwhm or {}
    success = success or {}

    wf_results = []
    for gid, members in groups.items():
        value = zk_um[gid] if isinstance(zk_um, dict) else zk_um
        ok = success.get(gid, True)
        donut_results = [
            WfDonutResult(
                donut_id=d.donut_id,
                det_name=d.det_name,
                visit_id=d.visit_id,
                # Meters in the record; the catalog converts to microns.
                zk_dev=np.full(_ZK_JMAX + 1, value * 1e-6),
                zk_intrinsic=np.zeros(_ZK_JMAX + 1),
                img=None,
                model_img=None,
                fit_success=ok,
                fit_elapsed=1.0,
                setup_elapsed=0.1,
                fit_nfev=10,
                fit_cost=1.0,
                fit_optimality=1e-9,
                fit_njev=10,
                fit_outcome="ok" if ok else "nonconvergent",
                fit_dx=0.0,
                fit_dy=0.0,
                fit_flux=1e5,
                fit_fwhm=fwhm.get(gid, 0.9),
                blend_frac=0.0,
                group_id=gid,
                group_size=len(members),
            )
            for d in members
        ]
        group = WfGroupResult.empty(gid, n_zk=len(_NOLL))
        group.group_size = len(members)
        group.donut_results = donut_results
        group.det_names = [d.det_name for d in members]
        group.success = ok
        wf_results.append(group)

    return _build_donut_catalog(
        results=[_result("R00_SW0"), _result("R00_SW1")],
        wf_results=wf_results,
        donuts=all_donuts,
        unmatched_donuts=list(unmatched),
        visit_id=_VISIT,
        options=_options(wf_mode=mode),
    )


def _build(catalog, mode="paired", **overrides):
    """Run the builder with the defaults these tests share."""
    kwargs = dict(
        noll_indices=_NOLL,
        wf_mode=mode,
        combine_zernikes=CombineZernikesSigmaClipTask(),
        visit_id=_VISIT,
        cam_name="LSSTCam",
    )
    kwargs.update(overrides)
    return build_zernikes_tables(catalog, **kwargs)


def _paired_groups():
    """Groups as `_build_wf_groups` makes them in paired mode: A-P, B-Q."""
    extra, intra = _fixture_donuts()
    a, b, c = extra
    p, q = intra
    return {"R00_1_11": [a, p], "R00_2_12": [b, q]}, [c]


def _unpaired_groups():
    extra, intra = _fixture_donuts()
    return {f"{d.det_name}_{d.donut_id}": [d] for d in extra + intra}, []


def _full_detector_groups():
    extra, intra = _fixture_donuts()
    return {"R00_SW0": extra, "R00_SW1": intra}, []


def _full_corner_groups():
    extra, intra = _fixture_donuts()
    return {"R00": extra + intra}, []


def _data_rows(table: QTable) -> QTable:
    return table[table["label"] != "average"]


def _xy(row, column, unit):
    """One structured (x, y) cell as a plain float pair.

    ``float()`` on a dimensional Quantity raises, so the unit has to be
    stripped explicitly rather than relying on a bare cast.
    """
    cell = row[column]
    return (
        float(cell["x"].to_value(unit)),
        float(cell["y"].to_value(unit)),
    )


class TestSchema(unittest.TestCase):
    """The column set, units and row layout."""

    def setUp(self) -> None:
        groups, unmatched = _paired_groups()
        self.table = _build(_catalog(groups, unmatched))[191]

    def testKeyedByExtraFocalDetector(self) -> None:
        """Tables key on SW0, never on the intra-focal detector."""
        groups, unmatched = _paired_groups()
        tables = _build(_catalog(groups, unmatched))
        self.assertEqual(sorted(tables), [191])
        self.assertNotIn(192, tables)

    def testAverageRowIsFirst(self) -> None:
        self.assertEqual(self.table["label"][0], "average")
        self.assertEqual(list(_data_rows(self.table)["label"]), ["pair1", "pair2"])

    def testLabelsFitTheColumnWidth(self) -> None:
        """<U12 is the schema's width; a longer prefix would truncate."""
        self.assertLessEqual(self.table["label"].dtype.itemsize // 4, 12)
        for label in self.table["label"]:
            self.assertLessEqual(len(str(label)), 12)

    def testNoOpdOrIntrinsicColumns(self) -> None:
        """Asserted explicitly: restoring them would be silently dishonest.

        A joint fit spans N field positions, so there is no single intrinsic to
        report and hence no meaningful total OPD either.
        """
        for j in _NOLL:
            self.assertIn(f"Z{j}_deviation", self.table.colnames)
            self.assertNotIn(f"Z{j}", self.table.colnames)
            self.assertNotIn(f"Z{j}_intrinsic", self.table.colnames)

    def testOmittedQualityMetricColumns(self) -> None:
        """blitz computes none of these, so they are omitted, not NaN."""
        for side in ("intra", "extra"):
            for metric in ("entropy", "frac_bad_pix", "max_power_grad"):
                self.assertNotIn(f"{side}_{metric}", self.table.colnames)

    def testColumnUnits(self) -> None:
        self.assertEqual(self.table["intra_field"].unit, u.deg)
        self.assertEqual(self.table["extra_field"].unit, u.deg)
        self.assertEqual(self.table["intra_centroid"].unit, u.pixel)
        self.assertEqual(self.table["extra_centroid"].unit, u.pixel)
        for j in _NOLL:
            self.assertEqual(self.table[f"Z{j}_deviation"].unit, u.nm)

    def testDeviationsAreCatalogMicronsInNanometers(self) -> None:
        table = _build(_catalog(*_paired_groups(), zk_um=2.5))[191]
        for j in _NOLL:
            self.assertAlmostEqual(
                float(_data_rows(table)[f"Z{j}_deviation"][0].to_value(u.nm)), 2500.0, places=3
            )


class TestFieldAngleConvention(unittest.TestCase):
    """The DVCS/CCS axis swap, which is silent when wrong."""

    def testFieldAxesAreSwappedRelativeToCcs(self) -> None:
        """``*_field`` is DVCS, i.e. (thy_ccs, thx_ccs) in degrees.

        `DonutStamp.calcFieldXY` -- what the non-blitz table is built
        from -- returns DVCS, and `CalcZernikesTask` unpacks it as
        (ccs_y, ccs_x). blitz stores CCS, so the builder has to swap
        back. Getting this wrong yields plausible numbers in the wrong
        orientation.
        """
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        row = _data_rows(table)[0]
        # Donut A: donut_id 1, so thx_ccs=0.01 rad, thy_ccs=0.02 rad.
        x, y = _xy(row, "extra_field", u.deg)
        self.assertAlmostEqual(x, np.degrees(0.02), places=4)
        self.assertAlmostEqual(y, np.degrees(0.01), places=4)
        # Donut P: donut_id 11.
        x, y = _xy(row, "intra_field", u.deg)
        self.assertAlmostEqual(x, np.degrees(0.02 * 11), places=4)
        self.assertAlmostEqual(y, np.degrees(0.01 * 11), places=4)

    def testSidesAreNotTransposed(self) -> None:
        """intra columns come from SW1 and extra from SW0."""
        groups, unmatched = _paired_groups()
        row = _data_rows(_build(_catalog(groups, unmatched))[191])[0]
        self.assertEqual(str(row["extra_donut_id"]), "1")
        self.assertEqual(str(row["intra_donut_id"]), "11")
        self.assertAlmostEqual(_xy(row, "extra_centroid", u.pixel)[0], 101.0, places=4)
        self.assertAlmostEqual(_xy(row, "intra_centroid", u.pixel)[0], 111.0, places=4)
        self.assertAlmostEqual(float(row["extra_sn"]), _EXTRA_SNR[1], places=3)
        self.assertAlmostEqual(float(row["intra_sn"]), _INTRA_SNR[11], places=3)


class TestModeRowSemantics(unittest.TestCase):
    """One test per dispatch mode; this is where the schema is stretched."""

    def testPaired(self) -> None:
        """Two groups of two, both sides populated; surplus donut absent."""
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        rows = _data_rows(table)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows["label"]), ["pair1", "pair2"])
        for row in rows:
            self.assertNotEqual(str(row["extra_donut_id"]), "")
            self.assertNotEqual(str(row["intra_donut_id"]), "")
        # Donut C had no partner, so no row mentions it.
        self.assertNotIn("3", [str(v) for v in rows["extra_donut_id"]])

    def testUnpaired(self) -> None:
        """One row per donut, each with exactly one side populated."""
        groups, unmatched = _unpaired_groups()
        table = _build(_catalog(groups, unmatched, mode="unpaired"), mode="unpaired")[191]
        rows = _data_rows(table)
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows["label"]), [f"donut{i}" for i in range(1, 6)])
        for row in rows:
            populated = [side for side in ("intra", "extra") if str(row[f"{side}_donut_id"]) != ""]
            self.assertEqual(len(populated), 1, "exactly one side per unpaired row")
        # Donut C, dropped by paired mode, gets a row here.
        self.assertIn("3", [str(v) for v in rows["extra_donut_id"]])

    def testFullDetector(self) -> None:
        """One row per detector; the multi-donut side NaN, Zernikes not."""
        groups, unmatched = _full_detector_groups()
        table = _build(_catalog(groups, unmatched, mode="full_detector"), mode="full_detector")[191]
        rows = _data_rows(table)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows["label"]), ["group1", "group2"])
        for row in rows:
            self.assertEqual(str(row["extra_donut_id"]), "")
            self.assertEqual(str(row["intra_donut_id"]), "")
            self.assertTrue(np.isnan(_xy(row, "extra_field", u.deg)[0]))
            for j in _NOLL:
                self.assertFalse(np.isnan(float(row[f"Z{j}_deviation"].to_value(u.nm))))

    def testFullCorner(self) -> None:
        """A five-donut joint fit is ONE row, not five."""
        groups, unmatched = _full_corner_groups()
        table = _build(_catalog(groups, unmatched, mode="full_corner"), mode="full_corner")[191]
        # average + exactly one data row. The guard against a per-donut
        # explosion: the catalog replicates group_* onto all five member rows.
        self.assertEqual(len(table), 2)
        row = _data_rows(table)[0]
        self.assertEqual(str(row["label"]), "group1")
        for side in ("intra", "extra"):
            self.assertEqual(str(row[f"{side}_donut_id"]), "")
        for j in _NOLL:
            self.assertFalse(np.isnan(float(row[f"Z{j}_deviation"].to_value(u.nm))))

    def testGroupDedupeAcrossModes(self) -> None:
        """Row count tracks group count, never donut count."""
        for mode, (groups, unmatched) in (
            ("paired", _paired_groups()),
            ("unpaired", _unpaired_groups()),
            ("full_detector", _full_detector_groups()),
            ("full_corner", _full_corner_groups()),
        ):
            table = _build(_catalog(groups, unmatched, mode=mode), mode=mode)[191]
            self.assertEqual(len(_data_rows(table)), len(groups), mode)

    def testJointFitWithOneDonutPerSideIsPopulated(self) -> None:
        """The NaN rule keys off donut count, not off the mode.

        A sparse full_corner visit -- one donut per detector -- has an
        unambiguous field position on each side, so the row looks exactly
        like a paired one. Guards against implementing the rule as
        "joint mode => NaN".
        """
        extra, intra = _fixture_donuts()
        groups = {"R00": [extra[0], intra[0]]}
        table = _build(_catalog(groups, mode="full_corner"), mode="full_corner")[191]
        row = _data_rows(table)[0]
        self.assertEqual(str(row["extra_donut_id"]), "1")
        self.assertEqual(str(row["intra_donut_id"]), "11")
        self.assertFalse(np.isnan(_xy(row, "extra_field", u.deg)[0]))
        self.assertFalse(np.isnan(_xy(row, "intra_field", u.deg)[0]))

    def testSurplusDonutsProduceNoRow(self) -> None:
        """A donut no fit consumed has group_id "" and is skipped."""
        extra, intra = _fixture_donuts()
        groups = {"R00_1_11": [extra[0], intra[0]]}
        table = _build(_catalog(groups, unmatched=extra[1:]))[191]
        self.assertEqual(len(_data_rows(table)), 1)


class TestFitFailures(unittest.TestCase):
    """Failed fits are NaN'd; an all-failed corner still emits a table."""

    def testFailedGroupIsNaN(self) -> None:
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched, success={"R00_2_12": False}))[191]
        rows = _data_rows(table)
        for j in _NOLL:
            self.assertFalse(np.isnan(float(rows[f"Z{j}_deviation"][0].to_value(u.nm))))
            self.assertTrue(np.isnan(float(rows[f"Z{j}_deviation"][1].to_value(u.nm))))

    def testAllFailedStillHasAverageRow(self) -> None:
        """The consumer indexes [0] into the match, so it must exist."""
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched, success={gid: False for gid in groups}))[191]
        average = table[table["label"] == "average"]
        self.assertEqual(len(average), 1)
        self.assertFalse(bool(average["used"][0]))
        self.assertEqual(table.meta["estimatorInfo"]["blur_clipped"], [False] * len(_data_rows(table)))


class TestBlurClip(unittest.TestCase):
    """The blur-clip path and its row threshold."""

    def testOutlierBlurIsClipped(self) -> None:
        extra, intra = _fixture_donuts()
        groups = {
            "R00_1_11": [extra[0], intra[0]],
            "R00_2_12": [extra[1], intra[1]],
            "R00_3_13": [extra[2]],
        }
        # The third group's donut blur is a gross outlier.
        fwhm = {"R00_1_11": 0.90, "R00_2_12": 0.91, "R00_3_13": 9.0}
        table = _build(_catalog(groups, fwhm=fwhm))[191]
        self.assertIn("blur_clipped", table.meta["estimatorInfo"])
        self.assertEqual(table.meta["estimatorInfo"]["blur_clipped"], [False, False, True])
        self.assertFalse(bool(_data_rows(table)["used"][2]))

    def testSkippedBelowThreshold(self) -> None:
        """full_corner's single row cannot be clipped against anything.

        The average must still be the configured sigma-clip combine's, not the
        unweighted mean blurClipZkTable would substitute.
        """
        groups, unmatched = _full_corner_groups()
        catalog = _catalog(groups, unmatched, mode="full_corner", zk_um=3.0)
        table = _build(catalog, mode="full_corner")[191]
        self.assertEqual(len(_data_rows(table)), 1)
        self.assertEqual(table.meta["estimatorInfo"]["blur_clipped"], [False])
        average = table[table["label"] == "average"][0]
        for j in _NOLL:
            self.assertAlmostEqual(float(average[f"Z{j}_deviation"].to_value(u.nm)), 3000.0, places=3)

    def testThresholdIsConfigurable(self) -> None:
        """Raising the threshold past the row count skips the clip.

        Both paths leave ``blur_clipped`` all-False here, because mad_std
        cannot flag an outlier among two samples either way (see
        `testTwoSamplesCannotBeClipped`). What distinguishes them is the
        average: running the clip substitutes an unweighted mean for the
        configured combine's result, which is the cost the threshold
        exists to avoid.
        """
        extra, intra = _fixture_donuts()
        groups = {"R00_1_11": [extra[0], intra[0]], "R00_2_12": [extra[1], intra[1]]}
        # Distinct per-group Zernikes, so a mean over them is distinguishable
        # from a sigma-clipped combine that rejected one.
        catalog = _catalog(groups, zk_um={"R00_1_11": 1.0, "R00_2_12": 2.0})

        ran = _build(catalog, blur_clip_min_rows=2)[191]
        skipped = _build(catalog, blur_clip_min_rows=3)[191]
        for table in (ran, skipped):
            self.assertEqual(table.meta["estimatorInfo"]["blur_clipped"], [False, False])
        # Nothing was rejected on either path, so both averages are the mean of
        # 1 and 2 microns -- the point being that the skipped path reached it
        # without discarding the configured combine.
        for table in (ran, skipped):
            average = table[table["label"] == "average"][0]
            self.assertAlmostEqual(float(average["Z4_deviation"].to_value(u.nm)), 1500.0, places=3)

    def testTwoSamplesCannotBeClipped(self) -> None:
        """Why the default threshold is 3.

        ``mad_std`` of two points is symmetric about their own median, so
        neither can exceed the sigma bound however far apart they are. Blur
        clipping only becomes meaningful at three rows, which is what makes
        skipping it below that free rather than a compromise.
        """
        from astropy.stats import sigma_clip

        self.assertFalse(any(sigma_clip(np.array([0.9, 9.0]), stdfunc="mad_std", sigma_lower=99).mask))
        self.assertTrue(any(sigma_clip(np.array([0.90, 0.91, 9.0]), stdfunc="mad_std", sigma_lower=99).mask))

    def testDisabledEntirely(self) -> None:
        groups, unmatched = _paired_groups()
        catalog = _catalog(groups, unmatched, fwhm={"R00_1_11": 0.9, "R00_2_12": 9.0})
        table = _build(catalog, do_blur_clip=False, blur_clip_min_rows=1)[191]
        self.assertFalse(any(table.meta["estimatorInfo"]["blur_clipped"]))


class TestConsumerContract(unittest.TestCase):
    """The downstream reader's exact access pattern.

    Mirrors ``WavefrontCollection.pop``: read noll_indices, read the configured
    column list and reject it if empty, select the average row and index [0],
    then convert every column to microns. These are the failure modes
    that would otherwise only surface in production.
    """

    @staticmethod
    def _read(table: QTable, pattern: str = "deviation_columns") -> np.ndarray:
        """The consumer, reproduced."""
        table.meta["noll_indices"]
        z_columns = table.meta[pattern]
        if len(z_columns) == 0:
            raise ValueError(f"No zernike columns found for {pattern}.")
        average_row = table[table["label"] == "average"][0]
        return np.array([average_row[col].to(u.um).value for col in z_columns])

    def _all_mode_tables(self):
        for mode, (groups, unmatched) in (
            ("paired", _paired_groups()),
            ("unpaired", _unpaired_groups()),
            ("full_detector", _full_detector_groups()),
            ("full_corner", _full_corner_groups()),
        ):
            yield mode, _build(_catalog(groups, unmatched, mode=mode), mode=mode)[191]

    def testReadableInEveryMode(self) -> None:
        for mode, table in self._all_mode_tables():
            values = self._read(table)
            self.assertEqual(len(values), len(_NOLL), mode)
            self.assertTrue(np.all(np.isfinite(values)), mode)
            # 1 micron of deviation in, 1 micron back out.
            np.testing.assert_allclose(values, 1.0, rtol=1e-5)

    def testReadableWhenAllFitsFailed(self) -> None:
        """The degenerate table is read like any other; must not raise."""
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched, success={gid: False for gid in groups}))[191]
        values = self._read(table)
        self.assertEqual(len(values), len(_NOLL))
        self.assertTrue(np.all(np.isnan(values)))

    def testMisconfiguredPatternRaisesLoudly(self) -> None:
        """opd/intrinsic are present but empty, so the reader's guard fires.

        Those quantities are not meaningfully defined for blitz output, so a
        consumer pointed at them is misconfigured and must fail rather than
        silently average nonsense. Empty-but-present beats absent: a KeyError
        would be less legible than the reader's own message.
        """
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        for pattern in ("opd_columns", "intrinsic_columns"):
            self.assertIn(pattern, table.meta)
            self.assertEqual(table.meta[pattern], [])
            with self.assertRaises(ValueError):
                self._read(table, pattern)

    def testIsQTableWithQuantityColumns(self) -> None:
        """A plain Table loses .to(), breaking the consumer's last line."""
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        self.assertIsInstance(table, QTable)
        for j in _NOLL:
            self.assertIsInstance(table[f"Z{j}_deviation"], u.Quantity)

    def testSurvivesSerializationRoundTrip(self) -> None:
        """The contract must hold as read back, not merely as built.

        ECSV, since that is what the ``AstropyQTable`` storage class persists
        through -- and it is what carries the structured ``*_field`` columns,
        which parquet cannot represent. `CalcZernikesTask`'s own ``zernikes``
        has the same columns and persists the same way.
        """
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zernikes.ecsv")
            table.write(path, format="ascii.ecsv")
            restored = QTable.read(path, format="ascii.ecsv")

        values = self._read(restored)
        np.testing.assert_allclose(values, 1.0, rtol=1e-5)
        self.assertEqual(restored.meta["deviation_columns"], table.meta["deviation_columns"])
        # The empty lists have to survive too: they are what makes a
        # misconfigured consumer raise rather than KeyError.
        self.assertEqual(restored.meta["opd_columns"], [])
        self.assertEqual(restored.meta["intrinsic_columns"], [])
        self.assertEqual(restored["intra_field"].unit, u.deg)

    def testMetadataIdentifiesTheCorner(self) -> None:
        groups, unmatched = _paired_groups()
        table = _build(_catalog(groups, unmatched))[191]
        self.assertEqual(table.meta["extra"]["det_name"], "R00_SW0")
        self.assertEqual(table.meta["intra"]["det_name"], "R00_SW1")
        self.assertEqual(table.meta["extra"]["visit"], _VISIT)
        self.assertEqual(table.meta["cam_name"], "LSSTCam")
        self.assertEqual(table.meta["noll_indices"], list(_NOLL))
        self.assertEqual(table.meta["deviation_columns"], [f"Z{j}_deviation" for j in _NOLL])


class TestTaskWiring(unittest.TestCase):
    """The config flag and the output connection."""

    def testConnectionAbsentWhenDisabled(self) -> None:
        config = DonutBlitzCornerConfig()
        self.assertFalse(config.doZernikesOutput)
        self.assertNotIn("zernikes", DonutBlitzCornerConnections(config=config).outputs)

    def testConnectionPresentWhenEnabled(self) -> None:
        config = DonutBlitzCornerConfig()
        config.doZernikesOutput = True
        connections = DonutBlitzCornerConnections(config=config)
        self.assertIn("zernikes", connections.outputs)
        zernikes = connections.zernikes
        self.assertEqual(zernikes.name, "zernikes")
        self.assertEqual(zernikes.storageClass, "AstropyQTable")
        self.assertEqual(set(zernikes.dimensions), {"visit", "detector", "instrument"})
        # Load-bearing: a visit-dimensioned quantum gets a predicted ref per
        # corner detector and fills only the extra-focal ones.
        self.assertTrue(zernikes.multiple)

    def testCombineDefaultsToDeviationClipping(self) -> None:
        """The other two families are absent; clipping them cannot work."""
        config = DonutBlitzCornerConfig()
        config.doZernikesOutput = True
        self.assertEqual(config.combineZernikes.value.zkClipType, "deviation")


class TestEmptyInputs(unittest.TestCase):
    """Degenerate catalogs."""

    def testEmptyCatalogYieldsNoTables(self) -> None:
        self.assertEqual(_build(QTable()), {})

    def testCornerWithNoFitsIsOmitted(self) -> None:
        """A corner nothing was fit for is absent, not written empty.

        Its predicted output ref is simply left unfilled, which `multiple=True`
        permits.
        """
        groups, unmatched = _paired_groups()
        tables = _build(_catalog(groups, unmatched))
        self.assertEqual(sorted(tables), [191])
        for absent in (195, 199, 203):
            self.assertNotIn(absent, tables)

    def testIntraOnlyCornerStillKeysOnSw0(self) -> None:
        """An intra-only corner is keyed on SW0, which contributed nothing.

        The corner's ids are adjacent with SW0 first, so the intra detector's
        id - 1 supplies the key.
        """
        _, intra = _fixture_donuts()
        table = _build(_catalog({"R00": intra}, mode="full_corner"), mode="full_corner")
        self.assertEqual(sorted(table), [191])
        self.assertEqual(table[191].meta["extra"], {})
        self.assertEqual(table[191].meta["intra"]["det_name"], "R00_SW1")


if __name__ == "__main__":
    unittest.main()
