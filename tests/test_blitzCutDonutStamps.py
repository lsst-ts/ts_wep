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

"""A donut is a refcat source too, so it must not appear in its own neighbours.

The nearby-source lists come from a box query over the whole refcat, which
always contains the donut itself at zero offset. `CutDonutStampsTask` removes
it by refcat id -- exact, rather than a distance threshold -- which is only
sound because ``refcat`` is non-None precisely when the selections were drawn
from it. These tests pin that, and pin the donut's own magnitudes being carried
across from the selections table on the refcat path and left NaN off it.
"""

import unittest

import numpy as np
from astropy.table import QTable

import lsst.afw.image as afwImage
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.blitz.cutDonutStampsTask import (
    CutDonutStampsConfig,
    CutDonutStampsTask,
)
from lsst.ts.wep.blitz.utils import _REFCAT_COLUMNS

# Detector-frame positions of the two selected donuts, far enough apart that
# neither lands in the other's stamp box.
_D1_XY = (100.0, 100.0)
_D2_XY = (300.0, 300.0)
# A third refcat source, not selected as a donut, 4 px from the first donut and
# so inside its stamp box.
_NEIGHBOR_XY = (104.0, 100.0)
_STAMP_SIZE = 21


def _exposure():
    """A flat, unmasked exposure on a real detector (FIELD_ANGLE is needed)."""
    detector = LsstCam().getCamera()["R22_S11"]
    exposure = afwImage.ExposureF(detector.getBBox())
    exposure.setDetector(detector)
    exposure.image.array[:] = 1.0
    exposure.setFilter(afwImage.FilterLabel(band="r", physical="r_03"))
    exposure.getInfo().setVisitInfo(afwImage.VisitInfo(id=12345))
    return exposure


def _measurements(with_refcat_values: bool, d1_xy=_D1_XY) -> QTable:
    """The two selected donuts, as the measurement task would hand them over.

    ``with_refcat_values`` distinguishes the two upstream paths, but only in the
    *values*: the refcat path carries the donut's own magnitudes and sky position
    through from the refcat subset, while the blind-detection path has the same
    columns NaN-filled (see `_REFCAT_COLUMNS`, filled in `_cutout_one_exposure`).
    The schema is the same either way, which is what lets `CutDonutStampsTask`
    read them unconditionally.
    """
    table = QTable(
        {
            "donut_id": np.array([10, 20], dtype=np.int64),
            "centroid_x": np.array([d1_xy[0], _D2_XY[0]]),
            "centroid_y": np.array([d1_xy[1], _D2_XY[1]]),
            "flux": np.array([500.0, 400.0]),
            "inner_frac": np.array([0.0, 0.0]),
            "outer_frac": np.array([0.0, 0.0]),
            "outer_sector_minmax_frac": np.array([0.0, 0.0]),
            "snr": np.array([1e4, 1e4]),
            "bkg_std": np.array([1.0, 1.0]),
            "bkg": np.array([0.0, 0.0]),
        }
    )
    values = {
        "photo_mag": np.array([14.0, 15.0]),
        "astrom_mag": np.array([14.2, 15.2]),
        # Radians, as afw stores them.
        "coord_ra": np.radians(np.array([30.0, 30.1])),
        "coord_dec": np.radians(np.array([-20.0, -20.1])),
    }
    for column in _REFCAT_COLUMNS:
        table[column] = (
            values[column] if with_refcat_values else np.full(len(table), np.nan)
        )
    return table


def _refcat(d1_xy=_D1_XY) -> QTable:
    """Both donuts plus the unselected neighbour, as ids 10, 20 and 30."""
    return QTable(
        {
            "donut_id": np.array([10, 20, 30], dtype=np.int64),
            "centroid_x": np.array([d1_xy[0], _D2_XY[0], _NEIGHBOR_XY[0]]),
            "centroid_y": np.array([d1_xy[1], _D2_XY[1], _NEIGHBOR_XY[1]]),
            "photo_mag": np.array([14.0, 15.0, 18.0]),
            "astrom_mag": np.array([14.2, 15.2, 18.2]),
        }
    )


def _run(measurements, refcat):
    config = CutDonutStampsConfig()
    config.stampSize = _STAMP_SIZE
    task = CutDonutStampsTask(config=config)
    result = task.run(_exposure(), measurements, refcat, donutRadius=8.0)
    return {d.donut_id: d for d in result.donuts}


class TestNeighborSelfExclusion(unittest.TestCase):
    def test_donut_excluded_from_its_own_neighbors(self):
        """Only the unselected id-30 source is a neighbour of donut 10."""
        donuts = _run(_measurements(with_refcat_values=True), _refcat())

        self.assertEqual(len(donuts[10].nearby_photo), 1)
        dx, dy, mag = donuts[10].nearby_photo[0]
        # 4 px away in x, and the faint magnitude -- i.e. the id-30 source, not
        # the donut itself, which would be at (0, 0) with mag 14.
        self.assertAlmostEqual(dx, _NEIGHBOR_XY[0] - _D1_XY[0])
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(mag, 18.0)

    def test_isolated_donut_has_no_neighbors(self):
        """An isolated donut has an empty neighbour list, not one holding itself."""
        donuts = _run(_measurements(with_refcat_values=True), _refcat())

        self.assertEqual(donuts[20].nearby_photo, [])
        self.assertEqual(donuts[20].nearby_astrom, [])

    def test_own_magnitudes_carried_from_selections(self):
        donuts = _run(_measurements(with_refcat_values=True), _refcat())

        self.assertAlmostEqual(donuts[10].photo_mag, 14.0)
        self.assertAlmostEqual(donuts[10].astrom_mag, 14.2)
        self.assertAlmostEqual(donuts[20].photo_mag, 15.0)
        self.assertAlmostEqual(donuts[20].astrom_mag, 15.2)

    def test_own_sky_position_carried_from_selections(self):
        """Radians in, radians on the Donut -- the degrees conversion is the builder's."""
        donuts = _run(_measurements(with_refcat_values=True), _refcat())

        self.assertAlmostEqual(np.degrees(donuts[10].coord_ra), 30.0)
        self.assertAlmostEqual(np.degrees(donuts[10].coord_dec), -20.0)
        self.assertAlmostEqual(np.degrees(donuts[20].coord_ra), 30.1)
        self.assertAlmostEqual(np.degrees(donuts[20].coord_dec), -20.1)

    def test_blind_path_has_no_refcat_information(self):
        """No refcat means no neighbours, magnitudes or sky position -- NaN, not zero.

        The columns are still *present*: `_REFCAT_COLUMNS` is NaN-filled upstream
        so the task reads them without testing the schema.
        """
        donuts = _run(_measurements(with_refcat_values=False), None)

        for donut_id in (10, 20):
            donut = donuts[donut_id]
            self.assertEqual(donut.nearby_photo, [])
            self.assertEqual(donut.nearby_astrom, [])
            self.assertTrue(np.isnan(donut.photo_mag))
            self.assertTrue(np.isnan(donut.astrom_mag))
            self.assertTrue(np.isnan(donut.coord_ra))
            self.assertTrue(np.isnan(donut.coord_dec))


class TestRefcatColumnsAreRequired(unittest.TestCase):
    """The task reads `_REFCAT_COLUMNS` unconditionally, by design.

    It used to test ``measurements.colnames`` for them and substitute NaN, which
    made a schema gap indistinguishable from a genuine data gap. The guarantee now
    lives upstream, in `_cutout_one_exposure`, so a table arriving without them is
    a bug and should say so rather than quietly producing NaN donuts.
    """

    def test_each_column_is_load_bearing(self):
        for column in _REFCAT_COLUMNS:
            with self.subTest(column=column):
                measurements = _measurements(with_refcat_values=True)
                measurements.remove_column(column)
                with self.assertRaises(KeyError):
                    _run(measurements, _refcat())


class TestNearbyOffsetOrigin(unittest.TestCase):
    """Offsets are measured from x_det/y_det, so they compose with them.

    The stamp itself is cut on integer bounds around the rounded centroid, so a
    consumer measuring from there instead would be off by the rounding residual
    -- up to half a pixel, with nothing in the schema to warn them.
    """

    # 0.3 px off a pixel centre, so the rounded centroid the stamp was cut on
    # (100.0) differs measurably from the reported x_det (100.3).
    _FRACTIONAL_D1_XY = (100.3, 100.0)

    def _neighbor_of_donut_10(self):
        d1_xy = self._FRACTIONAL_D1_XY
        donuts = _run(
            _measurements(with_refcat_values=True, d1_xy=d1_xy),
            _refcat(d1_xy=d1_xy),
        )
        donut = donuts[10]
        self.assertEqual(len(donut.nearby_photo), 1)
        return donut, donut.nearby_photo[0]

    def test_offset_is_from_x_det(self):
        donut, (dx, dy, _) = self._neighbor_of_donut_10()

        self.assertAlmostEqual(dx, _NEIGHBOR_XY[0] - self._FRACTIONAL_D1_XY[0])
        self.assertAlmostEqual(dy, 0.0)

    def test_offset_is_not_from_the_rounded_centroid(self):
        """Stated explicitly, so a drift to the rounded centroid is unambiguous."""
        donut, (dx, _, _) = self._neighbor_of_donut_10()
        rounded = _NEIGHBOR_XY[0] - round(self._FRACTIONAL_D1_XY[0])

        self.assertAlmostEqual(rounded, 4.0)  # guard the fixture itself
        self.assertNotAlmostEqual(dx, rounded)

    def test_offset_composes_with_x_det(self):
        """The actual contract: x_det + dx is the neighbour's detector x."""
        donut, (dx, dy, _) = self._neighbor_of_donut_10()

        self.assertAlmostEqual(donut.x_det + dx, _NEIGHBOR_XY[0])
        self.assertAlmostEqual(donut.y_det + dy, _NEIGHBOR_XY[1])

    def test_rounding_residual_recovers_the_stamp_grid(self):
        """Adding ``x_det - round(x_det)`` back puts the offset on the stamp grid.

        This is the property `donutBlitzPlotTask._xform` relies on to draw refcat
        markers: the stamp was cut on integer bounds around the rounded centroid,
        so a neighbour sitting on an integer detector pixel must land exactly on
        a stamp pixel centre once the residual is applied. Omitting it puts the
        marker ~0.4 px off the source.
        """
        donut, (dx, dy, _) = self._neighbor_of_donut_10()
        residual_x = donut.x_det - round(donut.x_det)
        residual_y = donut.y_det - round(donut.y_det)

        for offset, residual in ((dx, residual_x), (dy, residual_y)):
            stamp_frame = offset + residual
            self.assertAlmostEqual(stamp_frame, round(stamp_frame))
        # And the uncorrected offset is genuinely not on the grid, so the
        # assertion above is not vacuous.
        self.assertNotAlmostEqual(dx, round(dx))

    def test_stamp_membership_uses_the_rounded_centroid(self):
        """A fractional centroid must not change which sources are in the box.

        Membership is pinned to the rounded centroid, because that is what the
        stamp bounds were cut on.
        """
        d1_xy = self._FRACTIONAL_D1_XY
        # Just inside the stamp: the box spans rounded centroid +/- (stampSize //
        # 2), so at stampSize 21 that is 100 +/- 10.
        edge_x = round(d1_xy[0]) + _STAMP_SIZE // 2
        refcat = _refcat(d1_xy=d1_xy)
        refcat["centroid_x"][2] = edge_x
        refcat["centroid_y"][2] = d1_xy[1]

        donuts = _run(_measurements(with_refcat_values=True, d1_xy=d1_xy), refcat)
        self.assertEqual(len(donuts[10].nearby_photo), 1)

        # One pixel further out and it drops, even though the distance from
        # x_det is under half the stamp.
        refcat["centroid_x"][2] = edge_x + 1
        donuts = _run(_measurements(with_refcat_values=True, d1_xy=d1_xy), refcat)
        self.assertEqual(donuts[10].nearby_photo, [])


if __name__ == "__main__":
    unittest.main()
