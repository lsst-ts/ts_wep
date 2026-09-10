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

"""The diagnostic plot task, driven from a real `build_donut_catalog` table.

The plot task is the only in-tree consumer of the output catalog, so it is what
breaks when a column is renamed.  These tests build the catalog through the
builder rather than by hand, so the two stay pinned to the same schema.
"""

import os
import tempfile
import unittest

import numpy as np

from lsst.ts.wep.blitz.catalogBuilder import CatalogOptions, build_donut_catalog
from lsst.ts.wep.blitz.dataStructures import Donut, WfResult
from lsst.ts.wep.blitz.donutBlitzPlotTask import (
    DonutBlitzPlotTask,
    DonutBlitzPlotTaskConfig,
)
from lsst.ts.wep.blitz.utils import _ZK_JMAX

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
        stamp=np.random.default_rng(donut_id).normal(100.0, 5.0, (_STAMP_SIZE, _STAMP_SIZE)).astype(np.float32),
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
    return {
        "det_name": det_name,
        "scatter_arcsec": 0.3,
        "wcs_refit_error": "",
        "cat_select_error": "",
        "rejected_catalog": list(rejected),
        "selection_source": "refcat",
        # Matches the Donut's own n_quarter: the stamps were rotated by it, and
        # the plot reads it back from det_meta to place the refcat overlays.
        "n_quarter": 1,
        "pair_path": "snr_rank",
    }


def _wf_result(donuts, group_id):
    """One joint fit over ``donuts``, with images the plot task can draw."""
    rng = np.random.default_rng(0)
    return {
        "donuts": [
            WfResult(
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
    }


def _options():
    return CatalogOptions(
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
    rejected = _donut(
        "R00_SW0", 4, rejected=True, rejected_snr=True, defocal_offsets=_EXTRA_OFFSETS
    )
    return build_donut_catalog(
        [_result("R00_SW0", rejected=[rejected]), _result("R00_SW1")],
        [_wf_result([extra, intra], "R00_1_2")],
        [extra, intra, surplus],
        [surplus],
        _VISIT_ID,
        _options(),
        rtp_rad=0.25,
        run_elapsed=30.0,
        refcat_elapsed=1.5,
        butler_elapsed=4.0,
        butler_times={"raw": 3.0, "bias": 1.0},
        cutout_elapsed=6.0,
        danish_elapsed=12.0,
    )


class TestDonutBlitzPlotTask(unittest.TestCase):
    """The plot task reads a builder-produced catalog end to end."""

    def testPlotsAreWrittenFromABuilderCatalog(self) -> None:
        catalog = _catalog()
        task = DonutBlitzPlotTask(config=DonutBlitzPlotTaskConfig())
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
        task = DonutBlitzPlotTask(config=DonutBlitzPlotTaskConfig())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                task.run(
                    build_donut_catalog([], [], [], [], _VISIT_ID, _options())
                )
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
