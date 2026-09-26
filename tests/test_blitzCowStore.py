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

"""The parent -> worker store contract, `CowStore`.

The submodule's central invariant is that workers read shared state from a
module-level object inherited copy-on-write, so nothing large crosses a pickle
boundary. That makes two things worth pinning that no other test covers:

* a forked child sees what the parent adopted, *through the binding the worker
  modules made at import time* -- which is why `CowStore.adopt` mutates in
  place instead of rebinding the module global; and
* the failure modes that replaced string-keyed lookups: a field the mode never
  set, and a field left over from the previous mode.
"""

import os
import pickle
import unittest

from lsst.ts.wep.blitz import cutoutPipeline, famPipeline
from lsst.ts.wep.blitz.utils import (
    _COW_STORE,
    CornerDetectorInputs,
    CowStore,
    FamDetectorInputs,
    IsrCalibs,
)

_INTRA = (0.0, -1.5e-3, 0.0)
_EXTRA = (0.0, +1.5e-3, 0.0)

# Enough of a telescope to satisfy the store; `for_fam` traces chief rays
# through it, so the real thing is needed there but not here.
_STUB_TELESCOPE = "stub-telescope"


def _corner_store(**overrides):
    """A corner store filled with stubs, since nothing here runs a subtask."""
    kwargs = dict(
        isr_task="isr",
        bkg_task="bkg",
        diam_task="diam",
        detect_task="detect",
        astrom_task="astrom",
        select_task="selector",
        measure_task="measure",
        cut_task="cut",
        wf_fit_task="wf",
        wf_estimation_mode="danish",
        max_fit_scatter=2.0,
        astrom_ref_filter="phot_g_mean",
        photo_ref_filter="lsst_r",
        telescope=_STUB_TELESCOPE,
        corner_detectors={
            "R00_SW0": CornerDetectorInputs(
                raw="raw",
                calibs=IsrCalibs(ptc="ptc", flat="flat", linearizer="lin", crosstalk="ct"),
            )
        },
        det_refcats={"R00_SW0": "refcat"},
    )
    kwargs.update(overrides)
    return CowStore.for_corner(**kwargs)


class TestStoreCrossesTheFork(unittest.TestCase):
    """The copy-on-write inheritance the whole submodule is built on."""

    def tearDown(self) -> None:
        _COW_STORE.__dict__.clear()

    def testChildSeesWhatTheParentAdopted(self) -> None:
        """A forked child reads the parent's store without a pickle.

        Read through ``cutoutPipeline._COW_STORE`` rather than the copy this
        test imported, because that module bound the name at import time. If
        `adopt` ever became a rebinding of the module global, the parent would
        see the new store and every worker the old one -- silently, and only in
        whichever mode populated last.
        """
        _COW_STORE.adopt(_corner_store(max_fit_scatter=7.5))

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            try:
                os.close(read_fd)
                seen = (
                    cutoutPipeline._COW_STORE.max_fit_scatter,
                    cutoutPipeline._COW_STORE.corner_detectors["R00_SW0"].calibs.flat,
                    famPipeline._COW_STORE.isr_task,
                )
                os.write(write_fd, pickle.dumps(seen))
                os._exit(0)
            except BaseException:  # noqa: BLE001 -- must not run the parent's teardown
                os._exit(1)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb") as stream:
            payload = stream.read()
        _, status = os.waitpid(pid, 0)

        self.assertEqual(os.waitstatus_to_exitcode(status), 0, "child could not read the store")
        self.assertEqual(pickle.loads(payload), (7.5, "flat", "isr"))

    def testChildDoesNotPropagateWritesBack(self) -> None:
        """The contract is one-directional: a child's write is its own.

        Not a limitation to work around but the reason the store holds no
        caches -- a worker-side memo would be invisible to every other worker
        and to the parent, so anything worth memoizing is computed before the
        fork instead.
        """
        _COW_STORE.adopt(_corner_store(max_fit_scatter=1.0))

        pid = os.fork()
        if pid == 0:  # child
            _COW_STORE.max_fit_scatter = 99.0
            os._exit(0)
        os.waitpid(pid, 0)

        self.assertEqual(_COW_STORE.max_fit_scatter, 1.0)


class TestStoreFailureModes(unittest.TestCase):
    """What replaced a `KeyError` from a string-keyed dict."""

    def tearDown(self) -> None:
        _COW_STORE.__dict__.clear()

    def testFieldTheModeNeverSetRaises(self) -> None:
        """A full-array field read in corner mode must not read as None.

        The fields have no defaults precisely so that this is an
        `AttributeError` rather than a silently plausible value.
        """
        _COW_STORE.adopt(_corner_store())
        with self.assertRaises(AttributeError):
            _COW_STORE.pair_match_tolerance
        with self.assertRaises(AttributeError):
            _COW_STORE.radial_scale_by_offsets

    def testAdoptClearsRatherThanMerges(self) -> None:
        """A `Task` instance is reused across quanta, so leftovers must go.

        Built by hand rather than through `for_fam`, which would want a real
        batoid telescope to trace through.
        """
        _COW_STORE.adopt(_corner_store())
        self.assertIn("R00_SW0", _COW_STORE.corner_detectors)

        fam = CowStore.uninitialized()
        fam.fam_detectors = {
            94: FamDetectorInputs(
                raws={101: "h", 102: "h"},
                ptc="ptc",
                flat="flat",
                linearizer="lin",
                crosstalk="ct",
                intrinsic_zernikes=None,
            )
        }
        _COW_STORE.adopt(fam)

        self.assertIn(94, _COW_STORE.fam_detectors)
        with self.assertRaises(AttributeError):
            _COW_STORE.corner_detectors

    def testForgottenFieldIsRejected(self) -> None:
        """A key one mode forgot is a TypeError here, not a worker crash.

        The point of spelling every field out in `for_corner` / `for_fam`
        instead of forwarding ``**kwargs``.
        """
        kwargs = dict(
            isr_task="isr",
            bkg_task="bkg",
            diam_task="diam",
            detect_task="detect",
            astrom_task="astrom",
            select_task="selector",
            measure_task="measure",
            cut_task="cut",
            wf_fit_task="wf",
            wf_estimation_mode="danish",
            astrom_ref_filter="phot_g_mean",
            photo_ref_filter="lsst_r",
            telescope=_STUB_TELESCOPE,
            corner_detectors={},
            det_refcats={},
        )  # max_fit_scatter omitted
        with self.assertRaisesRegex(TypeError, "max_fit_scatter"):
            CowStore.for_corner(**kwargs)


if __name__ == "__main__":
    unittest.main()
