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

"""Records passed between the blitz pipeline stages."""

__all__ = [
    "CutoutResult",
    "Donut",
    "FamDetectorResult",
    "WfDonutResult",
    "WfGroupResult",
]

import os
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from .utils import _ZK_JMAX, _dense_intrinsic

# Cap on a worker's `error` string. A traceback-derived message can run to
# kilobytes, and 189 of them cross a pickle boundary; the first few hundred
# characters carry the exception type and message, which is what anyone reads.
_ERROR_MAX_CHARS = 400

# `fit_status` for a group no `least_squares` call ever returned for. Outside
# scipy's own range, whose statuses run -2..4, so it cannot be confused with
# one: an empty or killed group has no status rather than a bad one.
_FIT_STATUS_ABSENT = -99


@dataclass
class Donut:
    """One cut donut stamp with its selection/quality metrics."""

    det_name: str
    stamp: npt.NDArray[np.float64] | None  #  CCS
    thx_ccs: float
    thy_ccs: float
    flux: float
    band: str
    det_id: int
    visit_id: int
    x_det: float
    y_det: float
    donut_id: int
    inner_frac: float
    outer_frac: float
    outer_sector_minmax_frac: float
    donut_radius: float
    snr: float
    bkg: float
    bkg_std: float
    n_quarter: int
    # This donut's own refcat magnitudes, NaN on the blitz-detection path where
    # there is no refcat to take them from.
    photo_mag: float
    astrom_mag: float
    # Neighboring refcat sources inside the stamp box as (dx, dy, mag), offset
    # from (x_det, y_det) so the offsets compose with it directly. Excludes
    # this donut itself. Stamp *membership* is decided on the rounded centroid,
    # which is what the stamp bounds were cut on.
    nearby_photo: list[tuple[float, float, float]]
    nearby_astrom: list[tuple[float, float, float]]
    # Refcat sky position in radians, NaN on the blitz-detection path. Refcat
    # truth, not a projection of (x_det, y_det) through the WCS -- so a finite
    # value here means this donut was matched to a catalog source.
    coord_ra: float = float("nan")
    coord_dec: float = float("nan")
    intrinsic_zk: npt.NDArray[np.float64] | None = None
    # The optic shifts that put this donut off focus: signed meters, ordered
    # (detector, camera, m2) -- see `_defocused_telescope`. Set by whichever
    # task builds the donut, because the two modes decide it differently:
    # corner mode from the detector id (SW0/SW1 sit either side of focus within
    # one exposure), FAM from which exposure of the pair it came from. This is
    # the only representation of defocal state; an intra/extra label would be
    # redundant with it, and derivable from det_name (corner) or visit_id
    # (FAM).
    defocal_offsets: tuple[float, float, float] | None = None
    # --- reject flags (default False = not rejected) ---
    rejected_sat: bool = False
    rejected_inner_frac: bool = False
    rejected_outer_frac: bool = False
    rejected_snr: bool = False
    rejected: bool = False


@dataclass
class CutoutResult:
    """One exposure of one detector, as `_cutout_one_exposure` returns it.

    Corner mode produces one per detector, full-array mode two -- one per
    exposure of the pair, each tagged with its own ``visit_id``.  Consumed by
    both tasks' logging and by `_build_donut_catalog`, which turns most of
    these fields into that detector's ``table.meta["det_meta"]`` entry.

    Every ``*_run`` field is seconds, and a stage that never ran is NaN rather
    than 0.0: a detector that bailed out early must not read as one whose
    remaining stages were instantaneous.  The stage fields are keyed by
    `lsst.ts.wep.blitz.utils._CUTOUT_STAGE_KEYS`, which is where their order
    comes from -- everything reporting them iterates that mapping rather than
    spelling them out.
    """

    det_name: str
    catalog: list[Donut]  # accepted `Donut`s
    rejected_catalog: list[Donut]  # `Donut`s that failed a selection or quality cut
    isr_run: float
    bkg_run: float
    diam_run: float
    detect_run: float
    wcs_refit_run: float
    catalog_select_run: float
    stamp_cut_run: float
    # On-sky scatter of the refit WCS in arcseconds, None where no refit ran.
    scatter_arcsec: float | None
    wcs_refit_error: str
    cat_select_error: str
    # Where this detector's donut ids came from, and so whether they are
    # comparable across the two exposures of a pair: full-array mode can only
    # pair on an exact refcat-id match when both exposures selected from the
    # refcat.
    selection_source: str
    n_quarter: int  # detector orientation; relates the CCS stamp to x_det/y_det
    # The WCS actually used, or None. Full-array mode reads it back when
    # deciding how to pair donuts between the two exposures.
    wcs: Any
    # Which pairing algorithm ran. Not decided here -- it is the grouping
    # stage's, and both modes overwrite it once they know. The default stands
    # so that every result reaching `_build_donut_catalog` carries a meaningful
    # string, including one from a full-array worker that died before grouping.
    pair_path: str = "n/a"
    # The exposure this result covers, set by full-array mode after the fact
    # because the two results of a pair share a `det_name`. Corner mode leaves
    # it None and the catalog falls back to the visit it is being built for.
    visit_id: int | None = None
    # Dispatch-to-arrival latency of the work unit that produced this, set by
    # the corner-mode pool entry point only.
    dispatch_to_arrival: float = float("nan")

    @classmethod
    def no_detections(
        cls,
        det_name: str,
        n_quarter: int,
        isr_run: float,
        bkg_run: float,
        diam_run: float,
        detect_run: float,
    ) -> "CutoutResult":
        """Result for a detector on which blitz detection found nothing.

        The four stages that did run are reported; the three below them never
        ran and stay NaN.  ``selection_source`` is ``"no_detections"``, which
        is distinct from the selector running and rejecting everything
        (``"blitz_failed"``).
        """
        return cls(
            det_name=det_name,
            catalog=[],
            rejected_catalog=[],
            isr_run=isr_run,
            bkg_run=bkg_run,
            diam_run=diam_run,
            detect_run=detect_run,
            wcs_refit_run=float("nan"),
            catalog_select_run=float("nan"),
            stamp_cut_run=float("nan"),
            scatter_arcsec=None,
            wcs_refit_error="No blitz detections",
            cat_select_error="",
            selection_source="no_detections",
            n_quarter=n_quarter,
            wcs=None,
        )

    @classmethod
    def dead(cls, det_name: str, reason: str) -> "CutoutResult":
        """Stand-in for a detector whose worker was killed outright.

        `_cutout_one_exposure` reports its own failures in ``wcs_refit_error``
        / ``cat_select_error`` and always returns one of these, but it cannot
        report a SIGKILL -- no Python runs in a process the kernel has already
        destroyed.  So the parent synthesizes the same shape on the worker's
        behalf, letting the visit proceed on the surviving detectors instead of
        being lost entirely.

        Every timing is NaN and both catalogs empty, matching
        `no_detections`: a stage that never ran must not read as one that was
        instantaneous.

        Parameters
        ----------
        det_name : `str`
            Detector whose worker died.
        reason : `str`
            Cause, from `_fork_map`'s `_WorkerDeath`.
        """
        return cls(
            det_name=det_name,
            catalog=[],
            rejected_catalog=[],
            isr_run=float("nan"),
            bkg_run=float("nan"),
            diam_run=float("nan"),
            detect_run=float("nan"),
            wcs_refit_run=float("nan"),
            catalog_select_run=float("nan"),
            stamp_cut_run=float("nan"),
            scatter_arcsec=None,
            wcs_refit_error=f"worker died: {reason}",
            cat_select_error="",
            selection_source="worker_died",
            # Unknown: the orientation is read off the post-ISR exposure, which
            # this worker never got far enough to produce.
            n_quarter=0,
            wcs=None,
        )


# The complete vocabulary of `WfDonutResult.fit_outcome`, which says which
# of the mutually exclusive paths through
# `WavefrontFittingTask._run_lstsq_fit` produced a result.
FitOutcome = Literal[
    "ok",  # least_squares converged (success=True)
    "nonconvergent",  # least_squares returned but success=False
    "timeout",  # SIGALRM fired; wfFitTimeoutPerDonut * group_size exceeded
    "exception",  # the fit raised
    "x0_only",  # wfInitialGuessOnly: the model was evaluated, never fit
    "",  # no fit consumed this donut (`_NULL_WF_DONUT`)
]

_FIT_OUTCOMES = get_args(FitOutcome)


@dataclass
class WfDonutResult:
    """One donut's wavefront-fit outputs, produced by `_wf_worker`.

    Consumed by `_build_donut_catalog`, keyed by ``(donut_id, det_name,
    visit_id)``. A fit that timed out or raised still produces a
    WfDonutResult with ``fit_success=False`` and all-NaN Zernikes;
    ``fit_outcome`` says which.

    ``visit_id`` is part of the key because full-array mode fits the same star
    on the same detector twice, once per side of focus, so ``(donut_id,
    det_name)`` alone is ambiguous there. Corner mode has one visit per
    quantum, so including it changes nothing.
    """

    donut_id: int
    det_name: str
    visit_id: int
    zk_dev: npt.NDArray[np.float64]  # dense Noll 0.._ZK_JMAX, meters, NaN where unfit
    zk_intrinsic: npt.NDArray[np.float64]  # dense Noll 0.._ZK_JMAX, meters
    img: np.ndarray | None
    model_img: np.ndarray | None
    fit_success: bool
    fit_elapsed: float
    setup_elapsed: float
    fit_nfev: int
    fit_cost: float
    fit_optimality: float
    fit_njev: int
    fit_outcome: FitOutcome
    fit_dx: float
    fit_dy: float
    fit_flux: float
    fit_fwhm: float
    blend_frac: float
    group_id: str
    group_size: int


# Sentinel for "no fit consumed this donut". All-NaN Zernikes, empty strings,
# fit_success=False -- so the catalog's empty group_id and empty
# group_fit_outcome both fall out naturally.
_NULL_WF_DONUT = WfDonutResult(
    donut_id=-1,
    det_name="",
    visit_id=-1,
    zk_dev=np.full(_ZK_JMAX + 1, np.nan),
    zk_intrinsic=np.full(_ZK_JMAX + 1, np.nan),
    img=None,
    model_img=None,
    fit_success=False,
    fit_elapsed=float("nan"),
    setup_elapsed=float("nan"),
    fit_nfev=0,
    fit_cost=float("nan"),
    fit_optimality=float("nan"),
    fit_njev=0,
    fit_outcome="",
    fit_dx=float("nan"),
    fit_dy=float("nan"),
    fit_flux=float("nan"),
    fit_fwhm=float("nan"),
    blend_frac=float("nan"),
    group_id="",
    group_size=0,
)


@dataclass
class _WfGroup:
    donuts: list[Donut]  # donut dicts, ordered; each carries its own "det_name" key
    group_id: str
    band: str
    rtp: float | None  # Boresight rotation (spider angle), degrees or None
    alt: float | None  # Boresight altitude, radians or None


@dataclass
class WfGroupResult:
    """One fit group's outputs, produced by `WavefrontFittingTask.run`.

    A group is what gets fit jointly -- a pair in paired mode, one donut
    unpaired, a whole corner in ``corner_group`` mode -- so everything here is
    a property of the *fit*, not of a donut.  The per-donut half of the same
    fit is the `WfDonutResult` list in ``donut_results``, one per member, each
    carrying its own copy of the group's values for the output catalog's
    ``group_*`` columns.

    The ``fit_*`` fields are flattened from what used to be a nested
    ``fit_info`` dict, so a group that never fit reports typed NaN/0/``""``
    rather than a missing key.  ``fit_outcome`` is one of `_FIT_OUTCOMES` and
    says which of the mutually exclusive paths ran; it subsumes ``success``.

    ``imgs`` and ``model_imgs`` are dropped by `_shed_images` in full-array
    mode when the config says not to save them, since they dominate the size of
    what a worker pickles home.
    """

    group_id: str
    group_size: int
    # Sparse, one entry per fitted Noll index -- not the dense Noll-indexed
    # array `WfDonutResult.zk_dev` carries. Meters.
    zk_dev: npt.NDArray[np.float64]
    success: bool
    donut_results: list[WfDonutResult]  # one `WfDonutResult` per group member
    det_names: list[str]  # each member's detector, in the same order as `donut_results`
    imgs: list[npt.NDArray[np.float64]]
    model_imgs: list[npt.NDArray[np.float64] | None] | None
    fit_elapsed: float
    fit_nfev: int
    fit_cost: float
    fit_optimality: float
    fit_njev: int
    fit_status: int
    fit_message: str
    fit_error: str
    fit_outcome: FitOutcome

    @classmethod
    def empty(cls, group_id: str, n_zk: int) -> "WfGroupResult":
        """Result for a group with no donuts in it.

        Nothing was fit, so every fit field takes its no-fit value and
        ``fit_outcome`` is ``""`` -- the same spelling `_NULL_WF_DONUT` uses
        for "no fit consumed this".
        """
        return cls(
            group_id=group_id,
            group_size=0,
            zk_dev=np.full(n_zk, np.nan),
            success=False,
            donut_results=[],
            det_names=[],
            imgs=[],
            model_imgs=None,
            fit_elapsed=float("nan"),
            fit_nfev=0,
            fit_cost=float("nan"),
            fit_optimality=float("nan"),
            fit_njev=0,
            fit_status=_FIT_STATUS_ABSENT,
            fit_message="",
            fit_error="",
            fit_outcome="",
        )

    @classmethod
    def dead(cls, group: _WfGroup, reason: str, n_zk: int) -> "WfGroupResult":
        """Stand-in for a fit group whose worker was killed outright.

        No Python runs in a process the kernel has already destroyed, so the
        parent reports the group as failed on its behalf and the surviving
        groups' Zernikes still reach the output.

        The group's donuts are carried through even though none was fit: they
        still get catalog rows, and dropping them here would lose them
        silently.  They arrive as `_NULL_WF_DONUT` copies rather than the
        `Donut`s themselves, since that is what `_build_donut_catalog` reads --
        identity and intrinsics survive, and everything the lost fit would have
        measured is NaN.  ``fit_outcome`` is ``"exception"`` and not ``""``,
        which would claim no group ever took the donut.

        Parameters
        ----------
        group : `_WfGroup`
            The group that was lost.
        reason : `str`
            Cause, from `_fork_map`'s `_WorkerDeath`.
        n_zk : `int`
            Length of the Zernike vector, so the NaN row matches its siblings
            and the output table stays rectangular.
        """
        out = cls.empty(group.group_id, n_zk)
        out.group_size = len(group.donuts)
        out.donut_results = [
            replace(
                _NULL_WF_DONUT,
                donut_id=d.donut_id,
                det_name=d.det_name,
                visit_id=d.visit_id,
                zk_intrinsic=_dense_intrinsic(d),
                fit_outcome="exception",
                group_id=group.group_id,
                group_size=len(group.donuts),
            )
            for d in group.donuts
        ]
        out.det_names = [d.det_name for d in group.donuts]
        out.fit_error = f"worker died: {reason}"[:_ERROR_MAX_CHARS]
        return out


@dataclass
class FamDetectorResult:
    """One full-array detector, as `_fam_detector_worker` returns it.

    The envelope around the pair of `CutoutResult`s in ``results``: a
    full-array worker fuses cutting, pairing and fitting into one work unit, so
    everything the parent needs about that detector has to come back through
    here.  Corner mode has no counterpart -- its work unit ends at the
    `CutoutResult`, and the fields below it would carry are parent-side locals
    in `DonutBlitzCornerTask.run` instead.

    Every ``*_run`` field is seconds and per *detector*, and NaN means the
    worker never reached that phase.  ``refcat_run`` in particular is not per
    exposure: one refcat load covers both sides of focus, so it belongs here
    and not on the two ``results``, where summing it across the quantum would
    double-count every detector.

    A worker that failed reports it in ``error`` rather than raising, so that
    one bad detector out of 189 cannot take down the pool.  ``skipped``
    separates the expected no-work outcome -- a dead CCD, which ip_isr raises
    `NoWorkFound` for -- from a genuine fault; both leave ``error`` set, so
    test ``skipped`` first.
    """

    det_id: int
    # Empty until the worker has read a raw, so a detector that died in butler
    # I/O has an id but no name; the parent's log lines fall back to the id.
    det_name: str
    results: list[CutoutResult]  # the pair of `CutoutResult`s, intra first
    wf_results: list[WfGroupResult]  # one per fit group
    donuts: list[Donut]  # accepted `Donut`s, both sides of focus
    unmatched_donuts: list[Donut]  # `Donut`s with no partner on the other side
    error: str
    skipped: bool
    pid: int
    dispatch_to_arrival: float
    io_run: float
    refcat_run: float
    cutout_run: float
    fit_run: float
    worker_wall: float
    # Which pairing algorithm ran; stays "n/a" for a detector that failed or
    # was skipped before it got as far as grouping.
    pair_path: str = "n/a"

    @classmethod
    def started(cls, det_id: int, dispatch_to_arrival: float) -> "FamDetectorResult":
        """The result a worker starts from and fills in as it goes.

        Every phase timing is NaN and every list empty, so a worker that dies
        part way through reports exactly the phases it reached.  `pid` is the
        worker's own, which is what ties a log line to a process.
        """
        return cls(
            det_id=det_id,
            det_name="",
            results=[],
            wf_results=[],
            donuts=[],
            unmatched_donuts=[],
            error="",
            skipped=False,
            pid=os.getpid(),
            dispatch_to_arrival=dispatch_to_arrival,
            io_run=float("nan"),
            refcat_run=float("nan"),
            cutout_run=float("nan"),
            fit_run=float("nan"),
            worker_wall=float("nan"),
        )

    @classmethod
    def dead(cls, det_id: int, reason: str) -> "FamDetectorResult":
        """Stand-in for a detector whose worker was killed outright.

        `_fam_detector_worker` catches `BaseException` so that it always
        returns one of these, but that cannot cover a SIGKILL: no Python runs
        in a process the kernel has already destroyed.  The parent fills in the
        same shape so one dead detector costs one detector, not the other 188
        and the hours already spent on them.

        Recorded as an error rather than a skip: `skipped` means an expected
        no-work outcome such as a dead CCD, whereas a killed worker is a fault.

        Parameters
        ----------
        det_id : `int`
            Detector whose worker died.
        reason : `str`
            Cause, from `_fork_map`'s `_WorkerDeath`.
        """
        out = cls.started(det_id, float("nan"))
        out.error = f"worker died: {reason}"[:_ERROR_MAX_CHARS]
        # The parent is filling this in, so its own pid would be a lie; -1 says
        # no worker process owns this record.
        out.pid = -1
        return out
