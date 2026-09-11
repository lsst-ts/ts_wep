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

"""The full-array-mode worker: one detector, both exposures of a pair.

Unlike corner mode -- where `runQuantum` loads every pixel and the pool workers
only compute -- a FAM worker does its *own* butler I/O from deferred handles the
parent resolved, so peak memory scales with the worker count rather than with the
visit.  Cutting and fitting are fused into one worker for the same reason: the
pixels are freed as soon as the stamps are cut, instead of being held for the
duration of Danish fitting that follow.

Everything shared comes from `lsst.ts.wep.blitz.utils._CALIB_STORE`, which the
parent populates before forking; workers inherit it by copy-on-write.
"""

__all__ = []

import copy
import logging
import os
import time

import numpy as np

from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.pipe.base import NoWorkFound

from .cutoutPipeline import _cutout_one_exposure
from .dataStructures import _WfGroup
from .utils import (
    _CALIB_STORE,
    _INSTRUMENT,
    _defocal_radial_scale,
)

_log = logging.getLogger(__name__)

# Radians of field angle per un-binned pixel. Used to express the pairing
# tolerance -- naturally a fraction of the donut radius, i.e. pixels -- in the
# field-angle space the radial defocus correction works in.
_RAD_PER_PIXEL = _INSTRUMENT.pixelSize / _INSTRUMENT.focalLength


def _fam_pool_initializer() -> None:
    """Reset the inherited DB connection pool, once per child, right after fork.

    A forked child inherits the parent's live psycopg2 socket.  Under
    ``pipetask run`` the butler behind a deferred handle is a full `Butler`, and
    if two children use that inherited SSL connection concurrently the stream
    corrupts::

        psycopg2.OperationalError: SSL error: ssl/tls alert bad record mac

    It is a race, so it is intermittent.  The triggering query is a
    ``file_datastore_records`` lookup, which is to say a resolved ref is *not*
    self-contained and this is not avoidable by resolving earlier.

    `SqlRegistry.resetConnectionPool` exists for exactly this and is documented to
    be called by the child immediately after the fork; `mp.Pool(initializer=...)`
    is that hook.
    """
    reset = getattr(
        getattr(_CALIB_STORE.get("butler"), "registry", None),
        "resetConnectionPool",
        None,
    )
    if reset is not None:
        reset()


def _common_frame_angles(donuts: list) -> np.ndarray:
    """Field angles of ``donuts`` mapped back to their in-focus positions.

    Shifting an optic along z moves an off-axis chief ray radially, and in
    *opposite* directions either side of focus, so the same star sits at
    measurably different field angles in the intra and extra exposures -- 27.2 px
    apart at 1.725 deg for a 1.5 mm camera shift, and zero on axis.  A tolerance
    tight enough to be safe at the field center therefore fails at the edge, which
    presents as "the outer rafts just don't pair".

    `_defocal_radial_scale` removes it exactly: the displacement is linear in field
    angle, hence a pure scale, so one factor per offset triplet corrects the whole
    focal plane.  Dividing by it puts both sides in a common frame where the same
    star lands at the same place.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` array of ``(thx, thy)`` in radians. Empty ``(0, 2)`` for no
        donuts.
    """
    if not donuts:
        return np.empty((0, 2))
    scales = np.array([_defocal_radial_scale(d.defocal_offsets) for d in donuts])
    angles = np.array([(d.thx_ccs, d.thy_ccs) for d in donuts], dtype=float)
    return angles / scales[:, None]


def _pair_donuts(
    intra: list,
    extra: list,
    tol_frac: float,
    intra_source: str,
    extra_source: str,
) -> tuple[list, list, str]:
    """Match the same star's intra and extra donuts on one detector.

    Corner mode must pair by SNR rank because SW0 and SW1 see different sky.  FAM
    sees the *same star on the same detector twice*, so it's natural to match by
    star ID.

    Parameters
    ----------
    intra, extra : list of Donut
        Accepted donuts from the two exposures.  Each must already carry its
        ``defocal_offsets``.
    tol_frac : float
        Spatial match tolerance as a fraction of the donut radius.
    intra_source, extra_source : str
        ``selection_source`` from each exposure's cutout result.

    Returns
    -------
    pairs : list of tuple
        ``(extra_donut, intra_donut)`` -- extra first, matching the order corner
        mode's paired groups use.
    unmatched : list of Donut
        Donuts from either side with no partner.  These flow through to the
        catalog as candidate-but-unused rows, exactly as corner mode's surplus
        donuts do.
    path : str
        ``"refcat_id"``, ``"spatial"``, or ``"empty"``.  Recorded per detector so a
        run that quietly fell back to spatial matching is visible in the logs
        rather than only in the Zernikes.
    """
    if not intra or not extra:
        return [], list(intra) + list(extra), "empty"

    # Fast path: when both exposures selected from the reference catalog, the
    # refcat id identifies the same star exactly, so there is nothing to infer.
    if intra_source == "refcat" and extra_source == "refcat":
        intra_by_id = {d.donut_id: d for d in intra}
        pairs = []
        matched_ids = set()
        for e in extra:
            i = intra_by_id.get(e.donut_id)
            if i is not None:
                pairs.append((e, i))
                matched_ids.add(e.donut_id)
        unmatched = [e for e in extra if e.donut_id not in matched_ids]
        unmatched += [i for i in intra if i.donut_id not in matched_ids]
        return pairs, unmatched, "refcat_id"

    # Fallback: match spatially in the common (de-defocused) field-angle frame.
    # Never on the raw header WCS -- it can be off by >10 arcsec, far more than
    # the separation being measured, so it cannot be used for association.
    intra_ang = _common_frame_angles(intra)
    extra_ang = _common_frame_angles(extra)
    dist = np.hypot(
        extra_ang[:, None, 0] - intra_ang[None, :, 0],
        extra_ang[:, None, 1] - intra_ang[None, :, 1],
    )
    # Tolerance is a fraction of the donut radius; take the smaller radius of the
    # candidate pair so a mis-measured radius on one side cannot loosen the cut.
    tol = tol_frac * _RAD_PER_PIXEL * np.minimum(
        np.array([e.donut_radius for e in extra])[:, None],
        np.array([i.donut_radius for i in intra])[None, :],
    )

    # Mutual nearest neighbors: a pair is accepted only if each is the other's
    # closest candidate. One-sided nearest-neighbor matching would happily assign
    # two extra donuts to the same intra donut.
    best_for_extra = np.argmin(dist, axis=1)
    best_for_intra = np.argmin(dist, axis=0)
    pairs = []
    matched_extra, matched_intra = set(), set()
    for ie, ii in enumerate(best_for_extra):
        if best_for_intra[ii] == ie and dist[ie, ii] <= tol[ie, ii]:
            pairs.append((extra[ie], intra[ii]))
            matched_extra.add(ie)
            matched_intra.add(int(ii))
    unmatched = [e for ie, e in enumerate(extra) if ie not in matched_extra]
    unmatched += [i for ii, i in enumerate(intra) if ii not in matched_intra]
    return pairs, unmatched, "spatial"


def _fam_group_donuts(
    mode: str,
    det_name: str,
    intra: list,
    extra: list,
    tol_frac: float,
    intra_source: str,
    extra_source: str,
    band: str,
    rtp_deg: float | None,
    alt_rad: float | None,
) -> tuple[list, list, str]:
    """Build `_WfGroup` work units for one detector of a FAM pair.

    All grouping is within a single detector, so unlike corner mode's
    `_build_wf_groups` -- which reaches across the two detectors of a corner --
    this runs inside the worker and needs no cross-detector state.

    ``group_id`` carries ``det_name`` so it stays unique across the quantum's
    detectors.

    Modes
    -----
    ``paired``
        One star, both sides.  The only mode that pairs, and so the only one that
        can leave donuts unmatched.
    ``unpaired``
        One star, one side.
    ``full_detector``
        Every star on the detector, one side: two groups.
    ``full_detector_pair``
        Every star on the detector, both sides, in one joint fit.  Deliberately
        does *not* pair: each donut carries its own ``defocal_offsets``, so Danish
        already knows which side of focus it is on and no association is needed.

    Returns
    -------
    groups : list of _WfGroup
    unmatched : list of Donut
        Non-empty only for ``paired``.
    path : str
        Pairing path taken, or ``"n/a"`` for the modes that do not pair.
    """
    def _group(donuts, gid):
        return _WfGroup(
            donuts=donuts, group_id=gid, band=band, rtp=rtp_deg, alt=alt_rad
        )

    if mode == "paired":
        pairs, unmatched, path = _pair_donuts(
            intra, extra, tol_frac, intra_source, extra_source
        )
        groups = [
            _group([e, i], f"{det_name}_{e.donut_id}_{i.donut_id}") for e, i in pairs
        ]
        return groups, unmatched, path

    if mode == "unpaired":
        groups = [
            _group([d], f"{det_name}_{d.visit_id}_{d.donut_id}") for d in extra + intra
        ]
        return groups, [], "n/a"

    if mode == "full_detector":
        # Skip a side with no donuts: an empty group fits nothing but still
        # reports success=False, which would skew the caller's success tally.
        groups = [
            _group(side, f"{det_name}_{side[0].visit_id}")
            for side in (extra, intra)
            if side
        ]
        return groups, [], "n/a"

    if mode == "full_detector_pair":
        all_donuts = extra + intra
        return ([_group(all_donuts, det_name)] if all_donuts else []), [], "n/a"

    raise ValueError(f"Unknown FAM WF mode {mode!r}")


def _fam_detector_worker(args: tuple) -> dict:
    """Cut and fit one detector across both exposures of a FAM pair.

    Never raises, for any ``BaseException`` short of ``KeyboardInterrupt`` or
    ``SystemExit``: a failure is reported in the returned ``error`` field so one
    bad detector out of 189 cannot take down the pool and lose the rest of the
    visit. ``NoWorkFound`` -- which is a ``BaseException``, and which ip_isr
    raises for every dead CCD -- is reported as ``skipped`` instead.

    Parameters
    ----------
    args : tuple
        ``(det_id, t_dispatch)``.  Only the detector id crosses the pickle
        boundary; everything else is read from ``_CALIB_STORE``.

    Returns
    -------
    dict
        Keys ``det_id``, ``det_name``, ``results`` (the two per-exposure cutout
        dicts, each tagged with its ``visit_id``), ``wf_results``,
        ``donuts`` (accepted, both sides), ``unmatched_donuts``, ``pair_path``,
        ``error``, and timings ``dispatch_to_arrival``, ``io_run``,
        ``refcat_run``, ``cutout_run``, ``fit_run``, ``worker_wall``, plus
        ``pid``.

        ``refcat_run`` is per *detector*, not per exposure: one refcat load
        covers both sides of focus (see below).  It therefore belongs here and
        not on the two entries of ``results``, where summing it across the
        quantum would double-count every detector.
    """
    det_id, t_dispatch = args
    t_arrival = time.time()
    t_wall0 = time.perf_counter()
    out: dict = {
        "det_id": det_id,
        "det_name": "",
        "results": [],
        "wf_results": [],
        "donuts": [],
        "unmatched_donuts": [],
        # Overwritten once grouping runs; stays "n/a" for a detector that failed
        # or was skipped before it got that far.
        "pair_path": "n/a",
        "error": "",
        "skipped": False,
        "dispatch_to_arrival": t_arrival - t_dispatch,
        "io_run": float("nan"),
        "refcat_run": float("nan"),
        "cutout_run": float("nan"),
        "fit_run": float("nan"),
        "worker_wall": float("nan"),
        "pid": os.getpid(),
    }

    try:
        entry = _CALIB_STORE["detectors"][det_id]
        mode = _CALIB_STORE["wfEstimationMode"]
        tol_frac = _CALIB_STORE["pairMatchTolerance"]
        offsets_by_exp = _CALIB_STORE["offsets_by_exposure"]
        intra_exp = _CALIB_STORE["intra_exposure"]
        extra_exp = _CALIB_STORE["extra_exposure"]

        # --- butler I/O, all of it, in this child ---
        t0 = time.perf_counter()
        raws = {exp: h.get() for exp, h in entry["raws"].items()}
        iz_handle = entry.get("intrinsicZernikes")
        intrinsic_calib = iz_handle.get() if iz_handle is not None else None
        t1 = time.perf_counter()
        io_elapsed = t1 - t0

        det_name = next(iter(raws.values())).getDetector().getName()
        out["det_name"] = det_name

        # One refcat load covers both exposures: either WCS plus pixelMargin=300
        # spans the few-arcsecond difference between them, and the loader itself
        # intersects each shard's region with the search box, so handing it all of
        # the visit's shard handles still reads only the ~2 that overlap.
        t_refcat0 = time.perf_counter()
        load_result = None
        refcat_handles = _CALIB_STORE["refcat_handles"]
        if refcat_handles:
            ref_raw = raws[extra_exp]
            loader = ReferenceObjectLoader(
                dataIds=[h.dataId for h in refcat_handles],
                refCats=refcat_handles,
            )
            loader.config.pixelMargin = 300  # extra tolerance for uncertain WCS
            try:
                load_result = loader.loadPixelBox(
                    bbox=ref_raw.getBBox(),
                    wcs=ref_raw.getWcs(),
                    filterName=_CALIB_STORE["astromRefFilter"],
                    epoch=ref_raw.getInfo().getVisitInfo().date.toAstropy(),
                )
            except Exception as exc:
                _log.warning("Failed to load refcat for %s: %s", det_name, exc)
            # ref_raw is a live reference to one of the raws; drop it so the
            # per-exposure `del raws[exp]` below can actually free those pixels.
            del ref_raw, loader
        # Recorded even when no shards were supplied (then it is ~0), so the
        # key distinguishes "no refcat" from "the worker died before this
        # point", which stays NaN.
        out["refcat_run"] = time.perf_counter() - t_refcat0

        # --- cutouts, one call per exposure ---
        cutout_elapsed = 0.0
        results = []
        for exp in (intra_exp, extra_exp):
            # Calibrations are fetched per exposure rather than once per detector.
            # ISR is the only consumer and it is called twice here, where corner
            # mode calls it once per loaded calib, so a calib modified in place
            # would be invisible there and would corrupt the second exposure here.
            # Two 197 MB flat reads cost ~1 s against 10s of seconds of fitting,
            # and peak memory is no worse -- each is freed after its own ISR.
            t_io = time.perf_counter()
            calibs = {
                key: entry[key].get() if entry.get(key) is not None else None
                for key in ("ptc", "flat", "linearizer", "crosstalk")
            }
            io_elapsed += time.perf_counter() - t_io

            t_cut = time.perf_counter()
            # Each call also gets its own copy of the refcat load result:
            # AstrometryTask.solve may mutate it, and letting the two exposures
            # share one would cross-contaminate them silently.
            result = _cutout_one_exposure(
                raw=raws[exp],
                calibs=calibs,
                refcat_load_result=(
                    copy.deepcopy(load_result) if load_result is not None else None
                ),
                det_name=det_name,
                maxFitScatter=_CALIB_STORE["maxFitScatter"],
                astromRefFilter=_CALIB_STORE["astromRefFilter"],
                photoRefFilter=_CALIB_STORE["photoRefFilter"],
            )
            cutout_elapsed += time.perf_counter() - t_cut
            # The exposure id, which is also the visit id for these data -- the
            # parent checks that and warns if it ever stops being true.
            # Two cutout results share a det_name here, so this is also what
            # separates them in the catalog's per-detector metadata.
            result["visit_id"] = exp
            results.append(result)

            # Free this exposure's pixels before moving to the next one.
            del raws[exp], calibs

        t2 = time.perf_counter()
        out["io_run"] = io_elapsed
        out["cutout_run"] = cutout_elapsed
        out["results"] = results

        # Free the remaining pixels before fitting. This is what makes fusing cut
        # and fit into one worker safe: ~850 MB is transient during the I/O phase
        # rather than held for the duration of Danish fitting that follow.
        raws.clear()
        del raws, load_result

        # --- annotate every donut, accepted and rejected ---
        # Rejected donuts get catalog rows too, and _prep_donut_for_danish raises
        # on a donut with no defocal_offsets, so both lists must be annotated.
        # There is no intra/extra label to set: the side is the offsets, and is
        # recoverable from visit_id.
        by_exp = {}
        for result in results:
            exp = result["visit_id"]
            offsets = offsets_by_exp[exp]
            accepted = result["catalog"]
            for d in accepted + result.get("rejected_catalog", []):
                d.defocal_offsets = offsets
                if intrinsic_calib is not None:
                    d.intrinsic_zk = np.squeeze(
                        intrinsic_calib.getIntrinsicZernikes(
                            np.degrees(d.thx_ccs), np.degrees(d.thy_ccs)
                        )
                    )
                else:
                    d.intrinsic_zk = None
            by_exp[exp] = accepted

        # --- group and fit, in this process ---
        groups, unmatched, path = _fam_group_donuts(
            mode=mode,
            det_name=det_name,
            intra=by_exp[intra_exp],
            extra=by_exp[extra_exp],
            tol_frac=tol_frac,
            intra_source=results[0]["selection_source"],
            extra_source=results[1]["selection_source"],
            band=_CALIB_STORE["band"],
            rtp_deg=_CALIB_STORE["rtp_deg"],
            alt_rad=_CALIB_STORE["boresight_alt_rad"],
        )
        out["pair_path"] = path
        # Also stamp it on both of this detector's cutout results: those are what
        # reach build_donut_catalog, so this is what gets pairing provenance into
        # the persisted table instead of only the parent's log line.
        for r in results:
            r["pair_path"] = path
        out["unmatched_donuts"] = unmatched
        out["donuts"] = by_exp[intra_exp] + by_exp[extra_exp]

        wf_task = _CALIB_STORE["wf_fitting_task"]
        wf_results = []
        for group in groups:
            r = wf_task.run(group)
            wf_results.append(r)
        out["wf_results"] = wf_results
        out["fit_run"] = time.perf_counter() - t2

        _shed_images(out)
    except NoWorkFound as exc:
        # NoWorkFound derives from BaseException, not Exception, so it sails
        # straight past the guard below -- and past multiprocessing's own
        # `except Exception` in pool.worker -- killing the worker outright.
        # That is not merely a lost detector: the dying child's interpreter
        # shutdown takes the QBB datastore-records sqlite DB with it, so every
        # detector dispatched afterwards fails with "no such table:
        # file_datastore_records", and the task the child was holding is never
        # re-queued, so Pool.imap blocks forever. This is not hypothetical --
        # ip_isr raises UnprocessableDataError for any CCD whose back-side bias
        # voltage is off, and between one and six CCDs have been dead at every
        # point in the observatory's life. Catch it first, and by base class:
        # a dead CCD is an expected outcome, not an error.
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["skipped"] = True
        _log.info(
            "FAM worker skipping detector %s (%s): %s",
            det_id,
            out["det_name"] or "?",
            out["error"],
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 -- one detector must not kill the pool
        # BaseException rather than Exception for the same reason as above: a
        # worker that dies instead of returning hangs the whole quantum, so the
        # contract here is that this function always returns its dict.
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
        _log.warning(
            "FAM worker failed on detector %s (%s): %s",
            det_id,
            out["det_name"] or "?",
            out["error"],
        )
    finally:
        out["worker_wall"] = time.perf_counter() - t_wall0
    return out


def _shed_images(out: dict) -> None:
    """Drop image arrays the output catalog will not use, before pickling back.

    The images are the dominant contributors to the size of the output dictionary,
    so trimming them when possible can significantly improve performance.
    """
    if not _CALIB_STORE["saveWfImages"]:
        for r in out["wf_results"]:
            for wd in r.get("donuts", []):
                wd.img = None
                wd.model_img = None
            r["imgs"] = []
            r["model_imgs"] = None
    if not _CALIB_STORE["saveStamps"]:
        for result in out["results"]:
            for d in result["catalog"] + result.get("rejected_catalog", []):
                d.stamp = None
