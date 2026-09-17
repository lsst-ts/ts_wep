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

"""A fork-per-work-unit process pool, plus the hang watchdog that guards it.

The blitz tasks run their per-detector and per-group work in parallel
processes, and the failure they have to survive is a worker being *killed
outright*: an OOM kill by the enclosing cgroup (batch jobs run under a memory
limit and a whole focal plane of pixels is not far from it), a segfault out of
a compiled dependency, an external SIGKILL. One killed worker should cost
exactly its own detector or group; every sibling should run to completion and
the lost unit should be named, so the visit still produces a catalog with a
documented hole in it rather than nothing at all.

Neither pool in the standard library can do that:

- `multiprocessing.Pool` sends results through one shared pipe guarded by a
  shared lock (`SimpleQueue._rlock`). A worker killed while holding it leaves
  the survivors blocked forever in `Queue.get`, `_maintain_pool` forks
  replacements that block identically, and the parent waits in `map` for a
  result that will never arrive. Nothing exits non-zero, so HTCondor sees no
  violation and never evicts or retries -- the job silently burns its walltime.
  This is what hung the BLOCK-T614 bulge visits.
- `ProcessPoolExecutor` does detect the death (~0.3 s), but recovers by killing
  every remaining worker and failing all pending futures. Measured on 8 units /
  4 workers with one killed: only the units that had *already* finished
  survived. Since the cutout pool runs 8 detectors concurrently in ~1.6 s, that
  is nothing -- one dead detector would lose the whole visit.

`_forkMap` gives each work unit its own anonymous pipe. That is the same
transport `Pool` already uses, minus the shared lock, so a dying child can only
ever truncate its own stream, and a length-framed payload makes a truncated
stream unambiguous. Nothing is written to disk.

A worker can also fail by never finishing, which needs its own bound: passing
`unitTimeout` gives each unit a deadline measured from its own fork, so a
pathologically slow unit -- or one whose pipe never reaches EOF because `func`
leaked its write end to a grandchild -- is killed and named like any other
loss rather than parking the drain loop. Callers with no bound on legitimate
runtime leave it unset and wait.

Forking is also load-bearing for memory here, not merely convenient: the
calibration products, telescope models and reference catalogs the workers need
are built once in the parent and inherited copy-on-write, which neither a
spawn-based pool nor a thread pool (the GIL aside) would give.

`_dumpStacksOnHang` remains as a backstop for hangs that come from somewhere
other than the pool, and turns the next one from something that has to be
caught live with `py-spy` into something the job log already explains.
"""

__all__ = []

import contextlib
import faulthandler
import glob
import os
import pickle
import selectors
import signal
import struct
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import Any, NamedTuple


class _WorkerDeath(NamedTuple):
    """One work unit whose worker died without returning a result.

    Attributes
    ----------
    unit : `Any`
        The element of ``args`` that was lost, so the caller can name the
        detector (or group) rather than guess at it.
    reason : `str`
        Human-readable cause, e.g. ``"killed by SIGKILL (typically the cgroup
        OOM killer)"``.
    """

    unit: Any
    reason: str


# Each child frames its pickled result with this length header, so the parent
# can tell a complete payload from one truncated by a mid-write kill.
_RESULT_HEADER = struct.Struct("!Q")
_READ_CHUNK = 1 << 16
# Only bounds how long the drain loop blocks before re-checking whether it can
# start another unit; a finished or dead child always shows up as pipe EOF.
_SELECT_TIMEOUT = 1.0


def _forkMap(
    func: Callable,
    args: Iterable,
    numWorkers: int,
    *,
    initializer: Callable | None = None,
    unitTimeout: float | None = None,
) -> tuple[list, list[_WorkerDeath]]:
    """Map `func` over `args` in forked workers, surviving a worker that dies.

    One fork per work unit, at most `numWorkers` alive at a time, each
    returning its result over its own anonymous pipe. A worker killed outright
    -- an OOM kill by the enclosing cgroup, a segfault, an external SIGKILL --
    costs exactly its own unit: every sibling runs to completion and the lost
    unit is named in the returned `_WorkerDeath` list. See the module docstring
    for why neither standard-library pool can do that.

    With `unitTimeout`, a unit that *overruns* costs its own unit too, on the
    same terms. That is a separate failure from being killed and needs its own
    deadline: a worker merely running long is indistinguishable from one making
    progress, so nothing in the loop would ever notice it.

    Parameters
    ----------
    func : `Callable`
        Called with one element of `args` per work unit. Its return value must
        be picklable.
    args : `Iterable`
        Work units. Materialized into a list, so a generator is fine.
    numWorkers : `int`
        Maximum number of workers alive at once. Clamped to ``[1, len(args)]``.
        Peak worker-side memory follows this rather than the number of units,
        which is the property `Pool.imap_unordered(chunksize=1)` was used for.
    initializer : `Callable`, optional
        Run in each worker immediately after the fork, before `func`. Mandatory
        for workers that touch the butler; see `_fam_pool_initializer`.
    unitTimeout : `float`, optional
        Seconds a single unit may run, measured from its own fork, before its
        worker is SIGKILLed and the unit recorded as a death. `None` (the
        default) waits indefinitely, which is what a caller with no bound on
        legitimate runtime wants.

        This bounds one *unit*, not the call: units run in waves of
        `numWorkers`, so the whole map can take up to
        ``ceil(len(args) / numWorkers) * unitTimeout``. A caller that also arms
        `_dumpStacksOnHang` needs that product to sit below the watchdog's
        timeout, or the watchdog still aborts the process first and the
        per-unit degradation never gets to happen.

    Returns
    -------
    results : `list`
        Results of the units that returned one, in the relative order of
        `args` with the dead units omitted -- no placeholder is left behind,
        so ``results[i]`` stops corresponding to ``args[i]`` from the first
        death onwards. Callers identify a result from its own contents (the
        detector or group it names), not from its position.
    deaths : `list` [`_WorkerDeath`]
        One entry per unit whose worker died or overran `unitTimeout`. Empty on
        a clean run. Callers are expected to fold these into whatever they
        already do with a failed unit rather than to raise.
    """
    argList = list(args)
    if not argList:
        return [], []
    numWorkers = max(1, min(numWorkers, len(argList)))

    buffers: dict[int, bytearray] = {}
    pidToIndex: dict[int, int] = {}
    fdToIndex: dict[int, int] = {}
    # Only used by the deadline sweep: when each unit was forked, how to signal
    # it, and which units the sweep has already given up on.
    startedAt: dict[int, float] = {}
    indexToPid: dict[int, int] = {}
    timedOut: set[int] = set()
    selector = selectors.DefaultSelector()
    nextIndex = 0
    openPipes = 0

    def spawn(index: int) -> None:
        nonlocal openPipes
        readFd, writeFd = os.pipe()
        pid = os.fork()
        if pid == 0:
            # ---------------- child ----------------
            # Exit via os._exit throughout: the parent's `finally` blocks,
            # atexit handlers and buffered streams are not ours to run.
            try:
                os.close(readFd)
                # Siblings' read ends come across the fork; close them so a
                # child does not accumulate up to numWorkers-1 stale
                # descriptors. Hygiene only -- a pipe reaches EOF when the last
                # *write* end closes, so a held read end delays nothing
                # (verified). The invariant that does matter is below in the
                # parent: writeFd is closed before the next fork, so no child
                # ever inherits a sibling's write end and no sibling's pipe can
                # be held open by an unrelated child.
                #
                # The same reasoning binds anything `func` itself starts: EOF
                # arrives only when the *last* copy of writeFd closes, so a
                # grandchild (a fork or a subprocess outliving the worker)
                # would inherit it and park the parent in `select` forever.
                # `func` must not leave one behind; `_dumpStacksOnHang` is the
                # only thing that would catch it if it did.
                for inherited in fdToIndex:
                    os.close(inherited)
                selector.close()
                if initializer is not None:
                    initializer()
                payload = pickle.dumps(func(argList[index]), protocol=pickle.HIGHEST_PROTOCOL)
                os.write(writeFd, _RESULT_HEADER.pack(len(payload)))
                sent = 0
                while sent < len(payload):
                    sent += os.write(writeFd, payload[sent:])
                os.close(writeFd)
            except BaseException:
                # The workers each mean to return their own errors as data;
                # reaching here means one did not, so report it and let the
                # parent record a death for this unit alone.
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        # ---------------- parent ----------------
        # Before the next fork, so no child inherits this write end: EOF on
        # this pipe then means *this* child and nothing else. See child's note.
        os.close(writeFd)
        pidToIndex[pid] = index
        fdToIndex[readFd] = index
        buffers[index] = bytearray()
        # Each unit's clock starts at its own fork, not at the start of the
        # map, so a unit that waited for a free slot is not charged for it.
        startedAt[index] = time.monotonic()
        indexToPid[index] = pid
        selector.register(readFd, selectors.EVENT_READ)
        openPipes += 1

    while nextIndex < len(argList) or openPipes:
        while openPipes < numWorkers and nextIndex < len(argList):
            spawn(nextIndex)
            nextIndex += 1
        for key, _ in selector.select(timeout=_SELECT_TIMEOUT):
            index = fdToIndex[key.fd]
            try:
                chunk = os.read(key.fd, _READ_CHUNK)
            except OSError:
                chunk = b""
            if chunk:
                buffers[index] += chunk
            else:
                # EOF: this child has finished writing or has died. Either way
                # its slot is free, so the next unit can start.
                selector.unregister(key.fd)
                os.close(key.fd)
                del fdToIndex[key.fd]
                openPipes -= 1
        if unitTimeout is not None:
            # `_SELECT_TIMEOUT` bounds the block above, so this runs at least
            # once a second without any timer of its own -- which is also the
            # granularity of the deadline.
            now = time.monotonic()
            for fd in list(fdToIndex):
                index = fdToIndex[fd]
                if now - startedAt[index] < unitTimeout:
                    continue
                # Kill, then drop the read end here rather than waiting for the
                # EOF the kill ought to produce. It need not: a pipe reaches
                # EOF only when the *last* write end closes, so a grandchild
                # that inherited this one (the hazard the child branch above
                # warns `func` against) keeps it open past the worker's death
                # and would park this loop forever. Termination must not depend
                # on who holds the write end.
                timedOut.add(index)
                try:
                    os.kill(indexToPid[index], signal.SIGKILL)
                except OSError:
                    pass  # already exited; still reaped by the waits below
                selector.unregister(fd)
                os.close(fd)
                del fdToIndex[fd]
                openPipes -= 1
    selector.close()

    # Every pipe is closed, so every child is at `_exit`, already gone, or
    # SIGKILLed by the sweep; these waits do not block for meaningful time.
    # The two closure paths justify that differently: EOF implies the child is
    # done with its write end, while a force-closed pipe implies only that we
    # killed the child ourselves -- equally sufficient, but it stops being so
    # if anything ever closes a pipe for a third reason.
    statuses: dict[int, int] = {}
    for pid, index in pidToIndex.items():
        try:
            statuses[index] = os.waitpid(pid, 0)[1]
        except ChildProcessError:
            statuses[index] = 0

    results: list = []
    deaths: list[_WorkerDeath] = []
    for index, unit in enumerate(argList):
        payload = _decodeResult(buffers[index])
        if payload is not _INCOMPLETE:
            # A complete payload counts even if the child was killed straight
            # after writing it -- the work was done.
            results.append(payload)
            continue
        deaths.append(
            _WorkerDeath(
                unit,
                _describeExit(
                    statuses[index],
                    len(buffers[index]),
                    timeout=unitTimeout if index in timedOut else None,
                ),
            )
        )
    return results, deaths


# Distinguishes "no usable payload" from a worker that legitimately returned
# None, which `None` alone could not.
_INCOMPLETE = object()


def _decodeResult(buffer: bytearray) -> Any:
    """Unpickle one framed result, or `_INCOMPLETE` if it is not all there."""
    if len(buffer) < _RESULT_HEADER.size:
        return _INCOMPLETE
    (declared,) = _RESULT_HEADER.unpack_from(buffer)
    body = bytes(buffer[_RESULT_HEADER.size :])
    if len(body) != declared:
        return _INCOMPLETE
    try:
        return pickle.loads(body)
    except Exception:  # noqa: BLE001 - corrupt payload = death, not a crash
        return _INCOMPLETE


def _describeExit(status: int, received: int, timeout: float | None = None) -> str:
    """Why a worker produced no usable result, in reviewable English.

    `timeout` is the deadline the unit overran, when that is what ended it. It
    takes precedence over `status`, which would otherwise report the sweep's
    own SIGKILL as an OOM kill and send the reader after a memory problem that
    was never involved.
    """
    if timeout is not None:
        # `:g` rather than a fixed precision: sub-second deadlines are
        # legitimate (tests use them) and would otherwise render as "0s".
        detail = f"exceeded the {timeout:g}s unit timeout and was killed"
    elif os.WIFSIGNALED(status):
        signum = os.WTERMSIG(status)
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        extra = " (typically the cgroup OOM killer)" if signum == signal.SIGKILL else ""
        detail = f"killed by {name}{extra}"
    elif os.WIFEXITED(status) and os.WEXITSTATUS(status):
        detail = f"exited {os.WEXITSTATUS(status)}"
    else:
        detail = "exited without writing a complete result"
    if received:
        detail += f"; {received} byte(s) of a partial result discarded"
    return detail


def _killChildProcesses() -> None:
    """SIGKILL every direct child of this process. Best effort.

    Called just before `os._exit`, which runs no `finally`, no `atexit` and no
    executor shutdown -- so without this the forked workers survive as orphans,
    holding the job's stdout pipe open (which stalls whatever is reading the
    log) and still counted against the enclosing cgroup.

    Children are read from ``/proc`` -- the kernel's own view at kill time --
    rather than from `multiprocessing.active_children`, which reports Python
    bookkeeping that a fork in progress can lag, and covers only children
    `multiprocessing` itself started.
    """
    pids: set[int] = set()
    # One `children` file per thread, since any thread may have forked.
    for path in glob.glob(f"/proc/{os.getpid()}/task/*/children"):
        try:
            with open(path) as handle:
                pids.update(int(pid) for pid in handle.read().split())
        except (OSError, ValueError):
            continue
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # already gone, or not ours to kill


@contextlib.contextmanager
def _dumpStacksOnHang(timeout: float | None, label: str, log: Any = None) -> Iterator[None]:
    """Dump every thread's stack and exit if the block outlives `timeout`.

    A backstop for hangs this process cannot otherwise detect, and a way to
    make the next one cheap to diagnose: without it a wedged pool looks
    identical to slow work and has to be caught live with `py-spy` or `gdb`.
    Only the calling process's threads are dumped -- enough to see the parent
    parked in `pool.map`/`Queue.get` -- not the workers'.

    `timeout` should sit well above the worst legitimate runtime, since
    tripping it aborts the job. Pass `None` to disable.

    Parameters
    ----------
    timeout : `float` or `None`
        Seconds before the traceback dump and `_exit(1)`. `None` disables.
    label : `str`
        Named in the log line so the dump can be tied back to a call site.
    log : optional
        Logger for the "armed" message. Nothing is logged when omitted.
    """
    if timeout is None or timeout <= 0:
        yield
        return
    if log is not None:
        log.debug("Hang watchdog armed for %s: %.0fs", label, timeout)
    done = threading.Event()

    def watch() -> None:
        if done.wait(timeout):
            return
        print(
            f"Hang watchdog: {label} exceeded {timeout:.0f}s; dumping stacks",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        _killChildProcesses()
        sys.stderr.flush()
        # Aborting is the only option left: the main thread is blocked in C on
        # a futex, where no exception can be delivered to it.
        os._exit(1)

    threading.Thread(target=watch, name=f"hang-watchdog[{label}]", daemon=True).start()
    try:
        yield
    finally:
        done.set()
