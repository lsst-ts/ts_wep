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

"""The crash isolation `_forkMap` exists to provide.

`_forkMap` is only worth its custom implementation if a worker killed outright
costs exactly its own work unit, so that is what these tests kill for: a child
SIGKILLed before it writes anything, and -- the case the length-framed payload
was designed for -- a child killed *between* its frame header and the end of
its payload, which is the one truncation an unframed stream could not tell
apart from a complete result. Both are forced deterministically by patching
`os.write` in the parent and letting the fork inherit the patch, rather than by
racing a sleep against a signal.

The complement matters just as much and is easy to lose in a refactor: a worker
that returns `None`, or that is killed *after* writing a complete payload, must
be counted as a success rather than swept into `deaths`.

`TestUnitTimeout` covers the other way a unit can fail to deliver: not dying
but overrunning, either because the work itself is pathologically slow or
because the unit's pipe never reaches EOF. The second is not hypothetical --
a worker that leaks its write end to a grandchild parks the pre-timeout
`_forkMap` forever, which is why those tests are bounded by `_failIfSlower`
rather than trusted to return.

`TestSignalsDuringWrite` pins the reason the child's write loop needs no
EINTR-retry helper: PEP 475 makes `os.write` retry on `EINTR` itself, and the
loop already absorbs the short write a signal can otherwise leave behind. That
is a property of the interpreter rather than of this module, so it is asserted
here to keep anyone from "fixing" the loop back into complexity.

Not covered: `_killChildProcesses`, which would SIGKILL every child of the
process running the tests (including the test runner's own), and the firing
path of `_dumpStacksOnHang`, which ends in `os._exit`.
"""

import os
import pickle
import signal
import threading
import time
import unittest
import unittest.mock
from contextlib import contextmanager

from lsst.ts.wep.blitz.forkPool import (
    _INCOMPLETE,
    _RESULT_HEADER,
    _decodeResult,
    _describeExit,
    _dumpStacksOnHang,
    _forkMap,
)

# Set by `_recordInitializer` in the child only; the parent's copy stays empty,
# which is how the tests show the initializer ran after the fork rather than
# before it.
_INITIALIZER_CALLS: list[int] = []


def _double(unit):
    return unit * 2


def _identity(unit):
    return unit


def _returnNone(unit):
    return None


def _suicide(unit):
    """Die before `_forkMap` has anything to write for this unit."""
    os.kill(os.getpid(), signal.SIGKILL)


def _dieIfThree(unit):
    if unit == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return unit * 10


def _raiseValueError(unit):
    raise ValueError("worker blew up")


def _raiseIfOne(unit):
    if unit == 1:
        raise ValueError("worker blew up")
    return unit


def _raiseBaseException(unit):
    # Not an `Exception`: the child's guard is written against `BaseException`
    # because the pipeline's own control-flow exceptions (`NoWorkFound` and
    # friends) sit outside `Exception`.
    raise SystemExit(7)


def _returnUnpicklable(unit):
    def notPicklable():
        pass

    return notPicklable


def _recordInitializer():
    _INITIALIZER_CALLS.append(os.getpid())


def _reportInitializerCalls(unit):
    return (unit, len(_INITIALIZER_CALLS))


def _timeInterval(unit):
    """Occupy a worker slot for a measurable span, and report when."""
    start = time.monotonic()
    time.sleep(0.15)
    return (unit, start, time.monotonic())


# Far past any deadline these tests set, so the timeout is what ends the unit
# rather than the work finishing on its own. Kept short all the same: a test
# cut short by `_failIfSlower` cannot reap what is still sleeping, so this is
# also how long a stray can outlive a failure.
_LONG_SLEEP = 6.0


def _sleepPastAnyDeadline(unit):
    """Alive, healthy and far too slow -- the hypothesis (a) failure."""
    time.sleep(_LONG_SLEEP)
    return unit


def _sleepIfZero(unit):
    if unit == 0:
        time.sleep(_LONG_SLEEP)
    return unit * 10


def _leakGrandchildHoldingWriteEnd(unit):
    """Do the work, write it in full, then leave the pipe held open anyway.

    The hypothesis (b) failure, and the reason a timeout may not wait for EOF
    after the `SIGKILL`. A pipe reaches EOF when the last *write* end closes,
    so a grandchild that outlives the worker keeps this unit's pipe open
    forever even though the worker exited 0 with a complete payload already
    buffered in the parent. That is precisely what forkPool.py's fd-hygiene
    comments warn `func` must not do -- this is the warning, executed.
    """
    if os.fork() == 0:
        # Grandchild. Never returns into `func`, and exits via `os._exit` so it
        # runs none of the test runner's teardown.
        time.sleep(_LONG_SLEEP)
        os._exit(0)
    return unit * 2


def _bigPayload(unit):
    # Comfortably more than both the 64 KiB pipe capacity and `_READ_CHUNK`, so
    # the parent has to accumulate the result over many reads while the child
    # blocks in `os.write` waiting for it.
    return b"z" * (4 << 20)


@contextmanager
def _silencedStderr():
    """Redirect the real fd 2 for the duration, forked children included.

    The workers that fail on purpose print a traceback by design. Patching
    `sys.stderr` would not reach them -- the child writes through the inherited
    descriptor -- so the descriptor itself is what has to be redirected.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


@contextmanager
def _killDuringWrite(afterBytes: int | None):
    """Patch `os.write` so a child dies mid-result, at a chosen point.

    The patch is installed in the parent and inherited through the fork, which
    makes the kill deterministic: no sleep is raced against a signal. The frame
    header is always let through -- it is the payload write that is cut short
    -- so `afterBytes` counts payload bytes only. `None` means write the
    payload in full and *then* die, which is the "killed after finishing" case.

    Only the children call `os.write` inside `_forkMap`; the parent reads. So
    the patch is inert in the process that installs it.
    """
    realWrite = os.write

    def fakeWrite(fd, data):
        if len(data) == _RESULT_HEADER.size:
            return realWrite(fd, data)
        written = realWrite(fd, data if afterBytes is None else data[:afterBytes])
        os.kill(os.getpid(), signal.SIGKILL)
        return written  # unreachable; SIGKILL is not catchable

    with unittest.mock.patch.object(os, "write", fakeWrite):
        yield


def _reapStrays() -> None:
    """Reap whatever `_forkMap` never got to, when a test cut it short.

    Only the already-exited: anything still sleeping is left to finish and be
    reaped by init, which is the price of interrupting the loop mid-flight.
    """
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


@contextmanager
def _failIfSlower(seconds: float):
    """Turn "this call never returns" into a failure instead of a wedged suite.

    A `SIGALRM` handler that raises is enough here, and that is itself worth
    pinning: the parent sits in `selector.select(timeout=_SELECT_TIMEOUT)`, so
    it wakes at least once a second and a Python-level exception can be
    delivered to it. (PEP 475 retries an `EINTR`-interrupted `select` only when
    the handler does *not* raise.) The claim in `_dumpStacksOnHang` that "the
    main thread is blocked in C on a futex, where no exception can be delivered
    to it" does not hold for this call path -- if it did, this guard could not
    work and these tests would hang the runner instead of failing it.

    `fork` resets interval timers in the child, so no worker inherits the
    alarm.
    """

    def raiseTimeout(signum, frame):
        raise TimeoutError(f"_forkMap did not return within {seconds}s")

    previous = signal.signal(signal.SIGALRM, raiseTimeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        _reapStrays()


def _maxConcurrent(intervals) -> int:
    """Peak overlap among ``(start, end)`` spans."""
    edges = [(start, 1) for _, start, _ in intervals]
    edges += [(end, -1) for _, _, end in intervals]
    peak = live = 0
    for _, delta in sorted(edges):
        live += delta
        peak = max(peak, live)
    return peak


class TestCleanRuns(unittest.TestCase):
    """The unexceptional paths, since the failure paths build on them."""

    def testMapsEveryUnit(self) -> None:
        results, deaths = _forkMap(_double, [1, 2, 3, 4], 2)
        self.assertEqual(deaths, [])
        self.assertEqual(results, [2, 4, 6, 8])

    def testEmptyArgs(self) -> None:
        self.assertEqual(_forkMap(_double, [], 4), ([], []))

    def testGeneratorArgs(self) -> None:
        results, deaths = _forkMap(_double, (i for i in range(3)), 2)
        self.assertEqual(deaths, [])
        self.assertEqual(results, [0, 2, 4])

    def testNoneIsAResultNotADeath(self) -> None:
        """The reason `_INCOMPLETE` exists: `None` alone could not say this."""
        results, deaths = _forkMap(_returnNone, [1, 2], 2)
        self.assertEqual(deaths, [])
        self.assertEqual(results, [None, None])

    def testPayloadLargerThanThePipeBuffer(self) -> None:
        """A result that cannot fit in the pipe at once still arrives whole."""
        (result,), deaths = _forkMap(_bigPayload, [0], 1)
        self.assertEqual(deaths, [])
        self.assertEqual(len(result), 4 << 20)
        self.assertEqual(result, b"z" * (4 << 20))

    def testInitializerRunsInEachChildAndNotInTheParent(self) -> None:
        results, deaths = _forkMap(_reportInitializerCalls, [0, 1, 2], 2, initializer=_recordInitializer)
        self.assertEqual(deaths, [])
        # Exactly once per child, and no leakage between them: unit 2 forks
        # after units 0 and 1 have run, and still sees a count of 1.
        self.assertEqual(sorted(results), [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(_INITIALIZER_CALLS, [])


class TestWorkerLimit(unittest.TestCase):
    """`numWorkers` is what bounds peak memory, so it has to bind."""

    def testAtMostNumWorkersAlive(self) -> None:
        results, deaths = _forkMap(_timeInterval, list(range(6)), 2)
        self.assertEqual(deaths, [])
        self.assertEqual(len(results), 6)
        self.assertEqual(_maxConcurrent(results), 2)

    def testClampedUpFromZero(self) -> None:
        results, deaths = _forkMap(_timeInterval, [0, 1], 0)
        self.assertEqual(deaths, [])
        self.assertEqual(_maxConcurrent(results), 1)

    def testClampedDownToUnitCount(self) -> None:
        results, deaths = _forkMap(_timeInterval, [0, 1], 32)
        self.assertEqual(deaths, [])
        self.assertEqual(_maxConcurrent(results), 2)


class TestKilledWorkers(unittest.TestCase):
    """One SIGKILL should cost exactly one unit."""

    def testKilledBeforeWritingAnything(self) -> None:
        results, deaths = _forkMap(_suicide, ["R01_S00"], 1)
        self.assertEqual(results, [])
        self.assertEqual(len(deaths), 1)
        self.assertEqual(deaths[0].unit, "R01_S00")
        self.assertIn("SIGKILL", deaths[0].reason)
        self.assertIn("OOM killer", deaths[0].reason)
        # Nothing arrived, so there is no partial payload to report.
        self.assertNotIn("partial result", deaths[0].reason)

    def testSiblingsSurviveAndTheDeadUnitIsNamed(self) -> None:
        """The whole point: a dead detector costs its own detector only."""
        units = list(range(8))
        results, deaths = _forkMap(_dieIfThree, units, 4)
        self.assertEqual([d.unit for d in deaths], [3])
        # Relative order of `args`, with the dead unit simply absent -- there
        # is no placeholder, which is why callers must not index `results` by
        # unit.
        self.assertEqual(results, [0, 10, 20, 40, 50, 60, 70])

    def testKilledMidPayloadIsADeathNotACorruptResult(self) -> None:
        """The case the length header is for: a truncated stream.

        Without framing, the parent would hand a half-written pickle to
        `pickle.loads` and take whatever came of that; with it, the shortfall
        against the declared length names the unit as lost.
        """
        with _killDuringWrite(afterBytes=64):
            results, deaths = _forkMap(_bigPayload, ["R22_S11"], 1)
        self.assertEqual(results, [])
        self.assertEqual(len(deaths), 1)
        self.assertEqual(deaths[0].unit, "R22_S11")
        self.assertIn("SIGKILL", deaths[0].reason)
        # The partial bytes are accounted for rather than silently dropped. The
        # count is of everything buffered, frame header included, so it is 8
        # more than the payload bytes that got through.
        self.assertIn(f"{_RESULT_HEADER.size + 64} byte(s) of a partial result", deaths[0].reason)

    def testKilledMidPayloadCostsOnlyItsOwnUnit(self) -> None:
        """A truncated stream must not desynchronize its siblings' pipes.

        Every unit here is truncated because the patch is global to the fork,
        which is the strongest version of the claim: N independent pipes means
        N independent failures, never one poisoned transport.
        """
        with _killDuringWrite(afterBytes=16):
            results, deaths = _forkMap(_bigPayload, list(range(4)), 2)
        self.assertEqual(results, [])
        self.assertEqual([d.unit for d in deaths], [0, 1, 2, 3])
        for death in deaths:
            self.assertIn(f"{_RESULT_HEADER.size + 16} byte(s) of a partial result", death.reason)

    def testKilledAfterWritingACompleteResultStillCounts(self) -> None:
        """The work was done; the exit status afterwards is not interesting."""
        with _killDuringWrite(afterBytes=None):
            results, deaths = _forkMap(_identity, ["done"], 1)
        self.assertEqual(deaths, [])
        self.assertEqual(results, ["done"])


class TestFailingWorkers(unittest.TestCase):
    """Deaths that are not signals: the child's own guard, reported as data."""

    def testExceptionInFunc(self) -> None:
        with _silencedStderr():
            results, deaths = _forkMap(_raiseValueError, ["R01_S01"], 1)
        self.assertEqual(results, [])
        self.assertEqual(deaths[0].unit, "R01_S01")
        self.assertEqual(deaths[0].reason, "exited 1")

    def testBaseExceptionInFunc(self) -> None:
        """`except Exception` in the child would let this one escape."""
        with _silencedStderr():
            results, deaths = _forkMap(_raiseBaseException, [0], 1)
        self.assertEqual(results, [])
        self.assertEqual(deaths[0].reason, "exited 1")

    def testUnpicklableResult(self) -> None:
        """A result that cannot cross the pipe is a death, not a crash."""
        with _silencedStderr():
            results, deaths = _forkMap(_returnUnpicklable, [0], 1)
        self.assertEqual(results, [])
        self.assertEqual(deaths[0].reason, "exited 1")

    def testOneFailureAmongSurvivors(self) -> None:
        with _silencedStderr():
            results, deaths = _forkMap(_raiseIfOne, [0, 1, 2], 2)
        self.assertEqual(results, [0, 2])
        self.assertEqual([d.unit for d in deaths], [1])


class TestUnitTimeout(unittest.TestCase):
    """A unit that overruns should cost its own unit, like a killed one.

    Without a per-unit deadline the only thing that notices an overrun is
    `_dumpStacksOnHang`, which fires from a side thread that cannot know which
    unit is late and so can only `os._exit(1)` -- taking every healthy sibling,
    and the whole quantum, with it. These tests say the pool degrades per unit
    instead, for both shapes of overrun: a worker that is merely far too slow,
    and one whose pipe never reaches EOF at all.

    Every deadline here is at least 1s because `_SELECT_TIMEOUT` is 1.0: the
    sweep can only run when `select` returns, so a sub-second deadline buys no
    sharpness and only makes the timings look tighter than they are.
    """

    def testSlowUnitBecomesADeathNotAHang(self) -> None:
        with _failIfSlower(20.0):
            results, deaths = _forkMap(_sleepPastAnyDeadline, ["R40_SW1"], 1, unitTimeout=1.0)
        self.assertEqual(results, [])
        self.assertEqual(len(deaths), 1)
        # The diagnostic the watchdog's thread dump could never give: the name
        # of the late detector.
        self.assertEqual(deaths[0].unit, "R40_SW1")
        self.assertIn("unit timeout", deaths[0].reason)

    def testTimeoutIsNotReportedAsAnOomKill(self) -> None:
        """A timed-out worker is `SIGKILL`ed, but it is not an OOM kill.

        Reading the plain `WIFSIGNALED` text here would send the next person
        chasing a memory limit that was never involved.
        """
        with _failIfSlower(20.0):
            _, deaths = _forkMap(_sleepPastAnyDeadline, [0], 1, unitTimeout=1.0)
        self.assertNotIn("OOM", deaths[0].reason)

    def testSlowUnitDoesNotDelayOrDisplaceItsSiblings(self) -> None:
        """The one that matters in production: 1 late detector, 7 delivered.

        Also that the timed-out slot is genuinely freed -- with 4 units and 2
        workers, units 2 and 3 only run if the deadline hands back unit 0's
        slot rather than merely marking it dead.
        """
        with _failIfSlower(20.0):
            results, deaths = _forkMap(_sleepIfZero, list(range(4)), 2, unitTimeout=1.0)
        self.assertEqual([d.unit for d in deaths], [0])
        self.assertEqual(results, [10, 20, 30])

    def testGrandchildHoldingTheWriteEndCannotParkTheLoop(self) -> None:
        """Termination must not depend on who holds the write end.

        Verified to hang forever against the pre-timeout implementation, which
        is the failure this whole change exists to bound. Note what the right
        answer is: the payload arrived *complete* before the deadline, so this
        is a result and not a death. A timeout that simply declared the unit
        dead on the way out would regress the existing "the work was done"
        semantics.
        """
        with _failIfSlower(20.0):
            results, deaths = _forkMap(_leakGrandchildHoldingWriteEnd, [21], 1, unitTimeout=1.0)
        self.assertEqual(deaths, [])
        self.assertEqual(results, [42])

    def testNoDeadlineWithoutOne(self) -> None:
        """`_forkMap` stays a pure mechanism for the callers that opt out.

        The wavefront and FAM pools pass no `unitTimeout`, so a slow unit there
        must still be waited for rather than acquiring a deadline by default.
        """
        results, deaths = _forkMap(_timeInterval, [0], 1)
        self.assertEqual(deaths, [])
        self.assertEqual(len(results), 1)

    def testDeadlineIsPerUnitNotPerPool(self) -> None:
        """Each unit gets the full deadline, measured from its own spawn.

        Units 2 and 3 start only as earlier slots free up, so a deadline
        measured from the start of the *pool* would kill them for their
        predecessors' runtime. This is also why the pool as a whole can outlive
        `unitTimeout` by a factor of the number of waves -- what has to stay
        below `hangTimeout` is that product, not `unitTimeout` itself.
        """
        with _failIfSlower(20.0):
            results, deaths = _forkMap(_timeInterval, list(range(4)), 2, unitTimeout=1.0)
        self.assertEqual(deaths, [])
        self.assertEqual(len(results), 4)


class TestSignalsDuringWrite(unittest.TestCase):
    """Why the child's write loop needs no EINTR-retry helper."""

    @staticmethod
    def _armRepeatingAlarm() -> None:
        # Signals land continuously while the child blocks writing a payload
        # far larger than the pipe, so every EINTR-capable write is interrupted
        # many times over.
        signal.signal(signal.SIGALRM, lambda *args: None)
        signal.setitimer(signal.ITIMER_REAL, 0.001, 0.001)

    def testResultSurvivesSignalsInterruptingTheWrite(self) -> None:
        """PEP 475 retries `os.write` on EINTR; the loop absorbs short writes.

        So a handled signal arriving mid-payload must not be able to turn a
        unit whose computation completed into a spurious death. If this ever
        fails, the fix is an EINTR-retry helper around the child's writes --
        until then one would be dead code asserting the opposite of the truth.
        """
        results, deaths = _forkMap(_bigPayload, [0], 1, initializer=self._armRepeatingAlarm)
        self.assertEqual(deaths, [])
        self.assertEqual(results[0], b"z" * (4 << 20))


class TestDecodeResult(unittest.TestCase):
    """The framing, exercised directly at every boundary."""

    def testRoundTrip(self) -> None:
        for value in ({"a": 1}, None, [], b"bytes", 0):
            with self.subTest(value=value):
                payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                buffer = bytearray(_RESULT_HEADER.pack(len(payload)) + payload)
                self.assertEqual(_decodeResult(buffer), value)

    def testEmptyBuffer(self) -> None:
        self.assertIs(_decodeResult(bytearray()), _INCOMPLETE)

    def testTruncatedHeader(self) -> None:
        self.assertIs(_decodeResult(bytearray(b"\x00" * (_RESULT_HEADER.size - 1))), _INCOMPLETE)

    def testHeaderOnly(self) -> None:
        self.assertIs(_decodeResult(bytearray(_RESULT_HEADER.pack(10))), _INCOMPLETE)

    def testShortBody(self) -> None:
        self.assertIs(_decodeResult(bytearray(_RESULT_HEADER.pack(10) + b"abc")), _INCOMPLETE)

    def testOverlongBody(self) -> None:
        """A body longer than declared is as untrustworthy as a short one."""
        self.assertIs(_decodeResult(bytearray(_RESULT_HEADER.pack(2) + b"abcd")), _INCOMPLETE)

    def testCorruptPayloadOfTheRightLength(self) -> None:
        body = b"not a pickle"
        self.assertIs(_decodeResult(bytearray(_RESULT_HEADER.pack(len(body)) + body)), _INCOMPLETE)


class TestDescribeExit(unittest.TestCase):
    """The text a human reads out of the job log to name what was lost."""

    def testSigkillCallsOutTheOomKiller(self) -> None:
        reason = _describeExit(signal.SIGKILL, 0)
        self.assertEqual(reason, "killed by SIGKILL (typically the cgroup OOM killer)")

    def testOtherSignalsAreNamedWithoutTheOomHint(self) -> None:
        reason = _describeExit(signal.SIGSEGV, 0)
        self.assertEqual(reason, "killed by SIGSEGV")

    def testUnknownSignalNumber(self) -> None:
        self.assertIn("signal 63", _describeExit(63, 0))

    def testNonZeroExit(self) -> None:
        self.assertEqual(_describeExit(3 << 8, 0), "exited 3")

    def testCleanExitWithoutAResult(self) -> None:
        self.assertEqual(_describeExit(0, 0), "exited without writing a complete result")

    def testPartialBytesAreReported(self) -> None:
        self.assertEqual(
            _describeExit(0, 42),
            "exited without writing a complete result; 42 byte(s) of a partial result discarded",
        )

    def testTimeoutOutranksTheSignalItWasKilledWith(self) -> None:
        """The timeout is the cause; the `SIGKILL` is only how it was enforced.

        Falling through to the `WIFSIGNALED` branch would report the deliberate
        kill as "typically the cgroup OOM killer" -- true of the signal, wrong
        about the cause.
        """
        reason = _describeExit(signal.SIGKILL, 0, timeout=30)
        self.assertEqual(reason, "exceeded the 30s unit timeout and was killed")

    def testTimeoutStillReportsPartialBytes(self) -> None:
        """A unit killed mid-write still accounts for what arrived."""
        reason = _describeExit(signal.SIGKILL, 42, timeout=30)
        self.assertEqual(
            reason,
            "exceeded the 30s unit timeout and was killed; 42 byte(s) of a partial result discarded",
        )


class TestDumpStacksOnHang(unittest.TestCase):
    """Only the paths that do not end in `os._exit`."""

    def testDisabledByNone(self) -> None:
        before = len(_threadNames())
        with _dumpStacksOnHang(None, "disabled"):
            self.assertEqual(len(_threadNames()), before)

    def testDisabledByNonPositiveTimeout(self) -> None:
        with _dumpStacksOnHang(0.0, "disabled"):
            self.assertNotIn("hang-watchdog[disabled]", _threadNames())

    def testWatchdogIsRetiredOnCleanExit(self) -> None:
        """A watchdog that outlived its block would abort the job later on."""
        with _dumpStacksOnHang(30.0, "armed"):
            self.assertIn("hang-watchdog[armed]", _threadNames())
        for _ in range(100):
            if "hang-watchdog[armed]" not in _threadNames():
                break
            time.sleep(0.01)
        self.assertNotIn("hang-watchdog[armed]", _threadNames())

    def testArmedMessageIsLogged(self) -> None:
        log = unittest.mock.MagicMock()
        with _dumpStacksOnHang(30.0, "labelled", log):
            pass
        log.debug.assert_called_once()
        self.assertEqual(log.debug.call_args.args[1:], ("labelled", 30.0))


def _threadNames() -> list[str]:
    return [thread.name for thread in threading.enumerate()]


if __name__ == "__main__":
    unittest.main()
