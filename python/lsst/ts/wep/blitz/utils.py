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

"""Constants and helpers shared by more than one blitz module."""

__all__ = []

import sys

import numpy as np

from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils import binArray

_CALIB_STORE: dict = {}  # populated in parent before fork; workers inherit via COW

# Hard coding global wavefront sensor geometry for now
_INSTRUMENT: Instrument = Instrument(configFile="policy:instruments/LsstCam.yaml")

_EXTRA_FOCAL_DET_IDS = frozenset({191, 195, 199, 203})
_INTRA_FOCAL_DET_IDS = frozenset({192, 196, 200, 204})

# SW0 = extra-focal, SW1 = intra-focal
CORNER_PAIRS = {
    "R00": ("R00_SW0", "R00_SW1"),
    "R04": ("R04_SW0", "R04_SW1"),
    "R40": ("R40_SW0", "R40_SW1"),
    "R44": ("R44_SW0", "R44_SW1"),
}
CORNER_DET_NAMES = frozenset(s for sw0, sw1 in CORNER_PAIRS.values() for s in (sw0, sw1))
# Detector name -> corner, derived from CORNER_PAIRS rather than re-encoded.
CORNER_BY_DET_NAME = {s: corner for corner, pair in CORNER_PAIRS.items() for s in pair}

# ANSI escape codes for colorizing log messages (see colorLog config field).
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BLUE = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN = "\033[36m"

# Maximum nearby sources to store in the output table for each donut.
_MAX_NEARBY = 5

# Maximum Noll index fit/reported. Dense Noll-indexed arrays are length
# _ZK_JMAX + 1: index j holds Zernike j, and indices 0-3 are always 0.
_ZK_JMAX = 66


def _resolveColorLogEnabled(colorLog: bool | None) -> bool:
    """Resolve the colorLog config value to a concrete enabled/disabled bool.

    If ``colorLog`` is None, color is enabled only when stdout is attached
    to an interactive terminal.
    """
    if colorLog is None:
        return sys.stdout.isatty()
    return colorLog


def _colorize(text: str, *codes: str, enabled: bool = True) -> str:
    """Wrap ``text`` in the given ANSI escape code(s) if ``enabled``.

    Parameters
    ----------
    text : str
        The text to colorize.
    *codes : str
        One or more ANSI escape codes (e.g. ``_ANSI_RED``, ``_ANSI_BOLD``).
    enabled : bool, optional
        If False, ``text`` is returned unchanged. (the default is True)

    Returns
    -------
    str
        The colorized (or original) text.
    """
    if not enabled or not codes:
        return text
    return "".join(codes) + text + _ANSI_RESET


def _resolveDonutRadius(donutRadius: float | None) -> float:
    """Return a usable donut radius in pixels, falling back to the nominal one.

    The per-exposure radius measured by `DonutDetectDiameterTask` is preferred,
    but it is NaN whenever the sizing curve could not be formed (no surviving
    peaks, monotonic curve). Rather than propagate NaN into every mask radius
    downstream, fall back to the nominal `_INSTRUMENT.donutRadius`.

    Parameters
    ----------
    donutRadius : float or None
        Measured donut radius in un-binned pixels, or None/NaN if unmeasured.

    Returns
    -------
    float
        ``donutRadius`` if finite and positive, else ``_INSTRUMENT.donutRadius``.
    """
    if donutRadius is None:
        return _INSTRUMENT.donutRadius
    if not np.isfinite(donutRadius) or donutRadius <= 0:
        return _INSTRUMENT.donutRadius
    return donutRadius


def _bin_stamp_odd(stamp: np.ndarray, binning: int) -> np.ndarray:
    """Bin a stamp and trim it to an odd pixel size.

    Danish wants an odd-sized image so the donut centre lands on a pixel
    centre. Shared by `_prep_donut_for_danish` and `_buildCatalog` so that
    donuts which never reached a fit (paired-mode surplus) still get a WF
    image on the same pixel grid as the fitted ones.
    """
    img = stamp.astype(float)
    if binning > 1:
        img = binArray(img, binning)
    if img.shape[0] % 2 == 0:
        img = img[:-1, :-1]
    return img


def _zk_cols(suffix: str, zk: np.ndarray) -> dict:
    """Return ``{f"Z{j}_{suffix}": um}`` columns for Noll 4..``_ZK_JMAX``.

    ``zk`` is a Noll-indexed array in metres of length ``_ZK_JMAX + 1``; values
    are converted to µm, with NaN passed through as NaN.
    """
    return {
        f"Z{j}_{suffix}": zk[j] * 1e6 if not np.isnan(zk[j]) else float("nan")
        for j in range(4, _ZK_JMAX + 1)
    }
