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

import batoid
import galsim
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
# Detector name -> intra/extra, likewise derived. Corner mode's defocal side is a
# property of the detector, so donuts carry only their optic offsets and anything
# needing the label (currently just plot layout) looks it up here. Full-array mode
# has no equivalent: there the side comes from which exposure of the pair.
CORNER_DEFOCAL_BY_DET_NAME = {
    name: ("extra" if name == sw0 else "intra")
    for sw0, sw1 in CORNER_PAIRS.values()
    for name in (sw0, sw1)
}

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

# Short stage label -> the key `_cutout_one_exposure` returns its elapsed time
# under, in the order the stages run.  That function defines the keys; this is
# the one ordered list of them, because four separate places report the same
# seven stages -- the corner-mode and full-array per-detector log lines, the
# plot task's per-detector panel, and the `det_meta` block of the output
# catalog -- and a stage added to the cutout pipeline should not be able to
# show up in some of them and silently not others.
_CUTOUT_STAGE_KEYS = {
    "isr": "isr_run",
    "bkg": "bkg_run",
    "diam": "diam_run",
    "detect": "blind_detect_run",
    "wcs": "wcs_refit_run",
    "select": "catalog_select_run",
    "cut": "stamp_cut_run",
}


# Optics that can be shifted along z to defocus, in the order the offset triplet
# carried on each `Donut` uses. Corner mode moves the detector plane inside the
# camera; full-array mode moves the whole camera; some data instead moves M2.
# Names match the batoid LSST model, where M1/M2/M3/LSSTCamera are top level and
# Detector is nested inside LSSTCamera.
_OFFSET_OPTICS = ("Detector", "LSSTCamera", "M2")


def _telescope_for_offsets(offsets: tuple[float, float, float]):
    """Return the telescope defocused by an offset triplet, memoised.

    Parameters
    ----------
    offsets : tuple of float
        Signed z shifts in meters, ordered as `_OFFSET_OPTICS`
        ``(detector, camera, m2)``.

    Returns
    -------
    batoid.Optic
        ``_CALIB_STORE["telescope"]`` with each non-zero component applied.

    Notes
    -----
    Callers are expected to pre-build every triplet they will need in the parent
    process before forking, so workers inherit the built telescopes via
    copy-on-write rather than each paying for them. A worker asking for a triplet
    the parent did not anticipate still gets a correct answer, it just builds it
    itself and the result does not propagate back.
    """
    store = _CALIB_STORE.setdefault("telescope_by_offsets", {})
    key = tuple(float(o) for o in offsets)
    telescope = store.get(key)
    if telescope is None:
        telescope = _CALIB_STORE["telescope"]
        for name, dz in zip(_OFFSET_OPTICS, key):
            if dz:
                telescope = telescope.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
        store[key] = telescope
    return telescope


# Field angle at which the defocal radial scale is evaluated. The displacement is
# linear in field angle (verified against batoid: 7.9/15.7/23.6 px at
# 0.5/1.0/1.5 deg for a 1.5 mm camera shift), so it is a pure scale and any
# non-zero reference angle gives the same answer.
_RADIAL_SCALE_REF_THETA = np.deg2rad(1.0)


def _defocal_radial_scale(offsets: tuple[float, float, float]) -> float:
    """Fractional radial stretch of the focal plane produced by a defocus.

    Shifting an optic along z moves an off-axis chief ray radially, so the *same*
    star lands at slightly different field angles either side of focus -- ~27 px
    apart at 1.725 deg for a 1.5 mm camera shift, and zero on axis. Any attempt
    to associate donuts between an intra and an extra exposure by position has to
    account for this, or it will work at the field centre and fail at the edge.

    Because the displacement is linear in field angle it is a pure scale, so one
    number per offset triplet corrects the whole focal plane.

    Parameters
    ----------
    offsets : tuple of float
        Signed z shifts in metres, ordered as `_OFFSET_OPTICS`.

    Returns
    -------
    float
        Ratio of defocused to in-focus radial position. Divide a measured field
        angle by this to recover the common frame. 1.0 for a null defocus.
    """
    store = _CALIB_STORE.setdefault("radial_scale_by_offsets", {})
    key = tuple(float(o) for o in offsets)
    scale = store.get(key)
    if scale is None:
        wavelength = _INSTRUMENT.wavelength[_INSTRUMENT.refBand]

        def _chief_ray_x(telescope):
            ray = batoid.RayVector.fromStop(
                0.0,
                0.0,
                optic=telescope,
                wavelength=wavelength,
                theta_x=_RADIAL_SCALE_REF_THETA,
                theta_y=0.0,
            )
            telescope.trace(ray)
            return float(ray.x[0])

        base = _chief_ray_x(_CALIB_STORE["telescope"])
        scale = _chief_ray_x(_telescope_for_offsets(key)) / base
        store[key] = scale
    return scale


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


def _rotate_zk(zk: np.ndarray, theta: float) -> np.ndarray:
    """Rotate dense Noll-indexed Zernike coefficients into a rotated frame.

    ``zk`` is ``(nrow, njmax + 1)``, Noll-indexed along axis 1 (index j holds
    Zernike j) and no longer than ``_ZK_JMAX + 1``; ``theta`` is the frame
    rotation in radians.

    Coefficients only mix within an (n, |m|) pair, so the matrix is built once at
    ``_ZK_JMAX`` -- a complete radial order, which galsim requires -- and the
    input is zero-padded up to it. Sizing the matrix from ``zk``'s own width
    would instead raise whenever that width splits a pair.

    NaN slots (Noll indices below 4, indices that were not fitted, unfit donuts)
    rotate as zero and are then restored, so the output is defined exactly where
    the input was.
    """
    rot = galsim.zernike.zernikeRotMatrix(_ZK_JMAX, theta)
    padded = np.zeros((len(zk), _ZK_JMAX + 1))
    padded[:, : zk.shape[1]] = np.nan_to_num(zk, nan=0.0)
    # Row-vector convention: each row is one donut's coefficient vector.
    out = (padded @ rot)[:, : zk.shape[1]]
    return np.where(np.isnan(zk), np.nan, out)
