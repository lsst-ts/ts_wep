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
from dataclasses import dataclass
from typing import Any

import batoid
import galsim
import numpy as np

from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils import binArray

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
# Detector name -> intra/extra, likewise derived. Corner mode's defocal side is
# a property of the detector, so donuts carry only their optic offsets and
# anything needing the label (currently just plot layout) looks it up here.
# Full-array mode has no equivalent: there the side comes from which exposure
# of the pair.
CORNER_DEFOCAL_BY_DET_NAME = {
    name: ("extra" if name == sw0 else "intra") for sw0, sw1 in CORNER_PAIRS.values() for name in (sw0, sw1)
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

# Columns that come from the reference catalog, and so exist only on the refcat
# selection path -- but which every table reaching `CutDonutStampsTask` carries
# regardless, NaN-filled on the blitz-detection path. The blitz path is a
# *data* gap, not a schema difference, so the consumer reads these
# unconditionally instead of testing `colnames` for their presence. Filled in
# `_cutout_one_exposure`.
_REFCAT_COLUMNS = (
    "coord_ra",
    "coord_dec",
    "photo_mag",
    "astrom_mag",
)

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
#
# Each label is also the stem of the `CowStore` field holding the subtask that
# runs that stage (`isr` -> `isr_task`, `astrom` -> `astrom_task`, ...), so the
# two vocabularies are one vocabulary.  `measure_task` and `wf_fit_task` are
# the only store subtasks with no label here, because neither is a timed
# cutout stage.
_CUTOUT_STAGE_KEYS = {
    "isr": "isr_run",
    "bkg": "bkg_run",
    "diam": "diam_run",
    "detect": "detect_run",
    "astrom": "wcs_refit_run",
    "select": "catalog_select_run",
    "cut": "stamp_cut_run",
}


# Optics that can be shifted along z to defocus, in the order the offset
# triplet carried on each `Donut` uses. Corner mode moves the detector plane
# inside the camera; full-array mode moves the whole camera; some data instead
# moves M2. Names match the batoid LSST model, where M1/M2/M3/LSSTCamera are
# top level and Detector is nested inside LSSTCamera.
_OFFSET_OPTICS = ("Detector", "LSSTCamera", "M2")

# An offset triplet: signed z shifts in meters, ordered as `_OFFSET_OPTICS`.
_Offsets = tuple[float, float, float]


def _defocused_telescope(telescope: batoid.Optic, offsets: _Offsets) -> batoid.Optic:
    """Return ``telescope`` with each non-zero offset component applied.

    Parameters
    ----------
    telescope : batoid.Optic
        The in-focus telescope, from `CowStore.telescope`.
    offsets : tuple of float
        Signed z shifts in meters, ordered as `_OFFSET_OPTICS`
        ``(detector, camera, m2)``.

    Returns
    -------
    batoid.Optic
        The defocused telescope.

    Notes
    -----
    Deliberately not memoized. Loading the base telescope from YAML costs
    ~41 ms and is done once per quantum in the parent, but shifting an optic on
    top of it costs only 20 µs (detector) to 134 µs (camera) -- less than one
    of the two chief-ray traces in `_defocal_radial_scale`, and nothing at all
    beside the danish fit that follows. An earlier version cached these per
    triplet in the COW store, which bought no measurable time and made the
    store's lifetime ambiguous, since the cached value silently depended on
    which telescope had been loaded.
    """
    for name, dz in zip(_OFFSET_OPTICS, offsets):
        if dz:
            telescope = telescope.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
    return telescope


# Field angle at which the defocal radial scale is evaluated. The displacement
# is linear in field angle (verified against batoid: 7.9/15.7/23.6 px at
# 0.5/1.0/1.5 deg for a 1.5 mm camera shift), so it is a pure scale and any
# non-zero reference angle gives the same answer.
_RADIAL_SCALE_REF_THETA = np.deg2rad(1.0)


def _defocal_radial_scale(telescope: batoid.Optic, offsets: _Offsets) -> float:
    """Fractional radial stretch of the focal plane produced by a defocus.

    Shifting an optic along z moves an off-axis chief ray radially, so the
    *same* star lands at slightly different field angles either side of focus
    -- ~27 px apart at 1.725 deg for a 1.5 mm camera shift, and zero on axis.
    Any attempt to associate donuts between an intra and an extra exposure by
    position has to account for this, or it will work at the field center and
    fail at the edge.

    Because the displacement is linear in field angle it is a pure scale, so
    one number per offset triplet corrects the whole focal plane.

    Two chief-ray traces, ~862 µs. That is cheap once per offset triplet and
    wasteful once per donut, so full-array mode does not call this in its
    workers: the parent evaluates it for both sides of focus and hands the
    answers over as `CowStore.radial_scale_by_offsets`.

    Parameters
    ----------
    telescope : batoid.Optic
        The in-focus telescope, from `CowStore.telescope`.
    offsets : tuple of float
        Signed z shifts in meters, ordered as `_OFFSET_OPTICS`.

    Returns
    -------
    float
        Ratio of defocused to in-focus radial position. Divide a measured field
        angle by this to recover the common frame. 1.0 for a null defocus.
    """
    wavelength = _INSTRUMENT.wavelength[_INSTRUMENT.refBand]

    def _chief_ray_x(optic):
        ray = batoid.RayVector.fromStop(
            0.0,
            0.0,
            optic=optic,
            wavelength=wavelength,
            theta_x=_RADIAL_SCALE_REF_THETA,
            theta_y=0.0,
        )
        optic.trace(ray)
        return float(ray.x[0])

    return _chief_ray_x(_defocused_telescope(telescope, offsets)) / _chief_ray_x(telescope)


@dataclass
class IsrCalibs:
    """The four materialized calibrations ISR needs for one exposure.

    The one detector-shaped thing both modes hand to `_cutout_one_exposure`,
    and separate from the two store entries below because the modes build it at
    different times: corner mode in the parent, where its calibrations are
    already objects, and full-array mode in each worker, once it has resolved
    its own deferred handles.
    """

    ptc: Any
    flat: Any
    linearizer: Any
    crosstalk: Any


@dataclass
class CornerDetectorInputs:
    """One corner sensor's raw exposure and calibrations, already read.

    Corner mode does its butler I/O in the parent, so these are materialized
    objects that the worker uses directly.
    """

    raw: Any
    calibs: IsrCalibs


@dataclass
class FamDetectorInputs:
    """One science CCD's deferred handles, one raw per exposure of the pair.

    Full-array mode reads no pixels in the parent -- each worker resolves these
    itself -- so every field here is a handle rather than an object.

    ``intrinsic_zernikes`` appears here and not on `CornerDetectorInputs`
    because of *where* each mode annotates its donuts, not because corner mode
    lacks intrinsics: it has the same ``intrinsicZernikes`` prerequisite input,
    but applies it in the parent once the cutout pool returns, so the
    calibration never has to cross a fork. It is optional because a detector
    may legitimately have no intrinsic calibration, in which case the parent
    warns and the fit references the nominal design optics instead.
    """

    raws: dict[int, Any]  # exposure id -> deferred handle
    ptc: Any
    flat: Any
    linearizer: Any
    crosstalk: Any
    intrinsic_zernikes: Any | None


@dataclass
class CowStore:
    """Everything the parent hands to the forked workers.

    Populated once before the fork and read-only afterwards, so the children
    inherit it by copy-on-write instead of receiving it through a pickle. That
    one direction is the whole contract: nothing here is written after the
    fork, which is why no field has a default and why there is nothing to
    invalidate between quanta.

    Because a field without a default is not a class attribute either, reading
    one this mode never set raises `AttributeError` -- the equivalent of the
    `KeyError` the string-keyed dict used to give, with no ``Optional``
    standing in for "wrong mode".

    Keys are snake_case throughout: the store is not framework-introspected,
    so the submodule's rule applies.  Scalars mirror their camelCase config
    field directly (``config.maxFitScatter`` becomes ``max_fit_scatter``), but
    **the subtask fields deliberately do not.**  Each is ``<stage>_task``,
    where the stage is the short label `_CUTOUT_STAGE_KEYS` already uses, so
    ``config.subtractBackground`` arrives as ``bkg_task`` and
    ``config.donutSelector`` as ``select_task``.  `wf_fit_task` is the only
    three-word name.  Do not "fix" these to track the config field names --
    the point is that the seven cutout stages, their timing keys, and the
    subtasks that run them all share one vocabulary.

    `for_corner` and `for_fam` spell every field out rather than sharing a
    ``**kwargs`` helper. The duplication is deliberate -- the point of the
    dataclass is that mypy checks both call sites, and ``**kwargs`` is exactly
    the hole that would reopen.
    """

    # --- Subtasks, shared by both modes.
    isr_task: Any
    bkg_task: Any
    diam_task: Any
    detect_task: Any
    astrom_task: Any
    select_task: Any
    measure_task: Any
    cut_task: Any
    wf_fit_task: Any
    # --- Config scalars and the telescope, shared by both modes.
    wf_estimation_mode: str
    max_fit_scatter: float
    astrom_ref_filter: str
    photo_ref_filter: str
    telescope: batoid.Optic
    # --- Corner mode only.
    corner_detectors: dict[str, CornerDetectorInputs]
    det_refcats: dict[str, Any]
    # --- Full-array mode only.
    fam_detectors: dict[int, FamDetectorInputs]
    pair_match_tolerance: float
    save_stamps: bool
    save_wf_images: bool
    band: str
    rtp_deg: float | None
    boresight_alt_rad: float | None
    intra_exposure: int
    extra_exposure: int
    offsets_by_exposure: dict[int, _Offsets]
    radial_scale_by_offsets: dict[_Offsets, float]
    refcat_handles: list
    butler: Any

    @classmethod
    def uninitialized(cls) -> "CowStore":
        """Return a store with its fields declared but none of them set.

        Bypasses the generated ``__init__`` deliberately: the two per-mode
        constructors below are the only sanctioned way to fill a store, and
        each sets a different subset of the fields.
        """
        return object.__new__(cls)

    @classmethod
    def for_corner(
        cls,
        *,
        isr_task: Any,
        bkg_task: Any,
        diam_task: Any,
        detect_task: Any,
        astrom_task: Any,
        select_task: Any,
        measure_task: Any,
        cut_task: Any,
        wf_fit_task: Any,
        wf_estimation_mode: str,
        max_fit_scatter: float,
        astrom_ref_filter: str,
        photo_ref_filter: str,
        telescope: batoid.Optic,
        corner_detectors: dict[str, CornerDetectorInputs],
        det_refcats: dict[str, Any],
    ) -> "CowStore":
        """Build the store corner mode's cutout and fit workers read."""
        store = cls.uninitialized()
        store.isr_task = isr_task
        store.bkg_task = bkg_task
        store.diam_task = diam_task
        store.detect_task = detect_task
        store.astrom_task = astrom_task
        store.select_task = select_task
        store.measure_task = measure_task
        store.cut_task = cut_task
        store.wf_fit_task = wf_fit_task
        store.wf_estimation_mode = wf_estimation_mode
        store.max_fit_scatter = max_fit_scatter
        store.astrom_ref_filter = astrom_ref_filter
        store.photo_ref_filter = photo_ref_filter
        store.telescope = telescope
        store.corner_detectors = corner_detectors
        store.det_refcats = det_refcats
        return store

    @classmethod
    def for_fam(
        cls,
        *,
        isr_task: Any,
        bkg_task: Any,
        diam_task: Any,
        detect_task: Any,
        astrom_task: Any,
        select_task: Any,
        measure_task: Any,
        cut_task: Any,
        wf_fit_task: Any,
        wf_estimation_mode: str,
        max_fit_scatter: float,
        astrom_ref_filter: str,
        photo_ref_filter: str,
        telescope: batoid.Optic,
        fam_detectors: dict[int, FamDetectorInputs],
        pair_match_tolerance: float,
        save_stamps: bool,
        save_wf_images: bool,
        band: str,
        rtp_deg: float | None,
        boresight_alt_rad: float | None,
        intra_exposure: int,
        extra_exposure: int,
        offsets_by_exposure: dict[int, _Offsets],
        refcat_handles: list,
        butler: Any,
    ) -> "CowStore":
        """Build the store the full-array per-detector workers read.

        ``radial_scale_by_offsets`` is derived here rather than passed in: the
        parent knows every triplet a worker can ask about, because a donut only
        ever carries back what ``offsets_by_exposure`` gave it.
        """
        store = cls.uninitialized()
        store.isr_task = isr_task
        store.bkg_task = bkg_task
        store.diam_task = diam_task
        store.detect_task = detect_task
        store.astrom_task = astrom_task
        store.select_task = select_task
        store.measure_task = measure_task
        store.cut_task = cut_task
        store.wf_fit_task = wf_fit_task
        store.wf_estimation_mode = wf_estimation_mode
        store.max_fit_scatter = max_fit_scatter
        store.astrom_ref_filter = astrom_ref_filter
        store.photo_ref_filter = photo_ref_filter
        store.telescope = telescope
        store.fam_detectors = fam_detectors
        store.pair_match_tolerance = pair_match_tolerance
        store.save_stamps = save_stamps
        store.save_wf_images = save_wf_images
        store.band = band
        store.rtp_deg = rtp_deg
        store.boresight_alt_rad = boresight_alt_rad
        store.intra_exposure = intra_exposure
        store.extra_exposure = extra_exposure
        store.offsets_by_exposure = offsets_by_exposure
        store.radial_scale_by_offsets = {
            tuple(float(o) for o in offsets): _defocal_radial_scale(telescope, offsets)
            for offsets in offsets_by_exposure.values()
        }
        store.refcat_handles = refcat_handles
        store.butler = butler
        return store

    def adopt(self, other: "CowStore") -> None:
        """Become ``other`` in place, discarding whatever was here before.

        Never rebind `_COW_STORE` instead of calling this: the worker modules
        bind the name at import time, so a rebind would leave them reading the
        previous quantum's store while the parent reads the current one.

        Clearing rather than merging is also what keeps a `Task` instance safe
        to reuse across quanta -- a field the previous mode set and this one
        does not goes back to raising `AttributeError`.
        """
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)


# Populated in the parent before the fork; workers inherit it via COW.
_COW_STORE: CowStore = CowStore.uninitialized()


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
        ``donutRadius`` if finite and positive, else
        ``_INSTRUMENT.donutRadius``.
    """
    if donutRadius is None:
        return _INSTRUMENT.donutRadius
    if not np.isfinite(donutRadius) or donutRadius <= 0:
        return _INSTRUMENT.donutRadius
    return donutRadius


def _dense_intrinsic(donut) -> np.ndarray:
    """Return intrinsic Zernikes in meters, dense over Noll 0..``_ZK_JMAX``.

    Indices with no supplied value are 0.0; ``Donut.intrinsic_zk`` is in µm and
    starts at Noll 4.

    Lives here rather than beside its original caller in
    `WavefrontFittingTask` because `_build_donut_catalog` needs the same
    conversion: intrinsics are a function of field position, so a donut no fit
    consumed still has them, and the catalog falls back to this for that row.
    """
    out = np.zeros(_ZK_JMAX + 1)
    raw = donut.intrinsic_zk  # µm, Noll 4.._ZK_JMAX
    if raw is not None:
        n_slots = _ZK_JMAX + 1 - 4  # Noll 4.._ZK_JMAX inclusive
        for idx in range(min(len(raw), n_slots)):
            out[idx + 4] = raw[idx] * 1e-6
    return out


def _bin_stamp_odd(stamp: np.ndarray, binning: int) -> np.ndarray:
    """Bin a stamp and trim it to an odd pixel size.

    Danish wants an odd-sized image so the donut center lands on a pixel
    center. Shared by `_prep_donut_for_danish` and `_buildCatalog` so that
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

    Coefficients only mix within an (n, |m|) pair, so the matrix is built once
    at ``_ZK_JMAX`` -- a complete radial order, which galsim requires -- and
    the input is zero-padded up to it. Sizing the matrix from ``zk``'s own
    width would instead raise whenever that width splits a pair.

    NaN slots (Noll indices below 4, indices that were not fitted, unfit
    donuts) rotate as zero and are then restored, so the output is defined
    exactly where the input was.
    """
    rot = galsim.zernike.zernikeRotMatrix(_ZK_JMAX, theta)
    padded = np.zeros((len(zk), _ZK_JMAX + 1))
    padded[:, : zk.shape[1]] = np.nan_to_num(zk, nan=0.0)
    # Row-vector convention: each row is one donut's coefficient vector.
    out = (padded @ rot)[:, : zk.shape[1]]
    return np.where(np.isnan(zk), np.nan, out)
