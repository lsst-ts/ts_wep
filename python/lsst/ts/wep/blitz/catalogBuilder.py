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

"""The per-donut output catalog, shared by corner and full-array mode.

Both modes emit the same schema, so this lives in one place rather than in each
task.
"""

__all__ = ["CatalogOptions", "build_donut_catalog", "build_noll_pairs", "transform_eb"]

import importlib
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import QTable
from galsim.zernike import noll_to_zern

from .dataStructures import _NULL_WF
from .utils import (
    _CUTOUT_STAGE_KEYS,
    _INSTRUMENT,
    _MAX_NEARBY,
    _OFFSET_OPTICS,
    _ZK_JMAX,
    _bin_stamp_odd,
    _dense_intrinsic,
    _rotate_zk,
)


def build_noll_pairs(jmax):
    """Return (pairs, singles) for Noll indices 1..jmax.

    Assumes jmax is pair-complete (no truncated doublets).

    pairs   : list of (j_cos, j_sin, n, |m|) doublets
    singles : list of j with m==0
    """
    pairs, singles = [], []
    j = 1
    while j <= jmax:
        n, m = noll_to_zern(j)
        if m == 0:
            singles.append(j)
            j += 1
            continue
        if m > 0:
            pairs.append((j, j + 1, n, m))
        else:
            pairs.append((j + 1, j, n, abs(m)))
        j += 2
    return pairs, singles


def transform_eb(
    zk_list,
    thx,
    thy,
):
    """Transform Noll-indexed Zernike coefficients to E/B (cosine/sine) representation.

    Rotates each spin-m Zernike doublet by m*phi, phi = atan2(thy, thx),
    referenced to the same axes as the Zernike azimuth. Same Noll layout:
    cosine slot -> aligned (E), sine slot -> cross (B). The E/B values are
    axis-independent; i.e., independent of the frame (_ccs) names.

    NaN handling: an output doublet is defined only if BOTH members are
    finite and phi is valid; otherwise both slots are NaN. m==0 terms pass
    through (NaN preserved).
    """
    phi = np.arctan2(thy, thx)
    phi_bad = ~np.isfinite(phi) | ((thx == 0.0) & (thy == 0.0))

    out_list = []
    for zk in zk_list:
        unit = getattr(zk, "unit", None)
        zk_val = zk.value if isinstance(zk, u.Quantity) else np.asarray(zk)
        if zk_val.ndim != 2:
            raise ValueError(f"zk must be 2D [nrow, n_noll]; got {zk_val.shape}")

        pairs, _ = build_noll_pairs(zk_val.shape[1] - 1)
        out = zk_val.copy()          # m==0 slots pass through, incl. NaNs

        for (j_cos, j_sin, _, m_abs) in pairs:
            c_cos = zk_val[:, j_cos]
            c_sin = zk_val[:, j_sin]
            a = m_abs * phi
            ca, sa = np.cos(a), np.sin(a)
            e = ca * c_cos + sa * c_sin
            b = -sa * c_cos + ca * c_sin
            bad = (~np.isfinite(c_cos)) | (~np.isfinite(c_sin)) | phi_bad
            out[:, j_cos] = np.where(bad, np.nan, e)
            out[:, j_sin] = np.where(bad, np.nan, b)

        if unit is not None:
            out = out << unit
        out_list.append(out)

    return out_list


@dataclass(frozen=True)
class CatalogOptions:
    """Config-derived scalars the catalog needs, gathered from the calling task.

    Grouped into one object rather than a dozen positional arguments because the
    two callers assemble them from different config trees, and a silently
    mismatched argument order would be hard to spot in the output.

    Attributes
    ----------
    stamp_size : int
        Un-binned stamp side, from the stamp-cutting subtask.
    binning : int
        WF binning factor, from the fitting subtask. Also written to ``meta`` so
        the plots can convert raw-pixel quantities when they fall back to
        ``wf_img``.
    noll_indices : tuple of int
        Noll indices actually fitted. The deviation array column stops at the
        highest one.
    aperture_margin_frac : float
        Fractional margin on both edges of the photometric annulus, from the
        measurement subtask: the outer edge sits at ``radius * (1 + frac)``, the
        inner edge at ``radius * obscuration * (1 - frac)``.
    bkg_inner_disc_frac : float
        Outer edge of the inner background disc (inside the obscuration), from
        the measurement subtask.
    bkg_annulus_inner_frac, bkg_annulus_outer_frac : float
        Outer background annulus geometry, from the measurement subtask.
    max_donuts : int
        Per-detector accepted-donut cap.
    wf_mode : str
        WF dispatch mode label.
    save_stamps : bool
        Include the un-binned ``stamp`` column. Much the largest column.
    save_wf_images : bool
        Include ``wf_img`` and ``model_img``. Dropped as a pair: a model with no
        data to compare it against is not useful.
    """

    stamp_size: int
    binning: int
    noll_indices: tuple[int, ...]
    aperture_margin_frac: float
    bkg_inner_disc_frac: float
    bkg_annulus_inner_frac: float
    bkg_annulus_outer_frac: float
    max_donuts: int
    wf_mode: str
    save_stamps: bool = True
    save_wf_images: bool = True

    @property
    def zk_deviation_jmax(self) -> int:
        """Highest fitted Noll index; deviations are cut off here."""
        return max(self.noll_indices)

    @property
    def wf_img_size(self) -> int:
        """Binned WF image side, forced odd (see `_prep_donut_for_danish`)."""
        binned = self.stamp_size // self.binning
        return binned if binned % 2 == 1 else binned - 1


def _encode_nearby(entries):
    """Return (dx, dy, mag) arrays of length ``_MAX_NEARBY`` for one donut.

    The offsets are relative to the donut's detector-frame centroid (``x_det``,
    ``y_det``); the donut itself is excluded upstream, in `CutDonutStampsTask`.
    Neighbors are sorted brightest-first (ascending magnitude); entries with
    a NaN magnitude sort last. The brightest ``_MAX_NEARBY`` are kept and
    shorter lists are NaN-padded. True pre-truncation counts are captured
    separately (see n_nearby_*).
    """
    x = np.full(_MAX_NEARBY, np.nan, dtype=float)
    y = np.full(_MAX_NEARBY, np.nan, dtype=float)
    mag = np.full(_MAX_NEARBY, np.nan, dtype=float)
    brightest = sorted(entries, key=lambda e: (np.isnan(e[2]), e[2]))[:_MAX_NEARBY]
    n_nearby = len(brightest)
    x[:n_nearby] = [e[0] for e in brightest]
    y[:n_nearby] = [e[1] for e in brightest]
    mag[:n_nearby] = [e[2] for e in brightest]
    return x, y, mag


def _defocal_offsets(donut) -> np.ndarray:
    """Return one donut's optic z shifts as a length-3 array of meters.
    """
    offsets = donut.defocal_offsets
    if offsets is None:
        return np.full(len(_OFFSET_OPTICS), np.nan, dtype=float)
    return np.asarray(offsets, dtype=float)


def _observation_meta(visit_info) -> dict:
    """Return the observation keys for ``meta``, read off a `VisitInfo`."""
    if visit_info is None:
        return {
            "date": None,
            "exposure_time": np.nan * u.s,
            "boresight_alt": np.nan * u.deg,
            "boresight_az": np.nan * u.deg,
            "boresight_rot_angle": np.nan * u.deg,
            "boresight_par_angle": np.nan * u.deg,
        }
    # MJD purely for a legible repr; the Time itself is scale- and
    # format-agnostic once constructed.
    date = visit_info.date.toAstropy().tai
    date.format = "mjd"
    az_alt = visit_info.boresightAzAlt
    return {
        "date": date,
        "exposure_time": visit_info.exposureTime * u.s,
        "boresight_alt": az_alt.getLatitude().asDegrees() * u.deg,
        "boresight_az": az_alt.getLongitude().asDegrees() * u.deg,
        "boresight_rot_angle": visit_info.boresightRotAngle.asDegrees() * u.deg,
        "boresight_par_angle": visit_info.boresightParAngle.asDegrees() * u.deg,
    }


def _software_versions() -> dict:
    """Return the version keys for ``meta``.

    Best effort: a package that cannot report a version gets ``""`` rather
    than sinking the catalog.
    """
    out = {}
    for key, module in (
        ("ts_wep_version", "lsst.ts.wep"),
        ("danish_version", "danish"),
        ("batoid_version", "batoid"),
    ):
        try:
            out[key] = str(importlib.import_module(module).__version__)
        except Exception:  # noqa: BLE001 -- provenance only
            out[key] = ""
    return out


# Written verbatim to ``meta["notes"]``, keyed by the meta key each line is
# about. This is for what a unit cannot carry: every value in ``meta`` says
# what it is measured in, but not which instant, which visit or which clock it
# refers to. Deliberately short -- one line each, not a substitute for the
# docstrings.
_META_NOTES = {
    "date": "mid-exposure, as VisitInfo defines every date",
    "*_elapsed": (
        "wall clock in corner mode; summed over parallel per-detector workers,"
        " hence CPU time, in full-array mode (see meta['mode']), where they can"
        " exceed run_elapsed"
    ),
    "rot_tel_pos": (
        "the angle the _ocs columns were rotated by, derived as"
        " (boresight_par_angle - boresight_rot_angle - 90 deg) wrapped to"
        " (-180, 180] -- not either raw input"
    ),
    "ref_visit_id": (
        "the visit this table is for, the extra-focal one of the pair in"
        " full-array mode; with group_id it keys a fit across visits"
    ),
    "det_meta": (
        "keyed f'{det_name}_{visit_id}'; astrom_scatter is the WCS refit's"
        " on-sky scatter, NaN where no refit ran; n_quarter is the detector"
        " orientation needed to relate the CCS stamp to x_det/y_det"
    ),
}


def build_donut_catalog(
    results: list,
    wf_results: list,
    donuts: list,
    unmatched_donuts: list,
    visit_id: int,
    options: CatalogOptions,
    intra_visit_id: int | None = None,
    extra_visit_id: int | None = None,
    exposure_group: str = "",
    run_elapsed: float = 0.0,
    refcat_elapsed: float = 0.0,
    butler_elapsed: float = 0.0,
    butler_times: dict | None = None,
    cutout_elapsed: float = 0.0,
    danish_elapsed: float = 0.0,
    photo_filter_name: str = "",
    astrom_filter_name: str = "",
    rtp_rad: float = 0.0,
    mode: str = "",
    visit_info: Any = None,
    instrument: str = "",
) -> QTable:
    """Build a per-donut QTable covering every donut cut from this visit.

    Parameters
    ----------
    results : list
        Per-detector cutout dicts (supplies rejected donuts and per-detector
        metadata).  Full-array mode passes two per detector, one per exposure of
        the pair, each tagged with its own ``visit_id``; corner mode passes one
        per detector and omits ``visit_id``, defaulting to the ``visit_id``
        argument.  Either way the metadata is keyed by ``f"{det_name}_{visit_id}"``
        (see ``table.meta["det_meta"]``).
    wf_results : list
        Per-fit WF result dicts from the WF worker pool.
    donuts : list
        Donut records that passed selection.
    unmatched_donuts : list
        Donut records with no intra/extra partner.  These also appear in
        ``donuts``; the table carries one row per donut.
    visit_id : int
        Visit identifier; the visit this table is *for*, written to
        ``meta["ref_visit_id"]``.  Full-array mode passes the extra-focal visit
        of the pair.
    options : CatalogOptions
        Config-derived scalars from the calling task.
    intra_visit_id, extra_visit_id : int, optional
        The visits that supplied the intra- and extra-focal donuts.
    exposure_group : str, optional
        The butler ``group`` dimension of the exposure(s) this table covers.
    rtp_rad : float
        Camera rotator angle on sky (rotTelPos) in radians, used to rotate the
        Zernikes from the camera into the optical coordinate system.  Written
        to ``meta["rot_tel_pos"]``.
    mode : str, optional
        ``"corner"`` or ``"fam"``, written to ``meta["mode"]``.
    visit_info : lsst.afw.image.VisitInfo, optional
        Exposure header metadata, read for the observation keys in ``meta``
        (``date``, ``exposure_time``, ``boresight_alt``, ``boresight_az``,
        ``boresight_rot_angle``, ``boresight_par_angle``).  In full-array
        mode this is the *extra-focal* exposure's, matching
        ``meta["ref_visit_id"]``.
    instrument : str, optional
        Instrument name, e.g. ``"LSSTCam"``, written to ``meta["instrument"]``.

    Returns
    -------
    QTable
        Exactly one row per donut, keyed by ``(visit_id, det_name, donut_id)``.

        ``candidate`` is the one selection flag: the donut passed every
        selection and quality cut.  The ``rejected_*`` booleans say why not
        Whether a fit actually consumed the donut takes two columns:

        ``group_id`` empty
            No group claimed the donut -- paired-mode surplus with no partner.
        ``group_fit_success`` False
            A group claimed it but the fit timed out or raised.

        Either way the Zernike deviations are all-NaN.  ``group_fit_outcome``
        resolves the second case further: it is one of
        `lsst.ts.wep.blitz.dataStructures._FIT_OUTCOMES` --
        ``"ok"``, ``"nonconvergent"``, ``"timeout"``, ``"exception"``,
        ``"x0_only"``, or ``""`` when no group claimed the donut, so it
        subsumes both columns above.

        Every column named ``group_*`` is a property of the joint fit, not of
        the donut, and is replicated verbatim onto each row of the group; only
        ``fit_dx``, ``fit_dy``, ``fit_flux`` and ``blend_frac`` are per-donut.

        ``defocal_offsets`` is the donut's defocal state: the optic z shifts in
        meters that put it off focus, ordered ``_OFFSET_OPTICS`` = (detector,
        camera, m2).

        Zernikes are Noll-indexed array columns: ``zk_deviation_ccs`` and
        ``zk_intrinsic_ccs`` in the camera coordinate system (as fit), plus
        ``zk_deviation_ocs`` and ``zk_intrinsic_ocs`` in the optical coordinate
        system.  The deviations stop at the highest fitted Noll index, the
        intrinsics run to ``_ZK_JMAX``.  Deviation slots below Noll 4 are 0.0
        on fitted rows and NaN on rows no fit consumed; intrinsic slots below
        Noll 4 are always 0.0.

        The two differ in how they treat a row no fit consumed.  Deviations are
        all-NaN there, because there is no measurement.  Intrinsics are still
        populated: they are a function of field position, not of the fit.

        Visit-level scalars are stored in ``table.meta``; per-detector scalars
        in ``table.meta["det_meta"]``, keyed by ``f"{det_name}_{visit_id}"``.
        Further descriptions of meta values are present in ``meta["notes"]``,
        keyed by the meta key being annotated (see `_META_NOTES`).
    """
    # Build lookup: (donut_id, det_name, visit_id) -> wf donut entry.
    wf_by_id: dict = {}
    for r in wf_results:
        for wd in r.get("donuts", []):
            wf_by_id[(wd.donut_id, wd.det_name, wd.visit_id)] = wd

    # Build lookup: "{det_name}_{visit_id}" -> per-detector metadata from cutout
    # results.  The visit is part of the key because full-array mode passes one
    # cutout result per detector *per exposure*, so det_name alone would collide
    # and drop half the metadata.
    det_meta: dict = {}
    for r in results:
        det_key = f"{r['det_name']}_{r.get('visit_id', visit_id)}"
        det_meta[det_key] = {
            "astrom_scatter": (
                r["scatter_arcsec"] if r["scatter_arcsec"] is not None else np.nan
            ) * u.arcsec,
            "wcs_refit_error": r["wcs_refit_error"],
            "cat_select_error": r["cat_select_error"],
            # Where this detector's donut ids came from. "refcat" ids are refcat
            # source ids; the blind paths number donuts 1..N per detector per
            # exposure, so a donut_id is only comparable across exposures on the refcat
            # path. "no_detections" when no selector ran at all.
            "selection_source": r["selection_source"],
            # Detector orientation: the `k` in the `np.rot90(stamp, k=-n_quarter).T`
            # that put the stamps in CCS. Per-detector, and only meaningful
            # alongside x_det/y_det, so it lives here rather than replicated onto
            # every row -- but recorded, so undoing the transform does not mean
            # loading the camera model.
            "n_quarter": r["n_quarter"],
            # Which pairing algorithm ran: "refcat_id", "spatial" or "empty" in
            # full-array mode, "snr_rank" in corner mode, "n/a" in the modes that
            # do not pair.
            "pair_path": r["pair_path"],
            **{
                key: r.get(key, np.nan) * u.s
                for key in _CUTOUT_STAGE_KEYS.values()
            },
        }

    # Collect every donut exactly once, tagged with whether it passed
    # selection ("candidate"). Whether a fit actually consumed it comes from the
    # wavefront result per row below (group_id / group_fit_success).
    #
    # `donuts` and `unmatched_donuts` overlap: the surplus donuts on whichever
    # side detected more pass selection but have no partner, so they appear in
    # both lists. Keyed dedupe keeps one row per donut -- they stay candidates,
    # they just never got fitted.
    def _key(d):
        return (d.visit_id, d.det_name, d.donut_id)

    all_donuts = []
    _seen = set()
    for d, candidate in (
        [(d, True) for d in donuts]
        + [
            (d, False)
            for r in results
            for d in r.get("rejected_catalog", [])
        ]
        + [(d, True) for d in unmatched_donuts]
    ):
        k = _key(d)
        if k in _seen:
            continue
        _seen.add(k)
        all_donuts.append((d, candidate))

    if not all_donuts:
        return QTable()

    stamp_size = options.stamp_size
    wf_img_size = options.wf_img_size
    zk_deviation_jmax = options.zk_deviation_jmax

    rows = []
    zk_deviation_rows = []
    zk_int_rows = []
    for d, candidate in all_donuts:
        sid = d.donut_id
        # `_NULL_WF` carries group_id "", the "no fit claimed this donut" marker.
        wd = wf_by_id.get((sid, d.det_name, d.visit_id), _NULL_WF)

        # Both are dense Noll-indexed arrays in meters of length _ZK_JMAX + 1;
        # they become the zk_*_ccs array columns after the loop.
        zk_deviation_rows.append(wd.zk_dev[: zk_deviation_jmax + 1])
        # Deviations only exist where a fit ran, but intrinsics are a
        # function of field position alone, so a row no fit consumed still has
        # them -- take them off the donut rather than inheriting `_NULL_WF`'s
        # all-NaN. Donuts whose intrinsic calib was missing have
        # `intrinsic_zk is None` and get zeros, as an unsupplied index does.
        zk_int_rows.append(
            wd.zk_intrinsic if wd is not _NULL_WF else _dense_intrinsic(d)
        )

        # Image columns are skipped entirely when not saving, so we do not pay
        # for float64 copies of columns that are about to be discarded.
        stamp = None
        if options.save_stamps:
            stamp = (
                d.stamp.astype(float)
                if d.stamp is not None
                else np.full((stamp_size, stamp_size), np.nan, dtype=float)
            )

        wf_img = model_img = None
        if options.save_wf_images:
            # Donuts no fit consumed (surplus) have no WF image from the fitter,
            # so bin their stamp here with the same prep the fitter would have
            # applied. Keeps them plottable as data-only rows.
            if wd.img is not None:
                wf_img = wd.img.astype(float)
            elif d.stamp is not None:
                wf_img = _bin_stamp_odd(d.stamp, options.binning)
            else:
                wf_img = np.full((wf_img_size, wf_img_size), np.nan, dtype=float)

            model_img = (
                wd.model_img.astype(float)
                if wd.model_img is not None
                else np.full((wf_img_size, wf_img_size), np.nan, dtype=float)
            )

        nb_photo_dx, nb_photo_dy, nb_photo_mag = _encode_nearby(d.nearby_photo)
        nb_astrom_dx, nb_astrom_dy, nb_astrom_mag = _encode_nearby(d.nearby_astrom)
        row = {
            # --- identity ---
            "visit_id": d.visit_id,
            "det_id": d.det_id,
            "det_name": d.det_name,
            "donut_id": sid,
            "band": d.band,
            "candidate": bool(candidate), # Passed every selection/quality cut.
            # --- geometry ---
            "x_det": d.x_det * u.pix,
            "y_det": d.y_det * u.pix,
            "thx_ccs": d.thx_ccs * u.rad,
            "thy_ccs": d.thy_ccs * u.rad,
            "defocal_offsets": _defocal_offsets(d) * u.m,
            # --- this donut's own refcat values (NaN off the refcat path) ---
            "photo_mag": d.photo_mag * u.mag,
            "astrom_mag": d.astrom_mag * u.mag,
            # Refcat truth.  NaN rather than approximate wherever there was no refcat.
            "coord_ra": np.degrees(d.coord_ra) * u.deg,
            "coord_dec": np.degrees(d.coord_dec) * u.deg,
            # --- nearby refcat sources (brightest-first, padded to _MAX_NEARBY) ---
            # Excludes this donut itself, so a count of 0 means genuinely
            # isolated within the stamp box.
            "nearby_photo_dx_det": nb_photo_dx * u.pix,
            "nearby_photo_dy_det": nb_photo_dy * u.pix,
            "nearby_photo_mag": nb_photo_mag * u.mag,
            "nearby_astrom_dx_det": nb_astrom_dx * u.pix,
            "nearby_astrom_dy_det": nb_astrom_dy * u.pix,
            "nearby_astrom_mag": nb_astrom_mag * u.mag,
            "n_nearby_photo": len(d.nearby_photo),
            "n_nearby_astrom": len(d.nearby_astrom),
            # --- selection metrics ---
            "flux": d.flux,
            "snr": d.snr,
            "inner_frac": d.inner_frac,
            "outer_frac": d.outer_frac,
            "outer_sector_minmax_frac": d.outer_sector_minmax_frac,
            "bkg": d.bkg,
            "bkg_std": d.bkg_std,
            "donut_radius": d.donut_radius * u.pix,
            "rejected_sat": d.rejected_sat,
            "rejected_inner_frac": d.rejected_inner_frac,
            "rejected_outer_frac": d.rejected_outer_frac,
            "rejected_snr": d.rejected_snr,
            # --- fit results ---
            # group_*: a property of the joint fit, replicated onto every row of
            # the group. Dedupe on group_id before averaging or summing these.
            "group_id": wd.group_id,
            "group_size": wd.group_size,
            "group_fit_success": wd.fit_success,
            "group_fit_elapsed": wd.fit_elapsed * u.s,
            "group_setup_elapsed": wd.setup_elapsed * u.s,
            "group_fit_nfev": wd.fit_nfev,
            "group_fit_njev": wd.fit_njev,
            "group_fit_cost": wd.fit_cost,
            "group_fit_optimality": wd.fit_optimality,
            "group_fit_outcome": wd.fit_outcome,
            "group_fwhm": wd.fit_fwhm * u.arcsec,
            # The rest are per-donut, even inside a joint fit.
            "fit_dx": wd.fit_dx * u.arcsec,
            "fit_dy": wd.fit_dy * u.arcsec,
            "fit_flux": wd.fit_flux,
            "blend_frac": wd.blend_frac,
            # Zernikes are attached after construction.
            # --- embedded images, both optional ---
            **({"stamp": stamp} if options.save_stamps else {}),
            **(
                {"wf_img": wf_img, "model_img": model_img}
                if options.save_wf_images
                else {}
            ),
        }
        rows.append(row)

    table = QTable(rows)
    # Zernikes are Noll-indexed along axis 1 the way galsim orders Zernike
    # coefficients: [:, j] is Noll j across donuts and [i] is donut i's
    # coefficient vector. Slots below Noll 4 are carried for indexing only:
    # 0.0 for intrinsics, and for deviations 0.0 on fitted rows (`_dense_dev`)
    # but NaN on rows no fit consumed (`_NULL_WF` is all-NaN).
    zk_deviation_um = np.array(zk_deviation_rows) * 1e6
    zk_int_um = np.array(zk_int_rows) * 1e6
    # Camera coordinate system, i.e. as fit.
    table["zk_deviation_ccs"] = zk_deviation_um * u.micron
    table["zk_intrinsic_ccs"] = zk_int_um * u.micron
    # transform_eb is unit-agnostic in the field angle -- it only ever takes
    # an atan2 of the pair -- so hand it bare radians rather than teach it
    # to strip units from arbitrary angle Quantities.
    dev_eb, intrinsic_eb = transform_eb(
        [table["zk_deviation_ccs"], table["zk_intrinsic_ccs"]],
        table["thx_ccs"].to_value(u.rad), table["thy_ccs"].to_value(u.rad)
    )
    table["zk_deviation_eb"] = dev_eb
    table["zk_intrinsic_eb"] = intrinsic_eb
    # Optical coordinate system: the camera frame rotated by -rotTelPos, so
    # m != 0 terms are comparable across visits taken at different rotator
    # angles.
    table["thx_ocs"] = np.cos(rtp_rad) * table["thx_ccs"] - np.sin(rtp_rad) * table["thy_ccs"]
    table["thy_ocs"] = np.sin(rtp_rad) * table["thx_ccs"] + np.cos(rtp_rad) * table["thy_ccs"]
    table["zk_deviation_ocs"] = _rotate_zk(zk_deviation_um, -rtp_rad) * u.micron
    table["zk_intrinsic_ocs"] = _rotate_zk(zk_int_um, -rtp_rad) * u.micron
    # The visit this table is *for*. Full-array mode has two visits in the
    # visit_id column (one per side of focus) and this is the extra-focal one,
    # so it is deliberately not named visit_id: with group_id it forms the
    # cross-visit key for a fit, which the row's own visit_id cannot.
    table.meta["ref_visit_id"] = visit_id
    # Which visit supplied each side of focus.  In corner mode they are both this visit.
    table.meta["intra_visit_id"] = visit_id if intra_visit_id is None else intra_visit_id
    table.meta["extra_visit_id"] = visit_id if extra_visit_id is None else extra_visit_id
    # The butler group dimension value.
    table.meta["exposure_group"] = str(exposure_group)
    # Which mode produced this table. Also the key to reading the elapsed
    # values below; see meta["notes"].
    table.meta["mode"] = str(mode)
    table.meta["run_elapsed"] = run_elapsed * u.s
    table.meta["refcat_elapsed"] = refcat_elapsed * u.s
    table.meta["butler_elapsed"] = butler_elapsed * u.s
    table.meta["butler_times"] = {
        key: value * u.s for key, value in (butler_times or {}).items()
    }
    table.meta["cutout_elapsed"] = cutout_elapsed * u.s
    table.meta["danish_elapsed"] = danish_elapsed * u.s
    table.meta["photo_filter_name"] = photo_filter_name
    table.meta["astrom_filter_name"] = astrom_filter_name
    table.meta["noll_indices"] = list(options.noll_indices)
    # Needed by the plots to convert raw-pixel quantities (aperture radii,
    # refcat offsets) when they fall back to the binned wf_img.
    table.meta["binning"] = options.binning
    # `stamp` is in CCS, while x_det/y_det and the nearby_* offsets are in the
    # detector frame; relating them needs the transform noted above.
    table.meta["stamp_frame"] = "CCS"
    # The side of the stamp box, which is what n_nearby_photo and
    # n_nearby_astrom counted sources over. Those two columns are written
    # whatever `save_stamps` is set to, so without this their counts have no
    # area to be compared against. Not needed to interpret the `stamp` column
    # itself: the array columns are uniform by construction and survive the
    # ArrowAstropy round trip 2-D, so a present stamp carries its own shape.
    # wf_img_size is deliberately absent for that reason -- nothing outside
    # wf_img/model_img depends on it, and it is stamp_size // binning odd.
    table.meta["stamp_size"] = options.stamp_size
    # The nearby_* array length, otherwise a hardcoded 5 consumers must guess.
    table.meta["max_nearby"] = _MAX_NEARBY
    table.meta["zk_deviation_jmax"] = zk_deviation_jmax
    table.meta["zk_intrinsic_jmax"] = _ZK_JMAX
    # rotTelPos: the *derived* angle the _ocs columns were rotated by. The two
    # raw boresight angles it comes from are in the observation block below;
    # this is the one that describes the data, so it keeps the distinct name.
    table.meta["rot_tel_pos"] = (rtp_rad * u.rad).to(u.deg)
    # --- observation and provenance ---
    # Without these the catalog joins to nothing without going back to the
    # butler, which for the pipeline's final data product is the point.
    table.meta["instrument"] = str(instrument)
    table.meta.update(_observation_meta(visit_info))
    table.meta.update(_software_versions())
    table.meta["det_meta"] = det_meta
    table.meta["aperture_margin_frac"] = options.aperture_margin_frac
    table.meta["bkg_inner_disc_frac"] = options.bkg_inner_disc_frac
    table.meta["bkg_annulus_inner_frac"] = options.bkg_annulus_inner_frac
    table.meta["bkg_annulus_outer_frac"] = options.bkg_annulus_outer_frac
    # Global instrument constant, so it lives in meta.
    table.meta["obscuration"] = _INSTRUMENT.obscuration
    table.meta["max_donuts"] = options.max_donuts
    table.meta["wf_mode"] = options.wf_mode
    # The values above carry their own units; these are the few facts a unit
    # cannot express, keyed by the meta key they annotate.
    table.meta["notes"] = dict(_META_NOTES)
    return table
