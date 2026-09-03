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
task -- a divergence here would show up as two subtly different `blitzResults`
flavours for downstream consumers to discover the hard way. Everything the
builder used to read off the task's config is now passed in as `CatalogOptions`.
"""

__all__ = ["CatalogOptions", "build_donut_catalog", "build_noll_pairs", "transform_eb"]

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import QTable
from galsim.zernike import noll_to_zern

from .dataStructures import _NULL_WF
from .utils import _MAX_NEARBY, _ZK_JMAX, _bin_stamp_odd, _rotate_zk


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
    aperture_outer_margin_frac : float
        Outer edge of the photometric aperture, from the measurement subtask.
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
    aperture_outer_margin_frac: float
    bkg_inner_disc_frac: float
    bkg_annulus_inner_frac: float
    bkg_annulus_outer_frac: float
    max_donuts: int
    wf_mode: str
    save_stamps: bool = True
    save_wf_images: bool = True

    @property
    def zk_dev_jmax(self) -> int:
        """Highest fitted Noll index; deviations are cut off here."""
        return max(self.noll_indices)

    @property
    def wf_img_size(self) -> int:
        """Binned WF image side, forced odd (see `_prep_donut_for_danish`)."""
        binned = self.stamp_size // self.binning
        return binned if binned % 2 == 1 else binned - 1


def _encode_nearby(entries):
    """Return (x, y, mag) arrays of length ``_MAX_NEARBY`` for one donut.

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


def build_donut_catalog(
    results: list,
    wf_results: list,
    donuts: list,
    unmatched_donuts: list,
    visit_id: int,
    options: CatalogOptions,
    run_elapsed: float = 0.0,
    refcat_elapsed: float = 0.0,
    butler_elapsed: float = 0.0,
    butler_times: dict | None = None,
    cutout_elapsed: float = 0.0,
    danish_elapsed: float = 0.0,
    photo_filter_name: str = "",
    astrom_filter_name: str = "",
    rtp_rad: float = 0.0,
) -> QTable:
    """Build a per-donut QTable covering every donut cut from this visit.

    Parameters
    ----------
    results : list
        Per-detector cutout dicts (supplies rejected donuts and per-detector
        metadata).
    wf_results : list
        Per-fit WF result dicts from the WF worker pool.
    donuts : list
        Donut records that passed selection.
    unmatched_donuts : list
        Donut records with no intra/extra partner.  These also appear in
        ``donuts``; the table carries one row per donut.
    visit_id : int
        Visit identifier.
    options : CatalogOptions
        Config-derived scalars from the calling task.
    rtp_rad : float
        Camera rotator angle on sky (rotTelPos) in radians, used to rotate the
        Zernikes from the camera into the optical coordinate system.

    Returns
    -------
    QTable
        Exactly one row per donut, keyed by ``(det_name, id)``, with two
        independent flags:

        ``candidate``
            The donut passed every selection and quality cut.  See
            ``reject_reasons`` and the ``rejected_*`` booleans for why not.
        ``used``
            A wavefront fit consumed the donut and returned a result.  A
            candidate can be unused either because no group claimed it
            (surplus donut with no partner: ``fit_mode`` empty, ``group`` -1)
            or because its fit timed out or raised (``fit_success`` False).
            Unused rows have all-NaN Zernikes.

        Zernikes are Noll-indexed array columns in um: ``zk_dev_ccs`` and
        ``zk_intrinsic_ccs`` in the camera coordinate system (as fit), plus
        ``zk_dev_ocs`` and ``zk_intrinsic_ocs`` in the optical coordinate
        system (the camera frame rotated by ``-rtp_rad``).  The deviations stop
        at the highest fitted Noll index, the intrinsics run to ``_ZK_JMAX``.

        Array columns (``stamp``, ``model_img``, ``wf_img``) are zero-padded to
        a common shape.  ``stamp`` is present only when ``save_stamps`` is set,
        and ``wf_img``/``model_img`` only when ``save_wf_images`` is; both
        default on for corner mode and off for full-array mode, where the row
        count makes them dominate the file.  When present, ``wf_img`` is filled
        for every donut with a stamp -- from the fitter when a fit ran,
        otherwise by binning the stamp the same way -- so only ``model_img`` is
        all-NaN for unused donuts.  Visit-level and per-detector scalars are
        stored in ``table.meta``.
    """
    # Build lookup: (id, det_name) -> (wf donut entry, group index).
    # Key on both fields to handle the same refcat star on two detectors.
    wf_by_id: dict = {}
    for group_idx, r in enumerate(wf_results):
        for wd in r.get("donuts", []):
            wf_by_id[(wd.donut_id, wd.det_name)] = (wd, group_idx)

    # Build lookup: det_name -> per-detector metadata from cutout results.
    det_meta: dict = {}
    rejected_by_det: dict = {}
    for r in results:
        dname = str(r["det_name"])
        det_meta[dname] = {
            "scatter_arcsec": (
                r["scatter_arcsec"] if r["scatter_arcsec"] is not None else float("nan")
            ),
            "wcs_refit_error": r["wcs_refit_error"],
            "cat_select_error": r["cat_select_error"],
            "isr_run": r.get("isr_run", float("nan")),
            "bkg_run": r.get("bkg_run", float("nan")),
            "diam_run": r.get("diam_run", float("nan")),
            "blind_detect_run": r.get("blind_detect_run", float("nan")),
            "wcs_refit_run": r.get("wcs_refit_run", float("nan")),
            "catalog_select_run": r.get("catalog_select_run", float("nan")),
        }
        rejected_by_det[dname] = r.get("rejected_catalog", [])

    # Collect every donut exactly once, tagged with whether it passed
    # selection ("candidate"). Whether a fit actually consumed it ("used")
    # is derived per row below from the wavefront result.
    #
    # `donuts` and `unmatched_donuts` overlap: the surplus donuts on whichever
    # side detected more pass selection but have no partner, so they appear in
    # both lists. Keyed dedupe keeps one row per donut -- they stay candidates,
    # they just never got fitted.
    def _key(d):
        return (d.det_name, d.id)

    all_donuts = []
    _seen = set()
    for d, candidate in (
        [(d, True) for d in donuts]
        + [
            (d, False)
            for r in results
            for d in rejected_by_det.get(str(r["det_name"]), [])
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
    zk_dev_jmax = options.zk_dev_jmax

    rows = []
    zk_dev_rows = []
    zk_int_rows = []
    for d, candidate in all_donuts:
        sid = d.id
        wd, grp = wf_by_id.get((sid, d.det_name), (_NULL_WF, -1))

        # Both are dense Noll-indexed arrays in meters of length _ZK_JMAX + 1;
        # they become the zk_*_ccs array columns after the loop.
        zk_dev_rows.append(wd.zk_dev[: zk_dev_jmax + 1])
        zk_int_rows.append(wd.zk_intrinsic)

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

        photo_x, photo_y, photo_mag = _encode_nearby(d.nearby_photo)
        astrom_x, astrom_y, astrom_mag = _encode_nearby(d.nearby_astrom)
        row = {
            # --- identity ---
            "visit_id": d.visit_id,
            "det_id": d.det_id,
            "det_name": d.det_name,
            "id": sid,
            "band": d.band,
            # candidate: passed every selection/quality cut.
            # used: a fit consumed it and returned a wavefront.
            # A candidate that is not used has no Zernikes (all NaN) -- see
            # fit_mode/group/fit_success for which of the two reasons.
            "candidate": bool(candidate),
            "used": bool(wd.fit_success),
            # --- geometry ---
            "centroid_x_raw": d.centroid_x_raw,
            "centroid_y_raw": d.centroid_y_raw,
            "thx_ccs": d.thx_ccs,
            "thy_ccs": d.thy_ccs,
            "field_dist_deg": np.degrees(np.hypot(d.thx_ccs, d.thy_ccs)),
            "n_quarter": d.n_quarter,
            # --- nearby refcat sources (brightest-first, padded to _MAX_NEARBY) ---
            "nearby_photo_x": photo_x * u.pix,
            "nearby_photo_y": photo_y * u.pix,
            "nearby_photo_mag": photo_mag * u.mag,
            "nearby_astrom_x": astrom_x * u.pix,
            "nearby_astrom_y": astrom_y * u.pix,
            "nearby_astrom_mag": astrom_mag * u.mag,
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
            "donut_radius": d.donut_radius,
            "obscuration": d.obscuration,
            "nearest_neighbor_dist_px": d.nearest_neighbor_dist_px,
            "n_neighbors_in_stamp": d.n_neighbors_in_stamp,
            "catalog_centroid_offset_px": d.catalog_centroid_offset_px,
            "rejected_sat": d.rejected_sat,
            "rejected_inner_frac": d.rejected_inner_frac,
            "rejected_outer_frac": d.rejected_outer_frac,
            "rejected_snr": d.rejected_snr,
            "rejected": d.rejected,
            # --- fit results ---
            "fit_mode": wd.fit_mode,
            "group": grp,
            "group_size": wd.group_size,
            "fit_success": wd.fit_success,
            "fit_elapsed": wd.fit_elapsed,
            "setup_elapsed": wd.setup_elapsed,
            "fit_nfev": wd.fit_nfev,
            "fit_cost": wd.fit_cost,
            "fit_dx": wd.fit_dx,
            "fit_dy": wd.fit_dy,
            "fit_flux": wd.fit_flux,
            "fit_fwhm": wd.fit_fwhm,
            "blend_frac": wd.blend_frac,
            # Zernikes are attached after construction as the array columns
            # zk_dev_ccs / zk_intrinsic_ccs.
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
    # coefficient vector. Slots below Noll 4 are carried for indexing only
    # (NaN for deviations, 0 for intrinsics).
    zk_dev_um = np.array(zk_dev_rows) * 1e6
    zk_int_um = np.array(zk_int_rows) * 1e6
    # Camera coordinate system, i.e. as fit.
    table["zk_dev_ccs"] = zk_dev_um * u.micron
    table["zk_intrinsic_ccs"] = zk_int_um * u.micron
    dev_eb, intrinsic_eb = transform_eb(
        [table["zk_dev_ccs"], table["zk_intrinsic_ccs"]],
        table["thx_ccs"], table["thy_ccs"]
    )
    table["zk_dev_eb"] = dev_eb
    table["zk_intrinsic_eb"] = intrinsic_eb
    # Optical coordinate system: the camera frame rotated by -rotTelPos, so
    # m != 0 terms are comparable across visits taken at different rotator
    # angles.
    table["thx_ocs"] = np.cos(rtp_rad) * table["thx_ccs"] - np.sin(rtp_rad) * table["thy_ccs"]
    table["thy_ocs"] = np.sin(rtp_rad) * table["thx_ccs"] + np.cos(rtp_rad) * table["thy_ccs"]
    table["zk_dev_ocs"] = _rotate_zk(zk_dev_um, -rtp_rad) * u.micron
    table["zk_intrinsic_ocs"] = _rotate_zk(zk_int_um, -rtp_rad) * u.micron
    table.meta["visit_id"] = visit_id
    table.meta["run_elapsed"] = run_elapsed
    table.meta["refcat_elapsed"] = refcat_elapsed
    table.meta["butler_elapsed"] = butler_elapsed
    table.meta["butler_times"] = dict(butler_times or {})
    table.meta["cutout_elapsed"] = cutout_elapsed
    table.meta["danish_elapsed"] = danish_elapsed
    table.meta["photo_filter_name"] = photo_filter_name
    table.meta["astrom_filter_name"] = astrom_filter_name
    table.meta["noll_indices"] = list(options.noll_indices)
    # Needed by the plots to convert raw-pixel quantities (aperture radii,
    # refcat offsets) when they fall back to the binned wf_img.
    table.meta["binning"] = options.binning
    table.meta["zk_dev_jmax"] = zk_dev_jmax
    table.meta["zk_jmax"] = _ZK_JMAX
    table.meta["rot_tel_pos"] = np.degrees(rtp_rad)
    table.meta["det_meta"] = det_meta
    table.meta["aperture_outer_margin_frac"] = options.aperture_outer_margin_frac
    table.meta["bkg_inner_disc_frac"] = options.bkg_inner_disc_frac
    table.meta["bkg_annulus_inner_frac"] = options.bkg_annulus_inner_frac
    table.meta["bkg_annulus_outer_frac"] = options.bkg_annulus_outer_frac
    table.meta["max_donuts"] = options.max_donuts
    table.meta["wf_mode"] = options.wf_mode
    return table
