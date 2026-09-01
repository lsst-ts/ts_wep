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

"""Danish wavefront fitting: donut grouping, the fit task, and its worker."""

__all__ = ["WavefrontFittingTaskConfig", "WavefrontFittingTask"]

import ast
import contextlib
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

import batoid
import danish
import galsim
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .dataStructures import Donut, WfResult, _WfGroup
from .utils import (
    _CALIB_STORE,
    _EXTRA_FOCAL_DET_IDS,
    _INSTRUMENT,
    _INTRA_FOCAL_DET_IDS,
    _ZK_JMAX,
    _bin_stamp_odd,
    CORNER_PAIRS,
)

# DZMultiDonutModel's field_radius has no effect on our fit (we don't model any
# field-dependent optics term), so any value works; set to roughly the Rubin
# field of view for a physically sensible default.
_DANISH_FIELD_RADIUS_RAD = np.deg2rad(1.85)


class _WfFitTimeoutError(Exception):
    pass


@contextlib.contextmanager
def _fit_timeout(seconds):
    """SIGALRM-based timeout context manager.

    Works on any POSIX platform, macOS included; it is Windows that has no
    ``SIGALRM``.  The real constraint is threads: ``signal.signal`` raises
    ``ValueError: signal only works in main thread of the main interpreter``,
    so this is usable from the WF pool (each forked worker runs the fit on its
    own main thread) but would break if fitting moved to a thread pool.
    """

    def _handler(_signum, _frame):
        raise _WfFitTimeoutError(f"WF fit exceeded {seconds:.0f}s timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _bkg_free_model(
    model_img: np.ndarray, danish_model, fit_params, donut_idx: int, bkg_order: int
) -> np.ndarray:
    """Return model_img with the fitted background subtracted.

    fit_params is the dict returned by model.unpack_params(), or None on fit
    failure.  For bkgOrder <= 0 (constant or no background) this is a cheap
    scalar subtraction.  For bkgOrder > 0 we re-evaluate the model with zeroed
    backgrounds, relying on Danish's cached optics component.
    """
    if model_img is None:
        return None
    if fit_params is None or bkg_order < 0:
        return model_img
    if bkg_order == 0:
        bkgs = fit_params.get("bkgs") or fit_params.get("bkg")
        if bkgs is None:
            return model_img
        if np.ndim(bkgs[0]) > 0:
            bkg_val = np.ravel(bkgs[donut_idx])[0]
        else:
            bkg_val = bkgs[0]
        return model_img - bkg_val
    # bkgOrder > 0: re-evaluate with zeroed backgrounds
    try:
        is_multi = "fluxes" in fit_params
        nbkg = danish_model.nbkg
        zero_bkgs = [[0.0] * nbkg] * len(fit_params["fluxes"]) if is_multi else [0.0] * nbkg
        bkg_key = "bkgs" if is_multi else "bkg"
        kw = {k: v for k, v in fit_params.items() if k not in ("bkgs", "bkg")}
        kw[bkg_key] = zero_bkgs
        result = danish_model.model(**kw)
        if isinstance(result, list):
            return result[donut_idx]
        return result
    except Exception:
        return model_img


def _blend_frac(
    resid: np.ndarray,
    model_img_bkg_free: np.ndarray,
    bkg_std: float,
    faint_frac: float = 0.05,
    sig_thresh: float = 2.0,
) -> float:
    """Fraction of significant residual flux in model-faint pixels, normalised by total model flux."""
    if resid is None or model_img_bkg_free is None or not np.isfinite(bkg_std) or bkg_std <= 0:
        return float("nan")
    model_peak = np.nanmax(model_img_bkg_free)
    total_model_flux = np.sum(model_img_bkg_free[model_img_bkg_free > 0])
    if model_peak <= 0 or total_model_flux <= 0:
        return float("nan")
    faint_mask = model_img_bkg_free < faint_frac * model_peak
    sig_mask = np.abs(resid) > sig_thresh * bkg_std
    return np.sum(np.abs(resid[faint_mask & sig_mask])) / total_model_flux


def _dense_intrinsic(donut: dict) -> np.ndarray:
    """Return intrinsic Zernikes in metres, dense over Noll 0..``_ZK_JMAX``.

    Indices with no supplied value are 0.0.
    """
    out = np.zeros(_ZK_JMAX + 1)
    raw = donut.intrinsic_zk  # µm, Noll 4.._ZK_JMAX
    if raw is not None:
        n_slots = _ZK_JMAX + 1 - 4  # Noll 4.._ZK_JMAX inclusive
        for idx in range(min(len(raw), n_slots)):
            out[idx + 4] = raw[idx] * 1e-6
    return out


def _dense_dev(zk_dev: np.ndarray, nollIndices) -> np.ndarray:
    """Return deviations in metres, dense over Noll 0..``_ZK_JMAX``.

    Indices that were not fitted are ``np.nan``.
    """
    out = np.full(_ZK_JMAX + 1, np.nan)
    out[0:4] = 0.0
    for k, j in enumerate(nollIndices):
        if k < len(zk_dev):
            out[j] = zk_dev[k]
    return out


def _build_wf_groups(mode, results_by_det, band: str, rtp_deg: float | None, boresight_alt_rad: float | None):
    """Build _WfGroup list from per-detector catalogs, matching the mode dispatch logic.

    Groups for ``"paired"`` hold one extra/intra pair; all other modes group
    without regard to defocal type -- the invariant reported by
    `_mode_groups_are_pairs`.

    ``results_by_det`` covers only the detectors that were processed, so a
    partial corner set falls out naturally: groups are never emitted empty, and
    ``"full_corner"`` fits a corner from one defocal side alone when that is all
    that is present.

    Returns (groups, unmatched_donuts).
    """
    groups = []
    unmatched_donuts = []
    if mode == "paired":
        for _corner, (sw0, sw1) in CORNER_PAIRS.items():
            extra_donuts = sorted(results_by_det.get(sw0, []), key=lambda d: d.snr, reverse=True)
            intra_donuts = sorted(results_by_det.get(sw1, []), key=lambda d: d.snr, reverse=True)
            for extra, intra in zip(extra_donuts, intra_donuts):
                gid = f"{extra.id}_{intra.id}"
                groups.append(_WfGroup(donuts=[extra, intra], group_id=gid, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
            n_pairs = min(len(extra_donuts), len(intra_donuts))
            unmatched_donuts.extend(extra_donuts[n_pairs:])
            unmatched_donuts.extend(intra_donuts[n_pairs:])
    elif mode == "unpaired":
        for det_donuts in results_by_det.values():
            for d in det_donuts:
                groups.append(_WfGroup(donuts=[d], group_id=str(d.id), band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    elif mode == "full_detector":
        # Skip detectors with no donuts: an empty group fits nothing but still
        # reports success=False, which would skew the caller's success tally.
        for det_name, det_donuts in results_by_det.items():
            if not det_donuts:
                continue
            groups.append(_WfGroup(donuts=det_donuts, group_id=det_name, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    elif mode == "full_corner":
        # A corner contributes whichever of its two detectors have donuts; one
        # defocal side alone is still fit. Corners with neither are skipped.
        for corner, (sw0, sw1) in CORNER_PAIRS.items():
            all_donuts = results_by_det.get(sw0, []) + results_by_det.get(sw1, [])
            if not all_donuts:
                continue
            groups.append(_WfGroup(donuts=all_donuts, group_id=corner, band=band, rtp=rtp_deg, alt=boresight_alt_rad))
    else:
        raise ValueError(f"Unknown WF mode {mode!r}")
    return groups, unmatched_donuts

# Module-level logger for the worker functions below. They are module-level
# (not methods) so the fork-based pools can pickle them by name, which means
# there is no `self` and so no `Task.log`. Consequence: worker output goes to
# the `lsst.ts.wep.task.donutBlitzMonolith` logger rather than the task's own
# `donutBlitzMonolithTask` hierarchy, so it is not affected by that task's log
# level. Parent-process code should keep using `self.log`.
_log = logging.getLogger(__name__)

_DZ_MODEL_KEYS = ("fluxes", "dxs", "dys", "fwhm", "wavefront_params", "bkgs")


def _wf_fitting_worker(group: "_WfGroup") -> dict:
    """Wavefront fitting worker for multiprocessing pool.

    Retrieves the WavefrontFittingTask from _CALIB_STORE and calls it
    on the group. Designed to be used with multiprocessing.Pool.map().
    """
    task = _CALIB_STORE["wf_fitting_task"]
    result = task.run(group)
    # Set fit_mode on each donut result from main config
    fit_mode = _CALIB_STORE["wfEstimationMode"]
    for wd in result.get("donuts", []):
        wd.fit_mode = fit_mode
    return result


class WavefrontFittingTaskConfig(pexConfig.Config):
    """Configuration for wavefront fitting via Danish algorithm."""

    nollIndices: pexConfig.ListField = pexConfig.ListField(
        dtype=int,
        doc="Noll indices to fit with Danish.",
        default=list(range(4, 20)) + list(range(22, 27)),
    )
    lstsqKwargs: pexConfig.DictField = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        doc=(
            "Keyword arguments for scipy.optimize.least_squares passed to the Danish "
            "WF workers. Values are strings that will be eval()'d, e.g. "
            "{'method': 'trf', 'max_nfev': '200'}."
        ),
        default={
            "xtol": "1e-3",
            "ftol": "1e-3",
            "gtol": "1e-3",
            "x_scale": "'jac'",
        },
    )
    binning: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Binning factor applied to donut stamps before Danish fitting.",
    )
    modelSpiderShadows: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Include spider shadow modeling in Danish forward model.",
    )
    bkgOrder: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Background polynomial order for Danish (-1=none, 0=constant).",
    )
    doAoiThroughput: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply angle-of-incidence throughput correction in Danish forward model.",
    )
    systematicLossAlpha: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Fractional systematic uncertainty for Danish loss function (0=chi2).",
    )
    triangleMode: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Use DonutTriangleFactory instead of DonutFactory.",
    )
    wfFitTimeoutPerDonut: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=10.0,
        doc=(
            "Timeout in seconds per donut for a single WF fit. "
            "The total timeout for a work unit is this value times the number of donuts "
            "(1 for unpaired, 2 for paired, N for full_corner). "
            "Fits exceeding the limit are killed and return NaN Zernikes."
        ),
    )
    wfInitialGuessOnly: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=(
            "Skip the least-squares fit and return the initial-guess (x0) model only. "
            "Useful for diagnosing stamp orientation and model setup without "
            "waiting for convergence."
        ),
    )

    def validate(self):
        super().validate()
        indices = set(self.nollIndices)
        out_of_range = sorted(j for j in indices if j < 4 or j > _ZK_JMAX)
        if out_of_range:
            raise pexConfig.FieldValidationError(
                self.__class__.nollIndices, self,
                f"nollIndices must lie in 4..{_ZK_JMAX} (the dense Zernike arrays "
                f"reported by the catalog are sized to {_ZK_JMAX}); "
                f"got {out_of_range}",
            )
        # Rotating Zernikes between coordinate frames (CCS -> OCS) mixes each
        # (n, +m) coefficient with its (n, -m) partner, so a lone half of a pair
        # cannot be rotated: its rotated power belongs to a term that was never
        # fit. Requiring whole pairs keeps every reported frame well defined.
        missing = []
        for j in sorted(indices):
            n, m = galsim.zernike.noll_to_zern(j)
            if m == 0:
                continue
            partner = j + 1 if galsim.zernike.noll_to_zern(j + 1) == (n, -m) else j - 1
            if partner not in indices:
                missing.append((j, partner))
        if missing:
            raise pexConfig.FieldValidationError(
                self.__class__.nollIndices, self,
                "nollIndices must contain both halves of every +/-m Zernike pair so "
                "the coefficients can be rotated between coordinate frames; missing "
                + ", ".join(f"{p} (partner of {j})" for j, p in missing),
            )


@dataclass
class _LstsqFitResult:
    # All outputs from _run_lstsq_fit as first-class typed fields.
    zk_dev: npt.NDArray[np.float64]
    model_imgs: list
    blend_fracs: list
    success: bool
    elapsed: float
    fluxes: list
    dxs: list
    dys: list
    fwhm: float
    nfev: int = 0
    cost: float = float("nan")
    optimality: float = float("nan")
    njev: int = 0
    status: int = 0
    message: str = ""
    error: str = ""


class WavefrontFittingTask(pipeBase.Task):
    """Task to fit wavefront aberrations from grouped donut stamps using Danish algorithm.

    This task takes a pre-grouped collection of donuts (a _WfGroup) and performs
    joint wavefront fitting across all donuts in the group, returning Zernike
    coefficients for the wavefront error.
    """

    ConfigClass = WavefrontFittingTaskConfig
    _DefaultName = "wavefrontFittingTask"
    config: WavefrontFittingTaskConfig

    def run(self, group: "_WfGroup") -> dict:
        """Fit wavefront aberrations for a group of donuts.

        Parameters
        ----------
        group : _WfGroup
            Pre-grouped collection of donuts to fit jointly, including
            band, rtp, and alt for exposure-dependent parameters.

        Returns
        -------
        result : dict
            Dictionary containing:
            - group_id: str
            - group_size: int
            - zk_dev: ndarray
            - success: bool
            - fit_info: dict
            - donuts: list of WfResult
            - model_imgs: list or None
            - imgs: list
            - det_names: list of str
        """
        nollIndices = self.config.nollIndices
        all_donuts = group.donuts
        n = len(all_donuts)

        if not all_donuts:
            return {
                "group_id": group.group_id,
                "group_size": 0,
                "zk_dev": np.full(len(nollIndices), np.nan),
                "success": False,
                "fit_info": {},
                "donuts": [],
                "model_imgs": None,
                "imgs": [],
                "det_names": [],
            }

        t_setup0 = time.perf_counter()
        factory = self._build_wf_factory(group)
        preps = [self._prep_donut_for_danish(d) for d in all_donuts]
        imgs = [p[0] for p in preps]
        thxs = [p[1][0] for p in preps]
        thys = [p[1][1] for p in preps]
        zk_refs = [p[2] for p in preps]
        sky_lvl = [p[3] for p in preps]
        dz_terms = [(1, j) for j in nollIndices]

        npix = min(img.shape[0] for img in imgs)
        imgs = [img[:npix, :npix] for img in imgs]

        model = danish.DZMultiDonutModel(
            factory,
            z_refs=zk_refs,
            dz_terms=dz_terms,
            field_radius=_DANISH_FIELD_RADIUS_RAD,
            thxs=thxs,
            thys=thys,
            npix=npix,
            bkg_order=self.config.bkgOrder,
            loss_fn=self._build_loss_fn(),
        )
        fluxes_init = [np.clip(np.sum(img), 1e3, 1e9) for img in imgs]
        x0 = model.pack_params(
            fluxes=fluxes_init,
            dxs=[0.0] * n,
            dys=[0.0] * n,
            fwhm=1.0,
            bkgs=[[0.0] * model.nbkg] * n,
            wavefront_params=[0.0] * len(dz_terms),
        )
        bounds = model.pack_params(
            fluxes=[[0.0, np.inf]] * n,
            dxs=[[-np.inf, np.inf]] * n,
            dys=[[-np.inf, np.inf]] * n,
            fwhm=[0.1, 5.0],
            bkgs=[[[-np.inf, np.inf]] * model.nbkg] * n,
            wavefront_params=[[-np.inf, np.inf]] * len(dz_terms),
        )
        bounds = [list(b) for b in zip(*bounds)]
        x0 = np.clip(x0, bounds[0], bounds[1])
        timeout = self.config.wfFitTimeoutPerDonut * n
        _setup_elapsed = time.perf_counter() - t_setup0
        label = f"group={group.group_id} n={n}"
        self.log.info("WF %s npix=%d setup=%.2fs", label, npix, _setup_elapsed)

        fit_result = self._run_lstsq_fit(
            model, x0, bounds, imgs, sky_lvl, timeout, label
        )
        zk_dev_dense = _dense_dev(fit_result.zk_dev, nollIndices)

        donuts_out = []
        for i, d in enumerate(all_donuts):
            defocal = "intra" if int(d.det_id) in _INTRA_FOCAL_DET_IDS else "extra"
            _img = imgs[i] if i < len(imgs) else None
            donuts_out.append(
                WfResult(
                    donut_id=int(d.id),
                    det_name=d.det_name,
                    defocal=defocal,
                    zk_dev=zk_dev_dense,
                    zk_intrinsic=_dense_intrinsic(d),
                    img=_img,
                    model_img=fit_result.model_imgs[i],
                    fit_success=fit_result.success,
                    fit_elapsed=fit_result.elapsed,
                    setup_elapsed=_setup_elapsed,
                    fit_nfev=fit_result.nfev,
                    fit_cost=fit_result.cost,
                    fit_dx=float(fit_result.dxs[i]),
                    fit_dy=float(fit_result.dys[i]),
                    fit_flux=float(fit_result.fluxes[i]),
                    fit_fwhm=fit_result.fwhm,
                    blend_frac=fit_result.blend_fracs[i],
                    group_id=group.group_id,
                    group_size=n,
                    fit_mode="",  # Will be set by caller with wfEstimationMode
                )
            )
        return {
            "group_id": group.group_id,
            "group_size": n,
            "zk_dev": fit_result.zk_dev,
            "success": fit_result.success,
            "fit_info": {
                "elapsed": fit_result.elapsed,
                "nfev": fit_result.nfev,
                "cost": fit_result.cost,
                "optimality": fit_result.optimality,
                "njev": fit_result.njev,
                "status": fit_result.status,
                "message": fit_result.message,
                "error": fit_result.error,
            },
            "donuts": donuts_out,
            "model_imgs": fit_result.model_imgs,
            "imgs": imgs,
            "det_names": [d.det_name for d in all_donuts],
        }

    def _build_wf_factory(self, group: "_WfGroup") -> "danish.DonutFactory":
        """Build a Danish donut factory from config and group."""
        factory_class = danish.DonutTriangleFactory if self.config.triangleMode else danish.DonutFactory
        factory_kwargs = {}
        if self.config.doAoiThroughput and group.band:
            wavelength = _INSTRUMENT.wavelength.get(group.band)
            if wavelength:
                factory_kwargs["bandpass_filter"] = wavelength
            if group.alt is not None and np.isfinite(group.alt) and group.alt > 0:
                airmass = np.clip((1.0 / np.sin(group.alt)), 1.0, 2.5)
                factory_kwargs["airmass"] = airmass
            else:
                factory_kwargs["airmass"] = 1.2
        return factory_class(
            R_outer=_INSTRUMENT.radius,
            R_inner=_INSTRUMENT.radius * _INSTRUMENT.obscuration,
            mask_params=_INSTRUMENT.maskParams,
            focal_length=_INSTRUMENT.focalLength,
            pixel_scale=_INSTRUMENT.pixelSize * self.config.binning,
            spider_angle=group.rtp,
            **factory_kwargs,
        )

    def _build_loss_fn(self) -> Any:
        """Return a danish loss function from config, or None for standard chi-squared."""
        alpha = self.config.systematicLossAlpha
        if alpha <= 0:
            return None
        return danish.systematic_loss(alpha)

    def _prep_donut_for_danish(self, donut: "Donut") -> tuple:
        """Prepare a Donut for Danish fitting.

        Bins the stamp and forces it to an odd pixel size, estimates background
        noise, computes the reference Zernike array ``zk_ref`` from
        ``batoid.zernikeTA`` (with optional measured-intrinsics correction), and
        extracts the field angle.

        Parameters
        ----------
        donut : Donut
            Donut record. Uses ``stamp`` (2-D array), ``det_id``, ``band``,
            ``thx_ccs``, ``thy_ccs`` (field angles in radians), and
            ``intrinsic_zk`` (µm, Noll 4..``_ZK_JMAX``; ``None`` if uncalibrated).

        Returns
        -------
        img : np.ndarray
            ``(npix, npix)`` float stamp, binned and forced to odd size.
        angle_rad : np.ndarray
            ``[thx_ccs, thy_ccs]`` field angle in radians.
        zk_ref : np.ndarray
            Reference Zernike array in metres, Noll-indexed, shape
            ``(_ZK_JMAX + 1,)``.
            Equals ``W_TA_defoc`` at uncalibrated indices and
            ``W_TA_defoc + (W_meas - zk_opd_foc)`` at calibrated indices.
        bkg_var : float
            Background variance estimate (``bkg_std ** 2``) from pixel-difference
            MAD of the stamp.
        bkg_std : float
            Background standard deviation estimate.
        """
        binning = self.config.binning
        det_id = donut.det_id
        defocalSign = +1 if det_id in _EXTRA_FOCAL_DET_IDS else -1

        img = _bin_stamp_odd(donut.stamp, binning)
        diff = (img[1:] - img[:-1]).ravel()
        bkg_std = median_abs_deviation(diff, scale="normal") / np.sqrt(2.0)

        band = donut.band
        wavelength_by_band = {bl.value: wl for bl, wl in _INSTRUMENT.wavelength.items()}
        if band not in wavelength_by_band:
            raise RuntimeError(
                f"No wavelength configured for band {band!r}; the instrument supplies "
                f"{sorted(wavelength_by_band)}. Fitting with a wrong wavelength would "
                "bias the whole wavefront, so this is fatal rather than defaulted."
            )
        wavelength = wavelength_by_band[band]
        telescope = _CALIB_STORE["telescope"]
        telescope_dz = (
            _CALIB_STORE["telescope_extra"] if defocalSign > 0 else _CALIB_STORE["telescope_intra"]
        )
        eps = telescope.pupilObscuration
        nrad = 10
        zernikeTA_kwargs = dict(
            jmax=_ZK_JMAX,
            eps=eps,
            focal_length=_INSTRUMENT.focalLength,
            nrad=nrad,
            naz=int(2 * np.pi * nrad / (1 - eps)),
        )
        # W_TA_defoc: off-axis + nominal intrinsics + defocus in one call
        zk_ref = (
            batoid.zernikeTA(
                telescope_dz,
                donut.thx_ccs,
                donut.thy_ccs,
                wavelength,
                **zernikeTA_kwargs,
            )
            * wavelength
        )  # meters, shape (_ZK_JMAX + 1,)

        # Replace nominal on-axis model (zk_opd_foc) with measured intrinsics (W_meas)
        # for calibrated indices.
        intrinsic_zk = donut.intrinsic_zk
        if intrinsic_zk is not None:
            zk_opd_foc = (
                batoid.zernikeTA(
                    telescope,
                    donut.thx_ccs,
                    donut.thy_ccs,
                    wavelength,
                    **zernikeTA_kwargs,
                )
                * wavelength
            )  # meters
            calib_noll = np.arange(4, _ZK_JMAX + 1)
            for i, j in enumerate(calib_noll):
                if i < len(intrinsic_zk) and j <= _ZK_JMAX:
                    zk_ref[j] += intrinsic_zk[i] * 1e-6 - zk_opd_foc[j]

        angle_rad = np.array([donut.thx_ccs, donut.thy_ccs])
        return img, angle_rad, zk_ref, bkg_std**2, bkg_std

    def _run_lstsq_fit(self, model, x0, bounds, imgs, variances, timeout, label):
        """Run a DZMultiDonutModel least-squares fit and return results uniformly.

        Handles the ``wfInitialGuessOnly`` path, SIGALRM timeout, and all exception
        cases so each worker only needs to build the model and call this helper.

        Parameters
        ----------
        model : danish.DZMultiDonutModel
            Fully-constructed model ready to call ``.chi``, ``.jac``, ``.model``.
        x0 : np.ndarray
            Initial parameter vector from ``model.pack_params``.
        bounds : list
            Two-element ``[lower, upper]`` bound lists from ``model.pack_params``.
        imgs : list of np.ndarray
            Donut image stamps, one per model donut.
        variances : list of np.ndarray
            Per-pixel variance arrays aligned with ``imgs``.
        timeout : float
            SIGALRM timeout in seconds.
        label : str
            Short description used in log messages (e.g. ``"group=123 n=2"``).

        Returns
        -------
        _LstsqFitResult
            All fit outputs as first-class typed fields; see `_LstsqFitResult`.
        """
        nollIndices = list(self.config.nollIndices)
        n = len(imgs)
        _nan_blends = [float("nan")] * n
        t0 = time.perf_counter()
        if self.config.wfInitialGuessOnly:
            try:
                params = model.unpack_params(x0)
                zk_dev = np.zeros(len(nollIndices))
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in variances]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                elapsed = time.perf_counter() - t0
                self.log.info("WF %s (x0 only)", label)
                return _LstsqFitResult(
                    zk_dev=zk_dev,
                    model_imgs=model_imgs,
                    blend_fracs=blend_fracs,
                    success=True,
                    elapsed=elapsed,
                    fluxes=params["fluxes"],
                    dxs=params["dxs"],
                    dys=params["dys"],
                    fwhm=params["fwhm"],
                    nfev=0,
                    cost=float("nan"),
                    optimality=float("nan"),
                    njev=0,
                    status=0,
                    message="x0 only",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=str(exc),
                )
        else:
            galsim.errors.raise_fft_size_error = True
            try:
                with _fit_timeout(timeout):
                    lstsq_kwargs = {
                        k: ast.literal_eval(v)
                        for k, v in self.config.lstsqKwargs.items()
                    }
                    result = least_squares(
                        model.chi,
                        jac=model.jac,
                        x0=x0,
                        args=(imgs, variances),
                        bounds=bounds,
                        **lstsq_kwargs,
                    )
                elapsed = time.perf_counter() - t0
                params = model.unpack_params(result.x)
                zk_dev = np.array(params["wavefront_params"])
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in variances]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                self.log.info(
                    "WF %s success=%s nfev=%d elapsed=%.1fs",
                    label,
                    bool(result.success),
                    result.nfev,
                    elapsed,
                )
                return _LstsqFitResult(
                    zk_dev=zk_dev,
                    model_imgs=model_imgs,
                    blend_fracs=blend_fracs,
                    success=bool(result.success),
                    elapsed=elapsed,
                    fluxes=params["fluxes"],
                    dxs=params["dxs"],
                    dys=params["dys"],
                    fwhm=params["fwhm"],
                    nfev=result.nfev,
                    cost=result.cost,
                    optimality=result.optimality,
                    njev=result.njev,
                    status=result.status,
                    message=result.message,
                )
            except _WfFitTimeoutError:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s TIMED OUT after %.1fs", label, elapsed)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=f"timeout after {timeout:.0f}s",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult(
                    zk_dev=np.full(len(nollIndices), np.nan),
                    model_imgs=[None] * n,
                    blend_fracs=_nan_blends,
                    success=False,
                    elapsed=elapsed,
                    fluxes=[float("nan")] * n,
                    dxs=[float("nan")] * n,
                    dys=[float("nan")] * n,
                    fwhm=float("nan"),
                    error=str(exc),
                )
