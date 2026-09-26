# This file is part of ts_wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Danish wavefront fitting: donut grouping, the fit task, and its worker."""

__all__ = ["WavefrontFittingConfig", "WavefrontFittingTask"]

import contextlib
import logging
import os
import signal
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import batoid
import danish
import galsim
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ts.wep.utils.ioUtils import resolveRelativeConfigPath

from .dataStructures import (
    _FIT_OUTCOMES,
    _FIT_STATUS_ABSENT,
    Donut,
    FitOutcome,
    WfDonutResult,
    WfGroupResult,
    _WfGroup,
)
from .lsstCam import _LSSTCAM, _rescale_zk_domain
from .utils import (
    _COW_STORE,
    _ZK_JMAX,
    CORNER_PAIRS,
    _bin_stamp_odd,
    _defocused_telescope,
    _dense_intrinsic,
)

# DZMultiDonutModel's field_radius has no effect on our fit (we don't model any
# field-dependent optics term), so any value works; set to roughly the Rubin
# field of view for a physically sensible default.
_DANISH_FIELD_RADIUS_RAD = np.deg2rad(1.85)


def _jacobian_callable(model, jacobian_format: str):
    """The Jacobian to hand `least_squares`, per `jacobianFormat`.

    `least_squares` calls `jac(x, *args)` and its own `kwargs` argument goes to
    `fun` as well as `jac`, so the format cannot be forwarded through it --
    `model.chi` would reject it. Bind it here instead.
    """
    if jacobian_format == "dense":
        return model.jac
    return partial(model.jac_sparse, format=jacobian_format)


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
    """Fraction of significant residual flux in model-faint pixels.

    Normalized by the total model flux.
    """
    if resid is None or model_img_bkg_free is None or not np.isfinite(bkg_std) or bkg_std <= 0:
        return float("nan")
    model_peak = np.nanmax(model_img_bkg_free)
    total_model_flux = np.sum(model_img_bkg_free[model_img_bkg_free > 0])
    if model_peak <= 0 or total_model_flux <= 0:
        return float("nan")
    faint_mask = model_img_bkg_free < faint_frac * model_peak
    sig_mask = np.abs(resid) > sig_thresh * bkg_std
    return np.sum(np.abs(resid[faint_mask & sig_mask])) / total_model_flux


def _dense_dev(zk_dev: np.ndarray, noll_indices) -> np.ndarray:
    """Return deviations in meters, dense over Noll 0..``_ZK_JMAX``.

    Indices that were not fitted are ``np.nan``, except Noll 0..3, which are
    0.0: they are carried for indexing only and are never fitted.
    """
    out = np.full(_ZK_JMAX + 1, np.nan)
    out[0:4] = 0.0
    for k, j in enumerate(noll_indices):
        if k < len(zk_dev):
            out[j] = zk_dev[k]
    return out


def _build_wf_groups(mode, results_by_det, band: str, rtp_deg: float | None, boresight_alt_rad: float | None):
    """Build `_WfGroup` work units from per-detector catalogs, for corner mode.

    Runs in the parent process after every detector has returned, because
    ``paired`` and ``full_corner`` group *across* the two detectors of a
    corner. Full-array mode groups within a single detector and so has its own
    `_fam_group_donuts`, which runs inside the worker; the two mode
    vocabularies overlap but are not interchangeable.

    ``results_by_det`` covers only the detectors that were processed, so a
    partial corner set falls out naturally: groups are never emitted empty, and
    ``full_corner`` fits a corner from one defocal side alone when that is all
    that is present.

    Modes
    -----
    ``paired``
        One star, both sides of focus: SW0 and SW1 donuts of a corner zipped in
        descending SNR order.  The only mode that pairs, and so the only one
        that can leave donuts unmatched.
    ``unpaired``
        One donut per group.
    ``full_detector``
        Every donut on a detector in one joint fit -- a single defocal side,
        since side is a property of the detector here.
    ``full_corner``
        Every donut on both detectors of a corner in one joint fit.  Does not
        pair: each donut carries its own optic offsets, so Danish already knows
        which side of focus it is on and no association is needed.

    Parameters
    ----------
    mode : `str`
        One of the modes above.
    results_by_det : `dict` [`str`, `list`]
        Detector name -> accepted donuts on that detector.
    band : `str`
        Photometric band, passed through to every group.
    rtp_deg : `float` or `None`
        Boresight rotation (spider angle) in degrees.
    boresight_alt_rad : `float` or `None`
        Boresight altitude in radians.

    Returns
    -------
    groups : `list` [`_WfGroup`]
        Work units to dispatch; never contains an empty group.
    unmatched_donuts : `list`
        Donuts left over by pairing.  Non-empty only for ``paired``.
    path : `str`
        Pairing path taken: ``"snr_rank"`` for ``paired``, ``"n/a"`` for the
        modes that do not pair.  Unlike full-array mode's `_pair_donuts`, which
        chooses between refcat-id and spatial matching at runtime, corner mode
        has a single algorithm and so a constant here; it is returned anyway so
        both modes record pairing provenance in the same ``det_meta`` field.

    Raises
    ------
    ValueError
        Raised if ``mode`` is not one of the modes above -- in particular for
        the full-array-only ``full_detector_pair``.
    """
    groups = []
    unmatched_donuts = []
    path = "n/a"
    if mode == "paired":
        path = "snr_rank"
        for corner, (sw0, sw1) in CORNER_PAIRS.items():
            extra_donuts = sorted(results_by_det.get(sw0, []), key=lambda d: d.snr, reverse=True)
            intra_donuts = sorted(results_by_det.get(sw1, []), key=lambda d: d.snr, reverse=True)
            for extra, intra in zip(extra_donuts, intra_donuts):
                # Qualified by corner: under blitz detection the ids are
                # per-detector 1..N slots, so every corner would otherwise log
                # as group=1_1, 2_2, ...
                gid = f"{corner}_{extra.donut_id}_{intra.donut_id}"
                groups.append(
                    _WfGroup(
                        donuts=[extra, intra], group_id=gid, band=band, rtp=rtp_deg, alt=boresight_alt_rad
                    )
                )
            n_pairs = min(len(extra_donuts), len(intra_donuts))
            unmatched_donuts.extend(extra_donuts[n_pairs:])
            unmatched_donuts.extend(intra_donuts[n_pairs:])
    elif mode == "unpaired":
        for det_donuts in results_by_det.values():
            for d in det_donuts:
                gid = f"{d.det_name}_{d.donut_id}"
                groups.append(
                    _WfGroup(donuts=[d], group_id=gid, band=band, rtp=rtp_deg, alt=boresight_alt_rad)
                )
    elif mode == "full_detector":
        # Skip detectors with no donuts: an empty group fits nothing but still
        # reports success=False, which would skew the caller's success tally.
        for det_name, det_donuts in results_by_det.items():
            if not det_donuts:
                continue
            groups.append(
                _WfGroup(donuts=det_donuts, group_id=det_name, band=band, rtp=rtp_deg, alt=boresight_alt_rad)
            )
    elif mode == "full_corner":
        # A corner contributes whichever of its two detectors have donuts; one
        # defocal side alone is still fit. Corners with neither are skipped.
        for corner, (sw0, sw1) in CORNER_PAIRS.items():
            all_donuts = results_by_det.get(sw0, []) + results_by_det.get(sw1, [])
            if not all_donuts:
                continue
            groups.append(
                _WfGroup(donuts=all_donuts, group_id=corner, band=band, rtp=rtp_deg, alt=boresight_alt_rad)
            )
    else:
        raise ValueError(f"Unknown WF mode {mode!r}")
    return groups, unmatched_donuts, path


# Module-level logger for the worker functions below. They are module-level
# (not methods) so the fork-based pools can pickle them by name, which means
# there is no `self` and so no `Task.log`. Consequence: worker output goes to
# the `lsst.ts.wep.blitz.wavefrontFitting` logger rather than the task's
# own `donutBlitzCorner` hierarchy, so it is not affected by that task's
# log level. Parent-process code should keep using `self.log`.
_log = logging.getLogger(__name__)

_DZ_MODEL_KEYS = ("fluxes", "dxs", "dys", "fwhm", "wavefront_params", "bkgs")


def _wf_fitting_worker(group: "_WfGroup") -> WfGroupResult:
    """Wavefront fitting worker for multiprocessing pool.

    Retrieves the WavefrontFittingTask from _COW_STORE and calls it on the
    group, unwrapping the `pipeBase.Struct` it returns: what crosses the fork
    boundary is the record itself, and a Struct around it would only make every
    caller reach through it.
    """
    task = _COW_STORE.wf_fit_task
    return task.run(group).result


class WavefrontFittingConfig(pexConfig.Config):
    """Configuration for wavefront fitting via Danish algorithm."""

    nollIndices: pexConfig.ListField[int] = pexConfig.ListField[int](
        doc="Noll indices to fit with Danish.",
        default=list(range(4, 20)) + list(range(22, 27)),
    )
    lstsqKwargs: pexConfig.DictField[str, Any] = pexConfig.DictField[str, None](
        doc=(
            "Keyword arguments for scipy.optimize.least_squares passed to the Danish "
            "WF workers, e.g. {'method': 'trf', 'max_nfev': 200}. `fun`, `x0`, `jac`, "
            "`args` and `bounds` are supplied by the task and are rejected here."
        ),
        default={
            "xtol": 1e-3,
            "ftol": 1e-3,
            "gtol": 1e-3,
            "x_scale": "jac",
            # 'lsmr' is faster than the default of 'exact' for simultaneous
            # donut fitting and about the same for single donuts.  It also
            # appears to be slightly more robust, running into the unphysical
            # parameter space less often (as caught via the "very large FFT"
            # guard in Danish).
            "tr_solver": "lsmr",
        },
    )
    binning: pexConfig.Field[int] = pexConfig.Field[int](
        default=2,
        doc="Binning factor applied to donut stamps before Danish fitting.",
    )
    modelSpiderShadows: pexConfig.Field[bool] = pexConfig.Field[bool](
        default=False,
        doc="Include spider shadow modeling in Danish forward model.",
    )
    maskModel: pexConfig.Field[str] = pexConfig.Field[str](
        default="RubinObsc_v3.14_r_rtpp0_azp45_pp0d0.yaml",
        doc=(
            "Pupil mask model: a file name in the danish package's data "
            "directory, or a 'policy:'-prefixed path relative to the ts_wep "
            "policy directory. 'policy:masks/LsstCamLegacy.yaml' is the mask "
            "used before danish shipped fitted models."
        ),
    )
    bkgOrder: pexConfig.Field[int] = pexConfig.Field[int](
        default=0,
        doc="Background polynomial order for Danish (-1=none, 0=constant).",
    )
    doAoiThroughput: pexConfig.Field[bool] = pexConfig.Field[bool](
        default=False,
        doc="Apply angle-of-incidence throughput correction in Danish forward model.",
    )
    systematicLossAlpha: pexConfig.Field[float] = pexConfig.Field[float](
        default=0.0,
        doc="Fractional systematic uncertainty for Danish loss function (0=chi2).",
    )
    triangleMode: pexConfig.Field[bool] = pexConfig.Field[bool](
        default=True,
        doc="Use DonutTriangleFactory instead of DonutFactory.",
    )
    jacobianFormat: pexConfig.ChoiceField[str] = pexConfig.ChoiceField[str](
        default="dense",
        optional=False,
        # There is some weak evidence that one or the other sparse formats may
        # be modestly faster for large groups of donuts.  But since these modes
        # are primarily run offline, we just leave the default to dense here.
        doc=(
            "Storage for the Danish Jacobian passed to least_squares. "
            "All three give bit-identical Jacobian *values*; "
            "they differ only in how those values are stored. A sparse "
            "Jacobian also requires an iterative tr_solver: scipy rejects it "
            "under tr_solver='exact', which `validate` checks for."
        ),
        allowed={
            "dense": "model.jac -- a dense ndarray.",
            "csr": "model.jac_sparse in CSR.",
            "csc": "model.jac_sparse in CSC.",
        },
    )
    logPerGroup: pexConfig.Field[bool] = pexConfig.Field[bool](
        default=True,
        doc=(
            "Log a setup line and a result line per fit group at INFO.  Useful when "
            "a quantum holds tens of groups, as corner mode's do; full-array mode "
            "turns it off because 'paired' over 189 detectors is ~5000 groups, and "
            "~10k lines bury the per-detector summaries that supersede them.  "
            "Failures and timeouts are logged as warnings either way -- this "
            "silences narration, not problems."
        ),
    )
    wfFitTimeoutPerDonut: pexConfig.Field[float] = pexConfig.Field[float](
        default=10.0,
        doc=(
            "Timeout in seconds per donut for a single WF fit. "
            "The total timeout for a work unit is this value times the number of donuts "
            "(1 for unpaired, 2 for paired, N for "
            "full_detector/full_corner/full_detector_pair). "
            "Fits exceeding the limit are killed and return NaN Zernikes."
        ),
    )
    wfInitialGuessOnly: pexConfig.Field[bool] = pexConfig.Field[bool](
        default=False,
        doc=(
            "Skip the least-squares fit and return the initial-guess (x0) model only. "
            "Useful for diagnosing stamp orientation and model setup without "
            "waiting for convergence."
        ),
    )

    def validate(self):
        super().validate()
        # `_run_lstsq_fit` supplies these itself, so setting them here would
        # pass a duplicate keyword argument to least_squares.
        reserved = {"fun", "x0", "jac", "args", "bounds"} & set(self.lstsqKwargs)
        if reserved:
            raise pexConfig.FieldValidationError(
                self.__class__.lstsqKwargs,
                self,
                f"{sorted(reserved)} are supplied by the task and must not be set "
                "in lstsqKwargs; passing them would duplicate a keyword argument "
                "to scipy.optimize.least_squares",
            )

        if self.jacobianFormat != "dense" and self.lstsqKwargs.get("tr_solver") == "exact":
            raise pexConfig.FieldValidationError(
                self.__class__.jacobianFormat,
                self,
                f"jacobianFormat={self.jacobianFormat!r} needs an iterative "
                "tr_solver, but lstsqKwargs sets tr_solver='exact'. Use "
                "'lsmr', or omit tr_solver and scipy will choose lsmr for a "
                "sparse Jacobian.",
            )

        indices = set(self.nollIndices)
        out_of_range = sorted(j for j in indices if j < 4 or j > _ZK_JMAX)
        if out_of_range:
            raise pexConfig.FieldValidationError(
                self.__class__.nollIndices,
                self,
                f"nollIndices must lie in 4..{_ZK_JMAX} (the dense Zernike arrays "
                f"reported by the catalog are sized to {_ZK_JMAX}); "
                f"got {out_of_range}",
            )
        # Rotating Zernikes between coordinate frames (CCS -> OCS) mixes each
        # (n, +m) coefficient with its (n, -m) partner, so a lone half of a
        # pair cannot be rotated: its rotated power belongs to a term that was
        # never fit. Requiring whole pairs keeps every reported frame well
        # defined.
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
                self.__class__.nollIndices,
                self,
                "nollIndices must contain both halves of every +/-m Zernike pair so "
                "the coefficients can be rotated between coordinate frames; missing "
                + ", ".join(f"{p} (partner of {j})" for j, p in missing),
            )


class _DanishDonutInputs(NamedTuple):
    """One donut, prepared for danish, as `_prep_donut_for_danish` returns it.

    A `NamedTuple` rather than a dataclass: `WavefrontFittingTask.run` builds
    one per group member and immediately transposes them into the per-field
    lists `danish.DZMultiDonutModel` wants, so the tuple-ness is used.
    """

    img: npt.NDArray[np.float64]  # (npix, npix), binned and forced to odd size
    angle_rad: npt.NDArray[np.float64]  # [thx_ccs, thy_ccs], radians
    # Reference Zernikes in meters, Noll-indexed, shape (_ZK_JMAX + 1,).
    # `W_TA_defoc` at uncalibrated indices, and
    # `W_TA_defoc + (W_meas - zk_opd_foc)` at calibrated ones.
    zk_ref: npt.NDArray[np.float64]
    # Background level: one scalar per donut, not a per-pixel array. `bkg_var`
    # is `bkg_std ** 2`, from the pixel-difference MAD of the stamp; danish's
    # `chi` takes the variance and `_blend_frac` the deviation, which is why
    # both are carried rather than one being re-derived at each use.
    bkg_var: float
    bkg_std: float


@dataclass
class _LstsqFitResult:
    # All outputs from _run_lstsq_fit as first-class typed fields.
    zk_dev: npt.NDArray[np.float64]
    model_imgs: list[npt.NDArray[np.float64] | None]
    blend_fracs: list[float]
    success: bool
    elapsed: float
    fluxes: list[float]
    dxs: list[float]
    dys: list[float]
    bkgs: list[npt.NDArray[np.float64] | None]
    fwhm: float
    nfev: int = 0
    cost: float = float("nan")
    optimality: float = float("nan")
    njev: int = 0
    status: int = _FIT_STATUS_ABSENT
    message: str = ""
    error: str = ""
    # Which of the mutually exclusive fit paths produced this result.
    outcome: FitOutcome = ""

    def __post_init__(self):
        if not self.outcome or self.outcome not in _FIT_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {tuple(o for o in _FIT_OUTCOMES if o)}; got {self.outcome!r}"
            )

    @classmethod
    def failed(
        cls,
        n_zk: int,
        n_donuts: int,
        elapsed: float,
        error: str,
        outcome: FitOutcome,
    ) -> "_LstsqFitResult":
        """Result for a fit that did not produce a wavefront.

        The three failure paths through `_run_lstsq_fit` -- a raise from the
        ``x0``-only branch, a SIGALRM timeout, and a raise from the fit itself
        -- differ only in ``error`` and ``outcome``; all the rest is the same
        wall of NaN.  ``elapsed`` is real and kept: how long a fit ran before
        failing is the difference between a timeout and an immediate raise.

        Parameters
        ----------
        n_zk : `int`
            Number of fitted Noll indices, so the NaN Zernike vector matches
            the length a successful fit would have returned.
        n_donuts : `int`
            Group size, for the per-donut lists.
        elapsed : `float`
            Seconds spent before failing.
        error : `str`
            What went wrong, for the catalog and the log.
        outcome : `FitOutcome`
            Says which path failed.
        """
        return cls(
            zk_dev=np.full(n_zk, np.nan),
            model_imgs=[None] * n_donuts,
            blend_fracs=[float("nan")] * n_donuts,
            success=False,
            elapsed=elapsed,
            fluxes=[float("nan")] * n_donuts,
            dxs=[float("nan")] * n_donuts,
            dys=[float("nan")] * n_donuts,
            bkgs=[None] * n_donuts,
            fwhm=float("nan"),
            error=error,
            outcome=outcome,
        )


class WavefrontFittingTask(pipeBase.Task):
    """Fit wavefront aberrations from grouped donut stamps with danish.

    This task takes a pre-grouped collection of donuts (a _WfGroup) and
    performs joint wavefront fitting across all donuts in the group, returning
    Zernike coefficients for the wavefront error.
    """

    ConfigClass = WavefrontFittingConfig
    _DefaultName = "wavefrontFitting"
    config: WavefrontFittingConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Resolved once here rather than per group. A bad name raises here, in
        # the parent, rather than in every worker.
        self._mask_params = danish.load_mask_params(self._maskModelPath())

    def _maskModelPath(self) -> str:
        """Absolute path to the configured mask file."""
        name = self.config.maskModel
        if name.startswith("policy:"):
            return resolveRelativeConfigPath(name)
        return os.path.join(danish.datadir, name)

    def resolvedMaskModel(self) -> str:
        """The mask file name for the catalog, with symlinks followed.

        ``RubinObsc.yaml`` is a symlink in the danish data directory, so the
        configured name can outlive the file it pointed at when the run
        happened. Recording the concrete target is what makes two catalogs
        comparable after the fact.
        """
        return os.path.basename(os.path.realpath(self._maskModelPath()))

    def run(self, group: "_WfGroup") -> pipeBase.Struct:
        """Fit wavefront aberrations for a group of donuts.

        Parameters
        ----------
        group : _WfGroup
            Pre-grouped collection of donuts to fit jointly, including
            band, rtp, and alt for exposure-dependent parameters.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``result``
                This group's fit, as a
                `lsst.ts.wep.blitz.dataStructures.WfGroupResult`.  See that
                class for what each field means.
        """
        noll_indices = self.config.nollIndices
        all_donuts = group.donuts
        n = len(all_donuts)

        if not all_donuts:
            return pipeBase.Struct(result=WfGroupResult.empty(group.group_id, len(noll_indices)))

        t_setup0 = time.perf_counter()
        factory = self._build_wf_factory(group)
        preps = [self._prep_donut_for_danish(d) for d in all_donuts]
        imgs = [p.img for p in preps]
        thxs = [p.angle_rad[0] for p in preps]
        thys = [p.angle_rad[1] for p in preps]
        zk_refs = [p.zk_ref for p in preps]
        bkg_vars = [p.bkg_var for p in preps]
        dz_terms = [(1, j) for j in noll_indices]

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
        setup_elapsed = time.perf_counter() - t_setup0
        label = f"group={group.group_id} n={n}"
        if self.config.logPerGroup:
            self.log.info("WF %s setup=%.2fs", label, setup_elapsed)

        fit_result = self._run_lstsq_fit(model, x0, bounds, imgs, bkg_vars, timeout, label)
        zk_dev_dense = _dense_dev(fit_result.zk_dev, noll_indices)

        donuts_out = []
        for i, d in enumerate(all_donuts):
            img_i = imgs[i] if i < len(imgs) else None
            donuts_out.append(
                WfDonutResult(
                    donut_id=int(d.donut_id),
                    det_name=d.det_name,
                    visit_id=int(d.visit_id),
                    zk_dev=zk_dev_dense,
                    zk_intrinsic=_dense_intrinsic(d),
                    img=img_i,
                    model_img=fit_result.model_imgs[i],
                    fit_success=fit_result.success,
                    fit_elapsed=fit_result.elapsed,
                    setup_elapsed=setup_elapsed,
                    fit_nfev=fit_result.nfev,
                    fit_cost=fit_result.cost,
                    fit_optimality=fit_result.optimality,
                    fit_njev=fit_result.njev,
                    fit_outcome=fit_result.outcome,
                    fit_dx=float(fit_result.dxs[i]),
                    fit_dy=float(fit_result.dys[i]),
                    fit_flux=float(fit_result.fluxes[i]),
                    fit_fwhm=fit_result.fwhm,
                    fit_bkg=fit_result.bkgs[i],
                    blend_frac=fit_result.blend_fracs[i],
                    group_id=group.group_id,
                    group_size=n,
                )
            )
        return pipeBase.Struct(
            result=WfGroupResult(
                group_id=group.group_id,
                group_size=n,
                zk_dev=fit_result.zk_dev,
                success=fit_result.success,
                donut_results=donuts_out,
                det_names=[d.det_name for d in all_donuts],
                imgs=imgs,
                model_imgs=fit_result.model_imgs,
                fit_elapsed=fit_result.elapsed,
                fit_nfev=fit_result.nfev,
                fit_cost=fit_result.cost,
                fit_optimality=fit_result.optimality,
                fit_njev=fit_result.njev,
                fit_status=fit_result.status,
                fit_message=fit_result.message,
                fit_error=fit_result.error,
                fit_outcome=fit_result.outcome,
            )
        )

    def _build_wf_factory(self, group: "_WfGroup") -> "danish.DonutFactory":
        """Build a Danish donut factory from config and group."""
        factory_class = danish.DonutTriangleFactory if self.config.triangleMode else danish.DonutFactory
        factory_kwargs = {}
        if self.config.doAoiThroughput and group.band:
            wavelength = _LSSTCAM.wavelength.get(group.band)
            if wavelength:
                factory_kwargs["bandpass_filter"] = wavelength
            if group.alt is not None and np.isfinite(group.alt) and group.alt > 0:
                airmass = np.clip((1.0 / np.sin(group.alt)), 1.0, 2.5)
                factory_kwargs["airmass"] = airmass
            else:
                factory_kwargs["airmass"] = 1.2
        return factory_class(
            R_outer=_LSSTCAM.zk_r_outer,
            R_inner=_LSSTCAM.zk_r_inner,
            mask_params=self._mask_params,
            focal_length=_LSSTCAM.focal_length,
            pixel_scale=_LSSTCAM.pixel_size * self.config.binning,
            spider_angle=group.rtp,
            **factory_kwargs,
        )

    def _build_loss_fn(self) -> Any:
        """Return a danish loss function from config, None for chi-squared."""
        alpha = self.config.systematicLossAlpha
        if alpha <= 0:
            return None
        return danish.systematic_loss(alpha)

    def _prep_donut_for_danish(self, donut: "Donut") -> _DanishDonutInputs:
        """Prepare a Donut for Danish fitting.

        Bins the stamp and forces it to an odd pixel size, estimates background
        noise, computes the reference Zernike array ``zk_ref`` from
        ``batoid.zernikeTA`` (with optional measured-intrinsics correction),
        and extracts the field angle.

        Parameters
        ----------
        donut : Donut
            Donut record. Uses ``stamp`` (2-D array), ``det_id``, ``band``,
            ``thx_ccs``, ``thy_ccs`` (field angles in radians), and
            ``intrinsic_zk`` (µm, Noll 4..``_ZK_JMAX``; ``None`` if
            uncalibrated).

        Returns
        -------
        `_DanishDonutInputs`
            The binned stamp, field angle, reference Zernikes and background
            level.  See that class for what each field means.
        """
        binning = self.config.binning

        if donut.stamp is None:
            raise RuntimeError(
                f"donut {donut.donut_id} on {donut.det_name} has no stamp; cannot fit a "
                "donut whose stamp has already been shed"
            )
        img = _bin_stamp_odd(donut.stamp, binning)
        diff = (img[1:] - img[:-1]).ravel()
        bkg_std = median_abs_deviation(diff, scale="normal") / np.sqrt(2.0)

        band = donut.band
        if band not in _LSSTCAM.wavelength:
            raise RuntimeError(
                f"No wavelength configured for band {band!r}; the instrument supplies "
                f"{sorted(_LSSTCAM.wavelength)}. Fitting with a wrong wavelength would "
                "bias the whole wavefront, so this is fatal rather than defaulted."
            )
        wavelength = _LSSTCAM.wavelength[band]
        telescope = _COW_STORE.telescope
        # The defocused telescope comes from the donut's own offset triplet, so
        # the fitter never needs to know which detectors sit on which side of
        # focus -- a rule that has no meaning in full-array mode, where every
        # detector appears on both sides.
        if donut.defocal_offsets is None:
            raise RuntimeError(
                f"Donut {donut.donut_id} on {donut.det_name} has no defocal_offsets; the "
                "task that built it must set them (see Donut.defocal_offsets)."
            )
        telescope_dz = _defocused_telescope(telescope, donut.defocal_offsets)
        eps = telescope.pupilObscuration
        nrad = 10
        zernikeTA_kwargs = dict(
            jmax=_ZK_JMAX,
            eps=eps,
            focal_length=_LSSTCAM.focal_length,
            nrad=nrad,
            naz=int(2 * np.pi * nrad / (1 - eps)),
        )
        # zernikeTA normalizes on the traced model's own pupil, which is not
        # necessarily the domain the danish factory is built on, so every array
        # it produces is moved onto blitz's domain at the point of production.
        # Rescaling here rather than at the point of use is what lets the
        # arithmetic below mix the two legs without tracking whose domain is
        # whose. A z shift leaves the pupil alone, so telescope_dz and
        # telescope share these radii.
        r_outer_from = telescope.pupilSize / 2

        def _zk_ta_meters(optic: batoid.Optic) -> np.ndarray:
            coef = (
                batoid.zernikeTA(
                    optic,
                    donut.thx_ccs,
                    donut.thy_ccs,
                    wavelength,
                    **zernikeTA_kwargs,
                )
                * wavelength
            )  # meters, shape (_ZK_JMAX + 1,)
            return _rescale_zk_domain(
                coef,
                r_outer_from=r_outer_from,
                r_inner_from=eps * r_outer_from,
                r_outer_to=_LSSTCAM.zk_r_outer,
                r_inner_to=_LSSTCAM.zk_r_inner,
            )

        # W_TA_defoc: off-axis + nominal intrinsics + defocus in one call
        zk_ref = _zk_ta_meters(telescope_dz)

        # Swap the nominal design intrinsics for the measured ones at
        # calibrated indices. zk_opd_foc is the same raytrace as zk_ref minus
        # the defocal offsets, so subtracting it leaves the defocus
        # contribution intact and only the static aberration field is replaced
        # by W_meas. Both are evaluated at this donut's field angle, so neither
        # is on-axis.
        #
        # intrinsic_zk is added as supplied: the IntrinsicZernikes calibration
        # carries no record of the radii it was normalized on, so there is no
        # domain to convert from. Assuming one would be a guess with the same
        # failure mode as the mismatch handled above.
        intrinsic_zk = donut.intrinsic_zk
        if intrinsic_zk is not None:
            zk_opd_foc = _zk_ta_meters(telescope)
            calib_noll = np.arange(4, _ZK_JMAX + 1)
            for i, j in enumerate(calib_noll):
                if i < len(intrinsic_zk) and j <= _ZK_JMAX:
                    zk_ref[j] += intrinsic_zk[i] * 1e-6 - zk_opd_foc[j]

        angle_rad = np.array([donut.thx_ccs, donut.thy_ccs])
        return _DanishDonutInputs(
            img=img,
            angle_rad=angle_rad,
            zk_ref=zk_ref,
            bkg_var=bkg_std**2,
            bkg_std=bkg_std,
        )

    def _run_lstsq_fit(self, model, x0, bounds, imgs, bkg_vars, timeout, label):
        """Run a DZMultiDonutModel least-squares fit with a uniform result.

        Handles the ``wfInitialGuessOnly`` path, SIGALRM timeout, and all
        exception cases so each worker only needs to build the model and call
        this helper.

        Parameters
        ----------
        model : danish.DZMultiDonutModel
            Fully-constructed model ready to call ``.chi``, ``.jac``,
            ``.model``.
        x0 : np.ndarray
            Initial parameter vector from ``model.pack_params``.
        bounds : list
            Two-element ``[lower, upper]`` bound lists from
            ``model.pack_params``.
        imgs : list of np.ndarray
            Donut image stamps, one per model donut.
        bkg_vars : list of float
            One background variance per donut, aligned with ``imgs`` -- a
            scalar each, not a per-pixel array.
        timeout : float
            SIGALRM timeout in seconds.
        label : str
            Short description used in log messages (e.g. ``"group=123 n=2"``).

        Returns
        -------
        _LstsqFitResult
            All fit outputs as first-class typed fields; see `_LstsqFitResult`.
        """
        noll_indices = list(self.config.nollIndices)
        n = len(imgs)
        t0 = time.perf_counter()
        if self.config.wfInitialGuessOnly:
            try:
                params = model.unpack_params(x0)
                zk_dev = np.zeros(len(noll_indices))
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in bkg_vars]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                elapsed = time.perf_counter() - t0
                if self.config.logPerGroup:
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
                    bkgs=[np.asarray(b, dtype=float) for b in params["bkgs"]],
                    fwhm=params["fwhm"],
                    nfev=0,
                    cost=float("nan"),
                    optimality=float("nan"),
                    njev=0,
                    status=0,
                    message="x0 only",
                    outcome="x0_only",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult.failed(
                    n_zk=len(noll_indices),
                    n_donuts=n,
                    elapsed=elapsed,
                    error=str(exc),
                    outcome="exception",
                )
        else:
            galsim.errors.raise_fft_size_error = True
            try:
                with _fit_timeout(timeout):
                    lstsq_kwargs = dict(self.config.lstsqKwargs)
                    result = least_squares(
                        model.chi,
                        jac=_jacobian_callable(model, self.config.jacobianFormat),
                        x0=x0,
                        args=(imgs, bkg_vars),
                        bounds=bounds,
                        **lstsq_kwargs,
                    )
                elapsed = time.perf_counter() - t0
                params = model.unpack_params(result.x)
                zk_dev = np.array(params["wavefront_params"])
                model_imgs = model.model(**{k: params[k] for k in _DZ_MODEL_KEYS})
                bkg_stds = [np.sqrt(v) for v in bkg_vars]
                blend_fracs = [
                    _blend_frac(
                        imgs[i] - model_imgs[i],
                        _bkg_free_model(model_imgs[i], model, params, i, self.config.bkgOrder),
                        bkg_stds[i],
                    )
                    for i in range(n)
                ]
                if self.config.logPerGroup:
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
                    bkgs=[np.asarray(b, dtype=float) for b in params["bkgs"]],
                    fwhm=params["fwhm"],
                    nfev=result.nfev,
                    cost=result.cost,
                    optimality=result.optimality,
                    njev=result.njev,
                    status=result.status,
                    message=result.message,
                    outcome="ok" if result.success else "nonconvergent",
                )
            except _WfFitTimeoutError:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s TIMED OUT after %.1fs", label, elapsed)
                return _LstsqFitResult.failed(
                    n_zk=len(noll_indices),
                    n_donuts=n,
                    elapsed=elapsed,
                    error=f"timeout after {timeout:.0f}s",
                    outcome="timeout",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.log.warning("WF %s FAILED in %.1fs: %s", label, elapsed, exc)
                return _LstsqFitResult.failed(
                    n_zk=len(noll_indices),
                    n_donuts=n,
                    elapsed=elapsed,
                    error=str(exc),
                    outcome="exception",
                )
