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

"""Records passed between the blitz pipeline stages."""

__all__ = ["Donut", "WfResult"]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .utils import _ZK_JMAX


@dataclass
class Donut:
    """One cut donut stamp with its selection/quality metrics."""

    det_name: str
    stamp: np.ndarray  #  CCS
    thx_ccs: float
    thy_ccs: float
    flux: float
    band: str
    det_id: int
    visit_id: int
    x_det: float
    y_det: float
    donut_id: int
    inner_frac: float
    outer_frac: float
    outer_sector_minmax_frac: float
    donut_radius: float
    snr: float
    bkg: float
    bkg_std: float
    n_quarter: int
    # This donut's own refcat magnitudes, NaN on the blind-detection path where
    # there is no refcat to take them from.
    photo_mag: float
    astrom_mag: float
    # Neighboring refcat sources inside the stamp box as (dx, dy, mag), offset
    # from (x_det, y_det) so the offsets compose with it directly. Excludes this
    # donut itself. Stamp *membership* is decided on the rounded centroid, which
    # is what the stamp bounds were cut on.
    nearby_photo: list[tuple[float, float, float]]
    nearby_astrom: list[tuple[float, float, float]]
    # Refcat sky position in radians, NaN on the blind-detection path. Refcat
    # truth, not a projection of (x_det, y_det) through the WCS -- so a finite
    # value here means this donut was matched to a catalog source.
    coord_ra: float = float("nan")
    coord_dec: float = float("nan")
    intrinsic_zk: npt.NDArray[np.float64] | None = None
    # The optic shifts that put this donut off focus: signed meters, ordered
    # (detector, camera, m2) -- see `_telescope_for_offsets`. Set by whichever
    # task builds the donut, because the two modes decide it differently: corner
    # mode from the detector id (SW0/SW1 sit either side of focus within one
    # exposure), FAM from which exposure of the pair it came from. This is the
    # only representation of defocal state; an intra/extra label would be
    # redundant with it, and derivable from det_name (corner) or visit_id (FAM).
    defocal_offsets: tuple[float, float, float] | None = None
    # --- reject flags (default False = not rejected) ---
    rejected_sat: bool = False
    rejected_inner_frac: bool = False
    rejected_outer_frac: bool = False
    rejected_snr: bool = False
    rejected: bool = False


# The complete vocabulary of `WfResult.fit_outcome`, which says which of the
# mutually exclusive paths through `WavefrontFittingTask._run_lstsq_fit`
# produced a result.
_FIT_OUTCOMES = (
    "ok",             # least_squares converged (success=True)
    "nonconvergent",  # least_squares returned but success=False
    "timeout",        # SIGALRM fired; wfFitTimeoutPerDonut * group_size exceeded
    "exception",      # the fit raised
    "x0_only",        # wfInitialGuessOnly: the model was evaluated, never fit
    "",               # no fit consumed this donut (`_NULL_WF`)
)


@dataclass
class WfResult:
    """One donut's wavefront-fit outputs, produced by `_wf_worker`.

    Consumed by `build_donut_catalog`, keyed by ``(donut_id, det_name,
    visit_id)``. A fit that timed out or raised still produces a WfResult with
    ``fit_success=False`` and all-NaN Zernikes; ``fit_outcome`` says which.

    ``visit_id`` is part of the key because full-array mode fits the same star on
    the same detector twice, once per side of focus, so ``(donut_id, det_name)``
    alone is ambiguous there. Corner mode has one visit per quantum, so including
    it changes nothing.
    """

    donut_id: int
    det_name: str
    visit_id: int
    zk_dev: npt.NDArray[np.float64]        # dense Noll 0.._ZK_JMAX, meters, NaN where unfit
    zk_intrinsic: npt.NDArray[np.float64]  # dense Noll 0.._ZK_JMAX, meters
    img: np.ndarray | None
    model_img: np.ndarray | None
    fit_success: bool
    fit_elapsed: float
    setup_elapsed: float
    fit_nfev: int
    fit_cost: float
    fit_optimality: float
    fit_njev: int
    fit_outcome: str                       # one of _FIT_OUTCOMES
    fit_dx: float
    fit_dy: float
    fit_flux: float
    fit_fwhm: float
    blend_frac: float
    group_id: str
    group_size: int


# Sentinel for "no fit consumed this donut". All-NaN Zernikes, empty strings,
# fit_success=False -- so the catalog's empty group_id and empty
# group_fit_outcome both fall out naturally.
_NULL_WF = WfResult(
    donut_id=-1,
    det_name="",
    visit_id=-1,
    zk_dev=np.full(_ZK_JMAX + 1, np.nan),
    zk_intrinsic=np.full(_ZK_JMAX + 1, np.nan),
    img=None,
    model_img=None,
    fit_success=False,
    fit_elapsed=float("nan"),
    setup_elapsed=float("nan"),
    fit_nfev=0,
    fit_cost=float("nan"),
    fit_optimality=float("nan"),
    fit_njev=0,
    fit_outcome="",
    fit_dx=float("nan"),
    fit_dy=float("nan"),
    fit_flux=float("nan"),
    fit_fwhm=float("nan"),
    blend_frac=float("nan"),
    group_id="",
    group_size=0,
)


@dataclass
class _WfGroup:
    donuts: list  # donut dicts, ordered; each carries its own "det_name" key
    group_id: str
    band: str
    rtp: float | None  # Boresight rotation (spider angle), degrees or None
    alt: float | None  # Boresight altitude, radians or None
