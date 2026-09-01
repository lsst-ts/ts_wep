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
    centroid_x_raw: float
    centroid_y_raw: float
    id: int
    inner_frac: float
    outer_frac: float
    outer_sector_minmax_frac: float
    field_dist_deg: float
    donut_radius: float
    obscuration: float
    snr: float
    bkg: float
    bkg_std: float
    nearest_neighbor_dist_px: float
    n_neighbors_in_stamp: int
    catalog_centroid_offset_px: float
    n_quarter: int
    nearby_photo: list[tuple[float, float, float]]
    nearby_astrom: list[tuple[float, float, float]]
    intrinsic_zk: npt.NDArray[np.float64] | None = None
    # --- reject flags (default False = not rejected) ---
    rejected_sat: bool = False
    rejected_inner_frac: bool = False
    rejected_outer_frac: bool = False
    rejected_snr: bool = False
    rejected: bool = False


@dataclass
class WfResult:
    """One donut's wavefront-fit outputs, produced by `_wf_worker`.

    Consumed by `_buildCatalog`, keyed by ``(donut_id, det_name)``. A fit that
    timed out or raised still produces a WfResult with ``fit_success=False``
    and all-NaN Zernikes. `_NULL_WF` is the sentinel for "no fit consumed this
    donut" (paired-mode surplus with no partner).
    """

    donut_id: int
    det_name: str
    defocal: str
    zk_dev: npt.NDArray[np.float64]        # dense Noll 0.._ZK_JMAX, metres, NaN where unfit
    zk_intrinsic: npt.NDArray[np.float64]  # dense Noll 0.._ZK_JMAX, metres
    img: np.ndarray | None
    model_img: np.ndarray | None
    fit_success: bool
    fit_elapsed: float
    setup_elapsed: float
    fit_nfev: int
    fit_cost: float
    fit_dx: float
    fit_dy: float
    fit_flux: float
    fit_fwhm: float
    blend_frac: float
    group_id: str
    group_size: int
    fit_mode: str


# Sentinel for "no fit consumed this donut". All-NaN Zernikes, empty strings,
# fit_success=False -- so _buildCatalog's `used` derivation is naturally False.
_NULL_WF = WfResult(
    donut_id=-1,
    det_name="",
    defocal="",
    zk_dev=np.full(_ZK_JMAX + 1, np.nan),
    zk_intrinsic=np.full(_ZK_JMAX + 1, np.nan),
    img=None,
    model_img=None,
    fit_success=False,
    fit_elapsed=float("nan"),
    setup_elapsed=float("nan"),
    fit_nfev=0,
    fit_cost=float("nan"),
    fit_dx=float("nan"),
    fit_dy=float("nan"),
    fit_flux=float("nan"),
    fit_fwhm=float("nan"),
    blend_frac=float("nan"),
    group_id="",
    group_size=0,
    fit_mode="",
)


@dataclass
class _WfGroup:
    donuts: list  # donut dicts, ordered; each carries its own "det_name" key
    group_id: str
    band: str
    rtp: float | None  # Boresight rotation (spider angle), degrees or None
    alt: float | None  # Boresight altitude, radians or None
