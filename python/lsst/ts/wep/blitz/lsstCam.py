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

"""Inventory of LSSTCam facts blitz depends on.

Blitz is LSSTCam-only by design -- no AuxTel, no ComCam -- which is what makes
a frozen dataclass of literals sufficient here.

This module imports nothing from blitz, so anything in blitz may import it.
"""

__all__ = []

from dataclasses import dataclass
from typing import Mapping

import galsim
import numpy as np
from frozendict import frozendict


@dataclass(frozen=True)
class _LsstCamConstants:
    """LSSTCam geometry for blitz."""

    # Radius the Zernike coefficients are normalized on, in meters.  Named for
    # that role because it is the load-bearing one: it is what the danish
    # factory is told, and what `_rescale_zk_domain` targets.
    #
    # Deliberately a blitz constant rather than the optics model's own pupil
    # radius.  batoid's `pupilSize/2` is the model's aperture, and it moves
    # between models -- 4.18 for LSST_r and Rubin_v3.14_r, 4.165 for
    # Rubin_v1000_r.  A normalization radius that moved with the model would
    # change what every reported coefficient means, so comparing two models'
    # output would be measuring the normalization as much as the optics.
    zk_r_outer: float = 4.18

    # Fractional pupil central obscuration.  Used as the annular Zernike
    # normalization, but also used as pupil geometry for the annular
    # detection template and photometric mask.  Distinct from the danish mask
    # model.
    obscuration: float = 0.612

    # Effective focal length in meters.  Also handed to `batoid.zernikeTA`,
    # which would otherwise derive its own value.  Fixing it here and in danish
    # keeps these consistent.
    focal_length: float = 10.312

    pixel_size: float = 1e-05

    # Detector z shift producing the nominal 1.5 mm defocus, in meters.
    #
    # Not exactly 1.5e-3: the nominal offset is specified as a shift of the
    # Detector in the batoid model, and the equivalent detector-plane offset
    # that reproduces its Z4 is solved numerically, landing 1.5e-9 m short.
    # Carrying the solved value keeps `donut_radius` and the reported
    # `dfc_dist` consistent with that definition.
    defocal_offset: float = 0.0014999985450571823

    # Band -> effective wavelength in meters.
    wavelength: Mapping[str, float] = frozendict(
        {
            "u": 3.709e-07,
            "g": 4.767e-07,
            "r": 6.194e-07,
            "i": 7.539e-07,
            "z": 8.668e-07,
            "y": 9.739e-07,
        }
    )

    @property
    def zk_r_inner(self) -> float:
        """Inner radius the coefficients are normalized on, in meters."""
        return self.zk_r_outer * self.obscuration

    @property
    def focal_ratio(self) -> float:
        """The f-number."""
        return self.focal_length / (2 * self.zk_r_outer)

    @property
    def donut_radius(self) -> float:
        """Nominal donut radius in un-binned pixels.

        A fallback: callers prefer the per-exposure radius measured by
        `DonutDetectDiameterTask` and come here when that could not be formed.
        """
        r_meters = self.defocal_offset / np.sqrt(4 * self.focal_ratio**2 - 1)
        return r_meters / self.pixel_size


_LSSTCAM = _LsstCamConstants()


def _rescale_zk_domain(
    coef: np.ndarray,
    *,
    r_outer_from: float,
    r_inner_from: float,
    r_outer_to: float,
    r_inner_to: float,
) -> np.ndarray:
    """Re-express annular Zernike coefficients on a different radial domain.

    Annular Zernike coefficients only mean something relative to the radii they
    are normalized on, and blitz normalizes on `_LSSTCAM.zk_r_outer` /
    `.zk_r_inner`.  `batoid.zernikeTA` instead always fits on the pupil of the
    model it traced -- ``pupilSize/2`` and ``eps*pupilSize/2``, see its
    ``_dZernikeBasis`` call -- so its output has to be brought onto blitz's
    domain before it can be combined with anything else.  Danish likewise adds
    reference coefficients to fitted deviations on the factory's domain
    (``danish/donut_model.py:605-606``), and the factory gets blitz's radii.

    The two domains coincide for the default optics model, where
    ``LSST_r.yaml``'s ``pupilSize`` is 8.36 and ``pupilSize/2`` equals 4.18
    bitwise.  They do not for ``Rubin_v1000_r.yaml`` (``pupilSize: 8.33``),
    where they differ by 0.36% and defocus shifts ~183 nm.

    Notes
    -----
    Uses private galsim API (``Zernike._coef_array_xy``,
    ``Zernike._from_coef_array_xy``) because the xy basis is the only
    domain-independent representation galsim exposes.  A domain argument on
    ``batoid.zernikeTA`` would make this function unnecessary.

    A ``coef`` not spanning a complete radial order returns a *longer* array,
    padded up to the next complete order; the padding is numerically zero (a
    length-61 input returns length 67 with its tail at 1e-11 nm).

    Parameters
    ----------
    coef : `np.ndarray`
        Noll-indexed annular Zernike coefficients; index 0 is meaningless.
    r_outer_from, r_inner_from : `float`
        Outer and inner radius, in meters, that ``coef`` is normalized on.
    r_outer_to, r_inner_to : `float`
        Outer and inner radius, in meters, to re-express ``coef`` on.

    Returns
    -------
    `np.ndarray`
        Coefficients on the ``*_to`` domain, describing the same wavefront as
        the input.
    """
    source = galsim.zernike.Zernike(coef, R_outer=r_outer_from, R_inner=r_inner_from)
    target = galsim.zernike.Zernike._from_coef_array_xy(
        source._coef_array_xy, R_outer=r_outer_to, R_inner=r_inner_to
    )
    return target.coef
