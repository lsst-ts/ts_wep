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

__all__ = ["ObservingConditions"]

from dataclasses import dataclass
from typing import Optional

from astropy.coordinates import Angle


@dataclass
class ObservingConditions:
    """Per-observation telescope pointing state for wavefront estimation.

    Parameters
    ----------
    rtp : Angle or None, optional
        Rotation angle of the camera on the telescope.  Note that this is
        not the same as BORESIGHT_ROT_ANG, which is the rotation angle of
        the camera on the sky.
        (the default is None, meaning unknown or not applicable)
    altitude : Angle or None, optional
        Boresight altitude.
        (the default is None, meaning unknown or not applicable)
    """

    rtp: Optional[Angle] = None
    altitude: Optional[Angle] = None
