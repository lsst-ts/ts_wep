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

__all__ = [
    "EnumDict",
    "BandLabel",
    "BscDbType",
    "CentroidFindType",
    "DeblendDonutType",
    "DefocalType",
    "ImageType",
    "PlaneType",
    "WfAlgorithmName",
]

from collections import UserDict
from enum import Enum
from typing import Any, Type


class EnumDict(UserDict):
    """A dictionary that aliases Enums and their corresponding values."""

    def __init__(self, enum: Type[Enum], regularDict: dict | None = None) -> None:
        super().__init__()
        self._enum = enum
        if regularDict is not None:
            for key, val in regularDict.items():
                self[key] = val

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(self._enum(key), value)

    def __getitem__(self, key: Any) -> None:
        return super().__getitem__(self._enum(key))


class BandLabel(Enum):
    LSST_U = "u"
    LSST_G = "g"
    LSST_R = "r"
    LSST_I = "i"
    LSST_Z = "z"
    LSST_Y = "y"
    REF = "ref"


class BscDbType(Enum):
    LocalDb = "localDb"
    LocalDbForStarFile = "file"


class CentroidFindType(Enum):
    RandomWalk = "randomWalk"
    Otsu = "otsu"
    ConvolveTemplate = "convolveTemplate"


class DeblendDonutType(Enum):
    Adapt = "adapt"


class DefocalType(Enum):
    Intra = "intra"
    Extra = "extra"


class ImageType(Enum):
    Amp = "amp"
    Eimg = "eimage"


class PlaneType(Enum):
    """Specifies whether the image is on the image or pupil plane."""

    Image = "image"
    Pupil = "pupil"


class WfAlgorithmName(Enum):
    TIE = "tie"
    Danish = "danish"
    AiDonut = "aidonut"
