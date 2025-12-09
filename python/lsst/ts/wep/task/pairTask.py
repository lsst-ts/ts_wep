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
    "ExposurePairerConfig",
    "ExposurePairer",
    "TablePairerConfig",
    "TablePairer",
    "GroupPairerConfig",
    "GroupPairer",
]

import typing
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.geom import radians


class IntraExtraIdxPair(typing.NamedTuple):
    intra: int
    extra: int


class ExposurePairerConfig(pexConfig.Config):
    timeThreshold: pexConfig.Field = pexConfig.Field[float](
        doc="Maximum time difference between paired intra- and extra-focal exposures (s)",
        default=300,
    )

    pointingThreshold: pexConfig.Field = pexConfig.Field[float](
        doc="Maximum pointing difference between paired intra- and extra-focal exposures (arcsec)",
        default=60,
    )

    rotationThreshold: pexConfig.Field = pexConfig.Field[float](
        doc="Maximum rotator angle difference between paired intra- and extra-focal exposures (deg)",
        default=1.0,
    )

    ignoreThresholds: pexConfig.Field = pexConfig.Field[bool](
        doc=(
            "If True, ignore time, pointing, and rotation thresholds.  Useful when grouping by exposure.group"
        ),
        default=False,
    )

    doOverrideSeparation: pexConfig.Field = pexConfig.Field[bool](
        doc="Whether to override expected intra-focal to focal separation",
        default=False,
    )

    overrideSeparation: pexConfig.Field = pexConfig.Field[float](
        doc="Expected extra-focal to focal separation (mm).  Note that this is signed such that "
        "a positive value means an extra-focal exposure has a greater focusZ value than an "
        "intra-focal exposure.",
        default=1.5,
    )

    groupingThreshold: pexConfig.Field = pexConfig.Field[float](
        doc="Threshold for assigning visit to intra/extra/focal group as a fraction of the expected "
        "intra-focal to focal separation.",
        default=0.1,
    )

    forceUniquePairs: pexConfig.Field = pexConfig.Field[bool](
        doc="If True, force each extra exposure to be paired with a unique intra exposure.",
        default=True,
    )


class ExposurePairer(pipeBase.Task):
    ConfigClass = ExposurePairerConfig
    _DefaultName = "exposurePairer"
    _needsPairTable = False
    _needsGroupDimension = False
    config: ExposurePairerConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def makeTables(
        self,
        visitInfos: typing.Dict[int, afwImage.VisitInfo],
    ) -> typing.Dict[str, Table]:
        """
        Make tables of intra/extra/focal/and pairs of exposures.

        Parameters
        ----------
        visitInfos : dict
            A dictionary of VisitInfo objects keyed by exposure.

        Returns
        -------
        dict
            A dictionary of astropy tables keyed by type.
        """
        if self.config.doOverrideSeparation:
            separation = self.config.overrideSeparation
        else:
            instrument = next(iter(visitInfos.values())).instrumentLabel
            match instrument:
                case "LSSTCam" | "LSSTComCam" | "LSSTComCamSim":
                    separation = 1.5
                case "LATISS":
                    separation = -0.8

        # Partition intra/extra/focal groups by finding a common offset that
        # maximizes the number of assigned visits.
        # If we're handed a pair of visitInfos with a focusZ difference
        # of ~1.5 mm, then we can't generally decide if that's an
        # intra/focal pair or a focal/extra pair.  Since the AOS pipeline
        # is keyed on extrafocal dataIds, we'll assume we're always handed
        # an extrafocal visitInfo in this case.

        table = Table(
            names=["exposure", "ra", "dec", "mjd", "focusZ", "rtp"],
            dtype=[int, float, float, float, float, float],
        )
        for exposure, visitInfo in visitInfos.items():
            radec = visitInfo.boresightRaDec
            ra = radec.getLongitude().asRadians()
            dec = radec.getLatitude().asRadians()
            mjd = visitInfo.date.toAstropy().mjd
            rtp = (
                visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)
            ).asDegrees()
            focusZ = visitInfo.focusZ
            table.add_row([exposure, ra, dec, mjd, focusZ, rtp])
        table["radec"] = SkyCoord(table["ra"], table["dec"], unit="radian")
        thresh = self.config.groupingThreshold * abs(separation)

        best_n_inliers = 0
        best_offset = None
        for offset in np.unique(table["focusZ"]):
            dz = table["focusZ"] - offset
            n_extra = np.sum(np.abs(dz) < thresh)
            n_focal = np.sum(np.abs(dz + separation) < thresh)
            n_intra = np.sum(np.abs(dz + 2 * separation) < thresh)
            if n_extra + n_focal + n_intra > best_n_inliers:
                best_n_inliers = n_extra + n_focal + n_intra
                best_offset = offset

        extraTable = table[np.abs(table["focusZ"] - best_offset) < thresh]
        focalTable = table[np.abs(table["focusZ"] - best_offset + separation) < thresh]
        intraTable = table[np.abs(table["focusZ"] - best_offset + 2 * separation) < thresh]

        # For each extra focal exposure, find the best intra focal exposure.
        # Note that it's possible that the same intra exposure is paired with
        # multiple extra exposures.
        dtype = [
            ("extra", "<i8"),
            ("intra", "<i8"),
        ]
        for prefix in ["extra", "intra"]:
            for name, dt in extraTable.dtype.fields.items():
                if name in ("exposure", "radec"):
                    continue
                dtype.append((prefix + "_" + name, dt[0]))

        pairTable = Table(dtype=dtype)

        for row in extraTable:
            if self.config.ignoreThresholds:
                nearby = np.ones(len(intraTable), dtype=bool)
            else:
                nearby = np.abs(intraTable["mjd"] - row["mjd"]) * 86400 < self.config.timeThreshold
                nearby &= np.abs(intraTable["rtp"] - row["rtp"]) < self.config.rotationThreshold
                nearby &= row["radec"].separation(intraTable["radec"]).arcsec < self.config.pointingThreshold
            if np.any(nearby):
                nearbyTable = intraTable[nearby]
                # Pick the nearest remaining point in radec
                nearest_idx = np.argmin(row["radec"].separation(nearbyTable["radec"]).deg)
                nearest_row = nearbyTable[nearest_idx]
                value = [row["exposure"], nearest_row["exposure"]]
                for r in [row, nearest_row]:
                    for name in r.dtype.names:
                        if name in ("exposure", "radec"):
                            continue
                        value.append(r[name])
                pairTable.add_row(np.array(value))
                if self.config.forceUniquePairs:
                    idx = np.where(intraTable["exposure"] == nearest_row["exposure"])[0]
                    intraTable.remove_rows(idx)

        # Adjust intra/extraTables to only hold unused exposures
        extraTable = extraTable[~np.isin(extraTable["exposure"], pairTable["extra"])]
        intraTable = intraTable[~np.isin(intraTable["exposure"], pairTable["intra"])]

        return {
            "pairTable": pairTable,
            "unusedIntraTable": intraTable,
            "unusedExtraTable": extraTable,
            "focalTable": focalTable,
        }

    def run(self, visitInfos: typing.Dict[int, afwImage.VisitInfo]) -> typing.List[IntraExtraIdxPair]:
        """
        Pair up the intra- and extra-focal exposures.
        """
        tables = self.makeTables(visitInfos)
        pairTable = tables["pairTable"]
        out = []
        for row in pairTable:
            out.append(IntraExtraIdxPair(row["intra"], row["extra"]))
        return out


class TablePairerConfig(pexConfig.Config):
    pass


class TablePairer(pipeBase.Task):
    ConfigClass = TablePairerConfig
    _DefaultName = "tablePairer"
    _needsPairTable = True
    _needsGroupDimension = False

    def run(
        self,
        visitInfos: typing.Dict[int, afwImage.VisitInfo],
        pairTable: Table,
    ) -> typing.List[IntraExtraIdxPair]:
        """
        Pair up the intra- and extra-focal exposures.
        """
        out = []
        for row in pairTable:
            if row["intra"] in visitInfos and row["extra"] in visitInfos:
                out.append(IntraExtraIdxPair(row["intra"], row["extra"]))
        return out


class GroupPairerConfig(ExposurePairerConfig):
    def setDefaults(self) -> None:
        super().setDefaults()
        self.ignoreThresholds = True

    def validate(self) -> None:
        super().validate()

        if not self.ignoreThresholds:
            raise pexConfig.FieldValidationError(
                self.__class__.ignoreThresholds,
                self,
                "ignoreThresholds must be True for GroupPairer",
            )


class GroupPairer(ExposurePairer):
    ConfigClass = GroupPairerConfig
    _DefaultName = "groupPairer"
    _needsGroupDimension = True

    def run(
        self,
        visitInfos: typing.Dict[int, afwImage.VisitInfo],
    ) -> typing.List[IntraExtraIdxPair]:
        # visitInfos should either be an intra/extra pair, or possibly an
        # intra/extra/in-focus triplet.  Catch that here.
        if len(visitInfos) not in (2, 3):
            self.log.warning("Expected 2 or 3 visits, but got %d.", len(visitInfos))
            self.log.warning("Received exposure ids:")
            for vi in visitInfos.keys():
                self.log.warning("  %d", vi)
            return []
        return super().run(visitInfos)
