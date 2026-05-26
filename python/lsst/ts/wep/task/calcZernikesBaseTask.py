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
    "CalcZernikesBaseTask",
]

import abc
from itertools import zip_longest
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import QTable

import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.donutStamps import DonutStamps

# Define the position 2D float dtype for the zernikes table
pos2f_dtype = np.dtype([("x", "<f4"), ("y", "<f4")])


class CalcZernikesBaseTask(pipeBase.PipelineTask, metaclass=abc.ABCMeta):
    """Base class for Zernike coefficient table creation and management.

    This class provides shared utility methods for creating and populating
    Zernike coefficient tables used by both paired (CalcZernikesTask) and
    neural network (CalcZernikesNeuralTask) implementations.

    Attributes
    ----------
    nollIndices : list[int]
        List of Noll indices for the Zernike coefficients.
    stampsIntra : DonutStamps
        Intra-focal donut stamps.
    stampsExtra : DonutStamps
        Extra-focal donut stamps.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base class.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments (reserved for subclass use).
        """
        self.nollIndices: list[int] = []
        self.stampsIntra: DonutStamps = DonutStamps([])
        self.stampsExtra: DonutStamps = DonutStamps([])

    def initZkTable(self) -> QTable:
        """Initialize the table to store the Zernike coefficients

        Returns
        -------
        table : `astropy.table.QTable`
            Table to store the Zernike coefficients
        """
        dtype = [
            ("label", "<U12"),
            ("used", np.bool_),
            ("intra_field", pos2f_dtype),
            ("extra_field", pos2f_dtype),
            ("intra_centroid", pos2f_dtype),
            ("extra_centroid", pos2f_dtype),
            ("intra_mag", "<f4"),
            ("extra_mag", "<f4"),
            ("intra_sn", "<f4"),
            ("extra_sn", "<f4"),
            ("intra_entropy", "<f4"),
            ("extra_entropy", "<f4"),
            ("intra_frac_bad_pix", "<f4"),
            ("extra_frac_bad_pix", "<f4"),
            ("intra_max_power_grad", "<f4"),
            ("extra_max_power_grad", "<f4"),
            ("intra_donut_id", "<U21"),
            ("extra_donut_id", "<U21"),
        ]
        for j in self.nollIndices:
            dtype.append((f"Z{j}", "<f4"))
        for j in self.nollIndices:
            dtype.append((f"Z{j}_intrinsic", "<f4"))
        for j in self.nollIndices:
            dtype.append((f"Z{j}_deviation", "<f4"))

        table = QTable(dtype=dtype)

        # Assign units where appropriate
        table["intra_field"].unit = u.deg
        table["extra_field"].unit = u.deg
        table["intra_centroid"].unit = u.pixel
        table["extra_centroid"].unit = u.pixel
        for j in self.nollIndices:
            table[f"Z{j}"].unit = u.nm
        for j in self.nollIndices:
            table[f"Z{j}_intrinsic"].unit = u.nm
        for j in self.nollIndices:
            table[f"Z{j}_deviation"].unit = u.nm

        return table

    def createZkTable(self, zkCoeffRaw: pipeBase.Struct) -> QTable:
        """Create the Zernike table to store Zernike Coefficients.

        Note this is written with the assumption that either extraStamps or
        intraStamps MIGHT be empty. This is because calcZernikesUnpairedTask
        also uses this method.

        Parameters
        ----------
        zkCoeffRaw: pipeBase.Struct
            All zernikes returned by calcZernikesTask.estimateZernikes.run(...)

        Returns
        -------
        table : `astropy.table.QTable`
            Table with the Zernike coefficients
        """
        zkTable = self.initZkTable()
        zkTable.add_row(
            {
                "label": "average",
                "used": True,
                **{f"Z{j}": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
                **{f"Z{j}_intrinsic": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
                **{f"Z{j}_deviation": np.nan * u.micron for i, j in enumerate(self.nollIndices)},
                "intra_field": np.nan,
                "extra_field": np.nan,
                "intra_centroid": np.nan,
                "extra_centroid": np.nan,
                "intra_mag": np.nan,
                "extra_mag": np.nan,
                "intra_sn": np.nan,
                "extra_sn": np.nan,
                "intra_entropy": np.nan,
                "extra_entropy": np.nan,
                "intra_frac_bad_pix": np.nan,
                "extra_frac_bad_pix": np.nan,
                "intra_max_power_grad": np.nan,
                "extra_max_power_grad": np.nan,
                "intra_donut_id": "",
                "extra_donut_id": "",
            }
        )
        for i, (intra, extra, zk) in enumerate(
            zip_longest(
                self.stampsIntra,
                self.stampsExtra,
                zkCoeffRaw.zernikes,
            )
        ):
            # If zk is None, we need to stop. This can happen when running
            # paired Zernike estimation and the number of intra/extra stamps
            # is not the same
            if zk is None:
                break

            # Assign units
            zk = zk * u.micron

            # Unpack data from stamps, handling cases with None stamps
            intraAngle, intraCentroid, intraIntrinsics = self._unpackStampData(intra)
            extraAngle, extraCentroid, extraIntrinsics = self._unpackStampData(extra)

            # Average the intrinsics
            intrinsics = np.nanmean((intraIntrinsics, extraIntrinsics), axis=0) * u.micron

            # Calculate the wavefront deviation
            deviation = zk - intrinsics

            row: dict = dict()
            row["label"] = f"pair{i + 1}"
            row["used"] = False  # Placeholder for now
            row.update({f"Z{j}": zk[i] for i, j in enumerate(self.nollIndices)})
            row.update({f"Z{j}_intrinsic": intrinsics[i] for i, j in enumerate(self.nollIndices)})
            row.update({f"Z{j}_deviation": deviation[i] for i, j in enumerate(self.nollIndices)})
            row["intra_field"] = intraAngle
            row["extra_field"] = extraAngle
            row["intra_centroid"] = intraCentroid
            row["extra_centroid"] = extraCentroid
            for key in ["MAG", "SN", "ENTROPY", "FRAC_BAD_PIX", "MAX_POWER_GRAD", "DONUT_ID"]:
                for stamps, foc in [
                    (self.stampsIntra, "intra"),
                    (self.stampsExtra, "extra"),
                ]:
                    if len(stamps) > 0 and key in stamps.metadata:
                        val = stamps.metadata.getArray(key)[i]
                    else:
                        val = "" if key == "DONUT_ID" else np.nan
                    row[f"{foc}_{key.lower()}"] = val
            zkTable.add_row(row)

        zkTable.meta = self.createZkTableMetadata()

        return zkTable

    def createZkTableMetadata(self) -> dict:
        """Create the metadata for the Zernike table.

        Returns
        -------
        metadata : dict
            Metadata for the Zernike table
        """
        meta: dict = {}
        meta["intra"] = {}
        meta["extra"] = {}
        cam_name = None

        if not self.stampsIntra.metadata and not self.stampsExtra.metadata:
            raise ValueError("No metadata in either DonutStamps object. Cannot create Zk table metadata.")

        for dict_, stamps in [
            (meta["intra"], self.stampsIntra),
            (meta["extra"], self.stampsExtra),
        ]:
            if not stamps.metadata:
                continue
            dict_["det_name"] = stamps.metadata["DET_NAME"]
            dict_["visit"] = stamps.metadata["VISIT"]
            dict_["dfc_dist"] = stamps.metadata["DFC_DIST"]
            dict_["band"] = stamps.metadata["BANDPASS"]
            dict_["boresight_rot_angle_rad"] = stamps.metadata["BORESIGHT_ROT_ANGLE_RAD"]
            dict_["boresight_par_angle_rad"] = stamps.metadata["BORESIGHT_PAR_ANGLE_RAD"]
            dict_["boresight_alt_rad"] = stamps.metadata["BORESIGHT_ALT_RAD"]
            dict_["boresight_az_rad"] = stamps.metadata["BORESIGHT_AZ_RAD"]
            dict_["boresight_ra_rad"] = stamps.metadata["BORESIGHT_RA_RAD"]
            dict_["boresight_dec_rad"] = stamps.metadata["BORESIGHT_DEC_RAD"]
            dict_["mjd"] = stamps.metadata["MJD"]
            if cam_name is None:
                cam_name = stamps.metadata["CAM_NAME"]

        meta["cam_name"] = cam_name
        meta["noll_indices"] = self.nollIndices.list()
        meta["opd_columns"] = [f"Z{j}" for j in self.nollIndices]
        meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in self.nollIndices]
        meta["deviation_columns"] = [f"Z{j}_deviation" for j in self.nollIndices]

        if self.stampsIntra.metadata and self.stampsExtra.metadata:
            assert self.stampsIntra.metadata["CAM_NAME"] == self.stampsExtra.metadata["CAM_NAME"]

        return meta
