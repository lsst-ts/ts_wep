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
__all__ = ["CombineZernikesWeightedTask", "CombineZernikesWeightedConfig"]

import numpy as np
from astropy.table import Table

from lsst.ts.wep.task.combineZernikesSigmaClipTask import CombineZernikesSigmaClipTask


def _weightedMean(vals: np.ndarray, weights: np.ndarray | None) -> np.float64:
    """Compute a weighted mean of finite values.

    Parameters
    ----------
    vals : `numpy.ndarray`
        Values to average.
    weights : `numpy.ndarray` or None
        Weights for each value. If None, an unweighted mean is used.

    Returns
    -------
    mean : `np.float64`
        Weighted mean of finite values, or NaN if none are finite.
    """
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.float64(np.nan)
    if weights is None:
        return np.nanmean(vals)
    return np.average(vals[finite], weights=weights[finite])


class CombineZernikesWeightedConfig(CombineZernikesSigmaClipTask.ConfigClass):
    """Configuration for CombineZernikesWeightedTask."""

    pass


class CombineZernikesWeightedTask(CombineZernikesSigmaClipTask):
    """Combine Zernike coefficients using sigma clipping then weighted average.

    Inherits sigma clipping from ``CombineZernikesSigmaClipTask`` to reject
    outlier pairs, then applies a confidence-weighted average over the
    remaining pairs using weights from
    ``zkTable.meta["estimatorInfo"]["weight"]``, set by the estimator task
    (e.g. AiDonut).
    """

    ConfigClass = CombineZernikesWeightedConfig

    def _combineZernikes(self, zkTable: Table) -> Table:
        """Combine Zernike coefficients with sigma clipping then weighted mean.

        Parameters
        ----------
        zkTable : `astropy.table.Table`
            Table of Zernike coefficients to combine.

        Returns
        -------
        zkTable : `astropy.table.Table`
            Table with the average row updated using weighted mean over
            sigma-clipped pairs.
        """
        # Let sigma clip parent set used flags and reject outlier pairs.
        super()._combineZernikes(zkTable)

        # Redo the average using weights over the sigma-clipped survivors.
        use_idx = np.where(zkTable[zkTable["label"] != "average"]["used"])[0]

        estimator_info = zkTable.meta.get("estimatorInfo", {})
        weight = estimator_info.get("weight", None)
        if weight is not None:
            weight = np.array(weight)
            if weight.ndim == 2:
                weight = weight.mean(axis=1)
            weight = weight[use_idx]
            if np.all(np.isfinite(weight)) and weight.sum() > 0:
                weight = weight / weight.sum()
            else:
                weight = None

        def _mean(vals: np.ndarray) -> np.float64:
            return _weightedMean(vals, weight)

        zk_columns = (
            zkTable.meta["opd_columns"]
            + zkTable.meta["intrinsic_columns"]
            + zkTable.meta["deviation_columns"]
        )
        for col in zk_columns:
            self._setAvg(zkTable, col, _mean, useIdx=use_idx)
        return zkTable
