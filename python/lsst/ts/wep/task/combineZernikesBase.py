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

__all__ = ["CombineZernikesBaseConfig", "CombineZernikesBaseTask"]

import abc
import logging
from typing import Any

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy.table import Table


class CombineZernikesBaseConfig(pexConfig.Config):
    pass


class CombineZernikesBaseTask(pipeBase.Task, metaclass=abc.ABCMeta):
    """
    Base class for algorithms that combine Zernikes from the individual
    pairs of donuts on a detector into a single array of Zernike values
    for that detector.
    """

    ConfigClass = CombineZernikesBaseConfig
    _DefaultName = "combineZernikes"

    def __init__(self, **kwargs: Any) -> None:
        pipeBase.Task.__init__(self, **kwargs)
        self.log = logging.getLogger(type(self).__name__)  # type: ignore

    @staticmethod
    def _setAvg(zkTable: Table, colName: str, function: callable, useIdx: list | None = None) -> None:
        """Set average value for a Zernike column.

        This is an abstract method meant to be used in subclasses.
        It sets the "average" for the given column using the function,
        only considering the rows indicated by useIdx.

        Parameters
        ----------
        zkTable : `astropy.table.Table`
            The full zernike table, to be altered in place.
        colName : `str`
            The name of the column to set the average value for.
        function : `callable`
            The function to use to calculate the average value.
            It should take a single argument which is an array
            of values to average.
        useIdx : `list` of `int` or `None`, optional
            The indices of the rows to use when calculating
            the average value. If None, all rows are used.
            (default is None)
        """
        if useIdx is None:
            useIdx = list(range(len(zkTable[zkTable["label"] != "average"])))

        avg = function(zkTable[zkTable["label"] != "average"][colName][useIdx])
        zkTable[colName][zkTable["label"] == "average"] = avg

    def run(self, zkTable: Table) -> pipeBase.Struct:
        """
        Combine the zernikes from the input array of Zernike
        coefficients from each individual donut pair.

        Parameters
        ----------
        zkTable : `astropy.table.Table`
            Table containing zernike coefficients for each donut (pair).

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
            - combinedTable : `astropy.table.Table`
                The input table with the averaged Zernike coefficients and
                combination flags added.
        """
        combinedTable = self.combineZernikes(zkTable)

        # Make sure that flags contains only integers
        flags = np.array(~combinedTable[combinedTable["label"] != "average"]["used"], dtype=int)
        self.log.info(
            f"Using {len(flags) - np.sum(flags)} pairs out of {len(flags)} in final Zernike estimate."
        )

        # Save flags and summary values in task metadata
        self.metadata["numDonutsTotal"] = len(flags)
        self.metadata["numDonutsUsed"] = len(flags) - np.sum(flags)
        self.metadata["numDonutsRejected"] = np.sum(flags)
        self.metadata["combineZernikesFlags"] = flags.tolist()
        return pipeBase.Struct(combinedTable=combinedTable, flags=flags)

    @abc.abstractmethod
    def _combineZernikes(self, zkTable: Table) -> None:
        """
        Abstract method to be implemented by subclasses.
        This method should implement the specific algorithm to
        combine the Zernike coefficients from each individual donut
        pair into a single set of coefficients for the detector.

        This function should modify the provided table in-place.

        Parameters
        ----------
        zkTable : `astropy.table.Table`
            Table containing zernike coefficients for each donut (pair).
        """
        raise NotImplementedError("Subclasses must implement _combineZernikes method.")

    def combineZernikes(self, zkTable: Table) -> Table:
        """Combine Zernikes from each donut (pair) into single set for detector.

        Parameters
        ----------
        zkTable : `astropy.table.Table`
            Table containing zernike coefficients for each donut (pair).

        Returns
        -------
        `astropy.table.Table`
            The input table with the averaged Zernike coefficients and
            combination flags added.
        """
        # This is to protect the input table from modification
        combinedTable = zkTable.copy()
        self._combineZernikes(combinedTable)
        return combinedTable
