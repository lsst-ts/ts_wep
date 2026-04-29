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
import unittest

import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.combineZernikesWeightedTask import CombineZernikesWeightedTask

ZK_COLS = [f"Z{i}" for i in range(4, 10)]
ZK_INTR = [f"Z{i}_intrinsic" for i in range(4, 10)]
ZK_DEV = [f"Z{i}_deviation" for i in range(4, 10)]
ALL_COLS = ZK_COLS + ZK_INTR + ZK_DEV


def _makeTable(nPairs: int, weights: np.ndarray, outlierPairIdx: int | None = None) -> Table:
    """Build a minimal zkTable compatible with CombineZernikesSigmaClipTask."""
    label = ["average"] + [f"pair{i}" for i in range(nPairs)]
    used = [True] + nPairs * [False]
    cols: dict = {"label": label, "used": used}
    for j, col in enumerate(ALL_COLS):
        # pair i in column j has value j + i, so pairs differ from each other
        vals = [float(j + i) for i in range(nPairs)]
        if outlierPairIdx is not None:
            vals[outlierPairIdx] = 1e6
        cols[col] = [-1.0] + vals
    table = Table(cols)
    table.meta["opd_columns"] = ZK_COLS
    table.meta["intrinsic_columns"] = ZK_INTR
    table.meta["deviation_columns"] = ZK_DEV
    table.meta["noll_indices"] = list(range(4, 10))
    table.meta["estimatorInfo"] = {"weight": list(weights)}
    return table


class TestCombineZernikesWeightedTask(unittest.TestCase):
    def setUp(self) -> None:
        self.task = CombineZernikesWeightedTask()

    def testCombineZernikes(self) -> None:
        nPairs = 10
        inTable = _makeTable(nPairs, np.ones(nPairs))
        outTable = self.task.combineZernikes(inTable)
        self.assertTrue(all(outTable["used"]))
        for j, col in enumerate(ALL_COLS):
            expected = float(j)
            self.assertAlmostEqual(float(outTable[outTable["label"] == "average"][col][0]), expected)

    def testCombineZernikesVaryingWeights(self) -> None:
        nPairs = 10
        weights = np.zeros(nPairs)
        weights[9] = 1.0
        inTable = _makeTable(nPairs, weights)
        outTable = self.task.combineZernikes(inTable)
        self.assertTrue(all(outTable["used"]))
        # All weight on pair 9: expected = j + 9, not unweighted j + 4.5
        for j, col in enumerate(ALL_COLS):
            self.assertAlmostEqual(float(outTable[outTable["label"] == "average"][col][0]), float(j + 9))

    def testCombineZernikesWithRejection(self) -> None:
        nPairs = 10
        inTable = _makeTable(nPairs, np.ones(nPairs), outlierPairIdx=0)
        outTable = self.task.combineZernikes(inTable)
        pairUsed = outTable[outTable["label"] != "average"]["used"]
        self.assertFalse(pairUsed[0])
        self.assertTrue(all(pairUsed[1:]))

    def testTaskRun(self) -> None:
        nPairs = 10
        inTable = _makeTable(nPairs, np.ones(nPairs))
        output = self.task.run(inTable.copy())
        self.assertEqual(type(output), pipeBase.Struct)
        outTable = output.combinedTable
        self.assertTrue(all(outTable["used"]))
        for j, col in enumerate(ALL_COLS):
            self.assertAlmostEqual(float(outTable[outTable["label"] == "average"][col][0]), float(j))
        flags = output.flags
        self.assertTrue(all(flags == 0))
