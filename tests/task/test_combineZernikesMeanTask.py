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
from lsst.ts.wep.task.combineZernikesMeanTask import CombineZernikesMeanTask


class TestCombineZernikesMeanTask(unittest.TestCase):
    def setUp(self) -> None:
        self.task = CombineZernikesMeanTask()

    def prepareTestTable(self) -> Table:
        label = ["average"] + [f"pair{i}" for i in range(10)]
        used = [True] + 10 * [True]
        z4 = [-1.0] + list(np.arange(10))
        table = Table([label, used, z4], names=["label", "used", "Z4"])
        table.meta["opd_columns"] = ["Z4"]
        table.meta["intrinsic_columns"] = []
        table.meta["deviation_columns"] = []
        return table

    def testCombineZernikes(self) -> None:
        inTable = self.prepareTestTable()
        outTable = self.task.combineZernikes(inTable)
        self.assertTrue(all(outTable["used"]))
        self.assertEqual(outTable[outTable["label"] == "average"]["Z4"], np.arange(10).mean())

    def testCombineZernikesWithRejection(self) -> None:
        inTable = self.prepareTestTable()
        inTable["used"][2] = False  # Reject one of the pairs
        outTable = self.task.combineZernikes(inTable)
        # Avg should be used, pairs should be used except the rejected one
        self.assertTrue([outTable["used"][:2]] + list(outTable["used"][3:]))
        expected_mean = (np.arange(10).sum() - 1) / 9  # Exclude the rejected pair
        self.assertEqual(outTable[outTable["label"] == "average"]["Z4"], expected_mean)

    def testTaskRun(self) -> None:
        inTable = self.prepareTestTable()
        output = self.task.run(inTable.copy())
        self.assertEqual(type(output), pipeBase.Struct)

        outTable = output.combinedTable
        self.assertTrue(all(outTable["used"]))
        self.assertEqual(outTable[outTable["label"] == "average"]["Z4"], np.arange(10).mean())

        flags = output.flags
        self.assertTrue(all(flags == 0))
