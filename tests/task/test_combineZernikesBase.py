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

import numbers
import unittest

import lsst.pipe.base as pipeBase
import numpy as np
from lsst.ts.wep.task.combineZernikesBase import CombineZernikesBaseTask
from astropy.table import Table


class TestCombineZernikesBaseTask(unittest.TestCase):
    def prepareTestTable(self) -> Table:
        label = ["average"] + [f"pair{i}" for i in range(10)]
        used = [True] + 10 * [False]
        z4 = [-1] + list(np.arange(10))
        table = Table([label, used, z4], names=["label", "used", "Z4"])
        table.meta["opd_columns"] = ["Z4"]
        table.meta["intrinsic_columns"] = []
        table.meta["deviation_columns"] = []
        return table

    def testAbstractClassTypeError(self) -> None:
        # Without a combineZernikes method the class
        # should not be built
        with self.assertRaises(TypeError):
            CombineZernikesBaseTask()

    def testSubclassWorks(self) -> None:
        class TestCombineClass(CombineZernikesBaseTask):
            def _combineZernikes(self, zkTable: Table) -> Table:
                return zkTable

        table = self.prepareTestTable()

        task = TestCombineClass()
        taskOutput = task.run(table)
        self.assertEqual(type(taskOutput), pipeBase.Struct)
        self.assertTrue(all(taskOutput.combinedTable == table))
        self.assertTrue(isinstance(taskOutput.flags[0], numbers.Integral))

        # Test Metadata stored
        self.assertEqual(task.metadata["numDonutsTotal"], 10)
        self.assertEqual(task.metadata["numDonutsUsed"], 0)
        self.assertEqual(task.metadata["numDonutsRejected"], 10)
        self.assertListEqual(task.metadata.arrays["combineZernikesFlags"], list(np.ones(10)))
