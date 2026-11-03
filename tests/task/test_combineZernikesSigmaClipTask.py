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

import unittest

import numpy as np
from astropy.table import Table

from lsst.ts.wep.task.combineZernikesSigmaClipTask import (
    CombineZernikesSigmaClipTask,
    CombineZernikesSigmaClipTaskConfig,
)


class TestCombineZernikesSigmaClipTask(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CombineZernikesSigmaClipTaskConfig()
        self.task = CombineZernikesSigmaClipTask()

    def prepareTestTable(self) -> Table:
        label = ["average"] + [f"pair{i}" for i in range(101)]
        used = [True] + 101 * [False]
        table = Table([label, used], names=["label", "used"])

        nollIndices = np.arange(4, 12)
        table.meta["noll_indices"] = nollIndices
        table.meta["opd_columns"] = [f"Z{j}" for j in nollIndices]
        table.meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in nollIndices]
        table.meta["deviation_columns"] = [f"Z{j}_deviation" for j in nollIndices]

        for i in table.meta["noll_indices"]:
            table[f"Z{i}"] = [np.nan] + [101.0] + 50 * [1.0] + 50 * [3.0]
        for i in table.meta["noll_indices"]:
            table[f"Z{i}_intrinsic"] = [np.nan] + 50 * [1.0] + 50 * [3.0] + [101.0]
        for i in table.meta["noll_indices"]:
            table[f"Z{i}_deviation"] = [np.nan] + 49 * [1.0] + 49 * [3.0] + 3 * [101.0]

        return table

    def testValidateConfigs(self) -> None:
        self.assertEqual(
            {"sigma": 3.0, "stdfunc": "mad_std", "maxiters": 1},
            self.task.sigmaClipKwargs,
        )
        self.assertEqual(3, self.task.maxZernClip)

        self.config.sigmaClipKwargs["sigma"] = 2.0
        self.config.stdMin = 0.005
        self.config.maxZernClip = 5
        task = CombineZernikesSigmaClipTask(config=self.config)
        self.assertEqual(2.0, task.sigmaClipKwargs["sigma"])
        self.assertEqual(0.005, task.stdMin)
        self.assertEqual(5, task.maxZernClip)

    def testCombineZernikes(self) -> None:
        inTable = self.prepareTestTable()
        outTable = self.task.combineZernikes(inTable)

        # Check all the averages
        avg = outTable[outTable["label"] == "average"]
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["opd_columns"]))
        self.assertTrue(all(avg[col] < 2 for col in avg.meta["intrinsic_columns"]))
        self.assertTrue(np.allclose([avg[col] for col in avg.meta["deviation_columns"]], 2.0))

        # Check used
        self.assertTrue(outTable["used"].tolist() == 99 * [True] + 3 * [False])

    def testCombineDeviationOnlyTable(self) -> None:
        """A table carrying only the deviation family must still combine.

        The means are driven by the ``*_columns`` metadata rather than by
        ``noll_indices``, so a table that omits the OPD and intrinsic families
        entirely -- as the blitz ``zernikes`` output does, since a joint
        fit has no single intrinsic to report -- combines instead of
        raising KeyError on the absent columns.
        """
        full = self.prepareTestTable()
        deviationColumns = list(full.meta["deviation_columns"])
        inTable = full[["label", "used"] + deviationColumns]
        inTable.meta = dict(full.meta)
        inTable.meta["opd_columns"] = []
        inTable.meta["intrinsic_columns"] = []

        outTable = self.task.combineZernikes(inTable)

        avg = outTable[outTable["label"] == "average"]
        self.assertTrue(np.allclose([avg[col] for col in deviationColumns], 2.0))
        self.assertTrue(outTable["used"].tolist() == 99 * [True] + 3 * [False])

    def testDeviationOnlyMatchesFullTable(self) -> None:
        """Dropping the unused families must not perturb the deviation result.

        Guards the column-list iteration against a reordering bug: ``_setAvg``
        touches one column at a time, so the deviation averages and ``used``
        flags have to come out identical whether or not the other two families
        are present.
        """
        full = self.task.combineZernikes(self.prepareTestTable())

        source = self.prepareTestTable()
        deviationColumns = list(source.meta["deviation_columns"])
        trimmed = source[["label", "used"] + deviationColumns]
        trimmed.meta = dict(source.meta)
        trimmed.meta["opd_columns"] = []
        trimmed.meta["intrinsic_columns"] = []
        deviationOnly = self.task.combineZernikes(trimmed)

        self.assertEqual(full["used"].tolist(), deviationOnly["used"].tolist())
        for col in deviationColumns:
            np.testing.assert_allclose(full[col], deviationOnly[col], equal_nan=True)

    def testEmptyClipFamilyRaises(self) -> None:
        """Clipping on a family the table does not carry fails loudly.

        Without the guard this reaches a 2-D slice of a 1-D array and raises an
        opaque IndexError instead.
        """
        source = self.prepareTestTable()
        source.meta["opd_columns"] = []
        self.config.zkClipType = "opd"
        task = CombineZernikesSigmaClipTask(config=self.config)
        with self.assertRaises(ValueError) as cm:
            task.combineZernikes(source)
        self.assertIn("opd_columns", str(cm.exception))

    def testCombineZernikesEffectiveMaxZernClip(self) -> None:
        inTable = self.prepareTestTable()
        inTable[1]["Z7_deviation"] = 1e2

        # Test that zernikes higher than maxZernClip don't trigger flagging
        outTable = self.task.combineZernikes(inTable)
        self.assertTrue(outTable[outTable["label"] == "average"]["Z7_deviation"] > 2)
        self.assertTrue(outTable["used"].tolist() == 99 * [True] + 3 * [False])

        # Test that raising maxZernClip does trigger flagging
        self.config.maxZernClip = 5
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(inTable)
        self.assertFalse(outTable[1]["used"])

    def testTaskRun(self) -> None:
        inTable = self.prepareTestTable()
        output = self.task.run(inTable)

        outTable = output.combinedTable

        # Check all the averages
        avg = outTable[outTable["label"] == "average"]
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["opd_columns"]))
        self.assertTrue(all(avg[col] < 2 for col in avg.meta["intrinsic_columns"]))
        self.assertTrue(np.allclose([avg[col] for col in avg.meta["deviation_columns"]], 2.0))

        # Check used
        self.assertTrue(outTable["used"].tolist() == 99 * [True] + 3 * [False])

        # Check flags
        flags = output.flags
        self.flags = flags
        np.allclose(flags, 98 * [0] + 3 * [1])

    def testZkClipType(self) -> None:
        inTable = self.prepareTestTable()

        # Test cutting on OPD
        self.config.zkClipType = "opd"
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(inTable)

        avg = outTable[outTable["label"] == "average"]
        self.assertTrue(np.allclose([avg[col] for col in avg.meta["opd_columns"]], 2.0))
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["intrinsic_columns"]))
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["deviation_columns"]))
        self.assertTrue(outTable["used"].tolist() == [True] + [False] + 100 * [True])

        # Test cutting on intrinsics
        self.config.zkClipType = "intrinsic"
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(inTable)

        avg = outTable[outTable["label"] == "average"]
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["opd_columns"]))
        self.assertTrue(np.allclose([avg[col] for col in avg.meta["intrinsic_columns"]], 2.0))
        self.assertTrue(all(avg[col] > 2 for col in avg.meta["deviation_columns"]))
        self.assertTrue(outTable["used"].tolist() == 101 * [True] + [False])
