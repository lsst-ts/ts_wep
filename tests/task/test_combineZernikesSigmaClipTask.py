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

import astropy.units as u
import numpy as np
from astropy.table import Table

from lsst.ts.wep.task.combineZernikesSigmaClipTask import (
    CombineZernikesSigmaClipTask,
    CombineZernikesSigmaClipTaskConfig,
)


class TestCombineZernikesSigmaClipTask(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CombineZernikesSigmaClipTaskConfig()
        self.config.stdMin = 0.005
        self.task = CombineZernikesSigmaClipTask(config=self.config)

    def prepareTestTable(self) -> Table:
        label = ["average"] + [f"pair{i}" for i in range(101)]
        used = [True] + 101 * [False]
        table = Table([label, used], names=["label", "used"])

        nollIndices = np.arange(4, 12)
        table.meta["noll_indices"] = nollIndices
        table.meta["opd_columns"] = [f"Z{j}" for j in nollIndices]
        table.meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in nollIndices]
        table.meta["deviation_columns"] = [f"Z{j}_deviation" for j in nollIndices]
        table.meta["estimatorInfo"] = {}

        for i in table.meta["noll_indices"]:
            table[f"Z{i}"] = np.array([np.nan] + [101.0] + 50 * [1.0] + 50 * [3.0]) * u.nm
        for i in table.meta["noll_indices"]:
            table[f"Z{i}_intrinsic"] = np.array([np.nan] + 50 * [1.0] + 50 * [3.0] + [101.0]) * u.nm
        for i in table.meta["noll_indices"]:
            table[f"Z{i}_deviation"] = np.array([np.nan] + 49 * [1.0] + 49 * [3.0] + 3 * [101.0]) * u.nm

        return table

    def testValidateConfigs(self) -> None:
        self.assertEqual(
            {"sigma": 3.0, "stdfunc": "mad_std", "maxiters": 1},
            self.task.sigmaClipKwargs,
        )
        self.assertEqual(3, self.task.maxZernClip)

        self.config.sigmaClipKwargs["sigma"] = 2.0
        self.config.stdMin = 0.05
        self.config.maxZernClip = 5
        task = CombineZernikesSigmaClipTask(config=self.config)
        self.assertEqual(2.0, task.sigmaClipKwargs["sigma"])
        self.assertEqual(0.05, task.stdMin)
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

    def testNollIdxRejected(self) -> None:
        """
        Test that nollIdxRejected correctly identifies clipped Zernike indices.
        """
        # Create a custom test table with specific outliers
        inTable = self.prepareTestTable()

        # Make the first donut (pair0) have an outlier in Z4
        # (Noll index 4, column index 0)
        # and Z5 (Noll index 5, column index 1) when using
        # deviation clipping
        nollIndices = inTable.meta["noll_indices"]
        col_z4 = f"Z{nollIndices[0]}_deviation"  # Should be Z4_deviation
        col_z5 = f"Z{nollIndices[1]}_deviation"  # Should be Z5_deviation

        # Modify the first data row to have extreme values that
        # will be clipped
        inTable[1][col_z4] = 1000.0  # Extreme outlier
        inTable[1][col_z5] = 1000.0  # Extreme outlier

        # The second donut should have an outlier only in Z4
        inTable[2][col_z4] = 1000.0

        # Run the task with deviation clipping
        self.config.zkClipType = "deviation"
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(inTable)

        # Verify that estimatorInfo exists and contains the noll indices
        self.assertIn("estimatorInfo", outTable.meta)
        self.assertIn("zern_clipped_rejected_noll_indices", outTable.meta["estimatorInfo"])
        self.assertIn("zern_clipped_max_noll_index", outTable.meta["estimatorInfo"])

        nollIdxRejected = outTable.meta["estimatorInfo"]["zern_clipped_rejected_noll_indices"]

        # Verify structure: list of lists
        self.assertIsInstance(nollIdxRejected, list)
        self.assertEqual(len(nollIdxRejected), len(inTable[inTable["label"] != "average"]))

        # First donut (index 0) should have Z4 and Z5 rejected
        self.assertEqual(sorted(nollIdxRejected[0]), sorted([nollIndices[0], nollIndices[1]]))

        # Second donut (index 1) should have only Z4 rejected
        self.assertEqual(nollIdxRejected[1], [nollIndices[0]])

        # Verify that nollIdxRejected entries are lists of
        # integers (Noll indices)
        for idx, rejected_indices in enumerate(nollIdxRejected):
            self.assertIsInstance(rejected_indices, list)
            for noll_idx in rejected_indices:
                self.assertIsInstance(noll_idx, int)
                self.assertIn(noll_idx, nollIndices)

    def testZernClippedMetadata(self) -> None:
        """Test that zern_clipped metadata is correctly added to zkTable."""
        inTable = self.prepareTestTable()

        # Run the task to add clipping information
        outTable = self.task.combineZernikes(inTable)

        # Verify that estimatorInfo exists and contains the required keys
        self.assertIn("estimatorInfo", outTable.meta)
        self.assertIn("zern_clipped", outTable.meta["estimatorInfo"])
        self.assertIn("zern_clipped_rejected_noll_indices", outTable.meta["estimatorInfo"])
        self.assertIn("zern_clipped_max_noll_index", outTable.meta["estimatorInfo"])

        # Get the clipped flag array and rejection info
        zernClipped = outTable.meta["estimatorInfo"]["zern_clipped"]
        nollIdxRejected = outTable.meta["estimatorInfo"]["zern_clipped_rejected_noll_indices"]
        self.assertEqual(len(outTable.meta["estimatorInfo"]["zern_clipped_max_noll_index"]), 1)
        maxNollIdx = outTable.meta["estimatorInfo"]["zern_clipped_max_noll_index"][0]

        # Verify zern_clipped_max_noll_idx is an integer and
        # within valid range
        self.assertIsInstance(maxNollIdx, (int, np.integer))
        self.assertGreaterEqual(maxNollIdx, 0)
        self.assertLessEqual(maxNollIdx, self.config.maxZernClip + 1)

        # Verify data types are lists
        self.assertIsInstance(zernClipped, list)
        self.assertIsInstance(nollIdxRejected, list)

        # Verify length matches number of data rows (excluding average)
        numDataRows = len(outTable[outTable["label"] != "average"])
        self.assertEqual(len(zernClipped), numDataRows)
        self.assertEqual(len(nollIdxRejected), numDataRows)

        # Verify consistency: if zern_clipped is True,
        # nollIdxRejected should have entries
        for idx, (isClipped, rejectedIndices) in enumerate(zip(zernClipped, nollIdxRejected)):
            if isClipped:
                self.assertGreater(
                    len(rejectedIndices), 0, f"Row {idx} marked as clipped but has no rejected indices"
                )
            else:
                self.assertEqual(
                    len(rejectedIndices),
                    0,
                    f"Row {idx} not marked as clipped but has rejected indices: {rejectedIndices}",
                )

        # Verify that the used column is consistent with zern_clipped
        usedValues = outTable["used"][outTable["label"] != "average"].tolist()
        for idx, (isClipped, isUsed) in enumerate(zip(zernClipped, usedValues)):
            if isClipped:
                self.assertFalse(isUsed, f"Row {idx} is clipped but marked as used")
            else:
                self.assertTrue(isUsed, f"Row {idx} is not clipped but marked as unused")
