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
        self.config.stdMin = 0.005
        self.config.maxZernClip = 5
        task = CombineZernikesSigmaClipTask(config=self.config)
        self.assertEqual(2.0, task.sigmaClipKwargs["sigma"])
        self.assertEqual(0.005, task.stdMin)
        self.assertEqual(5, task.maxZernClip)

    def prepareTightClusterTable(self, deviations_nm: list) -> Table:
        """Build a single-corner table with the given per-donut deviations.

        The deviation values are given in nm (the unit the Zernike columns are
        stored in) so that the nm->um conversion inside the task is exercised.
        Intrinsics are set to zero, so the opd and deviation columns match.
        """
        nDonuts = len(deviations_nm)
        label = ["average"] + [f"pair{i}" for i in range(nDonuts)]
        used = [True] + nDonuts * [False]
        table = Table([label, used], names=["label", "used"])

        nollIndices = np.arange(4, 12)
        table.meta["noll_indices"] = nollIndices
        table.meta["opd_columns"] = [f"Z{j}" for j in nollIndices]
        table.meta["intrinsic_columns"] = [f"Z{j}_intrinsic" for j in nollIndices]
        table.meta["deviation_columns"] = [f"Z{j}_deviation" for j in nollIndices]

        for j in nollIndices:
            table[f"Z{j}"] = np.array([np.nan] + list(deviations_nm)) * u.nm
            table[f"Z{j}_intrinsic"] = np.array([np.nan] + nDonuts * [0.0]) * u.nm
            table[f"Z{j}_deviation"] = np.array([np.nan] + list(deviations_nm)) * u.nm

        return table

    def testStdMinGate(self) -> None:
        # A tight cluster (~0.7 um) with a single donut offset by ~0.045 um.
        # The donut-to-donut scatter is ~0.018 um, which sits between the two
        # stdMin values tested below. stdMin is documented in um, while the
        # Zernike columns are stored in nm, so this also guards the nm->um
        # conversion in the task: without it the gate would compare a ~18 nm
        # scatter against a nm-scale threshold and never behave as intended.
        deviations_nm = [700.0, 705.0, 698.0, 702.0, 745.0]

        # With a small stdMin the offset donut exceeds the clip threshold and
        # is rejected.
        self.config.stdMin = 0.005
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(self.prepareTightClusterTable(deviations_nm))
        usedPairs = outTable["used"][outTable["label"] != "average"].tolist()
        self.assertEqual(usedPairs, [True, True, True, True, False])

        # Raising stdMin above the cluster scatter suppresses clipping
        # entirely, so the well-measured donut is kept.
        self.config.stdMin = 0.1
        task = CombineZernikesSigmaClipTask(config=self.config)
        outTable = task.combineZernikes(self.prepareTightClusterTable(deviations_nm))
        usedPairs = outTable["used"][outTable["label"] != "average"].tolist()
        self.assertEqual(usedPairs, [True, True, True, True, True])

        # A genuine large outlier is still rejected even with the high stdMin,
        # because its deviation dominates the (robust) scatter estimate.
        deviations_nm = [700.0, 705.0, 698.0, 702.0, 2000.0]
        outTable = task.combineZernikes(self.prepareTightClusterTable(deviations_nm))
        usedPairs = outTable["used"][outTable["label"] != "average"].tolist()
        self.assertEqual(usedPairs, [True, True, True, True, False])

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
