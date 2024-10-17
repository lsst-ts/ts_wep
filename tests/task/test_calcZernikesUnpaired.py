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

import os

import lsst.utils.tests
import numpy as np
from lsst.ts.wep.task import (
    CalcZernikesTask,
    CalcZernikesTaskConfig,
    CalcZernikesUnpairedTask,
    CalcZernikesUnpairedTaskConfig,
    EstimateZernikesTieTask,
    EstimateZernikesDanishTask,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    getModulePath,
)

class TestCalcZernikeUnpaired(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the stamps
        moduleDir = getModulePath()
        cls.testDataDir = os.path.join(moduleDir, "tests", "testData")
        cls.donutStampDir = os.path.join(cls.testDataDir, "donutImg", "donutStamps")
        cls.donutStampsExtra = DonutStamps.readFits(
            os.path.join(cls.donutStampDir, "R04_SW0_donutStamps.fits")
        )
        cls.donutStampsIntra = DonutStamps.readFits(
            os.path.join(cls.donutStampDir, "R04_SW1_donutStamps.fits")
        )

    def testWithAndWithoutPairs(self):
        # Look over EstimateZernikes subtasks
        for subtask in [EstimateZernikesTieTask, EstimateZernikesDanishTask]:
            # Calculate Zernikes with stamps paired
            config = CalcZernikesTaskConfig()
            config.estimateZernikes.retarget(subtask)
            pairedTask = CalcZernikesTask(config=config)

            pairedZk = pairedTask.run(self.donutStampsExtra, self.donutStampsIntra)
            pairedZk = pairedZk.outputZernikesAvg

            # Calculate Zernikes with stamps unpaired
            config = CalcZernikesUnpairedTaskConfig()
            config.estimateZernikes.retarget(subtask)
            unpairedTask = CalcZernikesUnpairedTask(config=config)

            extraZk = unpairedTask.run(self.donutStampsExtra).outputZernikesAvg
            intraZk = unpairedTask.run(self.donutStampsIntra).outputZernikesAvg
            meanZk = np.mean([extraZk, intraZk], axis=0)

            # Check that results are similar
            diff = np.sqrt(np.sum((meanZk - pairedZk) ** 2))
            self.assertLess(diff, 0.16)
