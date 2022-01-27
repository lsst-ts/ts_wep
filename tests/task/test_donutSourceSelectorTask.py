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
import unittest
import pandas as pd

from lsst.daf import butler as dafButler
from lsst.geom import Box2I, Point2I
from lsst.pex.config import FieldValidationError
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.DonutSourceSelectorTask import (
    DonutSourceSelectorTask,
    DonutSourceSelectorTaskConfig,
)


class TestDonutSourceSelectorTask(unittest.TestCase):
    def setUp(self):

        self.config = DonutSourceSelectorTaskConfig()
        self.task = DonutSourceSelectorTask()

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

    def _createTestCat(self):

        minimalCat = pd.DataFrame()
        minimalCat["flux"] = [30000.0, 30010.0, 30020.0, 30005.0]
        minimalCat["centroid_x"] = [100.0, 140.0, 190.0, 400.0]
        minimalCat["centroid_y"] = [100.0, 100.0, 100.0, 100.0]
        bbox = Box2I(Point2I(0, 0), Point2I(500, 500))
        return minimalCat, bbox

    def testValidateConfigs(self):

        # Check default configuration
        self.OrigTask = DonutSourceSelectorTask(config=self.config, name="Orig Task")
        self.assertEqual(self.OrigTask.config.xCoordField, "centroid_x")
        self.assertEqual(self.OrigTask.config.yCoordField, "centroid_y")
        self.assertFalse(self.OrigTask.config.xCoordField == "center_x")

        # Check configuration changes are passed through
        self.config.xCoordField = "center_x"
        self.ModifiedTask = DonutSourceSelectorTask(config=self.config, name="Mod Task")
        self.assertEqual(self.ModifiedTask.config.xCoordField, "center_x")
        self.assertEqual(self.ModifiedTask.config.yCoordField, "centroid_y")
        self.assertFalse(self.ModifiedTask.config.xCoordField == "centroid_x")

        # Check that error is raised if configuration type is incorrect.
        with self.assertRaises(FieldValidationError):
            # sourceLimit defined to be integer. Float should raise error.
            self.config.sourceLimit = 2.1

    def testSelectSources(self):

        minimalCat, bbox = self._createTestCat()

        # All donuts chosen since none overlap in this instance
        self.config.donutRadius = 15.0
        self.config.isoMagDiff = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 4)

        # The first three donuts overlap but none are more than
        # isoMagDiff brigher than the rest so none are chosen
        self.config.donutRadius = 50.0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 1)

        # Will now take the brightest of the three donuts
        # for a total of 2 selected donuts
        self.config.isoMagDiff = 0.0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 2)
        self.assertEqual(testCatSelected[0], False)
        self.assertEqual(testCatSelected[1], False)
        self.assertEqual(testCatSelected[2], True)

        # Test that number of sources in catalog limited by sourceLimit
        self.config.sourceLimit = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 1)

        # Test that sourceLimit can only be positive integer or -1
        self.config.sourceLimit = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        errMsg = str(
            "config.sourceLimit must be a positive integer "
            + "or turned off by setting it to '-1'"
        )
        with self.assertRaises(ValueError, msg=errMsg):
            testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 1)

        # Test that setting sourceLimit returns all selected sources
        self.config.sourceLimit = -1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 2)

        # Test that only the brightest blended on the end that only
        # blends with one other donut is not accepted if maxBlended
        # is still set to 0.
        self.config.isoMagDiff = 2.0
        self.config.donutRadius = 35.0
        self.config.maxBlended = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 1)

        # Test that only the brightest blended on the end that only
        # blends with one other donut is accepted even though
        # it does not meet the isoMagDiff criterion we allow it because
        # we now allow maxBlended up to 1.
        self.config.isoMagDiff = 2.0
        self.config.donutRadius = 35.0
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 2)
        # Make sure the one that gets through selection is
        # the brightest one.
        self.assertEqual(testCatSelected[0], False)
        self.assertEqual(testCatSelected[1], False)
        self.assertEqual(testCatSelected[2], True)

        # If we increase donutRadius back to 50 then our group of
        # 3 donuts should all be overlapping and blended. Therefore,
        # maxBlended set to one should not allow the brightest donut
        # through since it is blended with two objects.
        self.config.isoMagDiff = 2.0
        self.config.donutRadius = 50.0
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 1)

        # If we increase donutRadius back to 50 then our group of
        # 3 donuts should all be overlapping and blended. Therefore,
        # maxBlended set to one should not allow the brightest donut
        # through since it is blended with two objects.
        self.config.isoMagDiff = 2.0
        self.config.donutRadius = 50.0
        self.config.maxBlended = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, bbox)
        testCatSelected = testCatStruct.selected
        self.assertEqual(len(testCatSelected), 4)
        self.assertEqual(sum(testCatSelected), 2)
        # Make sure the one that gets through selection is
        # the brightest one.
        self.assertEqual(testCatSelected[0], False)
        self.assertEqual(testCatSelected[1], False)
        self.assertEqual(testCatSelected[2], True)
