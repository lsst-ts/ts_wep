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
import astropy.units as u

from lsst.daf import butler as dafButler
from lsst.pex.config import FieldValidationError
from lsst.ts.wep.Utility import getModulePath, getConfigDir
from lsst.ts.wep.ParamReader import ParamReader
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
        self.filterName = "g"

        self.magMax = 99.0
        self.magMin = -99.0

    def _createTestCat(self):

        minimalCat = pd.DataFrame()

        # Set magnitudes to test with policy file values
        magPolicyFile = os.path.join(getConfigDir(), "task", "magLimitStar.yaml")
        magPolicyDefaults = ParamReader(magPolicyFile).getContent()
        defaultFilterKey = f"filter{self.filterName.upper()}"
        self.magMax = magPolicyDefaults[defaultFilterKey]["high"]
        self.magMin = magPolicyDefaults[defaultFilterKey]["low"]
        magArray = [
            self.magMax + 1.0,
            self.magMax - 0.1,
            self.magMin - 1.0,
            self.magMax - 1.0,
        ]
        flux = (magArray * u.ABmag).to_value(u.nJy)

        minimalCat["g_flux"] = flux
        minimalCat["centroid_x"] = [100.0, 140.0, 190.0, 400.0]
        minimalCat["centroid_y"] = [100.0, 100.0, 100.0, 100.0]
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        detector = camera["R22_S11"]
        return minimalCat, detector

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

    def testSelectSourcesMagLimits(self):

        minimalCat, detector = self._createTestCat()

        # All donuts should pass since default mag limit is (-99.0, 99.0)
        self.config.useCustomMagLimit = True
        self.config.unblendedSeparation = 30
        self.config.isolatedMagDiff = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, True, True, True])

        # Test magMin
        self.config.magMin = self.magMin
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, True, False, True])

        # Test magMax
        self.config.magMin = -99.0
        self.config.magMax = self.magMax
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, True, True, True])

        # Test Defaults are used when useCustomMagLimit is turned off.
        self.config.useCustomMagLimit = False
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, True, False, True])

    def testSelectSources(self):

        minimalCat, detector = self._createTestCat()

        # All donuts chosen since none overlap in this instance
        self.config.useCustomMagLimit = True
        self.config.unblendedSeparation = 30
        self.config.isolatedMagDiff = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, True, True, True])

        # The first three donuts overlap but none are more than
        # isolatedMagDiff brighter than the rest so none are chosen
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 100
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])

        # Will now take the brightest of the three donuts
        # for a total of 2 selected donuts
        self.config.isolatedMagDiff = 0.0
        self.config.minBlendedSeparation = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])

        # Test that number of sources in catalog limited by sourceLimit
        # and that the brightest of the allowed donuts is chosen
        # as the only result
        self.config.sourceLimit = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, False])

        # Test that sourceLimit can only be positive integer or -1
        self.config.sourceLimit = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        errMsg = str(
            "config.sourceLimit must be a positive integer "
            + "or turned off by setting it to '-1'"
        )
        with self.assertRaises(ValueError, msg=errMsg):
            testCatStruct = self.task.selectSources(
                minimalCat, detector, self.filterName
            )

        # Test that setting sourceLimit returns all selected sources
        self.config.sourceLimit = -1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])

        # Test that only the brightest blended on the end that only
        # blends with one other donut is not accepted if maxBlended
        # is still set to 0.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 70
        self.config.maxBlended = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])

        # Test that only the brightest blended on the end that only
        # blends with one other donut is accepted even though
        # it does not meet the isolatedMagDiff criterion we allow
        # it because we now allow maxBlended up to 1.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 70
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        # Make sure the one that gets through selection is
        # the brightest one.
        self.assertListEqual(list(testCatSelected), [False, False, True, True])

        # If we increase unblendedSeparation back to 50 then our group of
        # 3 donuts should all be overlapping and blended. Therefore,
        # maxBlended set to one should not allow the brightest donut
        # through since it is blended with two objects.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 100
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])

        # If we increase unblendedSeparation back to 50 then our group of
        # 3 donuts should all be overlapping and blended. Therefore,
        # maxBlended set to one should not allow the brightest donut
        # through since it is blended with two objects.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 100
        self.config.maxBlended = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        # Make sure the one that gets through selection is
        # the brightest one.
        self.assertListEqual(list(testCatSelected), [False, False, True, True])

        # Donut furthest from center is over 0.15 degrees from field center
        # and should get cut out when setting maxFieldDist to 0.15
        self.config.unblendedSeparation = 30
        self.config.isolatedMagDiff = 10.0
        self.config.maxFieldDist = 0.15
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, True, True, True])

        # Blended donut separation should exclude all donuts that have a blend
        # less than minBlendedSeparation from themselves even if blends are
        # allowed. First check that it is excluded if closer than the limit.
        self.config.unblendedSeparation = 100
        self.config.minBlendedSeparation = 80
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])

        # Now test that it is allowed once the minBlendedSeparation is lowered.
        self.config.minBlendedSeparation = 45
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
