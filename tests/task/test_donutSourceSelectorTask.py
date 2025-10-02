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

import astropy.units as u
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler
from lsst.pex.config import FieldValidationError
from lsst.ts.wep.task.donutSourceSelectorTask import (
    DonutSourceSelectorTask,
    DonutSourceSelectorTaskConfig,
)
from lsst.ts.wep.utils import getModulePath, readConfigYaml


class TestDonutSourceSelectorTask(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DonutSourceSelectorTaskConfig()
        self.task = DonutSourceSelectorTask()

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry
        self.filterName = "g"

        self.magMax = 99.0
        self.magMin = -99.0

    def _createTestCat(self) -> tuple[pd.DataFrame, dict]:
        minimalCat = pd.DataFrame()

        # Set magnitudes to test with policy file values
        magPolicyDefaults = readConfigYaml("policy:magLimitStar.yaml")
        defaultFilterKey = f"filter{self.filterName.upper()}"
        self.magMax = magPolicyDefaults[defaultFilterKey]["high"]
        self.magMin = magPolicyDefaults[defaultFilterKey]["low"]
        self.magRange = self.magMax - self.magMin
        magArray = [
            self.magMax - 0.1,
            self.magMax + 1.0,
            self.magMin - 1.0,
            self.magMax - 1.0,
        ]
        flux = (magArray * u.ABmag).to_value(u.nJy)

        minimalCat["g_flux"] = flux
        minimalCat["centroid_x"] = [100.0, 140.0, 190.0, 360.0]
        minimalCat["centroid_y"] = [100.0, 100.0, 100.0, 100.0]
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        detector = camera["R22_S11"]
        return minimalCat, detector

    def testValidateConfigs(self) -> None:
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

    def testSelectSourcesMagLimits(self) -> None:
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
        self.assertListEqual(list(testCatSelected), [True, False, True, True])

        # Test Defaults are used when useCustomMagLimit is turned off.
        self.config.useCustomMagLimit = False
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, False, False, True])

    def testSelectSources(self) -> None:
        minimalCat, detector = self._createTestCat()

        # All donuts chosen since none overlap in this instance
        self.config.useCustomMagLimit = True
        self.config.unblendedSeparation = 30
        self.config.isolatedMagDiff = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, True, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[]] * 4)
        self.assertListEqual(list(testCatStruct.blendCentersY), [[]] * 4)

        # The first three donuts overlap but none are more than
        # isolatedMagDiff brighter than the rest so none are chosen
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 100
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[]])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[]])

        # Will now take the brightest of the three donuts
        # for a total of 2 selected donuts
        self.config.isolatedMagDiff = 0.0
        self.config.minBlendedSeparation = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[], []])

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

        # Test that two donuts on either end of the triplet that only
        # blend with one other donut are not accepted if maxBlended
        # is still set to 0.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 70
        self.config.maxBlended = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, False, True])

        # Test that two donuts on either end of the triplet that only
        # blend with one other donut are accepted even though
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
        self.assertListEqual(list(testCatSelected), [True, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[140.0], [140.0], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[100.0], [100.0], []])

        # Lower unblendedSeparation so that the first two donuts
        # are the only blended ones. Test that the brighter of the
        # two is accepted and that the blendCentersX and blendCentersY
        # values are consistent with the location of the first accepted
        # donut in the array.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 41
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        # Make sure the first accepted donut is the one with the
        # correct entries in blendCentersX and blendCentersY.
        self.assertListEqual(list(testCatSelected), [True, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[140.0], [], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[100.0], [], []])

        # If we increase unblendedSeparation back to 100 then our group of
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
        self.assertListEqual(list(testCatStruct.blendCentersX), [[]])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[]])

        # Decrease isolatedMagDiff so that only one of the donuts
        # blended with the brightest donut falls within the range.
        # Even though two fainter donuts fall within the range of
        # unblendedSeparation and 3 donuts are allowed to be blended
        # the returned results of blendCenters should only include
        # the donut that falls within the blended magnitude range.
        # Since the fainter donut should not affect Zernike calculation
        # we don't want to include its footprint in the deblending
        # masks which are created based upon results in blendCenters.
        self.config.isolatedMagDiff = self.magRange + 1.5
        self.config.unblendedSeparation = 100
        self.config.maxBlended = 3
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[100.0], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[100.0], []])

        # If we increase isolatedMagDiff to once again include
        # both sources overlapping the brightest donut as blended
        # then we should have both sources in the blendCenters lists.
        self.config.isolatedMagDiff = self.magRange + 2.5
        self.config.unblendedSeparation = 100
        self.config.maxBlended = 3
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        # Make sure the one that gets through selection is
        # the brightest one.
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
        blendCentersX = testCatStruct.blendCentersX
        blendCentersY = testCatStruct.blendCentersY
        self.assertEqual(len(blendCentersX), 2)
        self.assertEqual(len(blendCentersY), 2)
        for bX, trueBX in zip(blendCentersX, [[140.0, 100.0], []]):
            self.assertListEqual(list(bX), trueBX)
        for bY, trueBY in zip(blendCentersY, [[100.0, 100.0], []]):
            self.assertListEqual(list(bY), trueBY)

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

        # Test output if we have an empty catalog. Make sure that
        # all parts of the Struct are returned as expected.
        self.config.isolatedMagDiff = 10.0
        self.config.unblendedSeparation = 400
        self.config.maxBlended = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        emptyCat = pd.DataFrame(columns=minimalCat.columns)
        testCatStruct = self.task.selectSources(emptyCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [])
        self.assertEqual(testCatStruct.blendCentersX, None)
        self.assertEqual(testCatStruct.blendCentersY, None)

        # Test that when there are multiple blends but not all are within
        # isolatedMagDiff that the donut centers for only the blended
        # donuts within isolatedMagDiff are kept since these are the only ones
        # that should count as blended.
        self.config.minBlendedSeparation = 55
        self.config.unblendedSeparation = 100
        self.config.isolatedMagDiff = 9.0
        self.config.maxBlended = 1
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[100.0], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[100.0], []])

        # Test same as above but with multiple donuts kept and one donut that
        # should not appear.
        # Need to move donuts away from edge to test with needed separation
        minimalCat["centroid_y"] += 100.0
        minimalCat["centroid_x"] += 100.0
        self.config.unblendedSeparation = 180
        self.config.isolatedMagDiff = 9.0
        self.config.maxBlended = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.selectSources(minimalCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, False])
        for bX, trueBX in zip(testCatStruct.blendCentersX, [[200.0, 460.0]]):
            self.assertListEqual(list(bX), trueBX)
        for bY, trueBY in zip(testCatStruct.blendCentersY, [[200.0, 200.0]]):
            self.assertListEqual(list(bY), trueBY)

    def testTaskRun(self) -> None:
        minimalCat, detector = self._createTestCat()

        # All donuts should pass since default mag limit is (-99.0, 99.0)
        self.config.useCustomMagLimit = True
        self.config.unblendedSeparation = 30
        self.config.isolatedMagDiff = 2
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.run(minimalCat, detector, self.filterName)
        np.testing.assert_array_equal(testCatStruct.sourceCat, minimalCat)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [True, True, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[]] * 4)
        self.assertListEqual(list(testCatStruct.blendCentersY), [[]] * 4)

        # Will now take the brightest of the three donuts that overlap
        # for a total of 2 selected donuts
        self.config.unblendedSeparation = 100
        self.config.isolatedMagDiff = 0.0
        self.config.minBlendedSeparation = 0
        self.task = DonutSourceSelectorTask(config=self.config, name="Test Task")
        testCatStruct = self.task.run(minimalCat, detector, self.filterName)
        np.testing.assert_array_equal(testCatStruct.sourceCat, minimalCat.iloc[2:])
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [False, False, True, True])
        self.assertListEqual(list(testCatStruct.blendCentersX), [[], []])
        self.assertListEqual(list(testCatStruct.blendCentersY), [[], []])

        # Test output if we have an empty catalog. Make sure that
        # all parts of the Struct are returned as expected.
        emptyCat = pd.DataFrame(columns=minimalCat.columns)
        testCatStruct = self.task.run(emptyCat, detector, self.filterName)
        testCatSelected = testCatStruct.selected
        self.assertListEqual(list(testCatSelected), [])
        self.assertEqual(testCatStruct.blendCentersX, None)
        self.assertEqual(testCatStruct.blendCentersY, None)
