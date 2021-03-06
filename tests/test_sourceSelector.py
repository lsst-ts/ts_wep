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

from lsst.ts.wep.bsc.BaseBscTestCase import BaseBscTestCase
from lsst.ts.wep.SourceSelector import SourceSelector
from lsst.ts.wep.Utility import getModulePath, FilterType, CamType, BscDbType


class TestSourceSelector(BaseBscTestCase, unittest.TestCase):
    """Test the source selector class."""

    def setUp(self):

        self.createBscTest()

        # Get the path of module
        self.modulePath = getModulePath()
        self.sourSelc = SourceSelector(CamType.ComCam, BscDbType.LocalDb)

        # Set the survey parameters
        ra = 0.0
        dec = 63.0
        rotSkyPos = 0.0
        self.sourSelc.setObsMetaData(ra, dec, rotSkyPos)
        self.sourSelc.setFilter(FilterType.U)

        # Connect to database
        self.dbAdress = self.getPathOfBscTest()
        self.sourSelc.connect(self.dbAdress)

    def tearDown(self):

        self.sourSelc.disconnect()
        self.removeBscTest()

    def testInit(self):

        self.assertEqual(self.sourSelc.maxDistance, 157.5)
        self.assertEqual(self.sourSelc.maxNeighboringStar, 1)

    def testConfigNbrCriteria(self):

        starRadiusInPixel = 100
        spacingCoefficient = 2
        maxNeighboringStar = 3
        self.sourSelc.configNbrCriteria(
            starRadiusInPixel, spacingCoefficient, maxNeighboringStar=maxNeighboringStar
        )

        self.assertEqual(
            self.sourSelc.maxDistance, starRadiusInPixel * spacingCoefficient
        )
        self.assertEqual(self.sourSelc.maxNeighboringStar, maxNeighboringStar)

    def testSetAndGetFilter(self):

        filterType = FilterType.Z
        self.sourSelc.setFilter(filterType)

        self.assertEqual(self.sourSelc.getFilter(), filterType)

    def testGetTargetStarWithZeroOffset(self):

        self.sourSelc.configNbrCriteria(63.0, 2.5, maxNeighboringStar=99)
        neighborStarMap, starMap, wavefrontSensors = self.sourSelc.getTargetStar(
            offset=0
        )

        self.assertEqual(len(wavefrontSensors), 8)

    def testGetTargetStarWithNotZeroOffset(self):

        self.sourSelc.configNbrCriteria(63.0, 2.5, maxNeighboringStar=99)
        neighborStarMap, starMap, wavefrontSensors = self.sourSelc.getTargetStar(
            offset=-1000
        )

        self.assertEqual(len(wavefrontSensors), 3)

    def testGetTargetStarByFileWithWrongDbType(self):

        self.assertRaises(TypeError, self.sourSelc.getTargetStarByFile, "skyFile")

    def testGetTargetStarByFileForFilterG(self):

        neighborStarMap, starMap, wavefrontSensors = self._getTargetStarByFile(
            FilterType.G
        )

        self.assertEqual(len(wavefrontSensors), 8)

        for detector in wavefrontSensors:
            self.assertEqual(len(starMap[detector].getId()), 2)
            self.assertEqual(len(neighborStarMap[detector].getId()), 2)

    def testGetTargetStarByFileForFilterRef(self):

        neighborStarMap, starMap, wavefrontSensors = self._getTargetStarByFile(
            FilterType.REF
        )

        self.assertEqual(len(wavefrontSensors), 8)

        for detector in wavefrontSensors:
            self.assertEqual(len(starMap[detector].getId()), 2)
            self.assertEqual(len(neighborStarMap[detector].getId()), 2)

    def _getTargetStarByFile(self, filterType):

        self.sourSelc = SourceSelector(CamType.LsstCam, BscDbType.LocalDbForStarFile)
        self.sourSelc.setObsMetaData(0, 0, 0)
        self.sourSelc.setFilter(filterType)
        self.sourSelc.connect(self.dbAdress)

        skyFilePath = os.path.join(
            self.modulePath,
            "tests",
            "testData",
            "phosimOutput",
            "realWfs",
            "output",
            "skyWfsInfo.txt",
        )

        neighborStarMap, starMap, wavefrontSensors = self.sourSelc.getTargetStarByFile(
            skyFilePath, offset=0
        )

        return neighborStarMap, starMap, wavefrontSensors


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
