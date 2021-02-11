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

from lsst.ts.wep.Utility import (
    mapFilterRefToG,
    FilterType,
    getModulePath,
    getConfigDir,
    getObsLsstCmdTaskConfigDir,
    ImageType,
    getImageType,
    getBscDbType,
    BscDbType,
    getCentroidFindType,
    CentroidFindType,
    getDeblendDonutType,
    DeblendDonutType,
    getDonutTemplateType,
    DonutTemplateType,
)


class TestUtility(unittest.TestCase):
    """Test the Utility functions."""

    def testMapFilterRefToG(self):

        mappedFilterType = mapFilterRefToG(FilterType.REF)
        self.assertEqual(mappedFilterType, FilterType.G)

    def testMapFilterRefToGForFilterU(self):

        mappedFilterType = mapFilterRefToG(FilterType.U)
        self.assertEqual(mappedFilterType, FilterType.U)

    def testGetConfigDir(self):

        ansConfigDir = os.path.join(getModulePath(), "policy")
        self.assertEqual(getConfigDir(), ansConfigDir)

    def testGetObsLsstCmdTaskConfigDir(self):

        obsLsstCmdTaskConfirDir = getObsLsstCmdTaskConfigDir()
        configNormPath = os.path.normpath(obsLsstCmdTaskConfirDir)
        configNormPathList = configNormPath.split(os.sep)

        self.assertEqual(configNormPathList[-1], "config")
        self.assertTrue(("obs_lsst" in configNormPathList))

    def testGetBscDbType(self):

        self.assertEqual(getBscDbType("localDb"), BscDbType.LocalDb)
        self.assertEqual(getBscDbType("file"), BscDbType.LocalDbForStarFile)

    def testGetBscDbTypeWithWrongInput(self):

        self.assertRaises(ValueError, getBscDbType, "wrongType")

    def testGetImageType(self):

        self.assertEqual(getImageType("amp"), ImageType.Amp)
        self.assertEqual(getImageType("eimage"), ImageType.Eimg)

    def testGetImageTypeWithWrongInput(self):

        self.assertRaises(ValueError, getImageType, "wrongType")

    def testGetCentroidFindType(self):

        self.assertEqual(getCentroidFindType("randomWalk"), CentroidFindType.RandomWalk)
        self.assertEqual(getCentroidFindType("otsu"), CentroidFindType.Otsu)
        self.assertEqual(
            getCentroidFindType("convolveTemplate"), CentroidFindType.ConvolveTemplate
        )

    def testGetCentroidFindTypeWithWrongInput(self):

        self.assertRaises(ValueError, getCentroidFindType, "wrongType")

    def testGetDeblendDonutType(self):

        self.assertEqual(getDeblendDonutType("adapt"), DeblendDonutType.Adapt)

    def testGetDeblendDonutTypeWithWrongInput(self):

        self.assertRaises(ValueError, getDeblendDonutType, "wrongType")

    def testGetDonutTemplateType(self):

        self.assertEqual(getDonutTemplateType("model"), DonutTemplateType.Model)
        self.assertEqual(getDonutTemplateType("phosim"), DonutTemplateType.Phosim)

    def testGetDonutTemplateTypeWithWrongInput(self):

        self.assertRaises(ValueError, getDonutTemplateType, "wrongType")


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
