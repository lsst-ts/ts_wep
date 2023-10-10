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

from lsst.ts.wep.utils import (
    getAmpImagesFromDir,
    getConfigDir,
    getModulePath,
    getObsLsstCmdTaskConfigDir,
)


class TestIoUtils(unittest.TestCase):
    """Test the IO utility functions."""

    def testGetConfigDir(self):
        ansConfigDir = os.path.join(getModulePath(), "policy")
        self.assertEqual(getConfigDir(), ansConfigDir)

    def testGetObsLsstCmdTaskConfigDir(self):
        obsLsstCmdTaskConfirDir = getObsLsstCmdTaskConfigDir()
        configNormPath = os.path.normpath(obsLsstCmdTaskConfirDir)
        configNormPathList = configNormPath.split(os.sep)

        self.assertEqual(configNormPathList[-1], "config")
        self.assertTrue(("obs_lsst" in configNormPathList))

    def testGetAmpImagesFromDir(self):
        # path to repackaged phosim files
        # with amplifier images and e-images
        defocalImgDir = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "phosimOutput",
            "realComCam",
            "repackagedFiles",
            "extra",
        )
        # test that there are e-images in that dir
        filesInDir = os.listdir(defocalImgDir)
        self.assertTrue("MC_H_20211231_006001_R22_S11_e.fits.gz" in filesInDir)
        self.assertTrue("MC_H_20211231_006001_R22_S10_e.fits.gz" in filesInDir)

        # get names of amp files
        ampFiles = getAmpImagesFromDir(defocalImgDir)

        # assert the returned content
        self.assertIsInstance(ampFiles, list)

        # assert that amp images are on the returned list
        self.assertTrue("MC_H_20211231_006001_R22_S10.fits" in ampFiles)
        self.assertTrue("MC_H_20211231_006001_R22_S11.fits" in ampFiles)

        # assert that no other files are there
        # by checking that the length of list corresponds to
        # two files tested above
        self.assertEqual(len(ampFiles), 2)
