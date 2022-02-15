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
    getAmpImagesFromDir,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestUtility(unittest.TestCase):
    """Test the Utility functions."""

    def _writePipetaskCmd(
        self,
        repoName,
        instrument,
        collections,
        runName,
        taskName=None,
        pipelineName=None,
    ):

        # Write the part of the command that is always included
        testCmd = f"pipetask run -b {repoName} -i {collections} "
        testCmd += f"--instrument {instrument} "
        testCmd += f"--register-dataset-types --output-run {runName}"

        # Write with taskName
        if taskName is not None:
            testCmd += f" -t {taskName}"

        # Write with pipeline filename
        if pipelineName is not None:
            testCmd += f" -p {pipelineName}"

        return testCmd

    def _writeCleanUpCmd(self, repoName, runName):

        testCmd = f"butler remove-runs {repoName} {runName}"
        testCmd += " --no-confirm"

        return testCmd

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

    def testWritePipetaskCmd(self):

        repoName = "testRepo"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats"
        runName = "run2"

        # Test writing with task name
        taskName = "lsst.ts.wep.testTask"
        testCmdTask = self._writePipetaskCmd(
            repoName, instrument, collections, runName, taskName=taskName
        )

        pipeOutTask = writePipetaskCmd(
            repoName, runName, instrument, collections, taskName=taskName
        )
        self.assertEqual(testCmdTask, pipeOutTask)

        # Test writing with pipeline
        pipelineYamlFile = "testPipeOut.yaml"
        testCmdYaml = self._writePipetaskCmd(
            repoName, instrument, collections, runName, pipelineName=pipelineYamlFile
        )

        pipeOutYaml = writePipetaskCmd(
            repoName, runName, instrument, collections, pipelineYaml=pipelineYamlFile
        )
        self.assertEqual(testCmdYaml, pipeOutYaml)

        assertMsg = "At least one of taskName or pipelineYaml must not be None"
        with self.assertRaises(ValueError) as context:
            writePipetaskCmd(repoName, runName, instrument, collections)
        self.assertTrue(assertMsg in str(context.exception))

    def testWriteCleanUpRepoCmd(self):

        repoName = "testRepo"
        runName = "run2"

        testCmd = self._writeCleanUpCmd(repoName, runName)
        self.assertEqual(testCmd, writeCleanUpRepoCmd(repoName, runName))


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
