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
import numpy as np
from enum import IntEnum

from lsst.ts.wep.utility import (
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
    CamType,
    getCamType,
    getDefocalDisInMm,
    getCamTypeFromButlerName,
    getFilterTypeFromBandLabel,
    getCamNameFromCamType,
    createInstDictFromConfig,
    rotMatrix,
)
from lsst.afw.cameraGeom import DetectorType
from lsst.ts.wep.task.calcZernikesTask import CalcZernikesTaskConfig


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
        self.assertEqual(mappedFilterType, FilterType.LSST_G)

    def testMapFilterRefToGForFilterU(self):
        mappedFilterType = mapFilterRefToG(FilterType.LSST_U)
        self.assertEqual(mappedFilterType, FilterType.LSST_U)

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

    def testGetDefocalDisInMm(self):
        self.assertEqual(getDefocalDisInMm("lsst"), 1.5)
        self.assertEqual(getDefocalDisInMm("lsstfam"), 1.5)
        self.assertEqual(getDefocalDisInMm("comcam"), 1.5)
        self.assertEqual(getDefocalDisInMm("auxTel"), 0.8)
        instName = "telescope"
        assertMsg = f"Instrument name ({instName}) is not supported."
        with self.assertRaises(ValueError) as context:
            getDefocalDisInMm(instName)
        self.assertTrue(assertMsg in str(context.exception))

    def testGetCamType(self):
        self.assertEqual(getCamType("lsst"), CamType.LsstCam)
        self.assertEqual(getCamType("lsstfam"), CamType.LsstFamCam)
        self.assertEqual(getCamType("comcam"), CamType.ComCam)
        self.assertEqual(getCamType("auxTel"), CamType.AuxTel)
        instName = "telescope"
        assertMsg = f"Instrument name ({instName}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamType(instName)
        self.assertTrue(assertMsg in str(context.exception))

    def testGetCamTypeFromButlerName(self):
        self.assertEqual(
            getCamTypeFromButlerName("LSSTCam", DetectorType.WAVEFRONT), CamType.LsstCam
        )
        instName = "LSSTComCam"
        wfAssertMsg = (
            f"Wavefront sensors for instrument name ({instName}) are not supported."
        )
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, DetectorType.WAVEFRONT)
        self.assertTrue(wfAssertMsg in str(context.exception))

        self.assertEqual(
            getCamTypeFromButlerName("LSSTCam", DetectorType.SCIENCE),
            CamType.LsstFamCam,
        )
        self.assertEqual(
            getCamTypeFromButlerName("LSSTComCam", DetectorType.SCIENCE), CamType.ComCam
        )
        self.assertEqual(
            getCamTypeFromButlerName("LATISS", DetectorType.SCIENCE), CamType.AuxTel
        )
        instName = "telescope"
        sciAssertMsg = (
            f"Science sensors for instrument name ({instName}) are not supported."
        )
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, DetectorType.SCIENCE)
        self.assertTrue(sciAssertMsg in str(context.exception))

        detType = DetectorType.GUIDER
        detAssertMsg = f"Detector Type ({detType.name}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, detType)
        self.assertTrue(detAssertMsg in str(context.exception))

    def testGetFilterTypeFromBandLabel(self):
        # Test allowable filter band labels
        self.assertEqual(getFilterTypeFromBandLabel("u"), FilterType.LSST_U)
        self.assertEqual(getFilterTypeFromBandLabel("g"), FilterType.LSST_G)
        self.assertEqual(getFilterTypeFromBandLabel("r"), FilterType.LSST_R)
        self.assertEqual(getFilterTypeFromBandLabel("i"), FilterType.LSST_I)
        self.assertEqual(getFilterTypeFromBandLabel("z"), FilterType.LSST_Z)
        self.assertEqual(getFilterTypeFromBandLabel("y"), FilterType.LSST_Y)
        self.assertEqual(getFilterTypeFromBandLabel("SomeFilter"), FilterType.REF)

    def testGetCamNameFromCamType(self):
        # Test allowable CamType values
        self.assertEqual(getCamNameFromCamType(CamType.LsstCam), "lsst")
        self.assertEqual(getCamNameFromCamType(CamType.LsstFamCam), "lsstfam")
        self.assertEqual(getCamNameFromCamType(CamType.ComCam), "comcam")
        self.assertEqual(getCamNameFromCamType(CamType.AuxTel), "auxTel")
        self.assertEqual(getCamNameFromCamType(CamType.AuxTelZWO), "auxTelZWO")

        # Create a TestCamType that has a value greater than the last CamType
        class TestCamType(IntEnum):
            BadInst = len(CamType) + 1

        # Test the error is raised correctly with an incorrect CamType.
        badCamType = TestCamType.BadInst
        errMsg = f"CamType ({badCamType}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamNameFromCamType(badCamType)
        self.assertEqual(str(context.exception), errMsg)

    def testCreateInstDictFromConfig(self):
        # Test instDict creation in tasks
        testConfig = CalcZernikesTaskConfig()
        testInstDict = createInstDictFromConfig(testConfig)
        truthInstDict = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": None,
            "pixelSize": 10.0e-6,
        }

        self.assertDictEqual(truthInstDict, testInstDict)

    def testRotMatrix(self):
        # Test rotation with 0 degrees
        testTheta1 = 0
        rotMatrix1 = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(rotMatrix1, rotMatrix(testTheta1))

        # Test rotation with 90 degrees
        testTheta2 = 90
        rotMatrix2 = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(rotMatrix2, rotMatrix(testTheta2))

        # Test rotation with 45 degrees
        testTheta3 = 45
        rotMatrix3 = np.array([[0.707107, -0.707107], [0.707107, 0.707107]])
        np.testing.assert_array_almost_equal(rotMatrix3, rotMatrix(testTheta3))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
