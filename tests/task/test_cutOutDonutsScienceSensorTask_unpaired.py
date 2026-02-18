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
from copy import copy

import numpy as np
from astropy.table import QTable

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.task.generateDonutCatalogUtils import addVisitInfoToCatTable
from lsst.ts.wep.utils import (
    DefocalType,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestCutOutDonutsScienceSensorTask(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    visitNum: int
    baseRunName: str
    pairTableName: str
    run2Name: str
    run3Name: str
    cameraName: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run the pipeline only once since it takes a
        couple minutes with the ISR.
        """

        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.cameraName = "LSSTCam"

        # Check that runs don't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if "pretest_run_science" in collectionsList:
            cls.runName = "pretest_run_science"
        else:
            cls.runName = "run1"
            if "run1" in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, "run1")
                runProgram(cleanUpCmd)

        # Point to the collections for the reference catalogs,
        # the raw images and the camera model in the calib directory
        # that comes from `butler write-curated-calibrations`.
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        instrument = "lsst.obs.lsst.LsstCam"
        if cls.runName == "run1":
            pipelineYaml = os.path.join(testPipelineConfigDir, "testCutoutsFamPipeline_unpaired.yaml")
            pipeCmd = writePipetaskCmd(
                cls.repoDir,
                cls.runName,
                instrument,
                collections,
                pipelineYaml=pipelineYaml,
            )
            pipeCmd += " -d 'exposure IN (4021123106001..4021123106007)'"
            runProgram(pipeCmd)
        elif cls.runName == "pretest_run_science":
            collections += ",pretest_run_science"
            pipelineYaml = os.path.join(
                testPipelineConfigDir, "testCutoutsFamPipeline_unpaired.yaml#cutOutDonutsScienceSensorTask"
            )
            pipeCmd += " -d 'exposure IN (4021123106001..4021123106007)'"
            runProgram(pipeCmd)

    def setUp(self) -> None:
        self.config = CutOutDonutsScienceSensorTaskConfig(runPaired=False)
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.dataIdIntra = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106002,
            "visit": 4021123106002,
        }

    def testValidateConfigs(self) -> None:
        self.config.donutStampSize = 120
        self.config.initialCutoutPadding = 290
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutPadding, 290)

    def testAssignExtraIntraIdxLsstCam(self) -> None:
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZNegative, focusZPositive, "LSSTCam")
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZPositive, focusZNegative, "LSSTCam")
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxLsstComCam(self) -> None:
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZNegative, focusZPositive, "LSSTComCam")
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZPositive, focusZNegative, "LSSTComCam")
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxLsstComCamSim(self) -> None:
        focusZNegative = -1
        focusZPositive = 1

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZNegative, focusZPositive, "LSSTComCamSim")
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZPositive, focusZNegative, "LSSTComCamSim")
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)

    def testAssignExtraIntraIdxInvalidCamera(self) -> None:
        cameraName = "WrongCam"
        with self.assertRaises(ValueError) as context:
            self.task.assignExtraIntraIdx(-1, 1, cameraName)
        errorStr = str(
            f"Invalid cameraName parameter: {cameraName}. Camera must  "
            "be one of: 'LSSTCam', 'LSSTComCam', 'LSSTComCamSim' or 'LATISS'",
        )
        self.assertEqual(errorStr, str(context.exception))

    def testTaskRun(self) -> None:
        # Grab two exposures from the same detector at two different visits to
        # get extra and intra
        exposureExtra = self.butler.get("post_isr_image", dataId=self.dataIdExtra, collections=[self.runName])
        exposureIntra = self.butler.get("post_isr_image", dataId=self.dataIdIntra, collections=[self.runName])

        donutCatalogExtra = self.butler.get("donutTable", dataId=self.dataIdExtra, collections=[self.runName])
        donutCatalogIntra = self.butler.get("donutTable", dataId=self.dataIdIntra, collections=[self.runName])
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections="LSSTCam/calib/unbounded",
        )

        # Test return values when no sources in catalog
        columns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "source_flux",
            "detector",
        ]
        noSrcDonutCatalog = QTable({column: [] for column in columns})
        noSrcDonutCatalog = addVisitInfoToCatTable(exposureExtra, noSrcDonutCatalog)
        testOutNoSrc = self.task.run([exposureExtra, exposureIntra], [noSrcDonutCatalog] * 2, camera)

        self.assertEqual(len(testOutNoSrc.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra), 0)

        # Test normal behavior
        taskOut = self.task.run(
            [copy(exposureIntra), copy(exposureExtra)],
            [donutCatalogExtra, donutCatalogIntra],
            camera,
        )

        testExtraStamps = self.task.cutOutStamps(
            exposureExtra, donutCatalogExtra, DefocalType.Extra, camera.getName()
        )
        testIntraStamps = self.task.cutOutStamps(
            exposureIntra, donutCatalogIntra, DefocalType.Intra, camera.getName()
        )

        for donutStamp, cutOutStamp in zip(taskOut.donutStampsExtra, testExtraStamps):
            self.assertMaskedImagesAlmostEqual(donutStamp.stamp_im, cutOutStamp.stamp_im)  # type: ignore
        for donutStamp, cutOutStamp in zip(taskOut.donutStampsIntra, testIntraStamps):
            self.assertMaskedImagesAlmostEqual(donutStamp.stamp_im, cutOutStamp.stamp_im)  # type: ignore

        # Check that the new metadata is stored in butler
        donutStamps = self.butler.get(
            "donutStampsScienceSensor", dataId=self.dataIdExtra, collections=[self.runName]
        )
        metadata = list(donutStamps.metadata)
        expectedMetadata = [
            "RA_DEG",
            "DEC_DEG",
            "DET_NAME",
            "CAM_NAME",
            "DFC_TYPE",
            "DFC_DIST",
            "MAG",
            "CENT_X0",
            "CENT_Y0",
            "CENT_X",
            "CENT_Y",
            "CENT_DX",
            "CENT_DY",
            "CENT_DR",
            "BLEND_CX",
            "BLEND_CY",
            "X0",
            "Y0",
            "SN",
            "SIGNAL_MEAN",
            "SIGNAL_SUM",
            "NPX_MASK",
            "BKGD_STDEV",
            "SQRT_MEAN_VAR",
            "BKGD_VAR",
            "BACKGROUND_IMAGE_MEAN",
            "NOISE_VAR_BKGD",
            "NOISE_VAR_DONUT",
            "EFFECTIVE",
            "ENTROPY",
            "PEAK_HEIGHT",
            "MJD",
            "BORESIGHT_ROT_ANGLE_RAD",
            "BORESIGHT_PAR_ANGLE_RAD",
            "BORESIGHT_ALT_RAD",
            "BORESIGHT_AZ_RAD",
            "BORESIGHT_RA_RAD",
            "BORESIGHT_DEC_RAD",
            "BANDPASS",
        ]
        # Test that all expected metadata is included in the butler
        self.assertEqual(np.sum(np.in1d(expectedMetadata, metadata)), len(expectedMetadata))
        for measure in [
            "SIGNAL_SUM",
            "SIGNAL_MEAN",
            "NOISE_VAR_BKGD",
            "NOISE_VAR_DONUT",
            "EFFECTIVE",
            "ENTROPY",
            "PEAK_HEIGHT",
        ]:
            self.assertEqual(len(donutStamps), len(donutStamps.metadata.getArray(measure)))

    @staticmethod
    def compareMetadata(metadata1: dict, metadata2: dict) -> bool:
        for k, v in metadata1.items():
            if k.startswith("LSST BUTLER"):
                continue
            try:
                v2 = metadata2[k]
            except KeyError:
                # key not in metadata2, so unequal.
                return False

            if isinstance(v, (int, float, np.number)):
                if not isinstance(v2, (int, float, np.number)):
                    return False

                # Special case since nan != nan
                if np.isnan(v):
                    if np.isnan(v2):
                        continue
                    else:
                        return False

                if v != v2:
                    return False
            else:
                if v != v2:
                    return False
        return True

    @classmethod
    def tearDownClass(cls) -> None:
        tearDownRunList = list()
        if cls.runName == "run1":
            tearDownRunList.append(cls.runName)
        for runName in tearDownRunList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, runName)
            runProgram(cleanUpCmd)
