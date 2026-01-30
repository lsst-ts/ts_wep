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

import getpass
import os
import tempfile

import pytest

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.ts.wep.task.cutOutDonutsScienceSensorTask import (
    CutOutDonutsScienceSensorTask,
    CutOutDonutsScienceSensorTaskConfig,
)
from lsst.ts.wep.utils import (
    DefocalType,
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


@pytest.mark.skip
@pytest.mark.skipif(
    os.path.exists("/sdf/data/rubin/repo/main") is False,
    reason="requires access to data in /repo/main database",
)
@pytest.mark.skipif(
    not os.getenv("PGPASSFILE"),
    reason="requires access to butler db",
)
class TestCutOutDonutsLatissTask(lsst.utils.tests.TestCase):
    runName: str
    repoDir: str
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
        cls.repoDir = "/sdf/data/rubin/repo/main"

        # Create a temporary test directory
        # under /sdf/data/rubin/repo/main/u/$USER
        # to ensure write access is granted
        user = getpass.getuser()
        tempDir = os.path.join(cls.repoDir, "u", user)
        testDir = tempfile.TemporaryDirectory(dir=tempDir)
        testDirName = os.path.split(testDir.name)[1]  # temp dir name
        cls.runName = os.path.join("u", user, testDirName)

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        # Point to the collections with
        # the raw images and calibrations
        collections = "LATISS/raw/all,LATISS/calib"
        instrument = "lsst.obs.lsst.Latiss"
        cls.cameraName = "LATISS"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testCutoutsLatissPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += " -d 'exposure IN (2021090800487, 2021090800488) AND visit_system=0'"
        runProgram(pipeCmd)

    def setUp(self) -> None:
        self.config = CutOutDonutsScienceSensorTaskConfig()
        self.config.donutStampSize = 200
        self.config.opticalModel = "onAxis"
        self.config.initialCutoutPadding = 40
        self.task = CutOutDonutsScienceSensorTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.dataIdExtra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800487,
            "visit": 2021090800487,
        }
        self.dataIdIntra = {
            "instrument": "LATISS",
            "detector": 0,
            "exposure": 2021090800488,
            "visit": 2021090800488,
        }

    def testAssignExtraIntraIdx(self) -> None:
        focusZextra = -1.5
        focusZintra = -1.2

        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZextra, focusZintra, "LATISS")
        self.assertEqual(extraIdx, 0)
        self.assertEqual(intraIdx, 1)
        # invert the order
        extraIdx, intraIdx = self.task.assignExtraIntraIdx(focusZintra, focusZextra, "LATISS")
        self.assertEqual(extraIdx, 1)
        self.assertEqual(intraIdx, 0)

        with self.assertRaises(ValueError):
            self.task.assignExtraIntraIdx(focusZextra, focusZextra, "LATISS")
        with self.assertRaises(ValueError) as context:
            self.task.assignExtraIntraIdx(focusZintra, focusZintra, "LATISS")
        self.assertEqual(
            "Must have two images with different FOCUSZ parameter.",
            str(context.exception),
        )

    def testTaskRun(self) -> None:
        # Grab two exposures from the same detector at two different visits to
        # get extra and intra
        exposureExtra = self.butler.get("post_isr_image", dataId=self.dataIdExtra, collections=[self.runName])
        exposureIntra = self.butler.get("post_isr_image", dataId=self.dataIdIntra, collections=[self.runName])

        donutTableExtra = self.butler.get("donutTable", dataId=self.dataIdExtra, collections=[self.runName])
        donutTableIntra = self.butler.get("donutTable", dataId=self.dataIdIntra, collections=[self.runName])
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LATISS"},
            collections="LATISS/calib/unbounded",
        )

        # Test return values when no sources in catalog
        noSrcDonutTable = donutTableExtra.copy()
        noSrcDonutTable.remove_rows(slice(None))
        testOutNoSrc = self.task.run([exposureExtra, exposureIntra], [noSrcDonutTable] * 2, camera)

        self.assertEqual(len(testOutNoSrc.donutStampsExtra), 0)
        self.assertEqual(len(testOutNoSrc.donutStampsIntra), 0)

        # Test normal behavior
        taskOut = self.task.run(
            [exposureExtra, exposureIntra],
            [donutTableExtra, donutTableIntra],
            camera,
        )

        # Make sure donut catalog actually has sources to test
        self.assertGreater(len(donutTableExtra), 0)
        self.assertGreater(len(donutTableIntra), 0)
        # Test they have the same number of sources
        self.assertEqual(len(donutTableExtra), len(donutTableIntra))

        # Check that donut catalog sources are all cut out
        self.assertEqual(len(taskOut.donutStampsExtra), len(donutTableExtra))
        self.assertEqual(len(taskOut.donutStampsIntra), len(donutTableIntra))

        testExtraStamps = self.task.cutOutStamps(
            exposureExtra, donutTableExtra, DefocalType.Extra, camera.getName()
        )
        testIntraStamps = self.task.cutOutStamps(
            exposureIntra, donutTableIntra, DefocalType.Intra, camera.getName()
        )

        for donutStamp, cutOutStamp in zip(taskOut.donutStampsExtra, testExtraStamps):
            self.assertMaskedImagesAlmostEqual(  # type: ignore
                donutStamp.stamp_im, cutOutStamp.stamp_im, atol=1e-4
            )
        for donutStamp, cutOutStamp in zip(taskOut.donutStampsIntra, testIntraStamps):
            self.assertMaskedImagesAlmostEqual(  # type: ignore
                donutStamp.stamp_im, cutOutStamp.stamp_im, atol=1e-4
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
