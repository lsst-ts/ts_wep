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

from lsst.daf import butler as dafButler
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.EstimateZernikesTask import (
    EstimateZernikesTask,
    EstimateZernikesTaskConfig,
)
from lsst.ts.wep.Utility import (
    runProgram,
    DefocalType,
    writePipetaskCmd,
    writeCleanUpRepoCmd,
)


class TestEstimateZernikesTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")
        cls.runName = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.repoDir)
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testDataDir, "testTaskPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        runProgram(pipeCmd)

    def setUp(self):

        self.config = EstimateZernikesTaskConfig()
        self.task = EstimateZernikesTask(config=self.config)

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        datasetRefs = list(
            self.registry.queryDatasets(
                datasetType="postISRCCD", collections=[self.runName]
            ).expanded()
        )
        self.dataIds = [ref.dataId for ref in datasetRefs]

    def validateConfigs(self):

        self.config.donutTemplateSize = 120
        self.config.donutStampSize = 120
        self.config.initialCutoutSize = 290
        self.task = EstimateZernikesTask(config=self.config)

        self.assertEqual(self.task.donutTemplateSize, 120)
        self.assertEqual(self.task.donutStampSize, 120)
        self.assertEqual(self.task.initialCutoutSize, 290)

    def testGetTemplate(self):

        extra_template = self.task.getTemplate("R22_S11", DefocalType.Extra)
        self.assertEqual(
            np.shape(extra_template),
            (self.config.donutTemplateSize, self.config.donutTemplateSize),
        )

        self.config.donutTemplateSize = 180
        self.task = EstimateZernikesTask(config=self.config)
        intra_template = self.task.getTemplate("R22_S11", DefocalType.Intra)
        self.assertEqual(np.shape(intra_template), (180, 180))

    def testCutOutStamps(self):

        exposure = self.butler.get(
            "postISRCCD", dataId=self.dataIds[0], collections=[self.runName]
        )
        donutCatalog = self.butler.get(
            "donutCatalog", dataId=self.dataIds[0], collections=[self.runName]
        )
        donutStamps = self.task.cutOutStamps(exposure, donutCatalog, DefocalType.Extra)
        self.assertTrue(len(donutStamps), 4)

        stamp_centroid = donutStamps[0].centroid_position
        expCutOut = exposure.image.array[
            stamp_centroid.getX() - 80 : stamp_centroid.getX() + 80,
            stamp_centroid.getY() - 80 : stamp_centroid.getY() + 80,
        ].T
        np.testing.assert_array_equal(donutStamps[0].stamp_im.image.array, expCutOut)

    def testEstimateZernikes(self):

        donutStampsExtra = self.butler.get(
            "donutStampsExtra", dataId=self.dataIds[0], collection=[self.runName]
        )
        donutStampsIntra = self.butler.get(
            "donutStampsIntra", dataId=self.dataIds[0], collection=[self.runName]
        )

        zernCoeff = self.task.estimateZernikes(donutStampsExtra, donutStampsIntra)

        self.assertEqual(np.shape(zernCoeff), (len(donutStampsExtra), 19))

    @classmethod
    def tearDownClass(cls):

        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
