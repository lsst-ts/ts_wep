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

import numpy as np
from astropy.table import vstack

import lsst.geom
from lsst.daf.butler import Butler
from lsst.ts.wep.task.generateDonutCatalogWcsTask import (
    GenerateDonutCatalogWcsTask,
    GenerateDonutCatalogWcsTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)
from lsst.utils.tests import TestCase


class TestGenerateDonutCatalogWcsTask(TestCase):
    def setUp(self) -> None:
        self.config = GenerateDonutCatalogWcsTaskConfig()
        self.config.donutSelector.unblendedSeparation = 1
        self.task = GenerateDonutCatalogWcsTask(config=self.config)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.centerRaft = ["R22_S10", "R22_S11"]

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

    def _getRefCat(self) -> list:
        refCatList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            refCatList.append(self.butler.getDeferred(ref, collections=["refcats/gen2"]))

        return refCatList

    def testValidateConfigs(self) -> None:
        self.config.doDonutSelection = False
        self.config.anyFilterMapsToThis = "phot_g_mean"
        self.config.edgeMargin = 100
        self.task = GenerateDonutCatalogWcsTask(config=self.config)

        self.assertEqual(self.task.config.doDonutSelection, False)
        self.assertEqual(self.task.config.anyFilterMapsToThis, "phot_g_mean")
        self.assertEqual(self.task.config.edgeMargin, 100)

    def testAnyFilterMapsToThis(self) -> None:
        self.config.anyFilterMapsToThis = "r"
        self.task = GenerateDonutCatalogWcsTask(config=self.config)

        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        self.assertEqual(refObjLoader.config.anyFilterMapsToThis, "r")

    def testGetRefObjLoader(self) -> None:
        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        # Check that our refObjLoader loads the available objects
        # within a given search radius
        donutCatSmall = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(0.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatSmall.refCat), 8)

        donutCatFull = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(2.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatFull.refCat), 24)

    def testPipeline(self) -> None:
        """
        Test that the task runs in a pipeline. Also functions as a test of
        runQuantum function.
        """

        # Run pipeline command
        runName = "run1"
        instrument = "lsst.obs.lsst.LsstCam"
        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        exposureId = 4021123106001  # Exposure ID for test extra-focal image
        testPipelineConfigDir = os.path.join(self.testDataDir, "pipelineConfigs")
        pipelineYaml = os.path.join(testPipelineConfigDir, "testDonutCatWcsPipeline.yaml")
        pipetaskCmd = writePipetaskCmd(
            self.repoDir, runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Update task configuration to match pointing information
        pipetaskCmd += f" -d 'exposure IN ({exposureId})'"

        # Check that run doesn't already exist due to previous improper cleanup
        collectionsList = list(self.registry.queryCollections())
        if runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
            runProgram(cleanUpCmd)

        # Run pipeline task
        runProgram(pipetaskCmd)

        # Test instrument matches
        pipelineButler = Butler.from_config(self.repoDir)
        s11_wcs = pipelineButler.get(
            "post_isr_image.wcs",
            dataId={
                "instrument": "LSSTCam",
                "detector": 94,
                "visit": exposureId,
                "exposure": exposureId,
            },
            collections=[f"{runName}"],
        )
        s10_wcs = pipelineButler.get(
            "post_isr_image.wcs",
            dataId={
                "instrument": "LSSTCam",
                "detector": 93,
                "visit": exposureId,
                "exposure": exposureId,
            },
            collections=[f"{runName}"],
        )
        donutCatTable_S11 = pipelineButler.get(
            "donutTable",
            dataId={"instrument": "LSSTCam", "detector": 94, "visit": exposureId},
            collections=[f"{runName}"],
        )
        donutCatTable_S10 = pipelineButler.get(
            "donutTable",
            dataId={"instrument": "LSSTCam", "detector": 93, "visit": exposureId},
            collections=[f"{runName}"],
        )

        # Check 4 sources in each detector
        self.assertEqual(len(donutCatTable_S11), 4)
        self.assertEqual(len(donutCatTable_S10), 4)

        # Check correct detector names
        self.assertEqual(np.unique(donutCatTable_S11["detector"]), "R22_S11")
        self.assertEqual(np.unique(donutCatTable_S10["detector"]), "R22_S10")

        # Check outputs are correct
        outputTable = vstack([donutCatTable_S11, donutCatTable_S10])
        self.assertEqual(len(outputTable), 8)
        self.assertCountEqual(
            outputTable.columns,
            ["coord_ra", "coord_dec", "centroid_x", "centroid_y", "g_flux", "detector", "donut_id"],
        )
        self.assertCountEqual(
            outputTable.meta.keys(),
            ["blend_centroid_x", "blend_centroid_y", "visit_info"],
        )
        true_ra = [
            6.281628787,
            0.001158288,
            0.000188775,
            6.281628805,
            0.001158410,
            0.000188269,
            6.281627496,
            6.281627514,
        ]
        true_dec = [
            -0.001389369,
            0.0017140704,
            0.000744243,
            -0.001292381,
            -0.002390839,
            -0.003360248,
            -0.005492987,
            -0.005396017,
        ]
        np.testing.assert_allclose(np.sort(true_ra), np.sort(outputTable["coord_ra"].value), atol=1e-8)
        np.testing.assert_allclose(np.sort(true_dec), np.sort(outputTable["coord_dec"].value), atol=1e-8)
        s11_x, s11_y = s11_wcs.skyToPixelArray(true_ra[:4], true_dec[:4])
        s10_x, s10_y = s10_wcs.skyToPixelArray(true_ra[4:], true_dec[4:])
        true_x = np.sort(np.array([s11_x, s10_x]).flatten())
        true_y = np.sort(np.array([s11_y, s10_y]).flatten())
        np.testing.assert_allclose(
            true_x,
            np.sort(outputTable["centroid_x"].value),
            atol=1e-2,  # Small fractions of pixel okay since we abbreviated ra, dec positions above
        )
        np.testing.assert_allclose(true_y, np.sort(outputTable["centroid_y"].value), atol=1e-2)
        fluxTruth = np.ones(8)
        fluxTruth[:6] = 3630780.5477010026
        fluxTruth[6:] = 363078.0547701003
        self.assertCountEqual(outputTable["g_flux"].value, fluxTruth)

        # Clean up
        cleanUpCmd = writeCleanUpRepoCmd(self.repoDir, runName)
        runProgram(cleanUpCmd)

    def testDonutCatalogGeneration(self) -> None:
        """
        Test that task creates a QTable with detector information.
        """

        # Create list of deferred loaders for the ref cat
        deferredList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            deferredList.append(self.butler.getDeferred(ref, collections=["refcats/gen2"]))
        expGenerator = self.registry.queryDatasets(
            datasetType="raw",
            collections=["LSSTCam/raw/all"],
            dimensions=["exposure", "instrument"],
            dataId={"exposure": 4021123106001, "instrument": "LSSTCam"},
        ).expanded()
        expList = []
        for expRef in expGenerator:
            expList.append(self.butler.get("raw", dataId=expRef.dataId, collections=["LSSTCam/raw/all"]))

        # run task on all exposures
        donutCatTableList = []
        donutCatXPixelList = []
        donutCatYPixelList = []
        # Set task to take all donuts regardless of magnitude
        self.task.config.donutSelector.useCustomMagLimit = True
        for exposure in expList:
            taskOutput = self.task.run(deferredList, exposure)
            self.assertEqual(len(taskOutput.donutCatalog), 4)
            donutCatTableList.append(taskOutput.donutCatalog)
            # Get pixel locations with proper wcs
            donutX, donutY = exposure.wcs.skyToPixelArray(
                taskOutput.donutCatalog["coord_ra"],
                taskOutput.donutCatalog["coord_dec"],
            )
            donutCatXPixelList.append(donutX)
            donutCatYPixelList.append(donutY)

        # concatenate catalogs from each exposure into a single catalog
        # to compare against the test input reference catalog
        outputTable = donutCatTableList[0]
        for donutCat in donutCatTableList[1:]:
            outputTable = vstack([outputTable, donutCat])

        # Compare ra, dec info to original input catalog
        inputCat = np.genfromtxt(
            os.path.join(self.testDataDir, "phosimOutput", "realComCam", "skyComCamInfo.txt"),
            names=["id", "ra", "dec", "mag"],
        )

        self.assertEqual(len(outputTable), 8)
        self.assertCountEqual(np.radians(inputCat["ra"]), outputTable["coord_ra"].value)
        self.assertCountEqual(np.radians(inputCat["dec"]), outputTable["coord_dec"].value)
        np.testing.assert_allclose(
            np.sort(np.array(donutCatXPixelList).flatten()),
            np.sort(outputTable["centroid_x"]),
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            np.sort(np.array(donutCatYPixelList).flatten()),
            np.sort(outputTable["centroid_y"]),
            atol=1e-14,
            rtol=1e-14,
        )
        fluxTruth = np.ones(8)
        fluxTruth[:6] = 3630780.5477010026
        fluxTruth[6:] = 363078.0547701003
        self.assertCountEqual(outputTable["g_flux"].value, fluxTruth)
