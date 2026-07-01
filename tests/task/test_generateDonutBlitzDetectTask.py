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
from astropy.table import vstack

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.ts.wep.task.generateDonutBlitzDetectTask import (
    GenerateDonutBlitzDetectTask,
    GenerateDonutBlitzDetectTaskConfig,
)
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


class TestGenerateDonutBlitzDetectTask(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str
    baseRunName: str

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

        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry

        collectionsList = list(registry.queryCollections())
        cls.runName = "run_blitz"
        cls.baseRunName = "run_blitz"
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testDonutBlitzDetectPipeline.yaml")

        if "pretest_run_science" in collectionsList:
            pipelineYaml += "#generateDonutBlitzDetectTask"
            collections += ",pretest_run_science"
            cls.baseRunName = "pretest_run_science"

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        pipeCmd += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipeCmd += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
        runProgram(pipeCmd)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.runName == "run_blitz":
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.config = GenerateDonutBlitzDetectTaskConfig()
        self.task = GenerateDonutBlitzDetectTask(config=self.config)

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

        self.testDataIdS10 = {
            "instrument": "LSSTCam",
            "detector": 93,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.testDataIdS11 = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }
        self.camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )

    def testValidateConfigs(self) -> None:
        self.config.edgeMargin = 100
        self.config.detectionBinning = 2
        self.config.innerFracThreshold = 0.01
        self.config.outerFracThreshold = 0.01
        self.config.snrThreshold = 50.0
        self.config.maxFieldDist = 1.5
        task = GenerateDonutBlitzDetectTask(config=self.config)

        self.assertEqual(task.config.edgeMargin, 100)
        self.assertEqual(task.config.detectionBinning, 2)
        self.assertAlmostEqual(task.config.innerFracThreshold, 0.01)
        self.assertAlmostEqual(task.config.outerFracThreshold, 0.01)
        self.assertAlmostEqual(task.config.snrThreshold, 50.0)
        self.assertAlmostEqual(task.config.maxFieldDist, 1.5)

    def testEmptyTable(self) -> None:
        exposure_S11 = self.butler.get(
            "post_isr_image",
            dataId=self.testDataIdS11,
            collections=[self.baseRunName],
        )
        testTable = self.task.emptyTable(exposure_S11)

        self.assertEqual(len(testTable), 0)
        expected_columns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
        ]
        expected_columns.append("donut_id")
        self.assertCountEqual(testTable.columns, expected_columns)
        self.assertIn("blend_centroid_x", testTable.meta)
        self.assertIn("blend_centroid_y", testTable.meta)
        self.assertIn("visit_info", testTable.meta)

    def testDetectPeaks(self) -> None:
        """Test _detectPeaks finds a known synthetic donut."""
        rng = np.random.default_rng(42)
        size = 400
        donut_radius = 30.0
        center_row, center_col = 200, 200

        # Build a noiseless annular donut
        gy, gx = np.mgrid[0:size, 0:size]
        r = np.hypot(gx - center_col, gy - center_row)
        signal = np.where((r < donut_radius * 1.05) & (r > donut_radius * 0.6), 1000.0, 0.0)
        # Small background noise so the IQR estimator works
        signal += rng.normal(0.0, 1.0, signal.shape)

        image = afwImage.ImageF(size, size)
        image.array[:] = signal.astype(np.float32)
        exposure = afwImage.ExposureF(afwImage.MaskedImageF(image))

        peaks = self.task._detectPeaks(exposure, donut_radius, obscuration=0.612)

        self.assertGreater(len(peaks), 0, "No peaks found in synthetic donut image")
        distances = np.hypot(peaks[:, 0] - center_row, peaks[:, 1] - center_col)
        self.assertLess(
            np.min(distances),
            5,
            "Nearest peak is more than 5 px from the synthetic donut center",
        )

    def testMeasureFlux(self) -> None:
        """Test _measureFlux on a synthetic donut stamp."""
        rng = np.random.default_rng(7)
        size = 400
        donut_radius = 30.0
        center_row, center_col = 200, 200

        gy, gx = np.mgrid[0:size, 0:size]
        r = np.hypot(gx - center_col, gy - center_row)
        signal = np.where((r < donut_radius * 1.05) & (r > donut_radius * 0.6), 1000.0, 0.0)
        signal += rng.normal(0.0, 1.0, signal.shape)

        image = afwImage.ImageF(size, size)
        image.array[:] = signal.astype(np.float32)
        exposure = afwImage.ExposureF(afwImage.MaskedImageF(image))

        peaks = np.array([[center_row, center_col]])
        measTable = self.task._measureFlux(peaks, exposure, donut_radius, obscuration=0.612)

        self.assertEqual(len(measTable), 1)

        flux = float(measTable["flux"][0])
        inner_flux = float(measTable["inner_flux"][0])
        outer_flux = float(measTable["outer_flux"][0])
        snr = float(measTable["snr"][0])

        # Flux should be positive and substantial
        self.assertGreater(flux, 0)

        # Inner hole and outer annulus should carry very little flux relative
        # to the main ring (both are dark in a well-formed donut)
        self.assertLess(abs(inner_flux / flux), 0.005)
        self.assertLess(abs(outer_flux / flux), 0.005)

        # S/N should be well above the default threshold for a 1000-ADU signal
        self.assertGreater(snr, self.config.snrThreshold)

    def testTaskRun(self) -> None:
        """Test that the task runs interactively."""
        # Test on a noise-only exposure: expect empty catalog
        exposure_S11 = self.butler.get(
            "post_isr_image",
            dataId=self.testDataIdS11,
            collections=[self.baseRunName],
        )
        rng = np.random.default_rng(0)
        bkgnd = 100 * (rng.random(np.shape(exposure_S11.image.array)) - 0.5)
        image = afwImage.ImageF(exposure_S11.getBBox())
        image.array[:] = bkgnd
        maskedImage = afwImage.MaskedImageF(image)
        exposure_noSrc = copy(exposure_S11)
        exposure_noSrc.setMaskedImage(maskedImage)

        taskOutNoSrc = self.task.run(exposure_noSrc, self.camera)

        self.assertEqual(len(taskOutNoSrc.donutCatalog), 0)

        expected_columns = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "detector",
            "source_flux",
            "donut_id",
        ]
        self.assertCountEqual(taskOutNoSrc.donutCatalog.columns, expected_columns)

        expected_metakeys = ["blend_centroid_x", "blend_centroid_y", "visit_info"]
        self.assertCountEqual(taskOutNoSrc.donutCatalog.meta.keys(), expected_metakeys)

        # Run on real ISR data; use lenient thresholds to ensure detection
        self.task.config.innerFracThreshold = 0.05
        self.task.config.outerFracThreshold = 0.05
        self.task.config.snrThreshold = 20.0

        exposure_S10 = self.butler.get(
            "post_isr_image",
            dataId=self.testDataIdS10,
            collections=[self.baseRunName],
        )
        taskOut_S11 = self.task.run(exposure_S11, self.camera)
        taskOut_S10 = self.task.run(exposure_S10, self.camera)

        # Should find sources in both detectors
        self.assertGreater(len(taskOut_S11.donutCatalog), 0)
        self.assertGreater(len(taskOut_S10.donutCatalog), 0)

        outputTable = vstack(
            [taskOut_S11.donutCatalog, taskOut_S10.donutCatalog], metadata_conflicts="silent"
        )
        self.assertCountEqual(outputTable.columns, expected_columns)

        # Sources should be sorted by flux (brightest first)
        for cat in [taskOut_S11.donutCatalog, taskOut_S10.donutCatalog]:
            if len(cat) > 1:
                np.testing.assert_array_equal(
                    np.arange(len(cat)),
                    np.argsort(cat["source_flux"].value)[::-1],
                )

        # With maxFieldDist=0, the field angle cut should reject all sources
        self.task.config.maxFieldDist = 0
        taskOut_noField = self.task.run(exposure_S10, self.camera)
        self.assertEqual(len(taskOut_noField.donutCatalog), 0)
        self.assertCountEqual(taskOut_noField.donutCatalog.meta.keys(), expected_metakeys)

    def testTaskRunPipeline(self) -> None:
        """Test that the task runs end-to-end through the pipeline."""
        donutCatTable_S11 = self.butler.get(
            "donutTable",
            dataId=self.testDataIdS11,
            collections=[self.runName],
        )
        donutCatTable_S10 = self.butler.get(
            "donutTable",
            dataId=self.testDataIdS10,
            collections=[self.runName],
        )

        # Should detect at least one source per detector
        self.assertGreater(len(donutCatTable_S11), 0)
        self.assertGreater(len(donutCatTable_S10), 0)

        # Sources should be sorted brightest-first
        if len(donutCatTable_S10) > 1:
            np.testing.assert_array_equal(
                np.arange(len(donutCatTable_S10)),
                np.argsort(donutCatTable_S10["source_flux"].value)[::-1],
            )
        if len(donutCatTable_S11) > 1:
            np.testing.assert_array_equal(
                np.arange(len(donutCatTable_S11)),
                np.argsort(donutCatTable_S11["source_flux"].value)[::-1],
            )

        # Check correct detector names
        self.assertEqual(np.unique(donutCatTable_S11["detector"]), "R22_S11")
        self.assertEqual(np.unique(donutCatTable_S10["detector"]), "R22_S10")

        # Verify catalog schema
        outputTable = vstack([donutCatTable_S11, donutCatTable_S10], metadata_conflicts="silent")
        self.assertCountEqual(
            outputTable.columns,
            ["coord_ra", "coord_dec", "centroid_x", "centroid_y", "detector", "source_flux", "donut_id"],
        )
        self.assertCountEqual(
            outputTable.meta.keys(),
            ["blend_centroid_x", "blend_centroid_y", "visit_info"],
        )
