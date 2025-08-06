import os
import lsst.utils.tests
import numpy as np
from lsst.daf.butler import Butler
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.donutSizeCorrelator import DonutSizeCorrelator
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
    DefocalType,
    createTemplateForDetector,
    getTaskInstrument
)

class TestDonutSizeCorrelator(lsst.utils.tests.TestCase):
    runName: str
    testDataDir: str
    repoDir: str

    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate donut stamps needed for task.
        """
        # Run pipeline command
        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        cls.repoDir = os.path.join(testDataDir, "gen3TestRepo")

        butler = Butler.from_config(cls.repoDir)
        registry = butler.registry

        # Check that run doesn't already exist due to previous improper cleanup
        collectionsList = list(registry.queryCollections())
        cls.runName = "run1"
        if cls.runName in collectionsList:
            cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
            runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"
        pipelineYaml = os.path.join(testPipelineConfigDir, "testIsrPipeline.yaml")

        pipeCmd = writePipetaskCmd(
            cls.repoDir, cls.runName, instrument, collections, pipelineYaml=pipelineYaml
        )
        # Make sure we are using the right exposure+detector combinations
        pipeCmd += ' -d "exposure = 4021123106001 AND detector = 94"'
        runProgram(pipeCmd)

    def setUp(self) -> None:
        self.correlator = DonutSizeCorrelator()
        self.butler = Butler.from_config(self.repoDir)
        self.dataId = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
            "visit": 4021123106001,
        }

    def testRunCorrelator(self) -> None:
        exposure = self.butler.get(
            "post_isr_image",
            dataId=self.dataId,
            collections=[self.runName],
        )
        image = self.correlator.prepButlerExposure(exposure)
        diameter, *_ = self.correlator.getDonutDiameter(image)
        self.assertEqual(type(diameter), np.int64)

        # test that the diameter is correct
        self.assertEqual(120, diameter)

    def testPrepButlerExposure(self) -> None:
        exposure = self.butler.get(
            "post_isr_image",
            dataId=self.dataId,
            collections=[self.runName],
        )
        image = self.correlator.prepButlerExposure(exposure)

        self.assertIsInstance(image, np.ndarray)
        self.assertTrue(np.all(np.isfinite(image)))  # No NaNs or infs
        self.assertGreater(image.shape[0], 0)
        self.assertAlmostEqual(np.nanmax(image), 1.0, places=5)

    def testCropAndBinImage(self) -> None:
        # Make synthetic image with a bright donut near the center
        image = np.ones((2048, 2048))
        image[1024, 1024] = 100.0  # Simulate bright donut

        cropped_binned = self.correlator.cropAndBinImage(image, length=500, binning=10)

        self.assertEqual(cropped_binned.shape[0], 500 // 10)
        self.assertEqual(cropped_binned.shape[1], 500 // 10)

    def testCreateDonutTemplate(self) -> None:
        diameter = 40.0
        template = self.correlator.createDonutTemplate(diameter)

        self.assertIsInstance(template, np.ndarray)
        self.assertEqual(template.shape[0], template.shape[1])  # Square
        self.assertTrue(np.all((template >= 0.0) & (template <= 1.0)))  # Fractional mask
        self.assertGreater(np.sum(template), 0.0)

    def testCorrelateImage(self) -> None:
        # Create image with a bright ring at a known radius
        size = 512
        Y, X = np.indices((size, size))
        r = np.sqrt((X - size//2)**2 + (Y - size//2)**2)
        image = np.exp(-((r - 60.0)**2) / (2 * 3**2))  # Approximate donut

        diameters, correlations = self.correlator.correlateImage(
            image=image,
            resolution=4,
            dMin=40,
            dMax=100,
            length=256
        )

        self.assertIsInstance(diameters, np.ndarray)
        self.assertIsInstance(correlations, np.ndarray)
        self.assertEqual(diameters.shape, correlations.shape)
        self.assertGreater(len(diameters), 0)
        self.assertGreater(np.max(correlations), 0.0)

    def testGetDonutDiameter(self) -> None:
        # Create a donut template and compare to the expected radius
        camera = LsstCam.getCamera()
        detectorName = 'R44_SW1'
        instrument = getTaskInstrument(
                    'LSSTCam',
                detectorName,
            )

        # change offset from 0.0015 to create large donut
        instrument.defocalOffset = 0.0025

        template = createTemplateForDetector(
        detector=camera.get(detectorName),
        defocalType=DefocalType.Intra,
        bandLabel='r',
        instrument=instrument,
        opticalModel='offAxis',
        padding=5,
        isBinary=True,
        )
        diameter, diameters, correlations = self.correlator.getDonutDiameter(template)

        self.assertTrue(np.isfinite(diameter))
        self.assertIsInstance(diameter, (int, float, np.integer))
        self.assertGreater(diameter, 200)
        self.assertTrue(abs(diameter- instrument.donutDiameter)< 25)  # Loose tolerance


    @classmethod
    def tearDownClass(cls) -> None:
        cleanUpCmd = writeCleanUpRepoCmd(cls.repoDir, cls.runName)
        runProgram(cleanUpCmd)
