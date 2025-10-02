import os

from _pytest.config import Config
from _pytest.config.argparsing import Parser
from lsst.daf.butler import Butler
from lsst.ts.wep.utils import (
    getModulePath,
    runProgram,
    writeCleanUpRepoCmd,
    writePipetaskCmd,
)


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--run-pretest",
        action="store_true",
        default=False,
        help="Run WEP pipeline before all tests.",
    )


def pytest_configure(config: Config) -> None:
    if config.getoption("--run-pretest"):
        print("Running pre-test command...")

        # Set up the butler repository config
        config.testInfo = dict()
        moduleDir = getModulePath()
        testDataDir = os.path.join(moduleDir, "tests", "testData")
        testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
        config.testInfo["repoDir"] = os.path.join(testDataDir, "gen3TestRepo")
        config.testInfo["runNameCwfs"] = "pretest_run_cwfs"
        config.testInfo["runNameScience"] = "pretest_run_science"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = Butler.from_config(config.testInfo["repoDir"])
        registry = butler.registry
        collectionsList = list(registry.queryCollections())
        for runName in [
            config.testInfo["runNameCwfs"],
            config.testInfo["runNameScience"],
        ]:
            if runName in collectionsList:
                cleanUpCmd = writeCleanUpRepoCmd(config.testInfo["repoDir"], runName)
                runProgram(cleanUpCmd)

        collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all"
        instrument = "lsst.obs.lsst.LsstCam"

        # Run CWFS Pipeline
        pipelineYamlCwfs = os.path.join(
            testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml"
        )
        pipelineYamlScience = os.path.join(
            testPipelineConfigDir, "testCalcZernikesScienceSensorSetupPipeline.yaml"
        )

        print("Running CWFS pipeline...")
        pipeCmdCwfs = writePipetaskCmd(
            config.testInfo["repoDir"],
            config.testInfo["runNameCwfs"],
            instrument,
            collections,
            pipelineYaml=pipelineYamlCwfs,
        )
        pipeCmdCwfs += ' -d "detector IN (191, 192)"'
        runProgram(pipeCmdCwfs)

        print("Running Science pipeline...")
        pipeCmdScience = writePipetaskCmd(
            config.testInfo["repoDir"],
            config.testInfo["runNameScience"],
            instrument,
            collections,
            pipelineYaml=pipelineYamlScience,
        )
        pipeCmdScience += ' -d "exposure IN (4021123106001, 4021123106002) AND '
        pipeCmdScience += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
        runProgram(pipeCmdScience)


def pytest_unconfigure(config: Config) -> None:
    if config.getoption("--run-pretest"):
        print("Running cleanup...")
        for runName in [
            config.testInfo["runNameCwfs"],
            config.testInfo["runNameScience"],
        ]:
            cleanUpCmd = writeCleanUpRepoCmd(config.testInfo["repoDir"], runName)
            runProgram(cleanUpCmd)
