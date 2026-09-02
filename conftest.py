# This file is part of ts_wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os

from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.main import Session

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
        help="Force pre-test WEP pipeline run even if no pipeline tests are collected.",
    )
    parser.addoption(
        "--skip-pretest",
        action="store_true",
        default=False,
        help="Skip pre-test WEP pipeline run.",
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "pipeline: mark test as requiring the pre-run Butler pipeline",
    )
    if config.getoption("--run-pretest", default=False) and config.getoption("--skip-pretest", default=False):
        raise ValueError("--run-pretest and --skip-pretest are mutually exclusive")
    config._ran_pretest = False


def _run_pretest_pipelines(config: Config) -> None:
    """Run the CWFS and science pipelines that populate the test Butler repo.

    Populates config.testInfo with repo paths and run-collection names so that
    pytest_unconfigure knows what to clean up.
    """
    moduleDir = getModulePath()
    testDataDir = os.path.join(moduleDir, "tests", "testData")
    testPipelineConfigDir = os.path.join(testDataDir, "pipelineConfigs")
    config.testInfo = {
        "repoDir": os.path.join(testDataDir, "gen3TestRepo"),
        "runNameCwfs": "pretest_run_cwfs",
        "runNameScience": "pretest_run_science",
    }

    # Remove stale collections left by a previous interrupted run.
    butler = Butler.from_config(config.testInfo["repoDir"])
    existing = list(butler.registry.queryCollections())
    for runName in [config.testInfo["runNameCwfs"], config.testInfo["runNameScience"]]:
        if runName in existing:
            runProgram(writeCleanUpRepoCmd(config.testInfo["repoDir"], runName))

    collections = "refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic"
    instrument = "lsst.obs.lsst.LsstCam"

    print("Running CWFS pipeline...")
    pipeCmdCwfs = writePipetaskCmd(
        config.testInfo["repoDir"],
        config.testInfo["runNameCwfs"],
        instrument,
        collections,
        pipelineYaml=os.path.join(testPipelineConfigDir, "testCalcZernikesCwfsSetupPipeline.yaml"),
    )
    pipeCmdCwfs += ' -d "detector IN (191, 192)"'
    runProgram(pipeCmdCwfs)

    print("Running Science pipeline...")
    pipeCmdScience = writePipetaskCmd(
        config.testInfo["repoDir"],
        config.testInfo["runNameScience"],
        instrument,
        collections,
        pipelineYaml=os.path.join(testPipelineConfigDir, "testCalcZernikesScienceSensorSetupPipeline.yaml"),
    )
    pipeCmdScience += ' -d "exposure IN (4021123106001, 4021123106002) AND '
    pipeCmdScience += 'detector NOT IN (191, 192, 195, 196, 199, 200, 203, 204)"'
    runProgram(pipeCmdScience)


def pytest_collection_finish(session: Session) -> None:
    # Fires after collection but before any test runs — the earliest point at
    # which we know which tests were selected, so we can decide whether the
    # expensive pipeline setup is actually needed.
    config = session.config
    if config.option.collectonly or config.getoption("--skip-pretest"):
        return

    # Auto-run if any collected test carries the pipeline marker; --run-pretest
    # forces a run regardless (useful for re-populating a stale repo).
    has_pipeline = any(item.get_closest_marker("pipeline") for item in session.items)
    if not has_pipeline and not config.getoption("--run-pretest"):
        return

    _run_pretest_pipelines(config)
    config._ran_pretest = True


def pytest_unconfigure(config: Config) -> None:
    if not getattr(config, "_ran_pretest", False):
        return
    print("Running cleanup...")
    for runName in [config.testInfo["runNameCwfs"], config.testInfo["runNameScience"]]:
        cleanUpCmd = writeCleanUpRepoCmd(config.testInfo["repoDir"], runName)
        runProgram(cleanUpCmd)
