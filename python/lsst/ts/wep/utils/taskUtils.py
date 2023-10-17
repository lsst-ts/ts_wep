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

__all__ = [
    "runProgram",
    "writePipetaskCmd",
    "writeCleanUpRepoCmd",
    "getCameraFromButlerName",
]

import os
import subprocess

import lsst.obs.lsst as obs_lsst


def runProgram(command, binDir=None, argstring=None):
    """Run the program w/o arguments.

    Parameters
    ----------
    command : str
        Command of application.
    binDir : str, optional
        Directory of binary application. (the default is None.)
    argstring : str, optional
        Arguments of program. (the default is None.)

    Raises
    ------
    RuntimeError
        Error running of command.
    """

    # Directory of binary application
    if binDir is not None:
        command = os.path.join(binDir, command)

    # Arguments for the program
    if argstring is not None:
        command += " " + argstring

    # Call the program w/o arguments
    if subprocess.call(command, shell=True) != 0:
        raise RuntimeError("Error running: %s" % command)


def writePipetaskCmd(
    repoDir, runName, instrument, collections, taskName=None, pipelineYaml=None
):
    """
    Format a command line call to run a Gen 3 pipeline task.
    Can also be a set of tasks if specified in a pipeline yaml file.

    Parameters
    ----------
    repoDir: str
        Location of Gen 3 repository.
    runName: str
        Name of collection for data produced by the task.
    instrument: str
        The instrument to use for the task.
    collections: str
        The data collections needed for the task.
    taskName: str, optional
        Full task function name in lsst namespace. One of taskName
        or pipelineYaml must be specified to run. (The default is None).
    pipelineYaml: str, optional
        Yaml file that specifies a pipeline configuration to run
        instead of a single task. (The default is None.)

    Returns
    -------
    str
        Pipetask run command.

    Raises
    ------
    ValueError
        Need to at least specify name of task or name of pipeline file.
    """
    if (taskName is None) and (pipelineYaml is None):
        raise ValueError("At least one of taskName or pipelineYaml must not be None")

    pipetaskCmd = "pipetask run "
    pipetaskCmd += f"-b {repoDir} "  # Specify repo
    pipetaskCmd += f"-i {collections} "  # Specify collections with data to use
    pipetaskCmd += f"--instrument {instrument} "
    pipetaskCmd += f"--register-dataset-types --output-run {runName}"
    if taskName is not None:
        pipetaskCmd += f" -t {taskName}"
    if pipelineYaml is not None:
        pipetaskCmd += f" -p {pipelineYaml}"

    return pipetaskCmd


def writeCleanUpRepoCmd(repoDir, runName):
    """
    Format a command line call to clean up the data created by a pipeline.

    Parameters
    ----------
    repoDir: str
        Location of Gen 3 repository.
    runName: str
        Name of collection for data produced by the task.

    Returns
    -------
    str
        Butler prune-collection command.
    """

    cleanUpCmd = "butler remove-runs "
    cleanUpCmd += f"{repoDir} {runName} --no-confirm"

    return cleanUpCmd


def getCameraFromButlerName(camName):
    """
    Get the proper camera object for the donuts.

    Parameters
    ----------
    camName : str
        Name of instrument using butler convention. Available instruments
        are LSSTCam, LSSTComCam, and LATISS.

    Returns
    -------
    `lsst.afw.cameraGeom.Camera`
        Camera object for the exposures.

    Raises
    ------
    `ValueError`
        The camera is not supported.
    """

    if camName == "LSSTCam":
        return obs_lsst.LsstCam().getCamera()
    elif camName == "LSSTComCam":
        return obs_lsst.LsstComCam().getCamera()
    elif camName == "LATISS":
        return obs_lsst.Latiss.getCamera()
    else:
        raise ValueError(f"Camera {camName} is not supported.")
