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
    "getTaskInstrument",
    "createTemplateForDetector",
    "convertHistoryToMetadata",
    "convertMetadataToHistory",
    "convertDictToVisitInfo",
]

import os
import shlex
import subprocess
from contextlib import ExitStack
from typing import TextIO

import astropy.units as u
import lsst.obs.lsst as obs_lsst
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.cameraGeom import FIELD_ANGLE, Camera, Detector, DetectorType
from lsst.afw.coord import Observatory
from lsst.afw.image import RotType, VisitInfo
from lsst.daf.base import DateTime
from lsst.geom import SpherePoint, degrees, radians
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.image import Image
from lsst.ts.wep.imageMapper import ImageMapper
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType


def runProgram(
    command: str,
    binDir: str | None = None,
    argstring: str | None = None,
    stdout: TextIO | str | None = None,
    stderr: TextIO | str | None = None,
) -> None:
    """Run the program w/o arguments.

    Parameters
    ----------
    command : str
        Command of application.
    binDir : str, optional
        Directory of binary application. (the default is None.)
    argstring : str, optional
        Arguments of program. (the default is None.)
    stdout, stderr : str or _io.TextIOWrapper, optional
        Buffered text output/error streams or filenames

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

    with ExitStack() as stack:
        if isinstance(stdout, str):
            stdout = stack.enter_context(open(stdout, "w"))
            # Don't open competing filehandles on the same file
            if isinstance(stderr, str) and (stderr == stdout.name):
                stderr = stdout
        if isinstance(stderr, str):
            stderr = stack.enter_context(open(stderr, "w"))

        # Call the program w/o arguments
        if (
            subprocess.run(
                shlex.split(command),
                shell=False,
                stdout=stdout,
                stderr=stderr,
            ).returncode
            != 0
        ):
            raise RuntimeError("Error running: %s" % command)


def writePipetaskCmd(
    repoDir: str,
    runName: str,
    instrument: str,
    collections: str,
    taskName: str | None = None,
    pipelineYaml: str | None = None,
) -> str:
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


def writeCleanUpRepoCmd(repoDir: str, runName: str) -> str:
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


def getCameraFromButlerName(camName: str) -> Camera:
    """
    Get the proper camera object for the donuts.

    Parameters
    ----------
    camName : str
        Name of instrument using butler convention. Available instruments
        are LSSTCam, LSSTComCam, LSSTComCamSim, and LATISS.

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
    elif camName == "LSSTComCamSim":
        return obs_lsst.LsstComCamSim().getCamera()
    elif camName == "LATISS":
        return obs_lsst.Latiss.getCamera()
    else:
        raise ValueError(f"Camera {camName} is not supported.")


def getTaskInstrument(
    camName: str,
    detectorName: str,
    instConfigFile: str | None = None,
) -> Instrument:
    """Get the instrument to use for the task.

    The camera name is used to load a default instrument, and then the
    defocalOffset is override using the offset value, if provided.
    If instConfigFile is provided, that file is used instead of camName
    to load the instrument, and offset is only used if instConfigFile
    does not contain information for calculating the defocalOffset.

    Parameters
    ----------
    camName : str
        The name of the camera
    detectorName : str
        The name of the detector.
    instConfigFile : str or None
        An instrument config file to override the default instrument
        for the camName. If begins with "policy:", this path is understood
        as relative to the ts_wep policy directory.
        (the default is None)

    Returns
    -------
    Instrument
        The instrument object
    """
    # Load the starting instrument
    if instConfigFile is None:
        if camName == "LSSTCam":
            camera = LsstCam().getCamera()
            if camera[detectorName].getType() == DetectorType.WAVEFRONT:
                instrument = Instrument(configFile="policy:instruments/LsstCam.yaml")
            else:
                instrument = Instrument(configFile="policy:instruments/LsstFamCam.yaml")
        elif camName in ["LSSTComCam", "LSSTComCamSim"]:
            instrument = Instrument(configFile="policy:instruments/ComCam.yaml")
        elif camName == "LATISS":
            instrument = Instrument(configFile="policy:instruments/AuxTel.yaml")
        else:
            raise ValueError(f"No default instrument for camera {camName}")
    else:
        instrument = Instrument(configFile=instConfigFile)

    return instrument


def createTemplateForDetector(
    detector: Detector,
    defocalType: DefocalType,
    bandLabel: BandLabel | str = BandLabel.REF,
    instrument: Instrument | None = None,
    opticalModel: str = "offAxis",
    padding: int = 5,
    isBinary: bool = True,
    ccs: bool = False,
    nPixels: int | None = None,
) -> np.ndarray:
    """Create a donut template for the given detector.

    Parameters
    ----------
    detector : lsst.afw.cameraGeom.Detector
        The detector object for which the template is created
    defocalType : DefocalType or str
        The DefocalType enum (or corresponding string) specifying whether to
        create an intra- or extra-focal template.
    bandLabel : BandLabel or str, optional
        The BandLabel enum (or corresponding string) specifying the band for
        which the template is created. Note it is not expected that this
        will matter much for template creation. (the default is BandLabel.REF)
    instrument : Instrument, optional
        The ts.wep Instrument object. (the default is the default Instrument)
    opticalModel : str, optional
        The optical model for the ImageMapper. (the default is "offAxis")
    padding : int, optional
        Padding, in pixels, on each side of the template. (the default is 5)
    isBinary : bool, optional
        Whether to return a binary template. (the default is True)
    ccs : bool, optional
        Whether to return a template that is in the camera coordinate system
        (CCS). For templates on the CWFSs, this also includes de-rotation so
        they are in the same orientation as the science sensors. When False,
        the data is in the DVCS, and the CWFSs are in original orientation.
        This matches the data from the butler.
        (the default is False)
    nPixels : int, optional
        The number of pixels on a side of the template. Typically this number
        is calculated dynamically, but if supplied here, this number is used
        as an override. Note the padding is not applied if this is provided.

    Returns
    -------
    np.ndarray
        The donut template array.
    """
    if instrument is None:
        instrument = Instrument()

    # Create the Image mapper
    imageMapper = ImageMapper(instConfig=instrument, opticalModel=opticalModel)

    # Get the field angle for the center of the detector
    # Notice we swap the x and y coords here
    # so that the angle is in the camera coordinate system (CCS)
    detYDeg, detXDeg = np.rad2deg(detector.getCenter(FIELD_ANGLE))

    # Determine the template size
    if nPixels is None:
        nPixels = imageMapper.getProjectionSize((detXDeg, detYDeg), defocalType)
        nPixels += 2 * padding

    # Create a dummy Image for template creation
    dummyImage = Image(
        image=np.zeros((nPixels, nPixels)),
        fieldAngle=(detXDeg, detYDeg),
        defocalType=defocalType,
        bandLabel=bandLabel,
    )

    # Create the Donut template
    if isBinary:
        imageMapper.createImageMasks(dummyImage, isBinary=True)
        template = dummyImage.mask.astype(int)
    else:
        template = imageMapper.mapPupilToImage(dummyImage)

    # The template is created in the CCS, with CWFSs templates de-rotated to
    # account for the rotation of the CWFSs with respect to the science sensor

    # if not ccs
    # we want to transform back to DVCS (i.e. transpose)
    # and then rotate back to original orientation of CWFS
    # this will match the orientation of data from the butler
    if not ccs:
        # Transpose: CCS -> DVCS
        template = template.T

        # Rotate to original orientation of CWFS
        eulerZ = detector.getOrientation().getYaw().asDegrees()
        nRot = int(eulerZ // 90)
        template = np.rot90(template, nRot)

    return template


def convertHistoryToMetadata(history: dict) -> pipeBase.TaskMetadata:
    """Convert algorithm history to be saved in task metadata.

    Parameters
    ----------
    history : dict
        The nested dictionary representing the algorithm history.

    Returns
    -------
    lsst.pipe.base.TaskMetadata
        The reformatted history in a TaskMetadata object
    """
    if isinstance(history, dict):
        history = {
            str(key): ("None" if val is None else convertHistoryToMetadata(val))
            for key, val in history.items()
        }
        history = pipeBase.TaskMetadata.from_dict(history)
    elif isinstance(history, np.ndarray):
        history = {
            "shape": history.shape,
            "dtype": history.dtype.name,
            "values": history.flatten().tolist(),
        }

    return history


def convertMetadataToHistory(metadata: pipeBase.TaskMetadata | dict | str) -> dict:
    """Convert the history from the metadata back to original format.

    Parameters
    ----------
    metadata : pipeBase.TaskMetadata
        The metadata containing the history. Note this is meant to
        be only the nested Metadata that represents the algorithm
        history and nothing else.

    Returns
    -------
    dict
        The history dictionary
    """
    # If this is a TaskMetadata object, convert to dict and recurse
    if isinstance(metadata, pipeBase.TaskMetadata):
        metadata = convertMetadataToHistory(metadata.to_dict())

    # If this is a dict...
    elif isinstance(metadata, dict):
        # If these are the keys, convert to an array
        if set(metadata.keys()) == {"shape", "dtype", "values"}:
            _metadata = np.array(
                metadata["values"],
                dtype=metadata["dtype"],
            ).reshape([int(val) for val in metadata["shape"]])
        # Otherwise, recurse on keys and values
        else:
            _metadata = {
                convertMetadataToHistory(key): convertMetadataToHistory(val) for key, val in metadata.items()
            }

    # Convert "None" strings back to None and numeric strings to floats/ints
    elif isinstance(metadata, str):
        if metadata == "None":
            _metadata = None
        elif "." in metadata:
            try:
                _metadata = float(metadata)
            except:  # noqa: E722
                pass
        else:
            try:
                _metadata = int(metadata)
            except:  # noqa: E722
                pass

    return _metadata


def convertDictToVisitInfo(
    info: dict,
) -> VisitInfo:
    """
    Convert a dictionary (as from astropy metadata added by
    addVisitInfoToCatTable) back into a VisitInfo object.

    Parameters
    ----------
    info : dict
        Dictionary of visit info.

    Returns
    -------
    `lsst.afw.image.VisitInfo`
        VisitInfo object with the same information as the input dictionary.
    """
    kwargs = {
        "id": info["visit_id"],
        "date": DateTime(info["mjd"], system=DateTime.MJD, scale=DateTime.TAI),
        "boresightRaDec": SpherePoint(
            info["boresight_ra"].to_value(u.rad) * radians,
            info["boresight_dec"].to_value(u.rad) * radians,
        ),
        "boresightAzAlt": SpherePoint(
            info["boresight_az"].to_value(u.rad) * radians,
            info["boresight_alt"].to_value(u.rad) * radians,
        ),
        "boresightRotAngle": info["boresight_rot_angle"].to_value(u.rad) * radians,
        "rotType": RotType(info["rot_type_value"]),
        "focusZ": info["focus_z"].to_value(u.mm),
        "instrumentLabel": info["instrument_label"],
        "observatory": Observatory(
            info["observatory_longitude"].to_value(u.deg) * degrees,
            info["observatory_latitude"].to_value(u.deg) * degrees,
            info["observatory_elevation"].to_value(u.m),
        ),
        "era": info["ERA"].to_value(u.rad) * radians,
        "exposureTime": info["exposure_time"].to_value(u.s),
    }
    return VisitInfo(**kwargs)
