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
    "getOffsetFromExposure",
    "getTaskInstrument",
    "createTemplateForDetector",
    "createTemplateInFocus",
    "convertHistoryToMetadata",
    "convertMetadataToHistory",
]

import os
import shlex
import subprocess
from contextlib import ExitStack
from typing import Union

import lsst.obs.lsst as obs_lsst
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.cameraGeom import FIELD_ANGLE, Detector, DetectorType
from lsst.afw.image import Exposure
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.image import Image
from lsst.ts.wep.imageMapper import ImageMapper
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType


def runProgram(command, binDir=None, argstring=None, stdout=None, stderr=None):
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


def getOffsetFromExposure(
    exposure: Exposure,
    camName: str,
    defocalType: Union[DefocalType, str],
) -> float:
    """Get the offset from the exposure.

    For LSSTCam and ComCam this corresponds to the offset of the detector
    where the donut was imaged, while for AuxTel it corresponds to the offset
    of M2.

    Parameters
    ----------
    exposure : lsst.afw.image.Exposure
        The exposure
    camName : str
        Name of instrument using butler convention. Available instruments
        are LSSTCam, LSSTComCam, LSSTComCamSim, and LATISS.
    defocalType : DefocalType, str, or None
        The DefocalType enum, or corresponding string.

    Returns
    -------
    float
        The offset in mm

    Raises
    ------
    ValueError
        If the detector/camera combo is not supported
    """
    # Get the focus position from the exposure
    focusZ = exposure.visitInfo.focusZ  # mm

    # Get the detector type
    detectorType = exposure.detector.getType().name

    # If this isn't a wavefront sensor, we're done
    if detectorType != "WAVEFRONT":
        return focusZ

    # Wavefront sensors only supported for LsstCam
    if camName != "LSSTCam":
        raise ValueError(f"Wavefront sensors for camera {camName} are not supported.")

    # Cast defocalType to a DefocalType enum
    defocalType = DefocalType(defocalType)

    if defocalType == DefocalType.Extra:
        offset = focusZ + 1.5
    elif defocalType == DefocalType.Intra:
        offset = focusZ - 1.5
    else:
        raise ValueError(f"defocalType {defocalType} not supported.")

    return offset


def getTaskInstrument(
    camName: str,
    detectorName: str,
    offset: Union[float, None] = None,
    instConfigFile: Union[str, None] = None,
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
    offset : float or None, optional
        The true offset for the exposure in mm. For LSSTCam this corresponds
        to the offset of the detector, while for AuxTel it corresponds to the
        offset of M2. (the default is None)
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
        overrideOffset = True
    else:
        instrument = Instrument(configFile=instConfigFile)
        try:
            instrument.defocalOffset
        except ValueError:
            overrideOffset = True
        else:
            overrideOffset = False

    if offset is None or not overrideOffset:
        # We're done!
        return instrument

    # Override the defocalOffset
    if instrument.batoidOffsetOptic is None:
        instrument.defocalOffset = offset / 1e3
    else:
        instrument.batoidOffsetValue = offset / 1e3

    return instrument


def gkern(length=3, sig=0.5):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


def createTemplateInFocus(length: float = 11, sigma: float = 2.5):
    """Create an in-focus Gaussian PSF

    Parameters
    ----------
    length: float, optional
        Length of the correlation template.
    sigma: float, optional
        Sigma (spread) of the Gaussian PSF.

    Returns
    -------
    np.ndarray
        The donut template array.
    """
    psf_array = gkern(length=length, sig=sigma)
    psf_array = psf_array.astype(np.float64)
    return psf_array


def createTemplateForDetector(
    detector: Detector,
    defocalType: DefocalType,
    bandLabel: Union[BandLabel, str] = BandLabel.REF,
    instrument: Instrument = Instrument(),
    opticalModel: str = "offAxis",
    padding: int = 5,
    isBinary: bool = True,
    ccs: bool = False,
    nPixels: int = None,
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
            str(key): convertHistoryToMetadata(val) for key, val in history.items()
        }
        history = pipeBase.TaskMetadata.from_dict(history)
    elif isinstance(history, np.ndarray):
        history = {
            "shape": history.shape,
            "dtype": history.dtype.name,
            "values": history.flatten().tolist(),
        }

    return history


def convertMetadataToHistory(metadata: pipeBase.TaskMetadata) -> dict:
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
            metadata = np.array(
                metadata["values"],
                dtype=metadata["dtype"],
            ).reshape([int(val) for val in metadata["shape"]])
        # Otherwise, recurse on keys and values
        else:
            metadata = {
                convertMetadataToHistory(key): convertMetadataToHistory(val)
                for key, val in metadata.items()
            }

    # Convert numeric strings back to floats and ints
    elif isinstance(metadata, str):
        if "." in metadata:
            try:
                metadata = float(metadata)
            except:  # noqa: E722
                pass
        else:
            try:
                metadata = int(metadata)
            except:  # noqa: E722
                pass

    return metadata
