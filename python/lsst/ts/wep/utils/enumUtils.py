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
    "FilterType",
    "CamType",
    "BscDbType",
    "DefocalType",
    "ImageType",
    "CentroidFindType",
    "DonutTemplateType",
    "DeblendDonutType",
    "getFilterTypeFromBandLabel",
    "mapFilterRefToG",
    "getCamType",
    "getCamNameFromCamType",
    "getCamTypeFromButlerName",
    "getBscDbType",
    "getImageType",
    "getCentroidFindType",
    "getDonutTemplateType",
    "getDeblendDonutType",
]

from enum import IntEnum, auto

from lsst.afw.cameraGeom import DetectorType


class FilterType(IntEnum):
    LSST_U = 1
    LSST_G = auto()
    LSST_R = auto()
    LSST_I = auto()
    LSST_Z = auto()
    LSST_Y = auto()
    REF = auto()


class CamType(IntEnum):
    LsstCam = 1
    LsstFamCam = auto()
    ComCam = auto()
    AuxTel = auto()
    AuxTelZWO = auto()


class BscDbType(IntEnum):
    LocalDb = 1
    LocalDbForStarFile = auto()


class DefocalType(IntEnum):
    Intra = 1
    Extra = auto()


class ImageType(IntEnum):
    Amp = 1
    Eimg = auto()


class CentroidFindType(IntEnum):
    RandomWalk = 1
    Otsu = auto()
    ConvolveTemplate = auto()


class DonutTemplateType(IntEnum):
    Model = 1
    Phosim = auto()


class DeblendDonutType(IntEnum):
    Adapt = 1


def getFilterTypeFromBandLabel(bandLabel):
    """Get the FilterType associated with the name of the bandpass
    accessed in an exposure using `exposure.filter.bandLabel`.

    Parameters
    ----------
    bandLabel : str
        Bandpass label of the exposure.

    Returns
    -------
    filterType : enum `FilterType`
        Filter type.
    """
    filterLabelDict = {}
    filterLabelDict["u"] = FilterType.LSST_U
    filterLabelDict["g"] = FilterType.LSST_G
    filterLabelDict["r"] = FilterType.LSST_R
    filterLabelDict["i"] = FilterType.LSST_I
    filterLabelDict["z"] = FilterType.LSST_Z
    filterLabelDict["y"] = FilterType.LSST_Y

    return filterLabelDict.get(bandLabel, FilterType.REF)


def mapFilterRefToG(filterType):
    """Map the reference filter to the G filter.

    Parameters
    ----------
    filterType : enum 'FilterType'
        Filter type.

    Returns
    -------
    enum 'FilterType'
        Mapped filter type.
    """

    if filterType == FilterType.REF:
        return FilterType.LSST_G
    else:
        return filterType


def getCamType(instName):
    """Get the camera type from instrument name.

    Parameters
    ----------
    instName : str
         Instrument name.

    Returns
    -------
    camType : enum 'CamType'
        Camera type.

    Raises
    ------
    ValueError
        Instrument name is not supported.
    """
    if instName == "lsst":
        return CamType.LsstCam
    elif instName == "lsstfam":
        return CamType.LsstFamCam
    elif instName == "comcam":
        return CamType.ComCam
    elif instName == "auxTel":
        return CamType.AuxTel
    else:
        raise ValueError(f"Instrument name ({instName}) is not supported.")


def getCamNameFromCamType(camType):
    """Get the camera name for policy files from CamType.

    Parameters
    ----------
    camType : enum 'CamType'
        Camera Type.

    Returns
    -------
    str
        Instrument Name.

    Raises
    ------
    ValueError
        Camera Type is not supported.
    """

    if camType == CamType.LsstCam:
        return "lsst"
    elif camType == CamType.LsstFamCam:
        return "lsstfam"
    elif camType == CamType.ComCam:
        return "comcam"
    elif camType == CamType.AuxTel:
        return "auxTel"
    elif camType == CamType.AuxTelZWO:
        return "auxTelZWO"
    else:
        raise ValueError(f"CamType ({camType}) is not supported.")


def getCamTypeFromButlerName(instName, detectorType):
    """Get the camera type from instrument name used by the LSST DM
    middleware for each instrument.

    Parameters
    ----------
    instName : str
        Instrument name.
    detectorType : lsst.afw.cameraGeom.DetectorType
        Type of CCD. "SCIENCE" or "WAVEFRONT".

    Returns
    -------
    camType : enum 'CamType'
        Camera type.

    Raises
    ------
    ValueError
        Combination of instrument name and detector type is not supported.
    ValueError
        Detector type is not supported.
    """
    if detectorType == DetectorType.WAVEFRONT:
        if instName == "LSSTCam":
            return CamType.LsstCam
        else:
            raise ValueError(
                f"Wavefront sensors for instrument name ({instName}) are not supported."
            )
    elif detectorType == DetectorType.SCIENCE:
        if instName == "LSSTCam":
            return CamType.LsstFamCam
        elif instName == "LSSTComCam":
            return CamType.ComCam
        elif instName == "LATISS":
            return CamType.AuxTel
        else:
            raise ValueError(
                f"Science sensors for instrument name ({instName}) are not supported."
            )
    else:
        raise ValueError(f"Detector Type ({detectorType.name}) is not supported.")


def getBscDbType(bscDbType):
    """Get the bright star catalog (BSC) database type.

    Parameters
    ----------
    bscDbType : str
        BSC database type to use (localDb or file).

    Returns
    -------
    enum 'BscDbType'
        BSC database type.

    Raises
    ------
    ValueError
        The bscDb is not supported.
    """

    if bscDbType == "localDb":
        return BscDbType.LocalDb
    elif bscDbType == "file":
        return BscDbType.LocalDbForStarFile
    else:
        raise ValueError("The bscDb (%s) is not supported." % bscDbType)


def getImageType(imageType):
    """Get the image type.

    Parameters
    ----------
    imageType : str
        Image type to use (amp or eimage).

    Returns
    -------
    enum 'ImageType'
        ImageType enum.

    Raises
    ------
    ValueError
        The image type is not supported.
    """

    if imageType == "amp":
        return ImageType.Amp
    elif imageType == "eimage":
        return ImageType.Eimg
    else:
        raise ValueError("The %s is not supported." % imageType)


def getCentroidFindType(centroidFindType):
    """Get the centroid find type.

    Parameters
    ----------
    centroidFindType : str
        Centroid find algorithm to use (randomWalk, otsu, or convolveTemplate).

    Returns
    -------
    enum 'CentroidFindType'
        Centroid find type algorithm.

    Raises
    ------
    ValueError
        The centroid find type is not supported.
    """

    if centroidFindType == "randomWalk":
        return CentroidFindType.RandomWalk
    elif centroidFindType == "otsu":
        return CentroidFindType.Otsu
    elif centroidFindType == "convolveTemplate":
        return CentroidFindType.ConvolveTemplate
    else:
        raise ValueError("The %s is not supported." % centroidFindType)


def getDonutTemplateType(donutTemplateType):
    """Get the donut template type.

    Parameters
    ----------
    donutTemplateType : str
        Donut template type to use (model or phosim).

    Returns
    -------
    enum 'DonutTemplateType'
        Donut template type algorithm.

    Raises
    ------
    ValueError
        The donut template type is not supported.
    """

    if donutTemplateType == "model":
        return DonutTemplateType.Model
    elif donutTemplateType == "phosim":
        return DonutTemplateType.Phosim
    else:
        raise ValueError(f"The {donutTemplateType} is not supported.")


def getDeblendDonutType(deblendDonutType):
    """Get the deblend donut type.

    Parameters
    ----------
    deblendDonutType : str
        Deblend donut algorithm to use (adapt).

    Returns
    -------
    enum 'DeblendDonutType'
        Deblend donut type algorithm.

    Raises
    ------
    ValueError
        The deblend donut type is not supported.
    """

    if deblendDonutType == "adapt":
        return DeblendDonutType.Adapt
    else:
        raise ValueError("The %s is not supported." % deblendDonutType)
