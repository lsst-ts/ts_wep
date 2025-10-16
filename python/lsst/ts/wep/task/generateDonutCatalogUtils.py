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
    "runSelection",
    "donutCatalogToAstropy",
    "addVisitInfoToCatTable",
]

import astropy.units as u
import numpy as np
from astropy.table import QTable
from lsst.afw.cameraGeom import Detector
from lsst.afw.geom import SkyWcs
from lsst.afw.image import Exposure
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.ts.wep.task import DonutSourceSelectorTask
from lsst.ts.wep.donutSizeCorrelator import DonutSizeCorrelator


def runSelection(
    refObjLoader: ReferenceObjectLoader,
    detector: Detector,
    wcs: SkyWcs,
    filterName: str,
    donutSelectorTask: DonutSourceSelectorTask,
    edgeMargin: int,
) -> tuple[QTable, list, list]:
    """
    Match the detector area to the reference catalog
    and then run the LSST DM reference selection task.
    For configuration parameters on the reference selector
    see `lsst.meas.algorithms.ReferenceSourceSelectorConfig`.

    Parameters
    ----------
    refObjLoader : `meas.algorithms.ReferenceObjectLoader`
        Reference object loader to use in getting reference objects.
    detector : `lsst.afw.cameraGeom.Detector`
        Detector object from the camera.
    wcs : `lsst.afw.geom.SkyWcs`
        Wcs object defining the pixel to sky (and inverse) transform for
        the supplied ``bbox``.
    filterName : `str`
        Name of camera filter.
    donutSelectorTask : `lsst.ts.wep.task.DonutSourceSelectorTask` or None
        Task to run the donut source selection algorithm. If set to None,
        the catalog will be the exact same as the reference catalogs without
        any donut selection algorithm applied.
    edgeMargin: `int`
        The width of the margin by which we decrease bbox to avoid edge
        source selection.

    Returns
    -------
    referenceCatalog : `lsst.afw.table.SimpleCatalog`
        Catalog containing reference objects inside the specified bounding
        box and with properties within the bounds set by the
        `referenceSelector`.
    list
        X pixel location of centroids of blended donuts.
    list
        Y pixel location of centroids of blended donuts.
    """

    bbox = detector.getBBox()
    trimmedBBox = bbox.erodedBy(edgeMargin)
    donutCatalog = refObjLoader.loadPixelBox(trimmedBBox, wcs, filterName).refCat

    if donutSelectorTask is None:
        return donutCatalog, [[]] * len(donutCatalog), [[]] * len(donutCatalog)
    else:
        donutSelection = donutSelectorTask.run(donutCatalog, detector, filterName)
        return (
            donutCatalog[donutSelection.selected],
            donutSelection.blendCentersX,
            donutSelection.blendCentersY,
        )


def donutCatalogToAstropy(
    donutCatalog: QTable,
    filterName: str | list[str],
    blendCentersX: list | None = None,
    blendCentersY: list | None = None,
    sortFilterIdx: int = 0,
) -> QTable:
    """
    Reformat afwCatalog into an astropy QTable sorted by flux with
    the brightest objects at the top.

    Parameters
    ----------
    donutCatalog : `lsst.afw.table.SimpleCatalog` or `None`
        lsst.afw.table.SimpleCatalog object returned by the
        ReferenceObjectLoader search over the detector footprint.
        If None then it will return an empty QTable.
        (the default is None.)
    filterName : `str` or `list` of `str`
        Name of catalog flux filter(s). If donutCatalog is not None then
        this cannot be None. (the default is None.)
    blendCentersX : `list` or `None`, optional
        X pixel position of centroids for blended objects. List
        should be the same length as the donutCatalog. If
        blendCentersY is not None then this cannot be None. (the default
        is None.)
    blendCentersY : `list` or `None`, optional
        Y pixel position of centroids for blended objects. List
        should be the same length as the donutCatalog. If
        blendCentersX is not None then this cannot be None. (the default
        is None.)
    sortFilterIdx : int, optional
        Index for which filter in filterName to sort the entire catalog
        by brightness. (the default is 0.)

    Returns
    -------
    `astropy.table.QTable`
        Complete catalog of reference sources in the pointing.

    Raises
    ------
    `ValueError`
        Raised if filterName is None when donutCatalog is not None.
    `ValueError`
        Raised if blendCentersX and blendCentersY are not the same length.
    `ValueError`
        Raised if blendCentersX and blendCentersY are not both
        a list or are not both None.
    """

    ra = list()
    dec = list()
    centroidX = list()
    centroidY = list()

    # If just given a single filter, change to list for compatibility.
    # This is to ensure backwards compatibility with older versions.
    # If we want to break backwards compatibility for other things we
    # could eventually change this.
    if isinstance(filterName, str):
        filterName = [filterName]

    if donutCatalog is not None:
        ra = donutCatalog["coord_ra"]
        dec = donutCatalog["coord_dec"]
        centroidX = donutCatalog["centroid_x"]
        centroidY = donutCatalog["centroid_y"]
        sourceFlux = [donutCatalog[f"{fName}_flux"] for fName in filterName]

        if (blendCentersX is None) and (blendCentersY is None):
            blendCX: list = list()
            blendCY: list = list()
            for _ in range(len(donutCatalog)):
                blendCX.append(list())
                blendCY.append(list())
        elif isinstance(blendCentersX, list) and isinstance(blendCentersY, list):
            lengthErrMsg = "blendCentersX and blendCentersY need " + "to be same length as donutCatalog."
            if (len(blendCentersX) != len(donutCatalog)) or (len(blendCentersY) != len(donutCatalog)):
                raise ValueError(lengthErrMsg)
            xyMismatchErrMsg = (
                "Each list in blendCentersX must have the same "
                + "length as the list in blendCentersY at the "
                + "same index."
            )
            for listX, listY in zip(blendCentersX, blendCentersY):
                if len(listX) != len(listY):
                    raise ValueError(xyMismatchErrMsg)
            blendCX = blendCentersX
            blendCY = blendCentersY
        else:
            blendErrMsg = "blendCentersX and blendCentersY must be" + " both be None or both be a list."
            raise ValueError(blendErrMsg)

    fieldObjects = QTable()
    fieldObjects["coord_ra"] = ra * u.rad
    fieldObjects["coord_dec"] = dec * u.rad
    fieldObjects["centroid_x"] = centroidX
    fieldObjects["centroid_y"] = centroidY

    if len(fieldObjects) > 0:
        flux_sort = np.argsort(sourceFlux[sortFilterIdx])[::-1]
        for idx in range(len(filterName)):
            fieldObjects[f"{filterName[idx]}_flux"] = sourceFlux[idx] * u.nJy
        fieldObjects = fieldObjects[flux_sort]
        fieldObjects.meta["blend_centroid_x"] = [blendCX[idx] for idx in flux_sort]
        fieldObjects.meta["blend_centroid_y"] = [blendCY[idx] for idx in flux_sort]
    else:
        for idx in range(len(filterName)):
            fieldObjects[f"{filterName[idx]}_flux"] = list() * u.nJy
        fieldObjects.meta["blend_centroid_x"] = list()
        fieldObjects.meta["blend_centroid_y"] = list()

    return fieldObjects


def addVisitInfoToCatTable(exposure: Exposure, donutCat: QTable) -> QTable:
    """
    Add visit info from the exposure object to the catalog QTable metadata.
    This should include all information we will need downstream in the
    WEP / donut_viz tasks that would otherwise require loading VisitInfo from
    the butler.

    Parameters
    ----------
    exposure : lsst.afw.image.Exposure
        Image with donut sources that go in to the accompanying catalog.
    donutCat : astropy.table.QTable
        Donut catalog for given exposure.

    Returns
    -------
    `astropy.table.QTable`
        Catalog with relevant exposure metadata added to catalog metadata.
    """

    visitInfo = exposure.visitInfo

    catVisitInfo = dict()

    visitRaDec = visitInfo.boresightRaDec
    catVisitInfo["boresight_ra"] = visitRaDec.getRa().asDegrees() * u.deg
    catVisitInfo["boresight_dec"] = visitRaDec.getDec().asDegrees() * u.deg

    visitAzAlt = visitInfo.boresightAzAlt
    catVisitInfo["boresight_alt"] = visitAzAlt.getLatitude().asDegrees() * u.deg
    catVisitInfo["boresight_az"] = visitAzAlt.getLongitude().asDegrees() * u.deg

    catVisitInfo["boresight_rot_angle"] = visitInfo.boresightRotAngle.asDegrees() * u.deg
    catVisitInfo["rot_type_name"] = visitInfo.rotType.name
    catVisitInfo["rot_type_value"] = visitInfo.rotType.value

    catVisitInfo["boresight_par_angle"] = visitInfo.boresightParAngle.asDegrees() * u.deg

    catVisitInfo["focus_z"] = visitInfo.focusZ * u.mm
    catVisitInfo["mjd"] = visitInfo.date.toAstropy().tai.mjd
    catVisitInfo["visit_id"] = visitInfo.id
    catVisitInfo["instrument_label"] = visitInfo.instrumentLabel

    catVisitInfo["observatory_elevation"] = visitInfo.observatory.getElevation() * u.m
    catVisitInfo["observatory_latitude"] = visitInfo.observatory.getLatitude().asDegrees() * u.deg
    catVisitInfo["observatory_longitude"] = visitInfo.observatory.getLongitude().asDegrees() * u.deg
    catVisitInfo["ERA"] = visitInfo.era.asDegrees() * u.deg
    catVisitInfo["exposure_time"] = visitInfo.exposureTime * u.s

    # Estimate the donut diameter
    correlator = DonutSizeCorrelator()
    img = correlator.prepButlerExposure(exposure)
    diameter, *_ = correlator.getDonutDiameter(img)
    catVisitInfo["donut_radius"] = 0.5 * diameter

    donutCat.meta["visit_info"] = catVisitInfo

    return donutCat
