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

__all__ = ["runSelection", "donutCatalogToDataFrame"]

import pandas as pd


def runSelection(refObjLoader, detector, wcs, filterName, donutSelectorTask):
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

    Returns
    -------
    referenceCatalog : `lsst.afw.table.SimpleCatalog`
        Catalog containing reference objects inside the specified bounding
        box and with properties within the bounds set by the
        `referenceSelector`.
    """

    bbox = detector.getBBox()
    donutCatalog = refObjLoader.loadPixelBox(bbox, wcs, filterName).refCat

    if donutSelectorTask is None:
        return donutCatalog, [[]] * len(donutCatalog), [[]] * len(donutCatalog)
    else:
        donutSelection = donutSelectorTask.run(donutCatalog, detector, filterName)
        return (
            donutCatalog[donutSelection.selected],
            donutSelection.blendCentersX,
            donutSelection.blendCentersY,
        )


def donutCatalogToDataFrame(
    donutCatalog=None, filterName=None, blendCentersX=None, blendCentersY=None
):
    """
    Reformat afwCatalog into a pandas dataframe sorted by flux with
    the brightest objects at the top.

    Parameters
    ----------
    donutCatalog : `lsst.afw.table.SimpleCatalog` or `None`, optional
        lsst.afw.table.SimpleCatalog object returned by the
        ReferenceObjectLoader search over the detector footprint.
        If None then it will return an empty dataframe.
        (the default is None.)
    filterName : `str` or `None`, optional
        Name of camera filter. If donutCatalog is not None then
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

    Returns
    -------
    `pandas.DataFrame`
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

    ra = []
    dec = []
    centroidX = []
    centroidY = []
    sourceFlux = []
    blendCX = []
    blendCY = []

    if donutCatalog is not None:
        filterErrMsg = "If donutCatalog is not None then filterName cannot be None."
        if filterName is None:
            raise ValueError(filterErrMsg)
        ra = donutCatalog["coord_ra"]
        dec = donutCatalog["coord_dec"]
        centroidX = donutCatalog["centroid_x"]
        centroidY = donutCatalog["centroid_y"]
        sourceFlux = donutCatalog[f"{filterName}_flux"]

        if (blendCentersX is None) and (blendCentersY is None):
            blendCX = list()
            blendCY = list()
            for idx in range(len(donutCatalog)):
                blendCX.append(list())
                blendCY.append(list())
        elif isinstance(blendCentersX, list) and isinstance(blendCentersY, list):
            lengthErrMsg = (
                "blendCentersX and blendCentersY need "
                + "to be same length as donutCatalog."
            )
            if (len(blendCentersX) != len(donutCatalog)) or (
                len(blendCentersY) != len(donutCatalog)
            ):
                raise ValueError(lengthErrMsg)
            xyMismatchErrMsg = (
                "Each list in blendCentersX must have the same "
                + "length as the list in blendCentersY at the "
                + "same index."
            )
            for xList, yList in zip(blendCentersX, blendCentersY):
                if len(xList) != len(yList):
                    raise ValueError(xyMismatchErrMsg)
            blendCX = blendCentersX
            blendCY = blendCentersY
        else:
            blendErrMsg = (
                "blendCentersX and blendCentersY must be"
                + " both be None or both be a list."
            )
            raise ValueError(blendErrMsg)

    fieldObjects = pd.DataFrame([])
    fieldObjects["coord_ra"] = ra
    fieldObjects["coord_dec"] = dec
    fieldObjects["centroid_x"] = centroidX
    fieldObjects["centroid_y"] = centroidY
    fieldObjects["source_flux"] = sourceFlux
    fieldObjects["blend_centroid_x"] = blendCX
    fieldObjects["blend_centroid_y"] = blendCY

    fieldObjects = fieldObjects.sort_values("source_flux", ascending=False).reset_index(
        drop=True
    )

    return fieldObjects
