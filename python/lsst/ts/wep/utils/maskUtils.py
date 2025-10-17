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
    "fitMaskModel",
    "pruneMaskModel",
    "printMaskModel",
]

from copy import deepcopy
from typing import Optional, Tuple

import batoid
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _circleResid(params: tuple, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual function for fitting a circle.

    Note this is for a circle centered on the x-axis.

    Parameters
    ----------
    params : tuple
        Parameters being fit. params[0] is the center; params[1] the radius.
    x : np.ndarray
        x coordinates of the points
    y : np.ndarray
        y coordinates of the points

    Returns
    -------
    np.ndarray
        Array of finite residuals
    """
    xc, rc = params
    resid = (x - xc) ** 2 + y**2 - rc**2
    return resid[np.isfinite(resid)]


def _fitCircle(
    xPupil: np.ndarray,
    yPupil: np.ndarray,
    xItem: np.ndarray,
    yItem: np.ndarray,
    rPupilOut: float,
    rPupilIn: float,
    rEdge: float,
) -> tuple[float, float]:
    """Fit a circle to the item edge on the pupil.

    This function works by taking pairs of points on the pupil and
    item (e.g. the surface of one of the mirrors), and using these
    pairs to interpolate the edge of the item back to the pupil.
    It then fits a circle to the edge on the pupil. It returns
    NaNs if the edge does not cross the pupil.

    Parameters
    ----------
    xPupil : np.ndarray
        x coordinates (in meters) of points covering the pupil
    yPupil : np.ndarray
        y coordinates (in meters) of points covering the pupil
    xItem : np.ndarray
        x coordinates (in meters) of where the pupil rays intersect
        the optical item.
    yItem : np.ndarray
        y coordinates (in meters) of where the pupil rays intersect
        the optical item.
    rPupilOut : float
        The outer radius of the pupil in meters
    rPupilIn : float
        The inner radius of the pupil in meters
    rEdge : float
        The radius of the edge of the optical item in meters.

    Returns
    -------
    float
        The center of the fit circle in meters
    float
        The radius of the fit circle in meters
    """
    # Get the center of the pupil projection
    xCenterItem = (xItem.max() + xItem.min()) / 2

    # Re-center coordinates
    xItem = xItem - xCenterItem

    # Get the semi-major axes of the pupil projection
    xRadiusItem = (xItem.max() - xItem.min()) / 2
    yRadiusItem = (yItem.max() - yItem.min()) / 2

    # Create azimuth grid with same point spacing as item points
    # (only doing top half, and will reflect for bottom half)
    dAz = np.diff(yItem[:2]) / xRadiusItem
    azimuth = np.arange(0, np.pi, dAz)

    # Calculate points on edge of item
    xEdge = rEdge * np.cos(azimuth) - xCenterItem
    yEdge = rEdge * np.sin(azimuth)

    # Remove points outside the pupil projection
    idx = np.where((xEdge / xRadiusItem) ** 2 + (yEdge / yRadiusItem) ** 2 <= 1)
    xEdge = xEdge[idx]
    yEdge = yEdge[idx]

    # If there aren't enough points, return NaNs
    if xEdge.size < 5:
        return np.nan, np.nan

    # Select projected pupil points nearest to edge
    pupilPts = np.array([xPupil, yPupil]).T
    itemPts = np.array([xItem, yItem]).T
    edgePts = np.array([xEdge, yEdge]).T
    nbrs = NearestNeighbors(n_neighbors=3).fit(itemPts)
    nIdx = np.unique(nbrs.kneighbors(edgePts, return_distance=False))
    pupilPts = pupilPts[nIdx]
    itemPts = itemPts[nIdx]

    # Interpolate back to the pupil
    xEdgePupil = griddata(itemPts, pupilPts[:, 0], edgePts)
    yEdgePupil = griddata(itemPts, pupilPts[:, 1], edgePts)

    # Determine how many points are near the pupil
    rEdgePupil = np.sqrt(xEdgePupil**2 + yEdgePupil**2)
    nInside = np.sum((rEdgePupil >= 0.95 * rPupilIn) & (rEdgePupil <= 1.05 * rPupilOut))

    # If there are no points inside the pupil, just return NaN
    if nInside == 0:
        return np.nan, np.nan

    # Reflect to get bottom half of edge
    xEdgePupil = np.concatenate((xEdgePupil[::-1], xEdgePupil))
    yEdgePupil = np.concatenate((yEdgePupil[::-1], -yEdgePupil))

    # Now fit the center and radius of the circle
    # that matches the item edge on the pupil
    # Fit the circle
    result = least_squares(
        _circleResid,
        (-1, 5),
        args=(xEdgePupil, yEdgePupil),
        bounds=(
            (-np.inf, 0),
            (0, np.inf),
        ),
    )
    if result.success:
        return result.x
    else:
        return np.nan, np.nan


def _fitEdges(thx: float, optic: batoid.CompoundOptic, wavelength: float) -> dict:
    """Fit circles to edges of optical items on pupil for all items in optic.

    This function is for a specific field angle. It operates by covering
    the pupil in rays, tracing them through the whole optical system,
    and then for each optical subcomponent, calling _fitCircle.

    Parameters
    ----------
    thx : float
        Field angle in degrees
    optic : batoid.CompoundOptic
        The Batoid optic for which the edges of optical items are fit
    wavelength : float
        The wavelength of the photons

    Returns
    -------
    dict
        Dictionary containing the fit circles. There is an entry for each
        subcomponent in the optic (except for the detector). Each of these
        subcomponents maps to a dictionary that has a "outer" key, and, if
        the component has an inner radius, an "inner" key as well. Each of
        these is also a dictionary, containing a flag whether the component
        is clear, plus the fitted center and radius.

        For example, the dictionary might look like this:

            M1:
                outer:
                    clear: True
                    center: 0
                    radius: 4.18
                inner:
                    clear: False
                    center: 0
                    radius: 2.56
            Filter_entrance:
                outer:
                    clear: True
                    center: -18
                    radius: 19
            ...

    """
    # Get the pupil radii
    rPupilOut = optic.pupilSize / 2
    rPupilIn = optic.pupilObscuration * rPupilOut

    # Create rays on the pupil
    rays = batoid.RayVector.asPolar(
        optic,
        theta_x=np.deg2rad(thx),
        theta_y=0,
        wavelength=wavelength,
        inner=0,
        outer=1.1 * rPupilOut,
        nrad=100,
        naz=int(2 * np.pi * 100),
    )

    # Perform a full trace through the optic
    trace = optic.traceFull(rays.copy())

    # Propagate the rays to the pupil
    rays = rays.toCoordSys(optic.stopSurface.coordSys)
    optic.stopSurface.surface.intersect(rays)
    xPupil, yPupil = rays.x, rays.y

    # Create a dictionary to hold all the fits
    fits: dict = dict()

    # Loop over each item
    for name in trace:
        # Get the item optic
        itemName = optic._names[name]
        item = optic.itemDict[itemName]

        # Skip the detector
        if isinstance(item, batoid.Detector):
            continue

        # Get photon positions projected on this item
        xItem = trace[name]["out"].x
        yItem = trace[name]["out"].y

        # Get the obscuration
        if isinstance(item.obscuration, batoid.ObscNegation):
            obsc = item.obscuration.original
            outerClear, innerClear = True, False
        else:
            obsc = item.obscuration
            outerClear, innerClear = False, True

        # Get radii of obscuration
        rOuter = getattr(obsc, "radius", None)
        if rOuter is None:
            rOuter = getattr(obsc, "outer", np.nan)
        rInner = getattr(obsc, "inner", np.nan)

        # Fit the outer radius
        center, radius = _fitCircle(xPupil, yPupil, xItem, yItem, rPupilOut, rPupilIn, rOuter)
        fits[name] = dict()
        fits[name]["outer"] = dict(clear=outerClear, center=center, radius=radius)
        if np.isfinite(rInner):
            center, radius = _fitCircle(xPupil, yPupil, xItem, yItem, rPupilOut, rPupilIn, rInner)
            fits[name]["inner"] = dict(clear=innerClear, center=center, radius=radius)

    return fits


def fitMaskModel(
    optic: batoid.CompoundOptic,
    wavelength: float = 500e-9,
    deg: int = 3,
    thetaMax: float = 2,
    dTheta: float = 1e-2,
) -> Tuple[dict, dict]:
    """Fit the mask model for the telescope.

    This function is meant to produce mask models for Instruments.
    Use the printMaskModel to print the yaml-formatted mask model.
    Note this function assumes azimuthal symmetry.

    This function produces a non-minimal model. I.e. it might include
    elements that never actually vignette the pupil, and the thetaMin-
    thetaMax range might be larger than required. It is recommended
    you run pruneMaskModel on the output fitDict to get the minimal model.

    Parameters
    ----------
    optic : batoid.CompoundOptic
        The Batoid model for which to fit the mask model
    wavelength : float
        The wavelength of the photons in meters.
    deg : int, optional
        The degree of the polynomial to fit (the default is 3)
    thetaMax : float, optional
        The maximum field angle to fit (the default is 2)
    dTheta : float, optional
        The increment in field angle to use for fitting.
        (the default is 1e-2 degrees)

    Returns
    -------
    dict
        The mask model. Each item in the compound optic has an entry,
        except for the detector. Each item entry is also a dictionary
        that contains an "outer" key. Annular obscurations also contain
        an "inner" key. Each of these is a dictionary containing the keys:
        - clear: whether the item is clear or opaque
        - thetaMin: minimum field angle in degrees where the item vignettes
        - thetaMax: maximum field angle in degrees where the item vignettes
        - center: polynomial coefficients in meters fit to the mask center
        - radius: polynomial coefficients in meters fit to the mask radius
    dict
        A similarly nested dictionary containing the output of
        lsst.ts.wep.utils.maskUtils.fitMaskPoly for each item.
        See the docstring for that function to understand the
        contents of this dictionary.
    """
    # Create a grid of field angles
    thetas = np.arange(0, thetaMax + dTheta, dTheta)

    # Estimate for the first angle to create the data dictionary
    dataDict = _fitEdges(thetas[0], optic, wavelength)

    # Turn the relevant entries into lists
    for key, val in dataDict.items():
        for edge, fit in val.items():
            fit["center"] = [fit["center"]]
            fit["radius"] = [fit["radius"]]

    # Loop over other angles, fit circles to mask elements, append to dataDict
    for thx in tqdm(thetas[1:]):
        # Append these values to our dataDict
        for key, val in _fitEdges(thx, optic, wavelength).items():
            for edge, fit in val.items():
                dataDict[key][edge]["center"].append(fit["center"])
                dataDict[key][edge]["radius"].append(fit["radius"])

    # Loop over entries in dataDict and fit polynomial models
    fitDict = deepcopy(dataDict)
    for key, val in dataDict.items():
        for edge, fit in val.items():
            # Get mask for finite values
            mask = np.isfinite(fit["center"]) & np.isfinite(fit["radius"])
            fit["theta"] = thetas[mask]
            fit["center"] = np.array(fit["center"])[mask]
            fit["radius"] = np.array(fit["radius"])[mask]

            # Fit polynomials
            if np.sum(mask) > 0:
                fitDict[key][edge]["thetaMin"] = fit["theta"].min()
                fitDict[key][edge]["thetaMax"] = thetaMax
                fitDict[key][edge]["center"] = np.polyfit(fit["theta"], fit["center"], deg)
                fitDict[key][edge]["radius"] = np.polyfit(fit["theta"], fit["radius"], deg)
            else:
                fitDict[key][edge]["thetaMin"] = np.nan
                fitDict[key][edge]["thetaMax"] = np.nan
                fitDict[key][edge]["center"] = np.full(deg + 1, np.nan)
                fitDict[key][edge]["radius"] = np.full(deg + 1, np.nan)

    return fitDict, dataDict


def pruneMaskModel(
    maskModel: dict,
    thetaMax: Optional[float] = None,
    dTheta: float = 1e-3,
) -> dict:
    """Prune the mask model so only necessary elements remain.

    Parameters
    ----------
    maskModel : dict
        A mask model dictionary created by fitMaskModel.
    thetaMax : float or None, optional
        The maximum field angle to consider. If None, then the maximum thetaMax
        in the maskModel dictionary is used. (the default is None)
    dTheta : float, optional
        Increment in field angle to use for determining necessary items.
        (the default is 1e-3 degrees)

    Returns
    -------
    dict
        The mask model dictionary with unnecessary elements removed.
    """
    # Determine thetaMax
    if thetaMax is None:
        thetaMaxList = []
        for item in maskModel:
            for edge in maskModel[item]:
                thetaMaxList.append(maskModel[item][edge]["thetaMax"])
        thetaMax = np.nanmax(thetaMaxList)

    # Create a grid of field angles
    thetas = np.arange(0, thetaMax + dTheta, dTheta)

    # And a grid of azimuthal angles
    azimuth = np.linspace(0, 2 * np.pi, 1000)

    # Separate lists for the inner and outer edges
    innerKeys = []
    innerR = []

    outerKeys = []
    outerR = []

    # Loop over every item in the model
    for item in maskModel:
        for edge in maskModel[item]:
            # Get the fit for this item
            fit = maskModel[item][edge]

            # Calculate the radius and center of the mask in meters
            radius = np.polyval(fit["radius"], thetas)[:, None]
            center = np.polyval(fit["center"], thetas)[:, None]

            # Calculate x and y of circle
            # Using polar equation of a circle so points are gridded on theta
            A = center * np.cos(azimuth)
            B = radius**2 - center**2 + A**2
            with np.errstate(invalid="ignore"):
                r = np.sqrt(B) + A

            # When rTheta < thetaMin, set to NaN
            r[thetas < fit["thetaMin"]] = np.nan

            # Append keys and radii to appropriate lists
            if fit["clear"] is True:
                outerKeys.append((item, edge))
                outerR.append(r)
            else:
                innerKeys.append((item, edge))
                innerR.append(r)

    # Determine which items are required at each field angle
    innerReq = np.any(np.array(innerR) >= np.nanmax(innerR, axis=0), axis=2)
    outerReq = np.any(np.array(outerR) <= np.nanmin(outerR, axis=0), axis=2)

    # Create a new dict...
    newDict = deepcopy(maskModel)

    # Loop through all the unnecessary elements and remove from the model
    for i in np.where(innerReq.sum(axis=1) <= 1)[0]:
        item, edge = innerKeys[i]
        newDict[item].pop(edge)
    for i in np.where(outerReq.sum(axis=1) <= 1)[0]:
        item, edge = outerKeys[i]
        newDict[item].pop(edge)
    for item in list(newDict):
        if len(newDict[item]) == 0:
            newDict.pop(item)

    # Update the angle ranges
    for i in np.where(innerReq.sum(axis=1) > 1)[0]:
        item, edge = innerKeys[i]
        idx = np.where(innerReq[i] == True)[0]  # noqa: E712
        newDict[item][edge]["thetaMin"] = thetas[idx[0]]
        newDict[item][edge]["thetaMax"] = thetas[idx[-1]]
    for i in np.where(outerReq.sum(axis=1) > 1)[0]:
        item, edge = outerKeys[i]
        idx = np.where(outerReq[i] == True)[0]  # noqa: E712
        newDict[item][edge]["thetaMin"] = thetas[idx[0]]
        newDict[item][edge]["thetaMax"] = thetas[idx[-1]]

    return newDict


def printMaskModel(maskModel: dict) -> None:
    """Print the yaml-formatted mask model.

    Parameters
    ----------
    maskModel : dict
        A mask model dictionary created by fitMaskModel.
    """
    # Loop over the items in the model
    for key in maskModel:
        print(f"  {key}:")

        # Loop over outer/inner edges
        for edge, vals in maskModel[key].items():
            print(f"    {edge}:")
            print(f"      clear: {vals['clear']}")
            print(f"      thetaMin: {np.floor(1e3 * vals['thetaMin']) / 1e3:.3f}")
            print(f"      thetaMax: {np.ceil(1e3 * vals['thetaMax']) / 1e3:.3f}")

            print(f"      center: [{vals['center'][0]:.5e}", end="")
            for v in vals["center"][1:]:
                print(f", {v:.5e}", end="")
            print("]")

            print(f"      radius: [{vals['radius'][0]:.5e}", end="")
            for v in vals["radius"][1:]:
                print(f", {v:.5e}", end="")
            print("]")
