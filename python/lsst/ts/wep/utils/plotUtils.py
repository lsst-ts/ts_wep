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

__all__ = ["plotZernike", "plotPupilMaskElements", "plotRoundTrip"]

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from lsst.ts.wep.image import Image
from lsst.ts.wep.imageMapper import ImageMapper
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.enumUtils import BandLabel, DefocalType


def plotZernike(zkIdx, zk, unit, saveFilePath=None):
    """Plot the Zernike polynomials (zk).

    Parameters
    ----------
    zkIdx : list[int] or numpy.array[int]
        Index of zk.
    zk : numpy.array
        Zernike polynomials.
    unit : str
        Unit of Zernike polynomials.
    saveFilePath : str, optional
        File path to save the image. (the default is None.)
    """

    plt.plot(zkIdx, zk, marker="o", color="r", markersize=10)

    plt.xlabel("Zernike Index")
    plt.ylabel("Zernike coefficient (%s)" % unit)
    plt.grid()

    if saveFilePath is None:
        plt.show()
    else:
        plt.savefig(saveFilePath, bbox_inches="tight")
        plt.close()


def plotPupilMaskElements(
    instrument: Instrument,
    fieldAngle: tuple,
    legend: bool = True,
    minimal: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot the mask elements as circles.

    Outer and inner elements have the same color, with the inner elements dashed.
    The pupil is highlighted in yellow.

    Parameters
    ----------
    instrument : Instrument
        The ts_wep instrument that contains the mask parameters.
    fieldAngle : tuple
        Tuple of x and y field angles in degrees.
    legend : bool, optional
        Whether to draw the legend. (the default is True)
    minimal : bool, optional
        Whether to only draw the elements that presently determine the mask.
        (the default is True)
    ax : plt.Axes, optional
        A matplotlib axis to plot on. If None passed, plt.gca() is used.
        (the default is None)
    """
    # Get angle radius
    rTheta = np.sqrt(np.sum(np.square(fieldAngle)))

    # Generate angles around the circle
    theta = np.linspace(0, 2 * np.pi, 10_000)

    # Make lists to save the inner and outer radius at each angle
    innerR = []
    outerR = []

    # Create an empty dictionary for the colors
    colors = dict()

    # Make lists for plotting
    keys = []
    colorKeys = []
    xs = []
    ys = []
    rs = []

    # Loop over every mask element
    for key, val in instrument.maskParams.items():
        # Determine the color
        colorKey = key.removesuffix("Outer").removesuffix("Inner")
        if colorKey not in colors:
            colors[colorKey] = f"C{len(colors)}"

        # Skip elements for which we are not far enough out
        if rTheta < val["thetaMin"]:
            continue

        # Calculate the radius and center of the mask in meters
        radius = np.polyval(val["radius"], rTheta)
        rCenter = np.polyval(val["center"], rTheta)

        # Calculate x and y coordinates of the center
        xCenter = 0 if rTheta == 0 else rCenter * fieldAngle[0] / rTheta
        yCenter = 0 if rTheta == 0 else rCenter * fieldAngle[1] / rTheta

        # Calculate x and y of circle
        # Using the polar equation of a circle so that points are gridded on theta
        A = xCenter * np.cos(theta) + yCenter * np.sin(theta)
        B = radius**2 - (xCenter**2 + yCenter**2) + A**2
        with np.errstate(invalid="ignore"):
            r = np.sqrt(B) + A
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Save the radii to either the inner or outer lists
        if "Inner" in key:
            innerR.append(r)
        else:
            outerR.append(r)

        # Save the values for plotting
        keys.append(key)
        colorKeys.append(colorKey)
        xs.append(x)
        ys.append(y)
        rs.append(r)

    # Fill dummy values if inner or outer radii are missing
    innerR = [np.full_like(theta, 0)] if len(innerR) == 0 else innerR
    outerR = [np.full_like(theta, np.inf)] if len(outerR) == 0 else outerR

    # Determine the minimum and maximum radii at each angle
    innerR = np.nanmax(innerR, axis=0)
    outerR = np.nanmin(outerR, axis=0)

    # Select the axis
    ax = plt.gca() if ax is None else ax

    # Fill the pupil in yellow
    xIn = innerR * np.cos(theta)
    yIn = innerR * np.sin(theta)
    xOut = outerR * np.cos(theta)
    yOut = outerR * np.sin(theta)
    ax.fill(xOut, yOut, "gold", alpha=0.2)
    ax.fill(xIn, yIn, "w")

    # Loop over elements for plotting
    for key, colorKey, x, y, r in zip(keys, colorKeys, xs, ys, rs):
        # Determine the color
        c = colors[colorKey]

        # Determine the line style
        ls = "--" if "Inner" in key else "-"

        if not minimal:
            ax.plot(x, y, ls=ls, c=c, label=key)
            continue

        # Determine which points to plot
        idx = np.where((r >= innerR) & (r <= outerR))[0]
        if len(idx) == 0:
            continue

        # Split the points into consecutive segments
        cutPoints = np.where(np.diff(idx) != 1)[0] + 1
        splits = np.split(idx, cutPoints)
        for split in splits:
            ax.plot(x[split], y[split], ls=ls, c=c)

        # Add legend for this element
        ax.plot([], [], ls=ls, c=c, label=key)

    # Set the limits, axis labels, and title
    rmax = np.abs(outerR).max()
    rmax = rmax if np.isfinite(rmax) else np.abs(innerR).max()
    lim = 1.15 * rmax
    lim = lim if np.isfinite(lim) else None
    ax.set(
        xlim=(-lim, +lim),
        ylim=(-lim, +lim),
        xlabel="meters",
        ylabel="meters",
        aspect="equal",
        title=f"$\\theta\,=\,${rTheta:.2f}$\!^\circ$",
    )

    # Draw the legend?
    if legend:
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc="upper left")


def plotRoundTrip(
    fieldAngle: tuple,
    defocalType: Union[DefocalType, str],
    band: Union[BandLabel, str] = BandLabel.REF,
    opticalModel: str = "offAxis",
    zk: np.ndarray = np.zeros(19),
    nPixels: int = 180,
):
    """Plot a roundtrip of the ImageMapper

    Parameters
    ----------
    fieldAngle : tuple
        Tuple of x and y field angles in degrees.
    defocalType : Union[DefocalType, str]
        The DefocalType (or corresponding string)
    band : Union[BandLabel, str], optional
        The BandLabel (or corresponding string)
        (the default is BandLabel.REF)
    opticalModel : str, optional
        A string specifying the optical model
        (the default is "offAxis")
    zk : np.ndarray, optional
        Array of Zernike coefficients for wavefront aberrations (in meters)
        (the default is an array of zeros)
    nPixels : int, optional
        The number of pixels on a side for the images.
        (the default is 180)

    Returns
    -------
    fig
        The matplotlib Figure
    axes
        The matplotlib Axes
    """
    # Create the image mapper
    mapper = ImageMapper(opticalModel=opticalModel)

    # Forward model an image
    image = Image(
        np.zeros((nPixels, nPixels)),
        fieldAngle,
        defocalType,
        band,
    )
    image = mapper.mapPupilToImage(image, zk)

    # Then map back to the pupil
    pupilRecon = mapper.mapImageToPupil(image, zk)

    # Create the pupil mask
    pupil = mapper.createPupilMask(image)

    # Plot everything!
    fig, axes = plt.subplots(1, 4, figsize=(10, 2), dpi=150)

    settings = {"origin": "lower", "vmin": 0, "vmax": 1}

    axes[0].imshow(pupil, **settings)
    axes[0].set(title="Original")

    axes[1].imshow(image.image, **settings)
    axes[1].set(title="Mapped to image")

    axes[2].imshow(pupilRecon.image, **settings)
    axes[2].set(title="Back to pupil")

    axes[3].imshow(np.abs(pupilRecon.image - pupil), **settings)
    axes[3].set(title="Abs Pupil difference")

    return fig, axes
