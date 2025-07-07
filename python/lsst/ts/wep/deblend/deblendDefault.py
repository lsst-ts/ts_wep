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

__all__ = ["DeblendDefault"]

from typing import Any

import numpy as np
from lsst.ts.wep.centroid import CentroidFindFactory
from lsst.ts.wep.utils import CentroidFindType


class DeblendDefault(object):
    """Default deblend class."""

    def generateMultiDonut(
        self, template: np.ndarray, spaceCoef: float, magRatio: float, theta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Gemerate multiple donut images.

        Only one neightboring star will be generated for test, which is the
        baseline of LSST.

        Parameters
        ----------
        template : numpy.ndarray
            Template donut image.
        spaceCoef : float
            Spacing coefficient to decide the distance between donuts.
        magRatio : float
            Magnitude ratio of new donut compared with the original one.
        theta : float
            Theta angle of generated neighboring star in degree.

        Returns
        -------
        numpy.ndarray
            Image of donuts.
        numpy.ndarray
            Image of bright star.
        numpy.ndarray
            Image of neighboring star.
        float
            Position x of neighboring star.
        float
            Position y of neighboring star.

        Raises
        ------
        ValueError
            spaceCoef should be greater than zero.
        ValueError
            magRatio should be postive and less than 1.
        """

        # Check the inputs
        if spaceCoef <= 0:
            raise ValueError("spaceCoef should be greater than zero.")
        elif (magRatio < 0) or (magRatio > 1):
            raise ValueError("magRatio should be postive and less than 1.")

        # Get the center and radius of self-donut
        centroidFind = CentroidFindFactory.createCentroidFind(CentroidFindType.Otsu)
        imgBinary = centroidFind.getImgBinary(template)
        selfX, selfY, selfR = centroidFind.getCenterAndRfromImgBinary(imgBinary)

        # Get the position of new donut based on spaceCoef and theta
        thetaInRad = np.deg2rad(theta)
        newX = selfX + spaceCoef * selfR * np.cos(thetaInRad)
        newY = selfY + spaceCoef * selfR * np.sin(thetaInRad)

        # Calculate the frame size and shift the center of donuts
        lengthX = max(selfX, newX) - min(selfX, newX) + 5 * selfR
        lengthY = max(selfY, newY) - min(selfY, newY) + 5 * selfR
        length = int(max(lengthX, lengthY))

        # Enforce the length to be even for the symmetry
        if length % 2 == 1:
            length += 1

        shiftX = length / 2 - (selfX + newX) / 2
        shiftY = length / 2 - (selfY + newY) / 2

        # Get the new coordinate
        selfX += shiftX
        selfY += shiftY

        newX += shiftX
        newY += shiftY

        # Generate image of multiple donuts
        imageMain = np.zeros([length, length])
        imageNeighbor = np.zeros([length, length])

        # Get the shifted main donut image
        m, n = template.shape
        imageMain[
            int(selfY - m / 2) : int(selfY + m / 2),
            int(selfX - n / 2) : int(selfX + n / 2),
        ] += template

        # Get the shifted neighboring donut image
        imageNeighbor[
            int(newY - m / 2) : int(newY + m / 2), int(newX - n / 2) : int(newX + n / 2)
        ] += (magRatio * template)

        # Get the synthesized multi-donut image
        image = imageMain + imageNeighbor

        return image, imageMain, imageNeighbor, newX, newY

    def deblendDonut(
        self,
        imgToDeblend: np.ndarray,
        iniGuessXY: list[tuple],
        **kwargs: Any,
    ) -> tuple[np.ndarray, float, float]:
        """Deblend the donut image.

        Parameters
        ----------
        imgToDeblend : numpy.ndarray
            Image to deblend.
        iniGuessXY : list[tuple]
            The list contains the initial guess of (x, y) positions of
            neighboring stars as [star 1, star 2, etc.].
        **kwargs : dict[str, any]
            Dictionary of input argument: new value for that input argument.

        Returns
        -------
        numpy.ndarray
            Deblended donut image.
        float
            Position x of donut in pixel.
        float
            Position y of donut in pixel.

        Raises
        ------
        NotImplementedError
            Child class should implement this.
        """
        raise NotImplementedError("Child class should implement this.")
