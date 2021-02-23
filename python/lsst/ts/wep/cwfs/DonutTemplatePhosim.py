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

import os
import numpy as np
from lsst.ts.wep.Utility import (
    getConfigDir,
    DefocalType,
)
from lsst.ts.wep.cwfs.DonutTemplateDefault import DonutTemplateDefault


class DonutTemplatePhosim(DonutTemplateDefault):
    """Class to make the donut templates from phosim template images."""

    def makeTemplate(self, sensorName, defocalType, imageSize, phosimTemplateDir=None):
        """
        Make the donut template image from phosim templates.
        The templates should have been created using
        `bin.src/runCreatePhosimDonutTemplates.py`.
        See the `ts_wep` docs for more information on
        how to generate the templates.

        Parameters
        ----------
        sensorName : str
            The camera detector for which we want to make a template. Should
            be in "Rxx_Sxx" format.
        defocalType : enum 'DefocalType'
            The defocal state of the sensor.
        imageSize : int
            Size of template in pixels. The template will be a square.
        phosimTemplateDir : str, optional
            Specify the location where the phosim templates are stored. If None
            then the code will look in `policy/cwfs/templateData/phosimTemplates`.
            (The default is None)

        Returns
        -------
        numpy.ndarray [int]
            The donut template as a binary image.
        """

        if phosimTemplateDir is None:
            configDir = getConfigDir()
            phosimTemplateDir = os.path.join(
                configDir, "cwfs", "donutTemplateData", "phosimTemplates"
            )

        if defocalType == DefocalType.Extra:
            templateFilename = os.path.join(
                phosimTemplateDir, "extra_template-%s.txt" % sensorName
            )
        else:
            templateFilename = os.path.join(
                phosimTemplateDir, "intra_template-%s.txt" % sensorName
            )
        templateArray = np.genfromtxt(templateFilename, dtype=np.int)

        # Make the template the desired square shape by trimming edges of template
        templateSize = np.shape(templateArray)

        # Find the excess number of pixels in x and y direction
        templateTrim = templateSize - np.array((imageSize, imageSize))

        # Find the left and right edges by trimming half pixels from left and right.
        # Then do the same for the top.
        leftEdge = np.int(templateTrim[0] / 2)
        rightEdge = np.int(templateSize[0] - (templateTrim[0] - leftEdge))
        topEdge = np.int(templateTrim[1] / 2)
        bottomEdge = np.int(templateSize[1] - (templateTrim[1] - topEdge))

        templateArray = templateArray[leftEdge:rightEdge, topEdge:bottomEdge]
        return templateArray
