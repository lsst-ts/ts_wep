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

__all__ = ["BaseCwfsTestCase"]

import os

import numpy as np
from lsst.ts.wep.cwfs.algorithm import Algorithm
from lsst.ts.wep.cwfs.compensableImage import CompensableImage
from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.utility import DefocalType, getConfigDir


class BaseCwfsTestCase(object):
    """Base class for CWFS tests"""

    def calcWfErr(
        self,
        centroidFindType,
        fieldXY,
        camType,
        algoName,
        opticalModel,
        instParams,
        imageIntra=None,
        imageExtra=None,
        imageFileIntra=None,
        imageFileExtra=None,
    ):
        """Calculate the wavefront error.

        Parameters
        ----------
        centroidFindType : enum 'CentroidFindType'
            Algorithm to find the centroid of donut.
        fieldXY : tuple or list
            Position of donut on the focal plane in degree (field x, field y).
        camType : enum 'CamType'
            Camera type.
        algoName : str
            Algorithm configuration file to solve the Poisson's equation in the
            transport of intensity equation (TIE). It can be "fft" or "exp"
            here.
        opticalModel : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        instParams : dict
            Instrument Configuration Parameters to use.
        imageIntra : numpy.ndarray, optional
            Array of intra-focal image. (the default is None.)
        imageExtra : numpy.ndarray, optional
            Array of extra-focal image. (the default is None.)
        imageFileIntra : str, optional
            Path of intra-focal image file. (the default is None.)
        imageFileExtra : str, optional
            Path of extra-focal image file. (the default is None.)

        Returns
        -------
        numpy.ndarray
            Zernike polynomials of z4-zn in nm.
        """

        # Set the defocal images
        imgIntra = CompensableImage(centroidFindType=centroidFindType)
        imgExtra = CompensableImage(centroidFindType=centroidFindType)

        imgIntra.setImg(
            fieldXY, DefocalType.Intra, image=imageIntra, imageFile=imageFileIntra
        )
        imgExtra.setImg(
            fieldXY, DefocalType.Extra, image=imageExtra, imageFile=imageFileExtra
        )

        # Set the instrument
        inst = Instrument()
        inst.configFromDict(
            instParams,
            imgIntra.getImgSizeInPix(),
            camType,
        )

        # Define the algorithm to be used.
        algoFolderPath = os.path.join(getConfigDir(), "cwfs", "algo")
        algo = Algorithm(algoFolderPath)
        algo.config(algoName, inst, debugLevel=0)

        # Run it
        algo.runIt(imgIntra, imgExtra, opticalModel, tol=1e-3)

        # Return the Zernikes Zn (n>=4)
        return algo.getZer4UpInNm()

    def compareDiffWithTol(self, wfErr, wfErrAns, tolMax, tolRms):
        """Compare the difference between calculated wavefront error and answer
        with the tolerance.

        Parameters
        ----------
        wfErr : numpy.ndarray
            Calculated wavefront error.
        wfErrAns : numpy.ndarray
            Answer of wavefront error.
        tolMax : float
            Tolerance of maximum difference of wavefront errors.
        tolRms : float
            Tolerance of difference of root mean square (RMS).
        """

        diffMax = np.max(np.abs(wfErr - wfErrAns))
        self.assertLess(diffMax, tolMax)

        diffRms = np.sqrt(np.sum(np.abs(wfErr - wfErrAns) ** 2) / len(wfErr))
        self.assertLess(diffRms, tolRms)
