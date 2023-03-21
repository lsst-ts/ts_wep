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

__all__ = ["WfEstimator"]

from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.cwfs.Algorithm import Algorithm
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.Utility import (
    DefocalType,
    CamType,
    CentroidFindType,
)


class WfEstimator(object):
    """Initialize the wavefront estimator class.

    Parameters
    ----------
    algoDir : str
        Path to algorithm directory.
    """

    def __init__(self, algoDir):
        self.inst = Instrument()
        self.algo = Algorithm(algoDir)

        self.imgIntra = CompensableImage()
        self.imgExtra = CompensableImage()

        self.opticalModel = ""
        self.sizeInPix = 0

    def getAlgo(self):
        """Get the algorithm object.

        Returns
        -------
        Algorithm
            Algorithm object.
        """

        return self.algo

    def getInst(self):
        """Get the instrument object.

        Returns
        -------
        Instrument
            Instrument object.
        """

        return self.inst

    def getIntraImg(self):
        """Get the intra-focal donut image.

        Returns
        -------
        CompensableImage
            Intra-focal donut image.
        """

        return self.imgIntra

    def getExtraImg(self):
        """Get the extra-focal donut image.

        Returns
        -------
        CompensableImage
            Extra-focal donut image.
        """

        return self.imgExtra

    def getOptModel(self):
        """Get the optical model.

        Returns
        -------
        str
            Optical model.
        """

        return self.opticalModel

    def getSizeInPix(self):
        """Get the donut image size in pixel defined by the config() function.

        Returns
        -------
        int
            Donut image size in pixel
        """

        return self.sizeInPix

    def reset(self):
        """

        Reset the calculation for the new input images with the same algorithm
        settings.
        """

        self.algo.reset()

    def config(
        self,
        instParams=None,
        solver="exp",
        camType=CamType.LsstCam,
        opticalModel="offAxis",
        sizeInPix=120,
        centroidFindType=CentroidFindType.RandomWalk,
        debugLevel=0,
    ):
        """Configure the TIE solver.

        Parameters
        ----------
        instParams : dict or None, optional
            Instrument Configuration Parameters to use. If None will default to
            files in policy/cwfs directory.
        solver : str, optional
            Algorithm to solve the Poisson's equation in the transport of
            intensity equation (TIE). It can be "fft" or "exp" here. (the
            default is "exp".)
        camType : enum 'CamType', optional
            Camera type. (the default is CamType.LsstCam.)
        opticalModel : str, optional
            Optical model. It can be "paraxial", "onAxis", or "offAxis". (the
            default is "offAxis".)
        sizeInPix : int, optional
            Wavefront image pixel size. (the default is 120.)
        centroidFindType : enum 'CentroidFindType', optional
            Algorithm to find the centroid of donut. (the default is
            CentroidFindType.RandomWalk.)
        debugLevel : int, optional
            Show the information under the running. If the value is higher,
            the information shows more. It can be 0, 1, 2, or 3. (the default
            is 0.)

        Raises
        ------
        ValueError
            Wrong Poisson solver name.
        ValueError
            Wrong optical model.
        ValueError
            Wrong optical model for AuxTel ("offAxis" is not implemented).
        """

        if solver not in ("exp", "fft"):
            raise ValueError("Poisson solver cannot be '%s'." % solver)

        if opticalModel not in ("paraxial", "onAxis", "offAxis"):
            raise ValueError("Optical model cannot be '%s'." % opticalModel)
        else:
            self.opticalModel = opticalModel

        if camType == CamType.AuxTel and opticalModel not in ("paraxial", "onAxis"):
            raise ValueError(f"Optical model cannot be {opticalModel} for AuxTel.")

        # Update the instrument name
        self.sizeInPix = int(sizeInPix)

        if instParams is None:
            self.inst.configFromFile(sizeInPix, camType)
        else:
            self.inst.configFromDict(instParams, sizeInPix, camType)

        self.algo.config(solver, self.inst, debugLevel=debugLevel)

        self.imgIntra = CompensableImage(centroidFindType=centroidFindType)
        self.imgExtra = CompensableImage(centroidFindType=centroidFindType)

    def setImg(
        self, fieldXY, defocalType, blendOffsets=None, image=None, imageFile=None
    ):
        """Set the wavefront image.

        Parameters
        ----------
        fieldXY : tuple or list
            Position of donut on the focal plane in degree for intra- and
            extra-focal images.
        defocalType : enum 'DefocalType'
            Defocal type of image.
        blendOffsets : list or None, optional
            Positions of blended donuts relative to location of center donut.
            Enter as [xCoordList, yCoordList].
            Length of xCoordList and yCoordList must be the same length.
            (the default is None).
        image : numpy.ndarray, optional
            Array of image. (the default is None.)
        imageFile : str, optional
            Path of image file. (the default is None.)
        """

        if defocalType == DefocalType.Intra:
            img = self.imgIntra
        elif defocalType == DefocalType.Extra:
            img = self.imgExtra

        img.setImg(
            fieldXY,
            defocalType,
            blendOffsets=blendOffsets,
            image=image,
            imageFile=imageFile,
        )

    def calWfsErr(self, tol=1e-3, showZer=False, showPlot=False):
        """Calculate the wavefront error.

        Parameters
        ----------
        tol : float, optional
            [description] (the default is 1e-3.)
        showZer : bool, optional
            Decide to show the annular Zernike polynomails or not. (the default
            is False.)
        showPlot : bool, optional
            Decide to show the plot or not. (the default is False.)

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22).

        Raises
        ------
        RuntimeError
            Input image shape is wrong.
        """

        # Check the image size
        for img in (self.imgIntra, self.imgExtra):
            d1, d2 = img.getImg().shape
            if (d1 != self.sizeInPix) or (d2 != self.sizeInPix):
                raise RuntimeError(
                    "Input image shape is (%d, %d), not required (%d, %d)"
                    % (d1, d2, self.sizeInPix, self.sizeInPix)
                )

        # Calculate the wavefront error.
        # Run cwfs
        self.algo.runIt(self.imgIntra, self.imgExtra, self.opticalModel, tol=tol)

        # Show the Zernikes Zn (n>=4)
        if showZer:
            self.algo.outZer4Up(showPlot=showPlot)

        return self.algo.getZer4UpInNm()
