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

from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.cwfs.algorithm import Algorithm
from lsst.ts.wep.cwfs.compensableImage import CompensableImage
from lsst.ts.wep.utility import (
    DefocalType,
    CamType,
    CentroidFindType,
    FilterType,
)
from lsst.ts.phosim.utils.ConvertZernikesToPsfWidth import convertZernikesToPsfWidth

import warnings


class WfEstimator(object):
    """Initialize the wavefront estimator class.

    Parameters
    ----------
    algoDir : str
        Path to algorithm directory.
    """

    def __init__(self, algoDir):

        self.algoDir = algoDir
        self.inst = Instrument()

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
        try:
            return self.algo
        except AttributeError as algoErr:
            raise RuntimeError(
                "The algorithm has not been configured yet. See the config() method."
            ) from algoErr

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

        # reset the algorithm
        self.algo.reset()

        # reset the images
        try:
            self.imgIntra = CompensableImage(centroidFindType=self._centroidFindType)
            self.imgExtra = CompensableImage(centroidFindType=self._centroidFindType)
        except AttributeError:
            self.imgIntra = CompensableImage()
            self.imgExtra = CompensableImage()

    def config(
        self,
        instParams=None,
        algo="exp",
        camType=CamType.LsstCam,
        opticalModel="offAxis",
        sizeInPix=120,
        centroidFindType=CentroidFindType.RandomWalk,
        units="nm",
        debugLevel=0,
        solver=None,
    ):
        """Configure the wavefront estimation algorithm.

        Parameters
        ----------
        instParams : dict or None, optional
            Instrument Configuration Parameters to use. If None will default to
            files in policy/cwfs directory.
        algo : str, optional
            Algorithm to estimate the wavefront. Options are "exp", "fft", or
            "ml". The first two use the cwfs.Algorithm class to solve the
            transport of intensity equation (TIE), while the third uses the
            cwfs.MachineLearningAlgorithm class. (the default is "exp".)
        camType : enum 'CamType', optional
            Camera type. (the default is CamType.LsstCam.)
        opticalModel : str, optional
            Optical model (irrelevant if algo=="ml"). Options are "paraxial",
            "onAxis", or "offAxis". (the default is "offAxis".)
        sizeInPix : int, optional
            Wavefront image pixel size. (the default is 120.)
        centroidFindType : enum 'CentroidFindType', optional
            Algorithm to find the centroid of donut. (the default is
            CentroidFindType.RandomWalk.)
        units : str, optional
            Units to return Zernikes in. Options are "microns", "nm", "waves",
            or "arcsecs". (the default is "nm".)
        debugLevel : int, optional
            Show the information under the running. If the value is higher,
            the information shows more. It can be 0, 1, 2, or 3. (the default
            is 0.)
        solver : str, optional
            Deprecated alias for algo. Use algo instead.

        Raises
        ------
        ValueError
            Invalid algo name.
        ValueError
            Invalid optical model.
        """

        # configure the algorithm
        if solver is not None:
            warnings.warn(
                "Argument `solver` is deprecated. Use `algo` instead.",
                DeprecationWarning,
            )
            algo = solver
        if algo in ["exp", "fft"]:
            self.algo = Algorithm(self.algoDir)
            self.algo.config(algo, self.inst, debugLevel=debugLevel)
        elif solver == "ml":
            raise NotImplementedError("ML method not yet implemented.")
        else:
            raise ValueError(f"algo cannot be {algo}.")

        # set the optical model
        if camType == CamType.AuxTel and opticalModel not in ("paraxial", "onAxis"):
            raise ValueError(f"Optical model cannot be {opticalModel} for AuxTel.")
        elif opticalModel not in ("paraxial", "onAxis", "offAxis"):
            raise ValueError(f"Optical model can not be {opticalModel}")

        if algo == "ml":
            opticalModel = ""

        self.opticalModel = opticalModel

        # configure the instrument
        self.sizeInPix = int(sizeInPix)
        if instParams is None:
            self.inst.configFromFile(sizeInPix, camType)
        else:
            self.inst.configFromDict(instParams, sizeInPix, camType)

        # create empty compensable images with the desired centroidFindType
        self._centroidFindType = centroidFindType
        self.imgIntra = CompensableImage(centroidFindType=centroidFindType)
        self.imgExtra = CompensableImage(centroidFindType=centroidFindType)

        # save the unit type
        if units not in ["microns", "nm", "waves", "arcsecs"]:
            raise ValueError(f"Invalid unit type {units}")
        if units == "waves":
            raise NotImplementedError(
                "Unit type 'waves' not supported until wavelength "
                "information added to CompensableImage class"
            )
        self.units = units

    def setImg(
        self,
        fieldXY,
        defocalType,
        filterLabel=FilterType.REF,
        blendOffsets=None,
        image=None,
        imageFile=None,
    ):
        """Set the wavefront image.

        Parameters
        ----------
        fieldXY : tuple or list
            Position of donut on the focal plane in degree for intra- and
            extra-focal images.
        defocalType : enum 'DefocalType'
            Defocal type of image.
        filterLabel : enum `FilterType`, optional
            Filter of the exposure. (the default is FilterType.REF)
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
            filterLabel,
            blendOffsets=blendOffsets,
            image=image,
            imageFile=imageFile,
        )

    def _calZkUsingTie(self, tol, showZer, showPlot):
        """Calculate the Zernikes using the TIE solver.

        Parameters
        ----------
        tol : float
            [description]
        showZer : bool, optional
            Decide to show the annular Zernike polynomials or not.
        showPlot : bool
            Decide to show the plot or not.

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in nanometers.

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

    def _calZkUsingMl(self):
        """Calculate the Zernikes using the ML model.

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in nanometers.
        """
        raise NotImplementedError("ML method not yet implemented.")

    def calWfsErr(self, tol=1e-3, showZer=False, showPlot=False):
        """Calculate the wavefront error.

        Parameters
        ----------
        tol : float, optional
            [description] (the default is 1e-3.)
        showZer : bool, optional
            Decide to show the annular Zernike polynomials or not.
            (the default is False.)
        showPlot : bool, optional
            Decide to show the plot or not. (the default is False.)

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in the units set
            during the config() method.

        Raises
        ------
        RuntimeError
            Input image shape is wrong.
        """

        # get Zernikes in nm
        if isinstance(self.algo, Algorithm):
            zk = self._calZkUsingTie(tol, showZer, showPlot)
        else:
            zk = self._calZkUsingMl()

        # convert zernikes to desired units
        if self.units in ["microns", "arcsecs"]:
            zk /= 1e3  # convert nm --> microns
        if self.units == "arcsecs":
            # get the aperture radii
            R_outer = self.inst.apertureDiameter / 2
            R_inner = R_outer * self.inst.obscuration
            # convert microns --> PSF FWHM contribution in arcseconds
            zk = convertZernikesToPsfWidth(zk, R_outer=R_outer, R_inner=R_inner)

        return zk
