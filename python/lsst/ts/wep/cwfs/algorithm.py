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

__all__ = ["Algorithm"]

import os
import sys
import numpy as np
import galsim

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    iterate_structure,
    laplace,
)
from scipy.signal import convolve2d

from lsst.ts.wep.paramReader import ParamReader
from lsst.ts.wep.cwfs.instrument import Instrument
from lsst.ts.wep.cwfs.tool import (
    padArray,
    extractArray,
    ZernikeAnnularEval,
    ZernikeMaskedFit,
)
from lsst.ts.wep.utility import DefocalType
from lsst.ts.wep.plotUtil import plotZernike


class Algorithm(object):
    """Initialize the Algorithm class.

    Algorithm used to solve the transport of intensity equation to get
    normal/ annular Zernike polynomials.

    Parameters
    ----------
    algoDir : str
        Algorithm configuration directory.
    """

    def __init__(self, algoDir):
        self.algoDir = algoDir
        self.algoParamFile = ParamReader()

        self._inst = Instrument()

        # Show the calculation message based on this value
        # 0 means no message will be showed
        self.debugLevel = 0

        # Image has the problem or not from the over-compensation
        self.caustic = False

        # Record the Zk coefficients in each outer-loop iteration
        # The actual total outer-loop iteration time is Num_of_outer_itr + 1
        self.converge = np.array([])

        # Current number of outer-loop iteration
        self.currentItr = 0

        # Record the coefficients of normal/ annular Zernike polynomials after
        # z4 in unit of nm
        self.zer4UpNm = np.array([])

        # Converged wavefront.
        self.wcomp = np.array([])

        # Calculated wavefront in previous outer-loop iteration.
        self.West = np.array([])

        # Converged Zk coefficients
        self.zcomp = np.array([])

        # Calculated Zk coefficients in previous outer-loop iteration
        self.zc = np.array([])

        # Padded mask for use at the offset planes
        self.mask_comp = None

        # Non-padded mask corresponding to aperture
        self.mask_pupil = None

        # Change the dimension of mask for fft to use
        self.mask_comp_pad = None
        self.mask_pupil_pad = None

        # Cache annular Zernike evaluations
        self._zk = None

        # Cache evaluations of X and Y annular Zernike gradients
        self._dzkdx = None
        self._dzkdy = None

        # Create an attribute to store the algorithm history
        self._history = dict()

        # True if at least one of images has a blended object
        self.blend_exists = False

    def reset(self):
        """Reset the calculation for the new input images with the same
        algorithm settings."""

        self.caustic = False
        self.converge = np.zeros(self.converge.shape)
        self.currentItr = 0
        self.zer4UpNm = np.zeros(self.zer4UpNm.shape)

        self.wcomp = np.zeros(self.wcomp.shape)
        self.West = np.zeros(self.West.shape)

        self.zcomp = np.zeros(self.zcomp.shape)
        self.zc = np.zeros(self.zc.shape)

        self.mask_comp = None
        self.mask_pupil = None

        self.mask_comp_pad = None
        self.mask_pupil_pad = None

        self._history = dict()
        self.blend_exists = False

    def config(self, algoName, inst, debugLevel=0):
        """Configure the algorithm to solve TIE.

        Parameters
        ----------
        algoName : str
            Algorithm configuration file to solve the Poisson's equation in the
            transport of intensity equation (TIE). It can be "fft" or "exp"
            here.
        inst : Instrument
            Instrument to use.
        debugLevel : int, optional
            Show the information under the running. If the value is higher, the
            information shows more. It can be 0, 1, 2, or 3. (the default is
            0.)
        """

        algoParamFilePath = os.path.join(self.algoDir, "%s.yaml" % algoName)
        self.algoParamFile.setFilePath(algoParamFilePath)

        self._inst = inst
        self.debugLevel = debugLevel

        self.caustic = False

        numTerms = self.getNumOfZernikes()
        outerItr = self.getNumOfOuterItr()
        self.converge = np.zeros((numTerms, outerItr + 1))

        self.currentItr = 0

        self.zer4UpNm = np.zeros(numTerms - 3)

        # Wavefront related parameters
        self.wcomp = np.zeros((self._inst.dimOfDonutImg, self._inst.dimOfDonutImg))
        self.West = self.wcomp.copy()

        # Used in model basis ("zer").
        self.zcomp = np.zeros(numTerms)
        self.zc = self.zcomp.copy()

        # Mask related variables
        self.mask_comp = None
        self.mask_pupil = None
        self.mask_comp_pad = None
        self.mask_pupil_pad = None

        # Reset the history
        self._history = dict()

    def setDebugLevel(self, debugLevel):
        """Set the debug level.

        If the value is higher, the information shows more. It can be 0, 1, 2,
        or 3.

        Parameters
        ----------
        debugLevel : int
            Show the information under the running.
        """

        self.debugLevel = int(debugLevel)

    def getDebugLevel(self):
        """Get the debug level.

        If the value is higher, the information shows more. It can be 0, 1, 2,
        or 3.

        Returns
        -------
        int
            Debug level.
        """

        return self.debugLevel

    def getZer4UpInNm(self):
        """Get the coefficients of Zernike polynomials of z4-zn in nm.

        Returns
        -------
        numpy.ndarray
            Zernike polynomials of z4-zn in nm.
        """

        return self.zer4UpNm

    def getPoissonSolverName(self):
        """Get the method name to solve the Poisson equation.

        Returns
        -------
        str
            Method name to solve the Poisson equation.
        """

        return self.algoParamFile.getSetting("poissonSolver")

    def getNumOfZernikes(self):
        """Get the maximum number of Zernike polynomials supported.

        Returns
        -------
        int
            Maximum number of Zernike polynomials supported.
        """

        return int(self.algoParamFile.getSetting("numOfZernikes"))

    def getZernikeTerms(self):
        """Get the Zernike terms in using.

        Returns
        -------
        list[int]
            Zernike terms in using.
        """

        numTerms = self.getNumOfZernikes()

        return list(range(numTerms))

    def getObsOfZernikes(self):
        """Get the obscuration of annular Zernike polynomials.

        Returns
        -------
        float
            Obscuration of annular Zernike polynomials
        """

        zobsR = self.algoParamFile.getSetting("obsOfZernikes")
        if zobsR == 1:
            zobsR = self._inst.obscuration

        return float(zobsR)

    def getNumOfOuterItr(self):
        """Get the number of outer loop iteration.

        Returns
        -------
        int
            Number of outer loop iteration.
        """

        return int(self.algoParamFile.getSetting("numOfOuterItr"))

    def getNumOfInnerItr(self):
        """Get the number of inner loop iteration.

        This is for the fast Fourier transform (FFT) solver only.

        Returns
        -------
        int
            Number of inner loop iteration.
        """

        return int(self.algoParamFile.getSetting("numOfInnerItr"))

    def getFeedbackGain(self):
        """Get the gain value used in the outer loop iteration.

        Returns
        -------
        float
            Gain value used in the outer loop iteration.
        """

        return self.algoParamFile.getSetting("feedbackGain")

    def getOffAxisPolyOrder(self):
        """Get the number of polynomial order supported in off-axis correction.

        Returns
        -------
        int
            Number of polynomial order supported in off-axis correction.
        """

        return int(self.algoParamFile.getSetting("offAxisPolyOrder"))

    def getCompensatorMode(self):
        """Get the method name to compensate the wavefront by wavefront error.

        Returns
        -------
        str
            Method name to compensate the wavefront by wavefront error.
        """

        return self.algoParamFile.getSetting("compensatorMode")

    def getCompSequence(self):
        """Get the compensated sequence of Zernike order for each iteration.

        Returns
        -------
        numpy.ndarray[int]
            Compensated sequence of Zernike order for each iteration.
        """

        compSequenceFromFile = self.algoParamFile.getSetting("compSequence")
        compSequence = np.array(compSequenceFromFile, dtype=int)

        # If outerItr is large, and compSequence is too small,
        # the rest in compSequence will be filled.
        # This is used in the "zer" method.
        outerItr = self.getNumOfOuterItr()
        compSequence = self._extend1dArray(compSequence, outerItr)
        compSequence = compSequence.astype(int)

        return compSequence

    def _extend1dArray(self, origArray, targetLength):
        """Extend the 1D original array to the target length.

        The extended value will be the final element of original array. Nothing
        will be done if the input array is not 1D or its length is less than
        the target.

        Parameters
        ----------
        origArray : numpy.ndarray
            Original array with 1 dimension.
        targetLength : int
            Target length of new extended array.

        Returns
        -------
        numpy.ndarray
            Extended 1D array.
        """

        if (len(origArray) < targetLength) and (origArray.ndim == 1):
            leftOver = np.ones(targetLength - len(origArray))
            extendArray = np.append(origArray, origArray[-1] * leftOver)
        else:
            extendArray = origArray

        return extendArray

    def getBoundaryThickness(self):
        """Get the boundary thickness that the computation mask extends beyond
        the pupil mask.

        It is noted that in Fast Fourier transform (FFT) algorithm, it is also
        the width of Neumann boundary where the derivative of the wavefront is
        set to zero

        Returns
        -------
        int
            Boundary thickness.
        """

        return int(self.algoParamFile.getSetting("boundaryThickness"))

    def getFftDimension(self):
        """Get the FFT pad dimension in pixel.

        This is for the fast Fourier transform (FFT) solver only.

        Returns
        -------
        int
            FFT pad dimension.
        """

        fftDim = int(self.algoParamFile.getSetting("fftDimension"))

        # Make sure the dimension is the order of multiple of 2
        if fftDim == 999:
            dimToFit = self._inst.dimOfDonutImg
        else:
            dimToFit = fftDim

        padDim = int(2 ** np.ceil(np.log2(dimToFit)))

        return padDim

    def getSignalClipSequence(self):
        """Get the signal clip sequence.

        The number of values should be the number of compensation plus 1.
        This is for the fast Fourier transform (FFT) solver only.

        Returns
        -------
        numpy.ndarray
            Signal clip sequence.
        """

        sumclipSequenceFromFile = self.algoParamFile.getSetting("signalClipSequence")
        sumclipSequence = np.array(sumclipSequenceFromFile)

        # If outerItr is large, and sumclipSequence is too small, the rest in
        # sumclipSequence will be filled.
        # This is used in the "zer" method.
        targetLength = self.getNumOfOuterItr() + 1
        sumclipSequence = self._extend1dArray(sumclipSequence, targetLength)

        return sumclipSequence

    def getMaskScalingFactor(self):
        """Get the mask scaling factor for fast beam.

        Returns
        -------
        float
            Mask scaling factor for fast beam.
        """

        # m = R'*f/(l*R), R': radius of the no-aberration image
        maskScalingFactor = self._inst.focalLength / self._inst.getMarginalFocalLength()
        return maskScalingFactor

    def getWavefrontMapEsti(self):
        """Get the estimated wavefront map.

        Returns
        -------
        numpy.ndarray
            Estimated wavefront map.
        """

        return self._getWavefrontMapWithMaskApplied(self.wcomp)

    def getWavefrontMapResidual(self):
        """Get the residual wavefront map.

        Returns
        -------
        numpy.ndarray
            Residual wavefront map.
        """

        return self._getWavefrontMapWithMaskApplied(self.West)

    def _getWavefrontMapWithMaskApplied(self, wfMap):
        """Get the wavefront map with mask applied.

        Parameters
        ----------
        wfMap : numpy.ndarray
            Wavefront map.

        Returns
        -------
        numpy.ndarray
            Wavefront map with mask applied.
        """

        self._checkNotItr0()

        wfMapWithMask = wfMap.copy()
        wfMapWithMask[self.mask_pupil == 0] = np.nan

        return wfMapWithMask

    def _recordItem(self, item, itemName, outerItr, innerItr=None, debugLevel=0):
        """Record the item in the algorithm history.

        If you use this method to store new items in the algorithm history, you
        should update the docstring of the getHistory() method below, which
        describes the (potential) contents of the algorithm history.

        Parameters
        ----------
        item : Any
            The item to record in the history.
        itemName : str
            The name of the item in the history.
        outerItr : int
            The iteration of the outer-loop under which to store the item.
        innerItr : int, optional
            The iteration of the inner-loop under which to store the item.
            Inner-loop iterations are stored in another dictionary, which
            is stored under the key "innerLoop".
        debugLevel : int, optional
            The debugLevel at which this item should be recorded.
            (the default is 0.)
        """
        # if debug level too low, do nothing
        if self.debugLevel < debugLevel:
            return

        # create the outerItr dictionary if it doesn't exist
        if outerItr not in self._history:
            self._history[outerItr] = dict()

        # decide whether to record at the level of the outer- or inner-loop
        if innerItr is None:
            self._history[outerItr][itemName] = item
        else:
            # create the inner-loop dictionary if it doesn't exist
            if "innerLoop" not in self._history[outerItr]:
                self._history[outerItr]["innerLoop"] = dict()

            # create the innerItr dictionary if it doesn't exist
            if innerItr not in self._history[outerItr]["innerLoop"]:
                self._history[outerItr]["innerLoop"][innerItr] = dict()

            # record the item
            self._history[outerItr]["innerLoop"][innerItr][itemName] = item

    def getHistory(self):
        """Get the algorithm history.

        Returns
        -------
        dict
            Algorithm history.

        Notes
        -----
        The algorithm history is a dictionary that contains an entry for each
        iteration of the outer-loop, beginning with history[0].

        The entry for each outer-loop iteration is itself a dictionary,
        containing the following keys:
            - initI1 - the initial I1 image
            - initI2 - the initial I2 image
            - compZk - the zernikes used for image compensation (units: nm)
            - compI1 - the compensated I1 image
            - compI2 - the compensated I2 image
            - pupilMask - pupil mask that is applied before zernike calculation
            - maskedI1 - the masked version of the compensated I1
            - maskedI2 - the masked version of the compensated I2
            - residZk - the estimated residual zernikes (units: nm)
            - residWf - the estimated residual wavefront (units: nm)
            - totZk - the current best estimate of the zernikes (units: nm)
            - totWf - the current best estimate of the wavefront (units: nm)
            - caustic - whether this iteration encountered a caustic
            - converged - whether the zernikes have converged

        Further, if you are running the FFT algorithm, there is an inner-loop
        history saved under the key "innerLoop". The inner-loop history is
        itself a dictionary that contains an entry for every iteration of
        the inner-loop. The entry for each iteration of the inner-loop is
        another dictionary, containing the following keys:
            - initS - the initial "signal"
            - FFT - the Fast Fourier transform of initS
            - estWf - the estimated wavefront
            - estS - the estimated signal

        So for example, if you want to get the estimated signal from the 3rd
        step of the innerloop during the 2nd step of the outer loop, you
        would do the following:
        history = algorithm.getHistory()
        signal = history[1]["innerLoop"][2]["estS"]
        """
        return self._history

    def _checkNotItr0(self):
        """Check not in the iteration 0.

        TIE: Transport of intensity equation.

        Raises
        ------
        RuntimeError
            Need to solve the TIE first.
        """

        if self.currentItr == 0:
            raise RuntimeError("Need to solve the TIE first.")

    def itr0(self, I1, I2, model):
        """Calculate the wavefront and coefficients of normal/ annular Zernike
        polynomials in the first iteration time.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        """

        # Reset the iteration time of outer loop and decide to reset the
        # defocal images or not
        self._reset(I1, I2)

        # Solve the transport of intensity equation (TIE)
        self._singleItr(I1, I2, model)

    def runIt(self, I1, I2, model, tol=1e-3):
        """Calculate the wavefront error by solving the transport of intensity
        equation (TIE).

        The inner (for fft algorithm) and outer loops are used. The inner loop
        is to solve the Poisson's equation. The outer loop is to compensate the
        intra- and extra-focal images to mitigate the calculation of wavefront
        (e.g. S = -1/(delta Z) * (I1 - I2)/ (I1 + I2)).

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        tol : float, optional
            Tolerance of difference of coefficients of Zk polynomials compared
            with the previous iteration of outer loop. (the default is 1e-3.)

        Raises
        ------
        ValueError
            Check the that we have an intra- and extra-focal image.
        """
        if I1.defocalType == I2.defocalType:
            raise ValueError(
                f"I1 and I2 are both {I1.defocalType.name}focal. "
                "Please pass an intra- and extra-focal image."
            )

        # To have the iteration time initiated from global variable is to
        # distinguish the manually and automatically iteration processes.
        itr = self.currentItr
        while itr <= self.getNumOfOuterItr():
            stopItr = self._singleItr(I1, I2, model, tol)

            # Stop the iteration of outer loop if converged
            if stopItr:
                break

            itr += 1

    def nextItr(self, I1, I2, model, nItr=1):
        """Run the outer loop iteration with the specific time defined in nItr.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        nItr : int, optional
            Outer loop iteration time. (the default is 1.)
        """

        #  Do the iteration
        ii = 0
        while ii < nItr:
            self._singleItr(I1, I2, model)
            ii += 1

    def _zernikeBasisCache(self):
        """Evaluate and cache annular Zernike polynomials and their gradients.

        Produces 3 different basis sets from which inner products may be
        rapidly computed. The first dimension of each basis set indexes which
        Zernike term is evaluated. The second and third dimension index the x
        and y coordinates at which the polynomials are evaluated.

        Returns
        -------
        zk : ndarray of shape (nZk, nx, ny)
            Annular Zernike basis set.
        dzkdx, dzkdy : ndarray of shape (nZk, nx, ny)
            Gradient of annular Zernike basis set.

        Notes
        -----
        The cache assumes that
            self._inst.getSensorCoor()
            self.getNumOfZernikes()
            self.getObsOfZernikes()
        are all immutable during the lifetime of self.
        """
        if self._zk is None:
            # I'm assuming here that self._inst is immutable.
            xSensor, ySensor = self._inst.getSensorCoor()

            # Here's the GalSim public interface for a basis of annular
            # Zernikes.
            jmax = self.getNumOfZernikes()
            eps = self.getObsOfZernikes()
            self._zk = galsim.zernike.zernikeBasis(jmax, xSensor, ySensor, R_inner=eps)

            # There isn't currently a public interface for a gradient basis.
            # Here's what one would look like though if it existed. Relying a
            # bit on GalSim implementation details here, but should be okay in
            # practice.
            noll_coef_x = galsim.zernike._noll_coef_array_xy_gradx(jmax, eps)
            self._dzkdx = np.zeros(tuple((jmax + 1,) + xSensor.shape), dtype=float)
            self._dzkdx[1:] = np.array(
                [
                    galsim.utilities.horner2d(xSensor, ySensor, nc, dtype=float)
                    for nc in noll_coef_x.transpose(2, 0, 1)
                ]
            )

            noll_coef_y = galsim.zernike._noll_coef_array_xy_grady(jmax, eps)
            self._dzkdy = np.zeros(tuple((jmax + 1,) + xSensor.shape), dtype=float)
            self._dzkdy[1:] = np.array(
                [
                    galsim.utilities.horner2d(xSensor, ySensor, nc, dtype=float)
                    for nc in noll_coef_y.transpose(2, 0, 1)
                ]
            )

        return self._zk[1:], self._dzkdx[1:], self._dzkdy[1:]

    def _singleItr(self, I1, I2, model, tol=1e-3):
        """Run the outer-loop with single iteration to solve the transport of
        intensity equation (TIE).

        This is to compensate the approximation of wavefront:
        S = -1/(delta Z) * (I1 - I2)/ (I1 + I2)).

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        model : str
            Optical model. It can be "paraxial", "onAxis", or "offAxis".
        tol : float, optional
            Tolerance of difference of coefficients of Zk polynomials compared
            with the previous iteration of outer loop. (the default is 1e-3.)

        Returns
        -------
        bool
            Status of iteration.
        """

        # Use the zonal mode ("zer")
        compMode = self.getCompensatorMode()

        # Define the gain of feedbackGain
        feedbackGain = self.getFeedbackGain()

        # Set the pre-condition
        if self.currentItr == 0:
            # Check this is the first time of running iteration or not
            if I1.getImgInit() is None or I2.getImgInit() is None:
                # Check the image dimension
                if I1.getImg().shape != I2.getImg().shape:
                    print(
                        "Error: The intra and extra image stamps need to be of same size."
                    )
                    sys.exit()

                # Check for blends
                if (np.sum(np.isnan(I1.blendOffsetX)) == 0) and (
                    np.sum(np.isnan(I2.blendOffsetX)) == 0
                ):
                    self.blend_exists = True

                # Create the pupil masks
                # These will be applied to compensated images, so we will
                # create the masks with compensated=True
                boundaryT = self.getBoundaryThickness()
                # If the compensable image has no blended centroids
                # this function will just create a single masked donut
                I1.makeBlendedMask(self._inst, model, boundaryT, 1, compensated=True)
                I2.makeBlendedMask(self._inst, model, boundaryT, 1, compensated=True)
                self._makeMasterMask(I1, I2, self.getPoissonSolverName())

                # Load the offAxis correction coefficients
                if model == "offAxis":
                    offAxisPolyOrder = self.getOffAxisPolyOrder()
                    I1.setOffAxisCorr(self._inst, offAxisPolyOrder)
                    I2.setOffAxisCorr(self._inst, offAxisPolyOrder)

                # Update the self-initial image
                I1.updateImgInit()
                I2.updateImgInit()

            # Initialize the variables used in the iteration.
            self.zcomp = np.zeros(self.getNumOfZernikes())
            self.zc = self.zcomp.copy()

            self.wcomp = np.zeros((self._inst.dimOfDonutImg, self._inst.dimOfDonutImg))
            self.West = self.wcomp.copy()

            self.caustic = False

        # Rename this index (currentItr) for the simplification
        jj = self.currentItr

        # Solve the transport of intensity equation (TIE)
        if not self.caustic:
            # Reset the images before the compensation
            I1.updateImage(I1.getImgInit().copy())
            I2.updateImage(I2.getImgInit().copy())

            # Record the initial images for this iteration
            self._recordItem(I1.getImg().copy(), "initI1", jj, debugLevel=1)
            self._recordItem(I2.getImg().copy(), "initI2", jj, debugLevel=1)

            if compMode == "zer":
                # Zk coefficient from the previous iteration
                ztmp = self.zc.copy()

                # Do the feedback of Zk from the lower terms first based on the
                # sequence defined in compSequence
                if jj != 0:
                    compSequence = self.getCompSequence()
                    ztmp[int(compSequence[jj - 1]) :] = 0

                # Add partial feedback of residual estimated wavefront in Zk
                self.zcomp = self.zcomp + ztmp * feedbackGain

                # Record the zernikes (in nm) used to compensate the images
                self._recordItem(self.zcomp * 1e9, "compZk", jj, debugLevel=1)

                # Remove the image distortion by forwarding the image to pupil
                I1.compensate(self._inst, self, self.zcomp, model)
                I2.compensate(self._inst, self, self.zcomp, model)

                # Record the compensated images for this iteration
                self._recordItem(I1.getImg().copy(), "compI1", jj, debugLevel=1)
                self._recordItem(I2.getImg().copy(), "compI2", jj, debugLevel=1)

            # Check the image condition. If there is a problem, done with
            # this _singleItr().
            if (I1.isCaustic() is True) or (I2.isCaustic() is True):
                self.converge[:, jj] = self.converge[:, jj - 1]
                self.caustic = True
                self._recordItem(True, "caustic", jj, debugLevel=1)
                self._recordItem(False, "converged", jj, debugLevel=1)
                return

            # Correct the defocal images if I1 and I2 belong to different
            # sources, which is determined by the (fieldX, fieldY)
            self._applyI1I2mask_pupil(I1, I2)

            # Record the pupil mask that was used in the previous function,
            # plus the newly masked images
            self._recordItem(self.mask_pupil.copy(), "pupilMask", jj, debugLevel=1)
            self._recordItem(I1.getImg().copy(), "maskedI1", jj, debugLevel=1)
            self._recordItem(I2.getImg().copy(), "maskedI2", jj, debugLevel=1)

            # Solve the Poisson's equation
            self.zc, self.West = self._solvePoissonEq(I1, I2, jj)

            # Record the residual Zernike coefficients (in nm) and wavefront
            self._recordItem(self.zc.copy() * 1e9, "residZk", jj, debugLevel=1)
            self._recordItem(self.West.copy(), "residWf", jj, debugLevel=1)

            # Record/ calculate the Zk coefficient and wavefront
            if compMode == "zer":
                self.converge[:, jj] = self.zcomp + self.zc

                xoSensor, yoSensor = self._inst.getSensorCoorAnnular()
                self.wcomp = self.West + ZernikeAnnularEval(
                    np.concatenate(([0, 0, 0], self.zcomp[3:])),
                    xoSensor,
                    yoSensor,
                    self.getObsOfZernikes(),
                )

                # Record the total Zernike coefficients (in nm) and wavefront
                self._recordItem(
                    self.converge[:, jj].copy() * 1e9, "totZk", jj, debugLevel=1
                )
                self._recordItem(self.wcomp.copy(), "totWf", jj, debugLevel=1)

        else:
            # Once we run into caustic, stop here, results may be close to real
            # aberration.
            # Continuation may lead to disastrous results.
            self.converge[:, jj] = self.converge[:, jj - 1]

        # Record the coefficients of normal/ annular Zernike polynomials after
        # z4 in unit of nm
        self.zer4UpNm = self.converge[3:, jj] * 1e9

        # Status of iteration
        stopItr = False

        # Calculate the difference
        if jj > 0:
            diffZk = (
                np.sum(np.abs(self.converge[:, jj] - self.converge[:, jj - 1])) * 1e9
            )

            # Check the Status of iteration
            if diffZk < tol:
                stopItr = True

        # Record the convergence
        if not self.caustic:
            self._recordItem(False, "caustic", jj, debugLevel=1)
            self._recordItem(stopItr, "converged", jj, debugLevel=1)

        # Update the current iteration time
        self.currentItr += 1

        # Show the Zk coefficients in integer in each iteration
        if self.debugLevel >= 2:
            print("itr = %d, z4-z%d" % (jj, self.getNumOfZernikes()))
            print(np.rint(self.zer4UpNm))

        return stopItr

    def _solvePoissonEq(self, I1, I2, iOutItr=0):
        """Solve the Poisson's equation by Fourier transform (differential) or
        serial expansion (integration).

        There is no convergence for fft actually. Need to add the difference
        comparison and X-alpha method. Need to discuss further for this.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        iOutItr : int, optional
            ith number of outer loop iteration which is important in "fft"
            algorithm. (the default is 0.)

        Returns
        -------
        numpy.ndarray
            Coefficients of normal/ annular Zernike polynomials.
        numpy.ndarray
            Estimated wavefront.
        """

        # Calculate the aperture pixel size
        sensorFactor = self._inst.getSensorFactor()
        aperturePixelSize = (
            self._inst.apertureDiameter * sensorFactor / self._inst.dimOfDonutImg
        )

        # Calculate the differential Omega
        dOmega = aperturePixelSize**2

        # Solve the Poisson's equation based on the type of algorithm
        numTerms = self.getNumOfZernikes()
        zobsR = self.getObsOfZernikes()
        PoissonSolver = self.getPoissonSolverName()
        if PoissonSolver == "fft":
            # Use the differential method by fft to solve the Poisson's
            # equation

            # Parameter to determine the threshold of calculating I0.
            sumclipSequence = self.getSignalClipSequence()
            cliplevel = sumclipSequence[iOutItr]

            # Generate the v, u-coordinates on pupil plane
            padDim = self.getFftDimension()
            v, u = np.mgrid[
                -0.5
                / aperturePixelSize : 0.5
                / aperturePixelSize : 1.0
                / padDim
                / aperturePixelSize,
                -0.5
                / aperturePixelSize : 0.5
                / aperturePixelSize : 1.0
                / padDim
                / aperturePixelSize,
            ]

            # Show the threshold and pupil coordinate information
            if self.debugLevel >= 3:
                print("iOuter=%d, cliplevel=%4.2f" % (iOutItr, cliplevel))
                print(v.shape)

            # Calculate the const of fft:
            # FT{Delta W} = -4*pi^2*(u^2+v^2) * FT{W}
            u2v2 = -4 * (np.pi**2) * (u * u + v * v)

            # Set origin to Inf to result in 0 at origin after filtering
            ctrIdx = int(np.floor(padDim / 2.0))
            u2v2[ctrIdx, ctrIdx] = np.inf

            # Calculate the wavefront signal
            Sini = self._createSignal(I1, I2, cliplevel)

            # Find the just-outside and just-inside indices of a ring in pixels
            # This is for the use in setting dWdn = 0
            boundaryT = self.getBoundaryThickness()

            struct = generate_binary_structure(2, 1)
            struct = iterate_structure(struct, boundaryT)

            ApringOut = np.logical_xor(
                binary_dilation(self.mask_pupil, structure=struct), self.mask_pupil
            ).astype(int)
            ApringIn = np.logical_xor(
                binary_erosion(self.mask_pupil, structure=struct), self.mask_pupil
            ).astype(int)

            bordery, borderx = np.nonzero(ApringOut)

            # Put the signal in boundary (since there's no existing Sestimate,
            # S just equals self.S as the initial condition of SCF
            S = Sini.copy()
            for jj in range(self.getNumOfInnerItr()):
                # Record the initial image
                self._recordItem(S.copy(), "initS", iOutItr, jj, debugLevel=1)

                # Calculate FFT{S}
                SFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(S)))

                # Record FFT{S}
                self._recordItem(SFFT.copy(), "FFT", iOutItr, jj, debugLevel=1)

                # Calculate W by W=IFT{ FT{S}/(-4*pi^2*(u^2+v^2)) }
                W = np.fft.fftshift(
                    np.fft.irfft2(np.fft.fftshift(SFFT / u2v2), s=S.shape)
                )

                # Estimate the wavefront (includes zeroing offset & masking to
                # the aperture size)

                # Take the estimated wavefront
                West = extractArray(W, self._inst.dimOfDonutImg)

                # Calculate the offset
                offset = West[self.mask_pupil == 1].mean()
                West = West - offset
                West[self.mask_pupil == 0] = 0

                # Record the estimated wavefront
                self._recordItem(West.copy(), "estWf", iOutItr, jj, debugLevel=1)

                # Set dWestimate/dn = 0 around boundary
                WestdWdn0 = West.copy()

                # Do a 3x3 average around each border pixel, including only
                # those pixels inside the aperture. This averaging can be
                # efficiently computed using 1 numpy/scipy vectorized
                # convolve2d instruction to first sum the values in the 3x3
                # region, and dividing by a second convolve2d which counts
                # the non-zero pixels in each 3x3 region.

                kernel = np.ones((1 + 2 * boundaryT, 1 + 2 * boundaryT))
                tmp = convolve2d(West * ApringIn, kernel, mode="same")
                tmp_norm = convolve2d(ApringIn, kernel, mode="same")
                WestdWdn0[borderx, bordery] = (
                    tmp[borderx, bordery] / tmp_norm[borderx, bordery]
                )

                # Take Laplacian to find sensor signal estimate (Delta W = S)
                del2W = laplace(WestdWdn0) / dOmega

                # Extend the dimension of signal to the order of 2 for "fft" to
                # use
                Sest = padArray(del2W, padDim)

                # Record the estimated signal
                self._recordItem(Sest.copy(), "estS", iOutItr, jj, debugLevel=1)

                # Put signal back inside boundary, leaving the rest of
                # Sestimate
                Sest[self.mask_pupil_pad == 1] = Sini[self.mask_pupil_pad == 1]

                # Need to recheck this condition
                S = Sest

            # Calculate the coefficient of normal/ annular Zernike polynomials
            if self.getCompensatorMode() == "zer":
                xSensor, ySensor = self._inst.getSensorCoor()
                zc = ZernikeMaskedFit(
                    West, xSensor, ySensor, numTerms, self.mask_pupil, zobsR
                )
            else:
                zc = np.zeros(numTerms)

        elif PoissonSolver == "exp":
            # Use the integration method by serial expansion to solve the
            # Poisson's equation

            # Calculate dI and I0
            dI, I0 = self._getdIandI0(I1, I2)

            # Get the x, y coordinate in mask. The element outside mask is 0.
            xSensor, ySensor = self._inst.getSensorCoor()
            xSensor = xSensor * self.mask_comp
            ySensor = ySensor * self.mask_comp

            # Create the F matrix and Zernike-related matrixes

            # Get Zernike and gradient bases from cache.  These are each
            # (nzk, npix, npix) arrays, with the first dimension indicating
            # the Noll index.
            zk, dzkdx, dzkdy = self._zernikeBasisCache()

            # Eqn. (19) from Xin et al., Appl. Opt. 54, 9045-9054 (2015).
            # F_j = \int (d_z I) Z_j d_Omega
            F = np.tensordot(dI, zk, axes=((0, 1), (1, 2))) * dOmega
            # Eqn. (20) from Xin et al., Appl. Opt. 54, 9045-9054 (2015).
            # M_ij = \int I (grad Z_j) . (grad Z_i) d_Omega
            #      =   \int I (dZ_i/dx) (dZ_j/dx) d_Omega
            #        + \int I (dZ_i/dy) (dZ_j/dy) d_Omega
            Mij = np.einsum("ab,iab,jab->ij", I0, dzkdx, dzkdx)
            Mij += np.einsum("ab,iab,jab->ij", I0, dzkdy, dzkdy)
            Mij *= dOmega / (self._inst.apertureDiameter / 2.0) ** 2

            # Calculate dz
            dz = (
                2
                * self._inst.focalLength
                * (self._inst.focalLength - self._inst.defocalDisOffsetInM)
                / self._inst.defocalDisOffsetInM
            )

            # Define zc
            zc = np.zeros(numTerms)

            # Consider specific Zk terms only
            idx = self.getZernikeTerms()

            # Solve the equation: M*W = F => W = M^(-1)*F
            zc_tmp = np.linalg.lstsq(Mij[:, idx][idx], F[idx], rcond=None)[0] / dz
            zc[idx] = zc_tmp

            # Estimate the wavefront surface based on z4 - z22
            # z0 - z3 are set to be 0 instead
            West = ZernikeAnnularEval(
                np.concatenate(([0, 0, 0], zc[3:])), xSensor, ySensor, zobsR
            )

        return zc, West

    def _createSignal(self, I1, I2, cliplevel):
        """Calculate the wavefront singal for "fft" to use in solving the
        Poisson's equation.

        Need to discuss the method to define threshold and discuss to use
        np.median() instead.
        Need to discuss why the calculation of I0 is different from "exp".

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        cliplevel : float
            Parameter to determine the threshold of calculating I0.

        Returns
        -------
        numpy.ndarray
            Approximated wavefront signal.
        """

        # Calculate I0 and dI
        dI, I0 = self._getdIandI0(I1, I2)

        # Wavefront signal S=-(1/I0)*(dI/dz) is approximated to be
        # -(1/I0) * dI/(2 * delta z) = (-dI)/(2 * I0) * (1/delta z)
        # we will ignore the (1/delta z) until later in this method
        num = -dI
        den = 2 * I0

        # Define the effective minimum central signal element by the threshold
        # ( I0=(I1+I2)/2 )

        # Calculate the threshold
        pixelList = den * self.mask_comp
        pixelList = pixelList[pixelList != 0]

        low = pixelList.min()
        high = pixelList.max()
        medianThreshold = (high - low) / 2.0 + low

        # Define the effective minimum central signal element
        den[den < medianThreshold * cliplevel] = 1.5 * medianThreshold

        # Calculate delta z = f(f-l)/l, f: focal length, l: defocus distance of
        # the image planes
        deltaZ = (
            self._inst.focalLength
            * (self._inst.focalLength - self._inst.defocalDisOffsetInM)
            / self._inst.defocalDisOffsetInM
        )

        # Calculate the wavefront signal. Enforce the element outside the mask
        # to be 0.
        den[den == 0] = np.inf

        # Calculate the wavefront signal
        S = num / den / deltaZ

        # Extend the dimension of signal to the order of 2 for "fft" to use
        padDim = self.getFftDimension()
        Sout = padArray(S, padDim) * self.mask_comp_pad

        return Sout

    def _getdIandI0(self, I1, I2):
        """Calculate the differential image and central image.

        It is noted that the images are assumed to be co-center already, and
        that the images have already been compensated, so that the orientation
        of the extrafocal matches that of the intrafocal image.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.

        Returns
        -------
        numpy.ndarray
            Differential image, dI = Extra - Intra
        numpy.ndarray
            I0 = (Extra + Intra) / 2
        """

        # Check the image dimensions
        self._checkImageDim(I1, I2)

        # Calculate the central image and differential image
        dI = I1.getImg() - I2.getImg()
        I0 = (I1.getImg() + I2.getImg()) / 2

        # if I2 is extrafocal, flip the sign of dI
        if I2.defocalType == DefocalType.Extra:
            dI = -dI

        return dI, I0

    def _checkImageDim(self, I1, I2):
        """Check the dimension of images.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.

        Raises
        ------
        ValueError
            Check the dimension of images is n by n or not.
        ValueError
            Check two defocal images have the same size or not.
        """

        # Check the condition of images
        m1, n1 = I1.getImg().shape
        m2, n2 = I2.getImg().shape

        if m1 != n1 or m2 != n2:
            raise ValueError("Image is not square.")

        if m1 != m2 or n1 != n2:
            raise ValueError("Images do not have the same size.")

    def _makeMasterMask(self, I1, I2, poissonSolver=None):
        """Calculate the common mask of defocal images.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        poissonSolver : str, optional
            Algorithm to solve the Poisson's equation. If the "fft" is used,
            the mask dimension will be extended to the order of 2 for the "fft"
            to use. (the default is None.)
        """

        # Get the overlap region of mask for intra- and extra-focal images.
        # This is to avoid the anomalous signal due to difference in
        # vignetting.
        self.mask_comp = I1.getPaddedMask() * I2.getPaddedMask()
        self.mask_pupil = I1.getNonPaddedMask() * I2.getNonPaddedMask()

        # Change the dimension of image for fft to use
        if poissonSolver == "fft":
            padDim = self.getFftDimension()
            self.mask_comp_pad = padArray(self.mask_comp, padDim)
            self.mask_pupil_pad = padArray(self.mask_pupil, padDim)

    def _applyI1I2mask_pupil(self, I1, I2):
        """Mask the images if I1 and I2 belong to different sources.

        Note I1 and I2 are mutated in-place.

        (There is a problem for this actually. If I1 and I2 come from different
        sources, what should the correction of TIE be? At this moment, the
        fieldX and fieldY of I1 and I2 should be different. And the sources are
        different also.)

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        """

        # Get the overlap region of images and do the normalization.
        # If there is a blend we need to include the mask overlap
        # because the blended donut will appear on opposite sides
        # of the central donut.
        if (I1.getFieldXY() != I2.getFieldXY()) or self.blend_exists:
            # Get the overlap region of image
            I1.updateImage(I1.getImg() * self.mask_pupil)
            I2.updateImage(I2.getImg() * self.mask_pupil)

            # Do the normalization of image.
            I1.updateImage(I1.getImg() / np.sum(I1.getImg()))
            I2.updateImage(I2.getImg() / np.sum(I2.getImg()))

    def _reset(self, I1, I2):
        """Reset the iteration time of outer loop and defocal images.

        Parameters
        ----------
        I1 : CompensableImage
            Intra- or extra-focal image.
        I2 : CompensableImage
            Intra- or extra-focal image.
        """

        # Reset the current iteration time to 0
        self.currentItr = 0

        # Show the reset information
        if self.debugLevel >= 3:
            print("Resetting images: I1 and I2")

        # Determine to reset the images or not based on the existence of
        # the attribute: Image.image0. Only after the first run of
        # inner loop, this attribute will exist.
        try:
            # Reset the images to the first beginning
            I1.updateImage(I1.getImgInit().copy())
            I2.updateImage(I2.getImgInit().copy())

            # Show the information of resetting image
            if self.debugLevel >= 3:
                print("Resetting images in inside.")

        except AttributeError:
            # Show the information of no image0
            if self.debugLevel >= 3:
                print("Image0 = None. This is the first time to run the code.")

            pass

    def outZer4Up(self, unit="nm", filename=None, showPlot=False):
        """Put the coefficients of normal/ annular Zernike polynomials on
        terminal or file and show the image if it is needed.

        Parameters
        ----------
        unit : str, optional
            Unit of the coefficients of normal/ annular Zernike polynomials. It
            can be m, nm, or um. (the default is "nm".)
        filename : str, optional
            Name of output file. (the default is None.)
        showPlot : bool, optional
            Decide to show the plot or not. (the default is False.)
        """

        # List of Zn,m
        Znm = [
            "Z0,0",
            "Z1,1",
            "Z1,-1",
            "Z2,0",
            "Z2,-2",
            "Z2,2",
            "Z3,-1",
            "Z3,1",
            "Z3,-3",
            "Z3,3",
            "Z4,0",
            "Z4,2",
            "Z4,-2",
            "Z4,4",
            "Z4,-4",
            "Z5,1",
            "Z5,-1",
            "Z5,3",
            "Z5,-3",
            "Z5,5",
            "Z5,-5",
            "Z6,0",
        ]

        # Decide the format of z based on the input unit (m, nm, or um)
        if unit == "m":
            z = self.zer4UpNm * 1e-9
        elif unit == "nm":
            z = self.zer4UpNm
        elif unit == "um":
            z = self.zer4UpNm * 1e-3
        else:
            print("Unknown unit: %s" % unit)
            print("Unit options are: m, nm, um")
            return

        # Write the coefficients into a file if needed.
        if filename is not None:
            f = open(filename, "w")
        else:
            f = sys.stdout

        for ii in range(4, len(z) + 4):
            f.write("Z%d (%s)\t %8.3f\n" % (ii, Znm[ii - 1], z[ii - 4]))

        # Close the file
        if filename is not None:
            f.close()

        # Show the plot
        if showPlot:
            zkIdx = range(4, len(z) + 4)
            plotZernike(zkIdx, z, unit)
