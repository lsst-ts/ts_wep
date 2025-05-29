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

__all__ = ["TieAlgorithm"]

import inspect
from typing import Iterable, Optional, Union

import numpy as np
from lsst.ts.wep import Image, ImageMapper, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.utils import DefocalType, binArray, makeSparse
from scipy.ndimage import gaussian_filter


class TieAlgorithm(WfAlgorithm):
    """Wavefront estimation algorithm class for the TIE solver.

    The following sources discuss the details of the TIE algorithm:
    - https://sitcomtn-111.lsst.io
    - Xin (2015): http://arxiv.org/abs/1506.04839

    Parameters
    ----------
    opticalModel : str, optional
        The optical model to use for mapping between the image and pupil
        planes. Can be "offAxis", "onAxis", or "paraxial". offAxis is a
        numerical model that is valid for all optical systems, but requires
        an accurate Batoid model. onAxis is an analytic model that is valid
        for all optical systems near the optical axis. paraxial is an
        analytic model that is valid for slow optical systems near the
        optical axis. offAxis is recommended when you have a Batoid model
        and onAxis is recommended when you do not. paraxial is primarily
        meant for testing (the default is "offAxis")
        (the default is "offAxis")
    maxIter : int, optional
        The maximum number of iterations of the TIE loop.
        (the default is 30)
    compSequence : iterable, optional
        An iterable that determines the maximum Noll index to compensate on
        each iteration of the TIE loop. For example, if compSequence = [4, 10],
        then on the first iteration, only Zk4 is used in image compensation and
        on iteration 2, Zk4-Zk10 are used. Once the end of the sequence has
        been reached, all Zernike coefficients are used during compensation.
        (the default is (4, 4, 6, 6, 13, 13, 13, 13, 22, 22, 22, 22))
    compGain : float, optional
        The gain used to update the Zernikes for image compensation.
        (the default is 0.6)
    centerTol : float, optional
        The maximum absolute change in any Zernike amplitude (in meters) for
        which the images need to be recentered. A smaller value causes the
        images to be recentered more often. If 0, images are recentered on
        every iteration.
        (the default is 1e-9)
    centerBinary : bool, optional
        Whether to use a binary template when centering the image.
        (the default is True)
    convergeTol : float, optional
        The maximum absolute change in any Zernike amplitude (in meters)
        between subsequent TIE iterations below which convergence is declared
        and iteration is stopped.
        (the default is 1e-9)
    maskKwargs : dict or None, optional
        Dictionary of mask keyword arguments to pass to mask creation.
        To see possibilities, see the docstring for
        lsst.ts.wep.imageMapper.ImageMapper.createPupilMasks().
        (the default is None)
    modelPupilKernelSize : float, optional
        The size of the Gaussian kernel to convolve with the model pupil
        when estimating Zernikes with a single donut.
        (the default is 2)
    binning : int, optional
        Binning factor to apply to the donut stamps before estimating
        Zernike coefficients. The default value of 1 means no binning.
        (the default is 1)
    requireConverge : bool, optional
        Whether to require that the TIE converges. If True, and the TIE
        did not converge, the TIE returns NaNs.
        (the default is False)
    optimizeLinAlg : bool, optional
        Whether to optimize the numpy einsum calls in the TIE solver. If
        True, the einsum calls are optimized for speed using the optimize
        keyword (the default is True).
    """

    def __init__(
        self,
        opticalModel: str = "offAxis",
        maxIter: int = 30,
        compSequence: Iterable = 2 * (4,) + 2 * (6,) + 4 * (13,) + 4 * (22,),
        compGain: float = 0.6,
        centerTol: float = 1e-9,
        centerBinary: bool = True,
        convergeTol: float = 1e-9,
        maskKwargs: Optional[dict] = None,
        modelPupilKernelSize: float = 2,
        binning: int = 1,
        requireConverge: bool = False,
        optimizeLinAlg: bool = True,
    ) -> None:
        self.opticalModel = opticalModel
        self.maxIter = maxIter
        self.compSequence = compSequence
        self.compGain = compGain
        self.centerTol = centerTol
        self.centerBinary = centerBinary
        self.convergeTol = convergeTol
        self.maskKwargs = maskKwargs
        self.modelPupilKernelSize = modelPupilKernelSize
        self.binning = binning
        self.requireConverge = requireConverge
        self.optimizeLinAlg = optimizeLinAlg

    @property
    def requiresPairs(self) -> bool:
        """Whether the algorithm requires pairs to estimate Zernikes."""
        return False

    @property
    def opticalModel(self) -> str:
        """The optical model to use for mapping the image to the pupil."""
        return self._opticalModel

    @opticalModel.setter
    def opticalModel(self, value: str) -> None:
        """Set the optical model to use for image mapping.

        Parameters
        ----------
        value : str
            The optical model to use for mapping between the image and pupil
            planes. Can be "offAxis", "onAxis", or "paraxial". offAxis is a
            numerical model that is valid for all optical systems, but requires
            an accurate Batoid model. onAxis is an analytic model that is valid
            for all optical systems near the optical axis. paraxial is an
            analytic model that is valid for slow optical systems near the
            optical axis. offAxis is recommended when you have a Batoid model
            and onAxis is recommended when you do not. paraxial is primarily
            meant for testing.

        Raises
        ------
        ValueError
            If the value is not one of the allowed values
        """
        allowedModels = ["paraxial", "onAxis", "offAxis"]
        if value not in allowedModels:
            raise ValueError(f"opticalModel must be one of {str(allowedModels)[1:-1]}.")

        self._opticalModel = value

    @property
    def solver(self) -> str:
        """The name of the TIE solver."""
        return self._solver

    @solver.setter
    def solver(self, value: str) -> None:
        """Set the TIE solver.

        Parameters
        ----------
        value : str
            Method used to solve the TIE. If "exp", the TIE is solved via
            directly expanding the wavefront in a Zernike series. If "fft",
            the TIE is solved using fast Fourier transforms.

        Raises
        ------
        ValueError
            If the value is not one of the allowed values
        """
        allowedSolvers = ["exp", "fft"]
        if value not in allowedSolvers:
            raise ValueError(f"solver must be one of {str(allowedSolvers)[1:-1]}.")

        self._solver = value

    @property
    def maxIter(self) -> int:
        """The maximum number of iterations in the TIE loop."""
        return self._maxIter

    @maxIter.setter
    def maxIter(self, value: int) -> None:
        """Set the maximum number of iterations in the TIE loop.

        Parameters
        ----------
        value : int
            The maximum number of iterations of the TIE loop.

        Raises
        ------
        TypeError
            If the value is not an integer
        ValueError
            If the value is negative
        """
        if not isinstance(value, int) or (isinstance(value, float) and value % 1 != 0):
            raise TypeError("maxIter must be an integer.")
        if value < 0:
            raise ValueError("maxIter must be non-negative.")

        self._maxIter = int(value)

    @property
    def compSequence(self) -> np.ndarray:
        """The compensation sequence for the TIE loop."""
        return self._compSequence

    @compSequence.setter
    def compSequence(self, value: Iterable) -> None:
        """Set the compensation sequence for the TIE loop.

        Parameters
        ----------
        value : iterable
            An iterable that determines the maximum Noll index to
            compensate on each iteration of the TIE loop. For example,
            if compSequence = [4, 10], then on the first iteration,
            only Zk4 is used in image compensation and on iteration 2,
            Zk4-Zk10 are used. Once the end of the sequence has been
            reached, all Zernike coefficients are used during compensation.

        Raises
        ------
        ValueError
            If the value is not an iterable
        """
        value = np.array(value, dtype=int)
        if value.ndim != 1:
            raise ValueError("compSequence must be a 1D iterable.")

        self._compSequence = value

    @property
    def compGain(self) -> float:
        """The compensation gain for the TIE loop."""
        return self._compGain

    @compGain.setter
    def compGain(self, value: float) -> None:
        """Set the compensation gain for the TIE loop.

        Parameters
        ----------
        value : float, optional
            The gain used to update the Zernikes for image compensation.

        Raises
        ------
        ValueError
            If the value is not positive
        """
        value = float(value)
        if value <= 0:
            raise ValueError("compGain must be positive.")

        self._compGain = value

    @property
    def centerTol(self) -> float:
        """Max abs. deviation in Zernike coeff. that requires re-centering."""
        return self._centerTol

    @centerTol.setter
    def centerTol(self, value: float) -> None:
        """Set max abs. deviation in Zernike coeff that requires re-centering.

        Parameters
        ----------
        value : float
            The maximum absolute change in any Zernike amplitude (in meters)
            for which the images need to be recentered. A smaller value causes
            the images to be recentered more often. If 0, images are recentered
            on every iteration.
        """
        self._centerTol = float(value)

    @property
    def centerBinary(self) -> bool:
        """Whether to center donuts using a binary template."""
        return self._centerBinary

    @centerBinary.setter
    def centerBinary(self, value: bool) -> None:
        """Set whether to center donuts using a binary template.

        Parameters
        ----------
        value : bool
            Whether to use a binary template when centering the image.
            Using a binary template is typically less accurate, but faster.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("centerBinary must be a boolean.")

        self._centerBinary = value

    @property
    def convergeTol(self) -> float:
        """Mean abs. deviation in Zernikes (meters) at which TIE terminates."""
        return self._convergeTol

    @convergeTol.setter
    def convergeTol(self, value: float) -> None:
        """Set the convergence tolerance of the TIE loop.

        Parameters
        ----------
        value : float
            The mean absolute deviation of Zernike coefficients (in meters),
            below which the TIE is deemed to have converged, and the TIE loop
            is terminated.

        Raises
        ------
        ValueError
            If the value is negative
        """
        value = float(value)
        if value < 0:
            raise ValueError("convergeTol must be greater than or equal to zero.")

        self._convergeTol = value

    @property
    def maskKwargs(self) -> dict:
        """Mask keyword arguments to pass to ImageMapper.createPupilMasks()."""
        return self._maskKwargs

    @maskKwargs.setter
    def maskKwargs(self, value: Union[dict, None]) -> None:
        """Set dict of keyword args passed to ImageMapper.createPupilMasks().

        Parameters
        ----------
        value : dict or None
            Dictionary of mask keyword arguments to pass to mask creation.
            To see possibilities, see the docstring for
            lsst.ts.wep.imageMapper.ImageMapper.createPupilMasks().

        Raises
        ------
        TypeError
            If you do not pass a dictionary or None
        ValueError
            If the dictionary contains keys that are not allowed
        """
        if value is None:
            value = dict()
        if not isinstance(value, dict):
            raise TypeError("maskKwargs must be a dictionary or None.")

        # Get the set of allowed keyword arguments
        sig = inspect.signature(ImageMapper.createPupilMasks)
        allowedKeys = set(sig.parameters)
        allowedKeys.remove("self")
        allowedKeys.remove("image")

        # Check that the passed keys are a subset of the allowed set
        keys = set(value.keys())
        if not keys.issubset(allowedKeys):
            raise ValueError(
                f"maskKwargs key(s) {keys - allowedKeys} are not allowed. "
                f"The allowed keys are {allowedKeys}."
            )

        self._maskKwargs = value

    @property
    def modelPupilKernelSize(self) -> float:
        """Return the Gaussian kernel size for the model pupil."""
        return self._modelPupilKernelSize

    @modelPupilKernelSize.setter
    def modelPupilKernelSize(self, value: float) -> None:
        """Set size of  Gaussian kernel that is convolved with the model pupil.

        This only matters when estimating Zernikes from a single side of focus.

        Parameters
        ----------
        value : float
            The size of the Gaussian kernel to convolve with the model pupil
            when estimating Zernikes with a single donut.
        """
        value = float(value)
        if value < 0:
            raise ValueError(
                "modelPupilKernelSize must be greater than or equal to zero."
            )

        self._modelPupilKernelSize = value

    @property
    def binning(self) -> int:
        """The binning factor to apply to donut stamps."""
        return self._binning

    @binning.setter
    def binning(self, value: int) -> None:
        """Set the binning factor to apply to the donut stamps.

        Parameters
        ----------
        value : int
            The binning factor. A value of 1 means no binning.
        """
        if not isinstance(value, int):
            raise TypeError("binning must be an integer.")
        if value < 1:
            raise ValueError("binning must be greater than or equal to 1.")
        self._binning = value

    @property
    def requireConverge(self) -> int:
        """Whether to require that the TIE converges."""
        return self._requireConverge

    @requireConverge.setter
    def requireConverge(self, value: int) -> None:
        """Whether to require that the TIE converges.

        Parameters
        ----------
        value : bool
            Whether to require that the TIE converges.
        """
        if not isinstance(value, bool):
            raise TypeError("requireConverge must be a bool.")
        self._requireConverge = value

    @property
    def optimizeLinAlg(self) -> bool:
        """Whether to optimize the linear algebra calls."""
        return self._optimizeLinAlg

    @optimizeLinAlg.setter
    def optimizeLinAlg(self, value: bool) -> None:
        """Set whether to optimize the linear algebra calls.

        Parameters
        ----------
        value : bool
            Whether to optimize the numpy einsum calls for speed. If True,
            the einsum calls are optimized for speed using the optimize
            keyword (the default is True).
        """
        if not isinstance(value, bool):
            raise TypeError("optimizeLinAlg must be a boolean.")
        self._optimizeLinAlg = value

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary saving the intermediate products from
        each iteration of the TIE solver.

        The initial products before the iteration begins are stored
        in history[0], which is a dictionary with the keys:
            - "intraInit" - the initial intrafocal image
            - "extraInit" - the initial extrafocal image
            - "zkStartIntra" - the starting intrafocal Zernikes
            - "zkStartExtra" - the starting extrafocal Zernikes
            - "zkStartMean" - the mean of the starting Zernikes. Note these
                              Zernikes are added to zkBest to estimate the
                              full OPD.
            - "pupil" - model pupil image, in the case where only one image
                        was passed to the estimator.
            - "nollIndices" - array of Noll indices corresponding to the
                              estimated Zernikes values.

        Each iteration of the solver is then stored under indices >= 1.
        The entry for each iteration is also a dictionary, containing
        the following keys:
            - "intraComp" - the compensated intrafocal image
            - "extraComp" - the compensated extrafocal image
            - "I0" - the estimate of the beam intensity on the pupil
            - "dIdz" - estimate of z-derivative of intensity across the pupil
            - "zkCompIntra" - Zernikes for compensating the intrafocal image
            - "zkCompExtra" - Zernikes for compensating the extrafocal image
            - "zkResid" - the estimated residual Zernikes
            - "zkBest" - the best cumulative estimate the wavefront residual.
            - "zkSum" - the sum of zkBest and zkStartMean from history[0].
                        This is the best estimate of the OPD at the end of
                        this iteration.
            - "converged" - flag indicating if Zernike estimation has converged
            - "caustic" - flag indicating if a caustic has been hit

        Note the units for all Zernikes are in meters, and the z-derivative
        in dIdz is also in meters. Furthermore, all Zernikes start with Noll
        index 4.
        """
        return super().history

    @staticmethod
    def _assignIntraExtra(
        I1: Image,
        I2: Union[Image, None],
        zkStartI1: np.ndarray,
        zkStartI2: Union[np.ndarray, None],
    ) -> tuple:
        """Assign I1 and I2 to intra and extra.

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image or None
            A second image, on the opposite side of focus from I1. Can be None.
        zkStartI1 : np.ndarray
            The starting Zernikes for I1 (in meters, for Noll indices >= 4)
        zkStartI2 : np.ndarray or None
            The starting Zernikes for I2 (in meters, for Noll indices >= 4)
            Can be None.

        Returns
        -------
        Image or None
            The intrafocal image, or None if no intrafocal image was passed
        Image or None
            The extrafocal image, or None if no intrafocal image was passed
        """
        # Start everything as None
        intra, extra, zkStartIntra, zkStartExtra = None, None, None, None

        # Replace intra and extra as appropriate
        if I1 is not None:
            if I1.defocalType == DefocalType.Intra:
                intra = I1.copy()
                zkStartIntra = zkStartI1.copy()
            else:
                extra = I1.copy()
                zkStartExtra = zkStartI1.copy()
        if I2 is not None:
            if I2.defocalType == DefocalType.Intra:
                intra = I2.copy()
                zkStartIntra = zkStartI2.copy()
            else:
                extra = I2.copy()
                zkStartExtra = zkStartI2.copy()

        return intra, extra, zkStartIntra, zkStartExtra

    def _createPupilImage(
        self,
        refImage: Image,
        imageMapper: ImageMapper,
    ) -> Image:
        """Create a pupil image to be used when one image is missing.

        Parameters
        ----------
        refImage : Image
            The other image for the pair. Used for getting metadata.
        imageMapper : ImageMapper
            The ImageMapper to be used for pupil dimensions and mask creation.

        Returns
        -------
        Image
            A Pupil image
        """
        # Get the pupil size
        nPix = imageMapper.instrument.nPupilPixels

        # And the defocal type for the pupil
        defocalType = "intra" if refImage.defocalType == DefocalType.Extra else "extra"

        # Create empty pupil image
        pupil = Image(
            image=np.zeros((nPix, nPix)),
            fieldAngle=refImage.fieldAngle,
            defocalType=defocalType,
            bandLabel=refImage.bandLabel,
            planeType="pupil",
        )

        # Create the pupil mask
        imageMapper.createPupilMasks(pupil)

        # Replace the image with the pupil mask
        # (Convolving with Gaussian gives better results)
        pupil.image = gaussian_filter(
            pupil.mask.astype(float),
            self.modelPupilKernelSize,
        )

        return pupil

    def _expSolve(
        self,
        I0: np.ndarray,
        dIdz: np.ndarray,
        nollIndices: np.ndarray,
        instrument: Instrument,
    ) -> np.ndarray:
        """Solve the TIE directly using a Zernike expansion.

        Parameters
        ----------
        I0 : np.ndarray
            The beam intensity at the exit pupil
        dIdz : np.ndarray
            The z-derivative of the beam intensity across the exit pupil
        nollIndices : np.ndarray
            Noll indices for which you wish to estimate Zernike coefficients.
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        jmax = nollIndices.max()
        zk = instrument.createZernikeBasis(jmax=jmax)
        dzkdu, dzkdv = instrument.createZernikeGradBasis(jmax=jmax)

        # Down-select to the Zernikes we are solving for
        zk = makeSparse(zk, nollIndices)
        dzkdu = makeSparse(dzkdu, nollIndices)
        dzkdv = makeSparse(dzkdv, nollIndices)

        # Calculate quantities for the linear system
        # See Equations 43-45 of https://sitcomtn-111.lsst.io
        b = np.einsum("ab,jab->j", dIdz, zk, optimize=self.optimizeLinAlg)
        M = np.einsum("ab,jab,kab->jk", I0, dzkdu, dzkdu, optimize=self.optimizeLinAlg)
        M += np.einsum("ab,jab,kab->jk", I0, dzkdv, dzkdv, optimize=self.optimizeLinAlg)
        M /= -instrument.radius**2

        # Invert to get Zernike coefficients in meters
        zkCoeff, *_ = np.linalg.lstsq(M, b, rcond=None)

        return zkCoeff

    def _estimateZk(
        self,
        I1: Image,
        I2: Optional[Image],  # type: ignore[override]
        zkStartI1: np.ndarray,
        zkStartI2: Optional[np.ndarray],
        nollIndices: np.ndarray,
        instrument: Instrument,
        saveHistory: bool,
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image or None
            A second image, on the opposite side of focus from I1. Can be None.
        zkStartI1 : np.ndarray
            The starting Zernikes for I1 (in meters, for Noll indices >= 4)
        zkStartI2 : np.ndarray or None
            The starting Zernikes for I2 (in meters, for Noll indices >= 4)
            Can be None.
        nollIndices : np.ndarray
            Noll indices for which you wish to estimate Zernike coefficients.
        instrument : Instrument
            The Instrument object associated with the DonutStamps.
        saveHistory : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from
            the images, in meters.

        Raises
        ------
        RuntimeError
            If the solver is not supported
        """
        # Rescale instrument pixel size to account for binning
        if self.binning > 1:
            instrument = instrument.copy()
            instrument.pixelSize *= self.binning

        # Create the ImageMapper for centering and image compensation
        imageMapper = ImageMapper(instConfig=instrument, opticalModel=self.opticalModel)

        # Re-assign I1/I2 to intra/extra
        intra, extra, zkStartIntra, zkStartExtra = self._assignIntraExtra(
            I1,
            I2,
            zkStartI1,
            zkStartI2,
        )

        # Bin images to reduce resolution?
        if self.binning > 1:
            if intra is not None:
                intra.image = binArray(intra.image, self.binning)
            if extra is not None:
                extra.image = binArray(extra.image, self.binning)

        # Create a pupil object for handling missing images
        if intra is None:
            pupil = self._createPupilImage(extra, imageMapper)
        elif extra is None:
            pupil = self._createPupilImage(intra, imageMapper)
        else:
            # No missing image
            pupil = None

        # Calculate the mean starting Zernikes, ignoring None values
        if zkStartIntra is None:
            zkStartMean = zkStartExtra.copy()
        elif zkStartExtra is None:
            zkStartMean = zkStartIntra.copy()
        else:
            zkStartMean = np.nanmean([zkStartIntra, zkStartExtra], axis=0)

        if saveHistory:
            # Save the initial images and intrinsic Zernikes in the history
            self._history[0] = {
                "intraInit": None if intra is None else intra.image.copy(),
                "extraInit": None if extra is None else extra.image.copy(),
                "zkStartIntra": None if intra is None else zkStartIntra.copy(),
                "zkStartExtra": None if extra is None else zkStartExtra.copy(),
                "zkStartMean": zkStartMean.copy(),
                "pupil": None if pupil is None else pupil.copy(),
                "nollIndices": nollIndices.copy(),
            }

        # Initialize Zernike arrays at zero
        zkComp = np.zeros_like(zkStartMean)  # Zernikes for compensation
        zkCenter = np.zeros_like(zkComp)  # Zernikes for centering the image
        zkResid = np.zeros_like(zkComp)  # Residual Zernikes after compensation
        zkBest = np.zeros_like(zkComp)  # Current best Zernike estimate
        zkSum = np.zeros_like(zkComp)  # Current best + starting Zernikes

        # Get the compensation sequence
        compSequence = iter(self.compSequence)

        # Determine the maximum Noll index we will solve for
        jmax = nollIndices.max()

        # Set the caustic and converged flags to False
        caustic = False
        converged = False

        # Loop through every iteration in the sequence
        for i in range(self.maxIter):
            # Determine the maximum Noll index to compensate
            # Once the compensation sequence is exhausted, jmaxComp = jmax
            jmaxComp = next(compSequence, jmax)

            # Calculate zkComp for this iteration
            # The gain scales how much of previous residual we incorporate
            # Everything past jmaxComp is set to zero
            zkComp += self.compGain * zkResid
            zkComp[nollIndices > jmaxComp] = 0

            # Center the images
            recenter = (i == 0) or (np.max(np.abs(zkComp - zkCenter)) > self.centerTol)
            if recenter:
                # Zernikes have changed enough that we should recenter images
                zkCenter = zkComp.copy()
                intraCent = (
                    None
                    if intra is None
                    else imageMapper.centerOnProjection(
                        intra,
                        zkCenter + zkStartIntra,
                        nollIndices,
                        isBinary=self.centerBinary,
                        **self.maskKwargs,
                    )
                )
                extraCent = (
                    None
                    if extra is None
                    else imageMapper.centerOnProjection(
                        extra,
                        zkCenter + zkStartExtra,
                        nollIndices,
                        isBinary=self.centerBinary,
                        **self.maskKwargs,
                    )
                )

            # Compensate images using the Zernikes
            intraComp = (
                pupil
                if intraCent is None
                else imageMapper.mapImageToPupil(
                    intraCent,
                    zkComp + zkStartIntra,
                    nollIndices,
                    masks=None if i == 0 else intraComp.masks,  # noqa: F821
                    **self.maskKwargs,
                )
            )
            extraComp = (
                pupil
                if extraCent is None
                else imageMapper.mapImageToPupil(
                    extraCent,
                    zkComp + zkStartExtra,
                    nollIndices,
                    masks=None if i == 0 else extraComp.masks,  # noqa: F821
                    **self.maskKwargs,
                )
            )

            # Apply a common pupil mask to each
            mask = (intraComp.mask >= 1) & (extraComp.mask >= 1)
            intraComp.image *= mask
            extraComp.image *= mask

            # Check for caustics
            if (
                intraComp.image.max() <= 0
                or extraComp.image.max() <= 0
                or not np.isfinite(intraComp.image).all()
                or not np.isfinite(extraComp.image).all()
            ):
                caustic = True

                # Dummy NaNs for the missing objects
                I0 = np.full_like(intraComp.image, np.nan)
                dIdz = np.full_like(intraComp.image, np.nan)
                zkResid = np.nan * zkResid

            # If no caustic, proceed with Zernike estimation
            else:
                # Normalize the images
                intraComp.image /= intraComp.image.sum()  # type: ignore
                extraComp.image /= extraComp.image.sum()  # type: ignore

                # Approximate I0 = I(u, 0) and dI/dz = dI(u, z)/dz at z=0
                I0 = (intraComp.image + extraComp.image) / 2  # type: ignore
                dIdz = (intraComp.image - extraComp.image) / (  # type: ignore
                    2 * instrument.pupilOffset
                )

                # Estimate the Zernikes
                zkResid = self._expSolve(I0, dIdz, nollIndices, instrument)

                # If estimating with a single donut, double the coefficients
                # This is because the simulated pupil image is aberration free
                # so it is effectively damping the detected aberrations
                # (effect is only a small increase in convergence rate)
                if intraCent is None or extraCent is None:
                    zkResid *= 2

                # Check for convergence
                # (1) The max absolute difference with the previous iteration
                #     must be below self.convergeTol
                # (2) We must be compensating all the Zernikes
                newBest = zkComp + zkResid
                converged = (jmaxComp >= jmax) & (
                    np.max(np.abs(newBest - zkBest)) < self.convergeTol
                )

                # Set the new best cumulative estimate of the residuals
                zkBest = newBest

                # Add the starting Zernikes to the best residuals
                zkSum = zkBest + zkStartMean

            # Time to wrap up this iteration!
            # Should we save intermediate products in the algorithm history?
            if saveHistory:
                # Save the images and Zernikes from this iteration
                self._history[i + 1] = {
                    "recenter": bool(recenter),
                    "intraCent": None if intraCent is None else intraCent.image.copy(),
                    "extraCent": None if extraCent is None else extraCent.image.copy(),
                    "intraComp": intraComp.image.copy(),
                    "extraComp": extraComp.image.copy(),
                    "mask": mask.copy(),  # type: ignore
                    "I0": I0.copy(),
                    "dIdz": dIdz.copy(),
                    "zkCompIntra": None if intraCent is None else zkComp + zkStartIntra,
                    "zkCompExtra": None if extraCent is None else zkComp + zkStartExtra,
                    "zkResid": zkResid.copy(),
                    "zkBest": zkBest.copy(),
                    "zkSum": zkSum.copy(),
                    "converged": bool(converged),
                    "caustic": bool(caustic),
                }

            # If we've hit a caustic or converged, we will stop early
            if caustic or converged:
                break

        # If we never converged, return NaNs?
        if self.requireConverge and not converged:
            zkSum *= np.nan

        zkMeta = {"caustic": bool(caustic), "converged": bool(converged)}

        return zkSum, zkMeta
