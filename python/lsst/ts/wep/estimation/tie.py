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

import warnings
from typing import Iterable, Optional, Union

import numpy as np
from lsst.ts.wep import Image, ImageMapper, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.utils import DefocalType, createZernikeBasis, createZernikeGradBasis


class TieAlgorithm(WfAlgorithm):
    """Wavefront estimation algorithm class for the TIE solver.

    Parameters
    ----------
    configFile : str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/estimation/tie.yaml)
    opticalModel : str, optional
        The optical model to use for mapping the images to the pupil plane.
        Can be either "onAxis" or "offAxis". It is recommended you use offAxis,
        as this model can account for wide-field distortion effects, and so
        is appropriate for a wider range of field angles. However, the offAxis
        model requires a Batoid model of the telescope. If you do not have such
        a model, you can use the onAxis model, which is analytic, but is only
        appropriate near the optical axis. The field angle at which the onAxis
        model breaks down is telescope dependent.
    solver : str, optional
        Method used to solve the TIE. If "exp", the TIE is solved via
        directly expanding the wavefront in a Zernike series. If "fft",
        the TIE is solved using fast Fourier transforms.
    maxIter : int, optional
        The maximum number of iterations of the TIE loop.
    compSequence : iterable, optional
        An iterable that determines the maximum Noll index to compensate on
        each iteration of the TIE loop. For example, if compSequence = [4, 10],
        then on the first iteration, only Zk4 is used in image compensation and
        on iteration 2, Zk4-Zk10 are used. Once the end of the sequence has
        been reached, all Zernike coefficients are used during compensation.
    compGain : float, optional
        The gain used to update the Zernikes for image compensation.
    centerTol : float, optional
        The maximum absolute change in any Zernike amplitude (in meters) for which
        the images need to be recentered. A smaller value causes the images to
        be recentered more often. If 0, images are recentered on every iteration.
    centerBinary : bool, optional
        Whether to use a binary template when centering the image.
        Using a binary template is typically less accurate, but faster.
    convergeTol : float, optional
        The maximum absolute change in any Zernike amplitude (in meters) between
        subsequent TIE iterations below which convergence is declared and iteration
        is stopped.
    saveHistory : bool, optional
        Whether to save the algorithm history in the self.history attribute.
        If True, then self.history contains information about the most recent
        time the algorithm was run.
    """

    def __init__(
        self,
        configFile: Union[str, None] = "policy/estimation/tie.yaml",
        opticalModel: Optional[str] = None,
        solver: Optional[str] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
        centerTol: Optional[float] = None,
        centerBinary: Optional[bool] = None,
        convergeTol: Optional[float] = None,
        saveHistory: Optional[bool] = None,
    ) -> None:
        super().__init__(
            configFile=configFile,
            opticalModel=opticalModel,
            solver=solver,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
            centerTol=centerTol,
            centerBinary=centerBinary,
            convergeTol=convergeTol,
            saveHistory=saveHistory,
        )

        # Instantiate an empty history
        self._history = {}  # type: ignore

    @property
    def opticalModel(self) -> str:
        """The optical model to use for"""
        return self._opticalModel

    @opticalModel.setter
    def opticalModel(self, value: str) -> None:
        """Set the optical model to use for image mapping.

        Parameters
        ----------
        value : str
            The optical model to use for mapping between the image and pupil planes.
            Can be "paraxial", "onAxis", or "offAxis". Paraxial and onAxis are both
            analytic models, appropriate for donuts near the optical axis. The former
            is only valid for slow optical systems, while the latter is also valid
            for fast optical systems. The offAxis model is a numerically-fit model
            that is valid for fast optical systems at wide field angles. offAxis
            requires an accurate Batoid model.

        Raises
        ------
        TypeError
            If the value is not a string
        ValueError
            If the value is not one of the allowed values
        """
        allowedModels = ["paraxial", "onAxis", "offAxis"]
        if not isinstance(value, str):
            raise TypeError("optical model must be a string.")
        elif value not in allowedModels:
            raise ValueError(f"opticalModel must be one of {str(allowedModels)[1:-1]}.")

        self._opticalModel = value

    @property
    def solver(self) -> Union[str, None]:
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
        TypeError
            If value is not a string
        ValueError
            If the value is not one of the allowed values
        """
        allowedSolvers = ["exp", "fft"]
        if not isinstance(value, str):
            raise TypeError("solver must be a string.")
        elif value not in allowedSolvers:
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
            An iterable that determines the maximum Noll index to compensate on
            each iteration of the TIE loop. For example, if compSequence = [4, 10],
            then on the first iteration, only Zk4 is used in image compensation and
            on iteration 2, Zk4-Zk10 are used. Once the end of the sequence has
            been reached, all Zernike coefficients are used during compensation.

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
        """Maximum abs. deviation in Zernike coefficients that requires re-centering."""
        return self._centerTol

    @centerTol.setter
    def centerTol(self, value: float) -> None:
        """Set max abs. deviation in Zernike coefficients that requires re-centering.

        Parameters
        ----------
        value : float
            The maximum absolute change in any Zernike amplitude (in meters) for which
            the images need to be recentered. A smaller value causes the images to
            be recentered more often. If 0, images are recentered on every iteration.
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
        """Mean abs. deviation in Zernikes (meters) at which the TIE loop terminates."""
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
    def saveHistory(self) -> bool:
        """Whether the algorithm history is saved."""
        return self._saveHistory

    @saveHistory.setter
    def saveHistory(self, value: bool) -> None:
        """Set the boolean that determines whether the algorithm history is saved.

        Parameters
        ----------
        value : bool
            Boolean that determines whether the algorithm history is saved.

        Raises
        ------
        TypeError
            If the value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError("saveHistory must be a boolean.")

        self._saveHistory = value

        # If we are turning history-saving off, delete any old history
        # This is to avoid confusion
        if value is False:
            self._history = {}

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary saving the intermediate products from
        each iteration of the TIE solver. The first iteration is saved as
        history[0].

        The entry for each iteration is itself a dictionary containing
        the following keys:
            - "intraComp" - the compensated intrafocal image
            - "extraComp" - the compensated extrafocal image
            - "I0" - the estimate of the beam intensity on the pupil
            - "dIdz" - estimate of z-derivative of intensity across the pupil
            - "zkComp" - the Zernikes used for image compensation
            - "zkResid" - the estimated residual Zernikes
            - "zkBest" - the best estimate of the Zernikes after this iteration
            - "converged" - flag indicating if Zernike estimation has converged
            - "caustic" - flag indicating if a caustic has been hit

        Note the units for all Zernikes are in meters, and the z-derivative
        in dIdz is also in meters.
        """
        if not self._saveHistory:
            warnings.warn(
                "saveHistory is False. If you want the history to be saved, "
                "run self.config(saveHistory=True)."
            )

        return self._history

    def _expSolve(
        self,
        I0: np.ndarray,
        dIdz: np.ndarray,
        jmax: int,
        instrument: Instrument,
    ) -> np.ndarray:
        """Solve the TIE directly using a Zernike expansion.

        Parameters
        ----------
        I0 : np.ndarray
            The beam intensity at the exit pupil
        dIdz : np.ndarray
            The z-derivative of the beam intensity across the exit pupil
        jmax : int
            The maximum Zernike Noll index to estimate
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        # Get Zernike Bases
        uPupil, vPupil = instrument.createPupilGrid()
        zk = createZernikeBasis(uPupil, vPupil, jmax, instrument.obscuration)
        dzkdu, dzkdv = createZernikeGradBasis(
            uPupil,
            vPupil,
            jmax,
            instrument.obscuration,
        )

        # Calculate quantities for the linear equation
        b = -np.einsum("ab,jab->j", dIdz, zk, optimize=True)
        M = np.einsum("ab,jab,kab->jk", I0, dzkdu, dzkdu, optimize=True)
        M += np.einsum("ab,jab,kab->jk", I0, dzkdv, dzkdv, optimize=True)
        M /= instrument.radius**2

        # Invert to get Zernike coefficients in meters
        zkCoeff, *_ = np.linalg.lstsq(M, b, rcond=None)

        return zkCoeff

    def _fftSolve(
        self,
        I0: np.ndarray,
        dIdz: np.ndarray,
        jmax: int,
        instrument: Instrument,
    ) -> np.ndarray:
        """Solve the TIE using fast Fourier transforms.

        Parameters
        ----------
        I0 : np.ndarray
            The beam intensity at the exit pupil
        dIdz : np.ndarray
            The z-derivative of the beam intensity across the exit pupil
        jmax : int
            The maximum Zernike Noll index to estimate
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        # TODO: Implement the fft solver
        raise NotImplementedError("The fft solver is not yet implemented.")

    def estimateZk(
        self,
        I1: Image,
        I2: Image,  # type: ignore[override]
        jmax: int = 28,
        instrument: Instrument = Instrument(),
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image
            A second image, on the opposite side of focus from I1.
        jmax : int, optional
            The maximum Zernike Noll index to estimate.
            (the default is 28)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from
            the images, in meters.
        """
        # Validate the inputs
        if I1 is None or I2 is None:
            raise ValueError(
                "TIEAlgorithm requires a pair of intrafocal and extrafocal "
                "donuts to estimate Zernikes. Please provide both I1 and I2."
            )
        self._validateInputs(I1, I2, jmax, instrument)

        # Create the ImageMapper for centering and image compensation
        imageMapper = ImageMapper(
            configFile=None,
            instConfig=instrument,
            opticalModel=self.opticalModel,
        )

        # Get the initial intrafocal and extrafocal stamps
        intra = I1.copy() if I1.defocalType == DefocalType.Intra else I2.copy()
        extra = I1.copy() if I1.defocalType == DefocalType.Extra else I2.copy()

        if self.saveHistory:
            # Save the initial images in the history
            self._history[0] = {
                "intraInit": intra.image.copy(),
                "extraInit": extra.image.copy(),
            }

        # Initialize Zernike arrays at zero
        zkComp = np.zeros(jmax - 4 + 1)  # Zernikes for compensation
        zkCenter = np.zeros_like(zkComp)  # Zernikes for centering the image
        zkResid = np.zeros_like(zkComp)  # Residual Zernikes after compensation
        zkBest = np.zeros_like(zkComp)  # Current best Zernike estimate

        # Get the compensation sequence
        compSequence = iter(self.compSequence)

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
            zkComp[(jmaxComp - 3) :] = 0

            # Center the images
            recenter = (i == 0) or (np.max(np.abs(zkComp - zkCenter)) > self.centerTol)
            if recenter:
                # Zernikes have changed enough that we should recenter the images
                zkCenter = zkComp.copy()
                intraCent = imageMapper.centerOnProjection(
                    intra,
                    zkCenter,
                    binary=self.centerBinary,
                )
                extraCent = imageMapper.centerOnProjection(
                    extra,
                    zkCenter,
                    binary=self.centerBinary,
                )

            # Compensate images using the Zernikes
            intraComp = imageMapper.mapImageToPupil(intraCent, zkComp)
            extraComp = imageMapper.mapImageToPupil(extraCent, zkComp)

            # Apply a common mask to each
            intraMask = intraComp.mask
            extraMask = extraComp.mask
            mask = (intraMask >= 1) & (extraMask >= 1)  # type: ignore
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

                # Approximate I0 = I(x, 0) and dI/dz = dI(x, z)/dz at z=0
                I0 = (intraComp.image + extraComp.image) / 2  # type: ignore
                dIdz = (intraComp.image - extraComp.image) / (  # type: ignore
                    2 * instrument.pupilOffset
                )

                # Estimate the Zernikes
                if self.solver == "exp":
                    zkResid = self._expSolve(I0, dIdz, jmax, instrument)
                elif self.solver == "fft":
                    zkResid = self._fftSolve(I0, dIdz, jmax, instrument)

                # Check for convergence
                # (1) The max absolute difference with the previous iteration
                #     must be below self.convergeTol
                # (2) We must be compensating all the Zernikes
                newBest = zkComp + zkResid
                converged = (jmaxComp >= jmax) & (
                    np.max(np.abs(newBest - zkBest)) < self.convergeTol
                )

                # Set the new best estimate
                zkBest = newBest

            # Time to wrap up this iteration!
            # Should we save intermediate products in the algorithm history?
            if self.saveHistory:
                # Save the images and Zernikes from this iteration
                self._history[i + 1] = {
                    "recenter": recenter,
                    "intraCent": intraCent.image.copy(),
                    "extraCent": extraCent.image.copy(),
                    "intraComp": intraComp.image.copy(),
                    "extraComp": extraComp.image.copy(),
                    "mask": mask.copy(),  # type: ignore
                    "I0": I0.copy(),
                    "dIdz": dIdz.copy(),
                    "zkComp": zkComp.copy(),
                    "zkResid": zkResid.copy(),
                    "zkBest": zkBest.copy(),
                    "converged": converged,
                    "caustic": caustic,
                }

                # If we are using the FFT solver, save the inner loop as well
                if self.solver == "fft":
                    # TODO: After implementing fft, add inner loop here
                    self._history[i]["innerLoop"] = None

            # If we've hit a caustic or converged, we will stop early
            if caustic or converged:
                break

        return zkBest
