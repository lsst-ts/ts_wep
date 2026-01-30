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

__all__ = ["EstimateZernikesBaseConfig", "EstimateZernikesBaseTask"]

import abc
import itertools
import logging
import multiprocessing as mp
from typing import Any, Callable, Iterable

import numpy as np

from astropy.coordinates import Angle
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ts.wep.estimation import WfAlgorithm, WfAlgorithmFactory, WfEstimator
from lsst.ts.wep.task.donutStamp import DonutStamp
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    WfAlgorithmName,
    convertHistoryToMetadata,
    getTaskInstrument,
)


def estimate_zk_pair(args: tuple[DonutStamp, DonutStamp, Angle, WfEstimator]) -> tuple[np.array, dict, dict]:
    """Estimate Zernike coefficients for a pair of donuts."""
    donutExtra, donutIntra, rtp, wfEstimator = args
    log = logging.getLogger(__name__)
    log.info(
        "Calculating Zernikes for Extra Donut %s, Intra Donut %s", *(donutExtra.donut_id, donutIntra.donut_id)
    )
    zk, zkMeta = wfEstimator.estimateZk(donutExtra.wep_im, donutIntra.wep_im, rtp)
    log.info(
        "Zernike estimation completed for Extra Donut %s, Intra Donut %s",
        *(donutExtra.donut_id, donutIntra.donut_id),
    )
    # Log number of function evaluations if available (currently only danish)
    if "lstsq_nfev" in zkMeta:
        log.info(
            "Num Iterations for Donut Pair (%s, %s): nfev = %i",
            *(donutExtra.donut_id, donutIntra.donut_id, zkMeta["lstsq_nfev"]),
        )
    return zk, zkMeta, wfEstimator.history


def estimate_zk_single(args: tuple[DonutStamp, Angle, WfEstimator]) -> tuple[np.array, dict, dict]:
    """Estimate Zernike coefficients for a single donut."""
    donut, rtp, wfEstimator = args
    log = logging.getLogger(__name__)
    log.info("Calculating Zernikes for Donut %s", donut.donut_id)
    zk, zkMeta = wfEstimator.estimateZk(donut.wep_im, None, rtp)
    log.info("Zernike estimation completed for Donut %s", donut.donut_id)
    # Log number of function evaluations if available (currently only danish)
    if "lstsq_nfev" in zkMeta:
        log.info("Num Iterations for Donut %s: nfev = %i", *(donut.donut_id, zkMeta["lstsq_nfev"]))
    return zk, zkMeta, wfEstimator.history


class EstimateZernikesBaseConfig(pexConfig.Config):
    instConfigFile: pexConfig.Field = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded.",
        dtype=str,
        optional=True,
    )
    nollIndices: pexConfig.Field = pexConfig.ListField(
        dtype=int,
        default=tuple(range(4, 29)),
        doc="Noll indices for which you wish to estimate Zernike coefficients. "
        + "Note these values must be unique, ascending, >= 4, and azimuthal pairs "
        + "must be complete. For example, if nollIndices contains 5, it must also "
        + "contain 6 (because 5 and 6 are the azimuthal pairs for astigmatism).",
    )
    startWithIntrinsic: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to start Zernike estimation from the intrinsic Zernikes.",
    )
    returnWfDev: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="If True, returns wavefront deviation. If False, returns full OPD.",
    )
    saveHistory: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to save the algorithm history in the task metadata. "
        + "Depending on the algorithm, saving the history might slow down "
        + "estimation, but doing so will provide intermediate products from "
        + "the estimation process.",
    )


class EstimateZernikesBaseTask(pipeBase.Task, metaclass=abc.ABCMeta):
    """Base class for estimating Zernike coefficients from DonutStamps."""

    ConfigClass = EstimateZernikesBaseConfig
    _DefaultName = "estimateZernikes"
    config: EstimateZernikesBaseConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    @abc.abstractmethod
    def wfAlgoName(self) -> WfAlgorithmName:
        """Return the WfAlgorithmName enum from the subclass."""
        ...

    @property
    def wfAlgoConfig(self) -> WfAlgorithm:
        """Return the configuration for the WfAlgorithm."""
        algoConfig = {
            key: val
            for key, val in self.config.toDict().items()
            if key not in EstimateZernikesBaseConfig._fields.keys()
        }

        return WfAlgorithmFactory.createWfAlgorithm(self.wfAlgoName, algoConfig)

    @staticmethod
    def _applyToList(fun: Callable, args: Iterable, numCores: int) -> list:
        """Apply a function to a list of arguments, optionally in parallel.

        If numCores is 1, multiprocessing is bypassed entirely to avoid issues
        with pickling AI models.

        Parameters
        ----------
        fun : Callable
            The function to apply. Must take a single argument.
        args : Iterable
            An iterable of arguments to apply the function to.
        numCores : int
            The number of cores to use. If 1, no multiprocessing is used.

        Returns
        -------
        list
            A list of results from applying the function to the arguments.
        """
        if numCores == 1:
            results = [fun(arg) for arg in args]
        else:
            with mp.Pool(processes=numCores) as pool:
                results = pool.map(fun, args)

        return results

    @staticmethod
    def _get_rtp(donutStamps):
        """Get the camera rotator angle

        Parameters
        ----------
        donutStamps : DonutStamps
            Donut postage stamps holding rotator metadata.

        Returns
        -------
        Angle
            The rotation angle of the camera on the telescope.
        """
        if not donutStamps:
            return Angle(np.nan, "rad")
        metadata = donutStamps.metadata
        try:
            rsp = metadata["BORESIGHT_ROT_ANGLE_RAD"]
            q = metadata["BORESIGHT_PAR_ANGLE_RAD"]
        except KeyError:
            return Angle(np.nan, "rad")
        return Angle(q - rsp - np.pi/2, "rad")

    def estimateFromPairs(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        wfEstimator: WfEstimator,
        numCores: int = 1,
    ) -> tuple[np.array, dict]:
        """Estimate Zernike coefficients from pairs of donut stamps.

        Parameters
        ----------
        donutStampsExtra : DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra : DonutStamps
            Intra-focal donut postage stamps.
        wfEstimator : WfEstimator
            The wavefront estimator object.
        numCores : int
            Number of cores to parallelize over.

        Returns
        -------
        np.ndarray
            Numpy array of estimated Zernike coefficients. The first
            axis indexes donut pairs while the second axis indexes the
            Noll coefficients.
        dict
            Metadata containing extra output from Zernike estimation.
            Each key is a type of output from the Zernike estimation
            method selected and contains a list of values,
            one for each pair of donuts.
        """
        self.log.info("Estimating paired Zernikes.")
        rtp = self._get_rtp(donutStampsExtra)
        # Loop over pairs in a multiprocessing pool
        args = [
            (donutExtra, donutIntra, rtp, wfEstimator)
            for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra)
        ]
        results = self._applyToList(estimate_zk_pair, args, numCores)

        zkList, zkMetaList, histories = zip(*results)
        zkMeta: dict = {key: [] for key in zkMetaList[0].keys()}
        for zkMetaSingle in zkMetaList:
            for key, value in zkMetaSingle.items():
                zkMeta[key].append(value)

        zkArray = np.array(zkList)

        # Save the histories (note if self.config.saveHistory is False,
        # this is just an empty dictionary)
        histories_dict = {f"pair{i}": convertHistoryToMetadata(hist) for i, hist in enumerate(histories)}
        self.metadata["history"] = histories_dict

        return zkArray, zkMeta

    def estimateFromIndivStamps(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        wfEstimator: WfEstimator,
        numCores: int = 1,
    ) -> tuple[np.array, dict]:
        """Estimate Zernike coefficients from individual donut stamps.

        Parameters
        ----------
        donutStampsExtra : DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra : DonutStamps
            Intra-focal donut postage stamps.
        wfEstimator : WfEstimator
            The wavefront estimator object.
        numCores : int
            Number of cores to parallelize over.

        Returns
        -------
        np.ndarray
            Numpy array of estimated Zernike coefficients. The first
            axis indexes donut stamps, starting with extrafocal stamps,
            followed by intrafocal stamps. The second axis indexes the
            Noll coefficients.
        dict
            Metadata containing extra output from Zernike estimation.
            Each key is a type of output from the Zernike estimation
            method selected and contains a list of values,
            one for each donut.
        """
        self.log.info("Estimating single sided Zernikes.")
        rtp = self._get_rtp(donutStampsExtra)
        # Loop over individual donut stamps with a process pool
        args = [(donut, rtp, wfEstimator) for donut in itertools.chain(donutStampsExtra, donutStampsIntra)]
        results = self._applyToList(estimate_zk_single, args, numCores)

        zkList, zkMetaList, histories = zip(*results)
        zkMeta: dict = {key: [] for key in zkMetaList[0].keys()}
        for zkMetaSingle in zkMetaList:
            for key, value in zkMetaSingle.items():
                zkMeta[key].append(value)

        zkArray = np.array(zkList)

        histories_dict = {}
        for i in range(len(donutStampsExtra)):
            histories_dict[f"extra{i}"] = convertHistoryToMetadata(histories[i])
        for i in range(len(donutStampsIntra)):
            histories_dict[f"intra{i}"] = convertHistoryToMetadata(histories[i + len(donutStampsExtra)])
        self.metadata["history"] = histories_dict

        return zkArray, zkMeta

    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        numCores: int = 1,
    ) -> pipeBase.Struct:
        """Estimate Zernike coefficients (in microns) from the donut stamps.

        Parameters
        ----------
        donutStampsExtra : DonutStamps
            Extra-focal donut postage stamps.
        donutStampsIntra : DonutStamps
            Intra-focal donut postage stamps.
        numCores : int
            Number of cores to parallelize over.

        Returns
        -------
        `lsst.pipe.base.Struct`
            A struct containing "zernikes", which is a 2D numpy array,
            where the first axis indexes the pair of DonutStamps and the
            second axis indexes the Zernikes coefficients. The units are
            microns. Also contains "wfEstInfo", which is a dictionary
            containing metadata with extra output from the wavefront
            estimation algorithm.
        """
        # Get the instrument
        if len(donutStampsExtra) > 0:
            refStamp = donutStampsExtra[0]
        else:
            refStamp = donutStampsIntra[0]
        camName = refStamp.cam_name
        detectorName = refStamp.detector_name
        instrument = getTaskInstrument(
            camName,
            detectorName,
            self.config.instConfigFile,
        )

        # Create the wavefront estimator
        wfEst = WfEstimator(
            algoName=self.wfAlgoName,
            algoConfig=self.wfAlgoConfig,
            instConfig=instrument,
            nollIndices=self.config.nollIndices,
            startWithIntrinsic=self.config.startWithIntrinsic,
            returnWfDev=self.config.returnWfDev,
            units="um",
            saveHistory=self.config.saveHistory,
        )

        self.log.info("Using %d cores", numCores)
        self.log.info("Noll indices: %s", self.config.nollIndices)
        if len(donutStampsExtra) > 0 and len(donutStampsIntra) > 0:
            zernikes, zkMeta = self.estimateFromPairs(
                donutStampsExtra, donutStampsIntra, wfEst, numCores=numCores
            )
        else:
            if wfEst.algo.requiresPairs:
                raise ValueError(
                    f"Wavefront algorithm `{wfEst.algo.__class__.__name__}` requires pairs of donuts."
                )
            zernikes, zkMeta = self.estimateFromIndivStamps(
                donutStampsExtra, donutStampsIntra, wfEst, numCores=numCores
            )

        return pipeBase.Struct(zernikes=zernikes, wfEstInfo=zkMeta)
