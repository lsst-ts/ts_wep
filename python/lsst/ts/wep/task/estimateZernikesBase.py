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
import multiprocessing as mp

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.ts.wep.estimation import WfAlgorithm, WfAlgorithmFactory, WfEstimator
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import (
    WfAlgorithmName,
    convertHistoryToMetadata,
    getTaskInstrument,
)


def estimate_zk_pair(args):
    """Estimate Zernike coefficients for a pair of donuts."""
    donutExtra, donutIntra, wfEstimator = args
    zk, zkMeta = wfEstimator.estimateZk(donutExtra.wep_im, donutIntra.wep_im)
    return zk, zkMeta, wfEstimator.history


def estimate_zk_single(args):
    """Estimate Zernike coefficients for a single donut."""
    donut, wfEstimator = args
    zk, zkMeta = wfEstimator.estimateZk(donut.wep_im)
    return zk, zkMeta, wfEstimator.history


class EstimateZernikesBaseConfig(pexConfig.Config):
    instConfigFile = pexConfig.Field(
        doc="Path to a instrument configuration file to override the instrument "
        + "configuration. If begins with 'policy:' the path will be understood as "
        + "relative to the ts_wep policy directory. If not provided, the default "
        + "instrument for the camera will be loaded.",
        dtype=str,
        optional=True,
    )
    nollIndices = pexConfig.ListField(
        dtype=int,
        default=tuple(range(4, 29)),
        doc="Noll indices for which you wish to estimate Zernike coefficients. "
        + "Note these values must be unique, ascending, >= 4, and azimuthal pairs "
        + "must be complete. For example, if nollIndices contains 5, it must also "
        + "contain 6 (because 5 and 6 are the azimuthal pairs for astigmatism).",
    )
    startWithIntrinsic = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to start Zernike estimation from the intrinsic Zernikes.",
    )
    returnWfDev = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="If True, returns wavefront deviation. If False, returns full OPD.",
    )
    saveHistory = pexConfig.Field(
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

    def __init__(self, **kwargs):
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

    def estimateFromPairs(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        wfEstimator: WfEstimator,
        numCores: int = 1,
    ) -> np.array:
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
        """
        # Loop over pairs in a multiprocessing pool
        args = [
            (donutExtra, donutIntra, wfEstimator)
            for donutExtra, donutIntra in zip(donutStampsExtra, donutStampsIntra)
        ]
        with mp.Pool(processes=numCores) as pool:
            results = pool.map(estimate_zk_pair, args)

        zkList, zkMetaList, histories = zip(*results)
        zkMeta = {key: [] for key in zkMetaList[0].keys()}
        for zkMetaSingle in zkMetaList:
            for key, value in zkMetaSingle.items():
                zkMeta[key].append(value)

        zkArray = np.array(zkList)

        # Save the histories (note if self.config.saveHistory is False,
        # this is just an empty dictionary)
        histories_dict = {
            f"pair{i}": convertHistoryToMetadata(hist)
            for i, hist in enumerate(histories)
        }
        self.metadata["history"] = histories_dict

        return zkArray, zkMeta

    def estimateFromIndivStamps(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        wfEstimator: WfEstimator,
        numCores: int = 1,
    ) -> np.array:
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
        """
        # Loop over individual donut stamps with a process pool
        args = [
            (donut, wfEstimator)
            for donut in itertools.chain(donutStampsExtra, donutStampsIntra)
        ]
        with mp.Pool(processes=numCores) as pool:
            results = pool.map(estimate_zk_single, args)

        zkList, zkMetaList, histories = zip(*results)
        zkMeta = {key: [] for key in zkMetaList[0].keys()}
        for zkMetaSingle in zkMetaList:
            for key, value in zkMetaSingle.items():
                zkMeta[key].append(value)

        zkArray = np.array(zkList)

        histories_dict = {}
        for i in range(len(donutStampsExtra)):
            histories_dict[f"extra{i}"] = convertHistoryToMetadata(histories[i])
        for i in range(len(donutStampsIntra)):
            histories_dict[f"intra{i}"] = convertHistoryToMetadata(
                histories[i + len(donutStampsExtra)]
            )
        self.metadata["history"] = histories_dict

        return zkArray, zkMeta

    def run(
        self,
        donutStampsExtra: DonutStamps,
        donutStampsIntra: DonutStamps,
        numCores: int = 1,
    ) -> np.ndarray:
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
            microns.
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
        if len(donutStampsExtra) > 0 and len(donutStampsIntra) > 0:
            zernikes, zkMeta = self.estimateFromPairs(
                donutStampsExtra, donutStampsIntra, wfEst, numCores=numCores
            )
        else:
            if wfEst.algo.requiresPairs:
                raise ValueError(
                    f"Wavefront algorithm `{wfEst.algo.__class__.__name__}` "
                    "requires pairs of donuts."
                )
            zernikes, zkMeta = self.estimateFromIndivStamps(
                donutStampsExtra, donutStampsIntra, wfEst, numCores=numCores
            )

        return pipeBase.Struct(zernikes=zernikes, wfEstInfo=zkMeta)
