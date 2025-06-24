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

__all__ = [
    "CutOutDonutsScienceSensorTaskConnections",
    "CutOutDonutsScienceSensorTaskConfig",
    "CutOutDonutsScienceSensorTask",
]

import typing

import lsst.afw.cameraGeom
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
from astropy.table import QTable
from lsst.ts.wep.task.cutOutDonutsBase import (
    CutOutDonutsBaseTask,
    CutOutDonutsBaseTaskConfig,
    CutOutDonutsBaseTaskConnections,
)
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.task.generateDonutCatalogUtils import convertDictToVisitInfo
from lsst.ts.wep.utils import DefocalType
from lsst.utils.timer import timeMethod

from .pairTask import ExposurePairer


class CutOutDonutsScienceSensorTaskConnections(
    CutOutDonutsBaseTaskConnections, dimensions=("detector", "instrument")
):
    exposures = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
        multiple=True,
    )
    donutVisitPairTable = ct.Input(
        doc="Visit pair table",
        dimensions=("instrument",),
        storageClass="AstropyTable",
        name="donutVisitPairTable",
    )
    donutStampsExtra = ct.Output(
        doc="Extra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
    )
    donutStampsIntra = ct.Output(
        doc="Intra-focal Donut Postage Stamp Images",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.pairer.target._needsPairTable:
            del self.donutVisitPairTable
        if config.pairer.target._needsGroupDimension:
            # Note that when running with the GroupPairer you might also need
            # to change the name of the task in the pipeline yaml file (the
            # yaml key under `tasks` that points to
            # `CutOutDonutsScienceSensorTask`) to avoid dataset type dimension
            # conflicts with CutOutDonutsScienceSensorTask executions that
            # were run without the group pairer (and hence without the `group`
            # input dimension).
            self.dimensions.add("group")


class CutOutDonutsScienceSensorTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutDonutsScienceSensorTaskConnections,
):
    pairer = pexConfig.ConfigurableField(
        target=ExposurePairer,
        doc="Task to pair up intra- and extra-focal exposures",
    )


class CutOutDonutsScienceSensorTask(CutOutDonutsBaseTask):
    """
    Run Zernike Estimation in full-array mode (FAM)
    """

    ConfigClass = CutOutDonutsScienceSensorTaskConfig
    _DefaultName = "CutOutDonutsScienceSensorTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("pairer")

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We need to be able to take pairs of detectors from the full
        set of detector exposures and run the task. Then we need to put
        the outputs back into the butler repository with
        the appropriate butler dataIds.

        For the `outputZernikesRaw` and `outputZernikesAvg`
        we only have one set of values per pair of wavefront detectors
        so we put this in the dataId associated with the
        extra-focal detector.
        """
        camera = butlerQC.get(inputRefs.camera)

        visitInfoDict = {
            v.dataId["visit"]: convertDictToVisitInfo(
                butlerQC.get(v).meta["visit_info"]
            )
            for v in inputRefs.donutCatalog
        }
        exposureHandleDict = {v.dataId["exposure"]: v for v in inputRefs.exposures}
        donutCatalogHandleDict = {v.dataId["visit"]: v for v in inputRefs.donutCatalog}
        donutStampsIntraHandleDict = {
            v.dataId["visit"]: v for v in outputRefs.donutStampsIntra
        }
        donutStampsExtraHandleDict = {
            v.dataId["visit"]: v for v in outputRefs.donutStampsExtra
        }

        if hasattr(inputRefs, "donutVisitPairTable"):
            pairs = self.pairer.run(
                visitInfoDict, butlerQC.get(inputRefs.donutVisitPairTable)
            )
        else:
            pairs = self.pairer.run(visitInfoDict)
        for pair in pairs:
            exposures = butlerQC.get(
                [exposureHandleDict[k] for k in [pair.intra, pair.extra]]
            )
            donutCats = butlerQC.get(
                [donutCatalogHandleDict[k] for k in [pair.intra, pair.extra]]
            )
            outputs = self.run(exposures, donutCats, camera)
            butlerQC.put(
                outputs.donutStampsExtra, donutStampsExtraHandleDict[pair.extra]
            )
            butlerQC.put(
                outputs.donutStampsIntra,
                donutStampsIntraHandleDict[
                    pair.extra
                ],  # Intentionally use extra id for intra stamps here
            )

    def assignExtraIntraIdx(self, focusZVal0, focusZVal1, cameraName):
        """
        Identify which exposure in the list is the extra-focal and which
        is the intra-focal based upon `FOCUSZ` parameter in header.

        Parameters
        ----------
        focusZVal0 : float
            The `FOCUSZ` parameter from the first exposure.
        focusZVal1 : float
            The `FOCUSZ` parameter from the second exposure.
        cameraName : str
            Name of camera for the exposure. Can accept "LSSTCam",
            "LSSTComCam", "LSSTComCamSim", "LATISS".

        Returns
        -------
        int
            Index in list which is extra-focal image.
        int
            Index in list which is intra-focal image.

        Raises
        ------
        ValueError
            Exposures must be a pair with one intra-focal
            and one extra-focal image.
        ValueError
            Invalid cameraName variable.
        """

        if cameraName in ["LSSTCam", "LSSTComCam", "LSSTComCamSim"]:
            if focusZVal0 < focusZVal1:
                extraExpIdx = 1
                intraExpIdx = 0
            else:
                extraExpIdx = 0
                intraExpIdx = 1
        elif cameraName == "LATISS":
            errorStr = "Must have two images with different FOCUSZ parameter."
            if focusZVal0 != focusZVal1:
                # the exposure with smallest focusZVal is extra-focal
                # just find the smallest value...
                focuszPair = (focusZVal0, focusZVal1)
                extraExp = min(focuszPair)
                intraExp = max(focuszPair)

                expDicIdx = {focusZVal0: 0, focusZVal1: 1}
                intraExpIdx = expDicIdx[intraExp]
                extraExpIdx = expDicIdx[extraExp]
            else:
                # Need two different focusz values
                raise ValueError(errorStr)
        else:
            errorStr = str(
                f"Invalid cameraName parameter: {cameraName}. Camera must  "
                "be one of: 'LSSTCam', 'LSSTComCam', 'LSSTComCamSim' or 'LATISS'",
            )
            raise ValueError(errorStr)

        return extraExpIdx, intraExpIdx

    @timeMethod
    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalog: typing.List[QTable],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:
        # Determine which exposure is intra-/extra-focal
        focusZ = (exposures[0].visitInfo.focusZ, exposures[1].visitInfo.focusZ)
        cameraName = camera.getName()
        extraExpIdx, intraExpIdx = self.assignExtraIntraIdx(*focusZ, cameraName)

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx],
            donutCatalog[extraExpIdx],
            DefocalType.Extra,
            cameraName,
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx],
            donutCatalog[intraExpIdx],
            DefocalType.Intra,
            cameraName,
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0:
            donutStampsExtra = DonutStamps([])
            donutStampsExtra = self.addVisitLevelMetadata(
                exposures[extraExpIdx],
                donutStampsExtra,
                donutCatalog[extraExpIdx],
                DefocalType.Extra,
            )
            donutStampsExtra.use_archive = False

        if len(donutStampsIntra) == 0:
            donutStampsIntra = DonutStamps([])
            donutStampsIntra = self.addVisitLevelMetadata(
                exposures[intraExpIdx],
                donutStampsIntra,
                donutCatalog[intraExpIdx],
                DefocalType.Intra,
            )
            donutStampsIntra.use_archive = False

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
