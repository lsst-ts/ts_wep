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
import numpy as np
import pandas as pd

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
from lsst.utils.timer import timeMethod
from lsst.pipe.base import connectionTypes

from lsst.ts.wep.Utility import DefocalType
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.task.CutOutDonutsBase import (
    CutOutDonutsBaseTaskConnections,
    CutOutDonutsBaseTaskConfig,
    CutOutDonutsBaseTask,
)


class CutOutDonutsScienceSensorTaskConnections(
    CutOutDonutsBaseTaskConnections, dimensions=("detector", "instrument")
):
    exposures = connectionTypes.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="postISRCCD",
        multiple=True,
    )


class CutOutDonutsScienceSensorTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutDonutsScienceSensorTaskConnections,
):
    pass


class CutOutDonutsScienceSensorTask(CutOutDonutsBaseTask):
    """
    Run Zernike Estimation in full-array mode (FAM)
    """

    ConfigClass = CutOutDonutsScienceSensorTaskConfig
    _DefaultName = "CutOutDonutsScienceSensorTask"

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
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

        exposures = butlerQC.get(inputRefs.exposures)
        focusZVals = [exp.visitInfo.focusZ for exp in exposures]
        extraIdx, intraIdx = self.assignExtraIntraIdx(focusZVals[0], focusZVals[1])

        donutCats = butlerQC.get(inputRefs.donutCatalog)
        camera = butlerQC.get(inputRefs.camera)

        outputs = self.run(exposures, donutCats, camera)

        butlerQC.put(outputs.donutStampsExtra, outputRefs.donutStampsExtra[extraIdx])
        butlerQC.put(outputs.donutStampsIntra, outputRefs.donutStampsIntra[extraIdx])

    def assignExtraIntraIdx(self, focusZVal0, focusZVal1):
        """
        Identify which exposure in the list is the extra-focal and which
        is the intra-focal based upon `FOCUSZ` parameter in header.

        Parameters
        ----------
        focusZVal0 : float
            The `FOCUSZ` parameter from the first exposure.
        focusZVal1 : float
            The `FOCUSZ` parameter from the second exposure.

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
        """

        errorStr = "Must have one extra-focal and one intra-focal image."
        if focusZVal0 < 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 <= 0:
                raise ValueError(errorStr)
            extraExpIdx = 1
            intraExpIdx = 0
        elif focusZVal0 > 0:
            # Check that other image does not have same defocal direction
            if focusZVal1 >= 0:
                raise ValueError(errorStr)
            extraExpIdx = 0
            intraExpIdx = 1
        else:
            # Need to be defocal images ('FOCUSZ != 0')
            raise ValueError(errorStr)

        return extraExpIdx, intraExpIdx

    @timeMethod
    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalog: typing.List[pd.DataFrame],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        # Use exposure focusZ to find which is extra and intra
        focusZ0 = exposures[0].visitInfo.focusZ
        focusZ1 = exposures[1].visitInfo.focusZ

        # Get defocal distance from focusZ.
        self._checkAndSetOffset(np.abs(focusZ0))

        extraExpIdx, intraExpIdx = self.assignExtraIntraIdx(focusZ0, focusZ1)

        # Get the donut stamps from extra and intra focal images
        # The donut catalogs for each exposure should be the same
        # so we just pick the one for the first exposure
        cameraName = camera.getName()
        donutStampsExtra = self.cutOutStamps(
            exposures[extraExpIdx], donutCatalog[0], DefocalType.Extra, cameraName
        )
        donutStampsIntra = self.cutOutStamps(
            exposures[intraExpIdx], donutCatalog[0], DefocalType.Intra, cameraName
        )

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if len(donutStampsExtra) == 0:
            return pipeBase.Struct(
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
