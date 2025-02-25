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

__all__ = ["reassignCwfsCutoutsTaskConfig", "reassignCwfsCutoutsTask"]

import typing
import numpy as np
import lsst.pipe.base as pipeBase
import lsst.obs.lsst as obs_lsst
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.utils.timer import timeMethod
from lsst.pipe.base import connectionTypes

class ReassignCwfsCutoutsTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("visit", "detector", "instrument")
):
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    donutStampsIntraIn = connectionTypes.Input(
        doc="Intra-focal Donut Postage Stamp Images with Intra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraIn",
        multiple=True,
    )
    donutStampsIntraOut = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )


class ReassignCwfsCutoutsTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ReassignCwfsCutoutsTaskConnections
):
    pass

class ReassignCwfsCutoutsTask(
    pipeBase.PipelineTaskConnections, dimensions=("exposure", "instrument")
):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = ReassignCwfsCutoutsTaskConfig
    _DefaultName = "ReassignCwfsCutoutsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set which sensors are extra and intra focal
        # See LCA-13381 for definition
        self.extraFocalNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        self.intraFocalNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

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

        # Get the detector IDs for the wavefront sensors so
        # that we can appropriately match up pairs of detectors
        if camera.getName() == "LSSTCam":
            detectorMap = (
                obs_lsst.translators.lsstCam.LsstCamTranslator.detector_mapping()
            )
        else:
            raise ValueError(f"{camera.getName()} is not a valid camera name.")

        extraFocalIds = [detectorMap[detName][0] for detName in self.extraFocalNames]
        intraFocalIds = [detectorMap[detName][0] for detName in self.intraFocalNames]

        detectorIdArr = np.array(
            [exp.dataId["detector"] for exp in inputRefs.donutStampsIntraIn]
        )

        # Find cwfs detectors in the list of detectors being processed
        runIntraIds = list(set(detectorIdArr).intersection(intraFocalIds))
        runIntraIds.sort()

        for intraId in runIntraIds:
            intraListIdx = np.where(detectorIdArr == intraId)[0][0]
            stampInputs = butlerQC.get(
                inputRefs.donutStampsIntraIn[intraListIdx]
            )
            # each time we pass exactly one pair of
            # exposures and donut catalogs
            extraListIdx = intraListIdx - 1
            if extraListIdx not in extraFocalIds:
                raise ValueError("Extrafocal Ids are not correct for given camera.")
            butlerQC.put(
                stampInputs, outputRefs.donutStampsIntra[intraListIdx-1]
            )
