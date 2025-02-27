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
    "ReassignCwfsCutoutsTaskConnections",
    "ReassignCwfsCutoutsTaskConfig",
    "ReassignCwfsCutoutsTask",
]

import lsst.obs.lsst as obs_lsst
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.fgcmcal.utilities import lookupStaticCalibrations
from lsst.pipe.base import connectionTypes


class ReassignCwfsCutoutsTaskConnections(
    pipeBase.PipelineTaskConnections, dimensions=("visit", "instrument")
):
    camera = connectionTypes.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=lookupStaticCalibrations,
    )
    donutStampsExtraIn = connectionTypes.Input(
        doc="Extra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtraCwfs",
        multiple=True,
    )
    donutStampsIntraIn = connectionTypes.Input(
        doc="Intra-focal Donut Postage Stamp Images with Intra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntraCwfs",
        multiple=True,
    )
    donutStampsIntraOut = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=True,
    )
    donutStampsExtraOut = connectionTypes.Output(
        doc="Extra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=True,
    )


class ReassignCwfsCutoutsTaskConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ReassignCwfsCutoutsTaskConnections
):
    pass


class ReassignCwfsCutoutsTask(pipeBase.PipelineTask):
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

        detectorIdArrIntra = np.array(
            [exp.dataId["detector"] for exp in inputRefs.donutStampsIntraIn]
        )
        detectorIdArrExtra = np.array(
            [exp.dataId["detector"] for exp in inputRefs.donutStampsExtraIn]
        )
        detectorIdArrOut = np.array(
            [exp.dataId["detector"] for exp in outputRefs.donutStampsExtraOut]
        )

        # Find cwfs detectors in the list of detectors being processed
        runExtraIds = sorted(set(detectorIdArrExtra).intersection(extraFocalIds))
        runIntraIds = sorted(set(detectorIdArrIntra).intersection(intraFocalIds))

        if len(runExtraIds) != len(runIntraIds):
            raise ValueError("Unequal number of intra and extra focal detectors.")

        for extraId, intraId in zip(runExtraIds, runIntraIds):
            if abs(extraId - intraId) != 1:
                raise ValueError("Intra and extra focal detectors not adjacent.")
            extraListIdx = np.where(detectorIdArrExtra == extraId)[0][0]
            intraListIdx = np.where(detectorIdArrIntra == intraId)[0][0]
            outputListIdx = np.where(detectorIdArrOut == extraId)[0][0]

            outputExtra = butlerQC.get(inputRefs.donutStampsExtraIn[extraListIdx])
            outputIntra = butlerQC.get(inputRefs.donutStampsIntraIn[intraListIdx])

            # Assign both outputs to the same dataId so that we can run
            # Zernike estimation fully in parallel through the dataIds
            # of the extra-focal detectors using CalcZernikesTask.
            butlerQC.put(outputExtra, outputRefs.donutStampsExtraOut[outputListIdx])
            butlerQC.put(outputIntra, outputRefs.donutStampsIntraOut[outputListIdx])
