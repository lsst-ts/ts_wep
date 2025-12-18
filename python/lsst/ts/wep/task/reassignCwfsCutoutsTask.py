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

from typing import Any

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.daf.butler import DataCoordinate
from lsst.pipe.base import connectionTypes

intra_focal_ids = set([192, 196, 200, 204])
extra_focal_ids = set([191, 195, 199, 203])


class ReassignCwfsCutoutsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "detector", "instrument"),  # type: ignore
):
    donutStampsIn = connectionTypes.Input(
        doc="Donut Postage Stamp Images with either Intra-focal or Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsCwfs",
        multiple=True,
    )
    donutStampsIntraOut = connectionTypes.Output(
        doc="Intra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsIntra",
        multiple=False,
    )
    donutStampsExtraOut = connectionTypes.Output(
        doc="Extra-focal Donut Postage Stamp Images with Extra-focal detector id.",
        dimensions=("visit", "detector", "instrument"),
        storageClass="StampsBase",
        name="donutStampsExtra",
        multiple=False,
    )

    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """This will drop intra quanta and assign
        them to the extra detector quanta
        """
        to_do = set(adjuster.iter_data_ids())
        seen = set()
        while to_do:
            data_id = to_do.pop()
            # Make sure the intra focal data id is not processed
            # by this task. The way RA runs the custom QG builder
            # will ensure the extra focal quantum has the intra
            # focal input.
            if data_id["detector"] in extra_focal_ids:
                seen.add(data_id)
            elif (data_id["detector"] in intra_focal_ids) and (not self.config.customQG):
                extra_focal_data_id = DataCoordinate.standardize(
                    data_id, detector=int(data_id["detector"]) - 1
                )

                assert extra_focal_data_id in seen or extra_focal_data_id in to_do, (
                    f"DataId {extra_focal_data_id} not found in seen or to_do sets."
                )

                inputs = adjuster.get_inputs(data_id)
                adjuster.add_input(extra_focal_data_id, "donutStampsIn", inputs["donutStampsIn"][0])
                adjuster.remove_quantum(data_id)

            else:
                adjuster.remove_quantum(data_id)


class ReassignCwfsCutoutsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ReassignCwfsCutoutsTaskConnections,  # type: ignore
):
    customQG: pexConfig.Field = pexConfig.Field[bool](
        doc="Whether this task is being run with a custom quantum graph builder. ",
        default=False,
    )


class ReassignCwfsCutoutsTask(pipeBase.PipelineTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = ReassignCwfsCutoutsTaskConfig
    _DefaultName = "ReassignCwfsCutoutsTask"
    config: ReassignCwfsCutoutsTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
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
        stamps = butlerQC.get(inputRefs.donutStampsIn)

        # We need to ensure we always have the stamps in the correct order
        # We know extra < intra
        detectors = [ref.dataId["detector"] for ref in inputRefs.donutStampsIn]
        if detectors[0] < detectors[1]:
            stamps.reverse()
        intra_stamp, extra_stamp = stamps

        butlerQC.put(extra_stamp, outputRefs.donutStampsExtraOut)
        butlerQC.put(intra_stamp, outputRefs.donutStampsIntraOut)
