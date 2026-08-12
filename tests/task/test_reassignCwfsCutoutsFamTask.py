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

import unittest
from collections.abc import Iterator
from unittest.mock import MagicMock

from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.ts.wep.task.reassignCwfsCutoutsFamTask import (
    ReassignCwfsCutoutsFamTask,
    ReassignCwfsCutoutsFamTaskConfig,
    ReassignCwfsCutoutsFamTaskConnections,
)


class FakeQuantaAdjuster:
    """Stand-in for `lsst.pipe.base.QuantaAdjuster`.

    Records the `add_input` and `remove_quantum` calls made by
    `adjust_all_quanta` so the reassignment logic can be checked without
    building a real quantum graph.
    """

    def __init__(self, dataIds: list[DataCoordinate]) -> None:
        self._dataIds = list(dataIds)
        self.removed: list[DataCoordinate] = []
        self.added: list[tuple[DataCoordinate, str, DataCoordinate]] = []

    def iter_data_ids(self) -> Iterator[DataCoordinate]:
        return iter(self._dataIds)

    def get_inputs(self, quantumDataId: DataCoordinate) -> dict[str, list[DataCoordinate]]:
        # Each quantum's own donutStampsIn input shares its data id.
        return {"donutStampsIn": [quantumDataId]}

    def add_input(
        self, quantumDataId: DataCoordinate, connectionName: str, datasetDataId: DataCoordinate
    ) -> None:
        self.added.append((quantumDataId, connectionName, datasetDataId))

    def remove_quantum(self, dataId: DataCoordinate) -> None:
        self.removed.append(dataId)


class TestReassignCwfsCutoutsFamTaskConnections(unittest.TestCase):
    def setUp(self) -> None:
        universe = DimensionUniverse()
        self.intraDataId = DataCoordinate.standardize(
            visit=100, detector=192, instrument="LSSTCam", universe=universe
        )
        self.extraDataId = DataCoordinate.standardize(
            visit=101, detector=191, instrument="LSSTCam", universe=universe
        )

    def _makeConnections(self, customQG: bool) -> ReassignCwfsCutoutsFamTaskConnections:
        config = ReassignCwfsCutoutsFamTaskConfig(customQG=customQG)
        return ReassignCwfsCutoutsFamTaskConnections(config=config)

    def testAdjustAllQuantaReassignsIntraFocalInputToNextVisit(self) -> None:
        connections = self._makeConnections(customQG=False)
        adjuster = FakeQuantaAdjuster([self.intraDataId, self.extraDataId])

        connections.adjust_all_quanta(adjuster)

        # The intra-focal quantum (visit=100, detector=192) is dropped...
        self.assertEqual(adjuster.removed, [self.intraDataId])
        # ...and its donutStampsIn dataset is added as an input to the
        # extra-focal quantum at visit+1, detector-1.
        self.assertEqual(adjuster.added, [(self.extraDataId, "donutStampsIn", self.intraDataId)])

    def testAdjustAllQuantaWithCustomQGOnlyDropsIntraFocalQuanta(self) -> None:
        connections = self._makeConnections(customQG=True)
        adjuster = FakeQuantaAdjuster([self.intraDataId, self.extraDataId])

        connections.adjust_all_quanta(adjuster)

        # With a custom quantum graph builder, the intra-focal quantum is
        # dropped but not reassigned since the builder already handles that.
        self.assertEqual(adjuster.removed, [self.intraDataId])
        self.assertEqual(adjuster.added, [])

    def testAdjustAllQuantaLeavesExtraFocalQuantaUntouched(self) -> None:
        connections = self._makeConnections(customQG=False)
        adjuster = FakeQuantaAdjuster([self.extraDataId])

        connections.adjust_all_quanta(adjuster)

        self.assertEqual(adjuster.removed, [])
        self.assertEqual(adjuster.added, [])


class TestReassignCwfsCutoutsFamTask(unittest.TestCase):
    def setUp(self) -> None:
        self.task = ReassignCwfsCutoutsFamTask(config=ReassignCwfsCutoutsFamTaskConfig())

    def _makeRef(self, detector: int) -> MagicMock:
        ref = MagicMock()
        ref.dataId = {"detector": detector}
        return ref

    def testRunQuantumOrdersStampsByDetectorWhenExtraFocalIsFirst(self) -> None:
        extraStamp = MagicMock(name="extraStamp")
        intraStamp = MagicMock(name="intraStamp")

        butlerQC = MagicMock()
        butlerQC.get.return_value = [extraStamp, intraStamp]

        inputRefs = MagicMock()
        inputRefs.donutStampsIn = [self._makeRef(191), self._makeRef(192)]
        outputRefs = MagicMock()

        self.task.runQuantum(butlerQC, inputRefs, outputRefs)

        butlerQC.put.assert_any_call(extraStamp, outputRefs.donutStampsExtraOut)
        butlerQC.put.assert_any_call(intraStamp, outputRefs.donutStampsIntraOut)

    def testRunQuantumOrdersStampsByDetectorWhenIntraFocalIsFirst(self) -> None:
        intraStamp = MagicMock(name="intraStamp")
        extraStamp = MagicMock(name="extraStamp")

        butlerQC = MagicMock()
        # Stamps come back in the same order as the refs, i.e. reversed
        # relative to the previous test.
        butlerQC.get.return_value = [intraStamp, extraStamp]

        inputRefs = MagicMock()
        inputRefs.donutStampsIn = [self._makeRef(192), self._makeRef(191)]
        outputRefs = MagicMock()

        self.task.runQuantum(butlerQC, inputRefs, outputRefs)

        butlerQC.put.assert_any_call(extraStamp, outputRefs.donutStampsExtraOut)
        butlerQC.put.assert_any_call(intraStamp, outputRefs.donutStampsIntraOut)


if __name__ == "__main__":
    unittest.main()
