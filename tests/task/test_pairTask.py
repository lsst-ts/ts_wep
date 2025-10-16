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

import typing
import unittest

import lsst.geom
import lsst.pex.config as pexConfig
from lsst.afw.coord import Observatory
from lsst.afw.image import VisitInfo
from lsst.daf.base import DateTime
from lsst.ts.wep.task import (
    ExposurePairer,
    ExposurePairerConfig,
    GroupPairer,
    GroupPairerConfig,
)


class TestPairTask(unittest.TestCase):
    def _createVisitInfoDict(
        self,
        ra_deg_start: float = 0.0,
        dec_deg_start: float = 0.0,
        time_mjd_start: float = 60432.0,
        rot_deg_start: float = 0.0,
        radec_sep_deg: float = 0.0,
        time_sep_sec: float = 0.0,
        rot_sep_deg: float = 0.0,
        intra_id: int = 0,
        extra_id: int = 1,
    ) -> typing.Dict[int, VisitInfo]:
        intra_boresight = lsst.geom.SpherePoint(ra_deg_start, dec_deg_start, lsst.geom.degrees)
        intra_mjd = DateTime(time_mjd_start)
        intra_focus_z = -1.5
        intra_rtp = lsst.geom.Angle(rot_deg_start, lsst.geom.degrees)

        intra_visit_info = VisitInfo(
            date=intra_mjd,
            boresightRaDec=intra_boresight,
            boresightRotAngle=intra_rtp,
            focusZ=intra_focus_z,
            instrumentLabel="LSSTCam",
            observatory=Observatory(
                lsst.geom.Angle(-30.2446, lsst.geom.degrees),
                lsst.geom.Angle(-70.7494, lsst.geom.degrees),
                2663,
            ),
            era=lsst.geom.Angle(1.7487, lsst.geom.radians),
        )

        extra_boresight = lsst.geom.SpherePoint(
            ra_deg_start, dec_deg_start + radec_sep_deg, lsst.geom.degrees
        )

        extra_mjd = DateTime(time_mjd_start + time_sep_sec / 86400)
        extra_focus_z = 1.5
        extra_rtp = lsst.geom.Angle(rot_deg_start + rot_sep_deg, lsst.geom.degrees)

        extra_visit_info = VisitInfo(
            date=extra_mjd,
            boresightRaDec=extra_boresight,
            boresightRotAngle=extra_rtp,
            focusZ=extra_focus_z,
            instrumentLabel="LSSTCam",
            observatory=Observatory(
                lsst.geom.Angle(-30.2446, lsst.geom.degrees),
                lsst.geom.Angle(-70.7494, lsst.geom.degrees),
                2663,
            ),
            era=lsst.geom.Angle(1.7487, lsst.geom.radians),
        )

        return {intra_id: intra_visit_info, extra_id: extra_visit_info}

    def testExposurePairer(self) -> None:
        task_config = ExposurePairerConfig()
        task = ExposurePairer(config=task_config)

        visit_info_dict = self._createVisitInfoDict()

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)
        self.assertEqual(task_output[0].intra, 0)
        self.assertEqual(task_output[0].extra, 1)

    def testExposurePairerRaDecThresh(self) -> None:
        task_config = ExposurePairerConfig(pointingThreshold=2.0 * 3600)
        task = ExposurePairer(config=task_config)

        visit_info_dict = self._createVisitInfoDict(radec_sep_deg=1.0)
        visit_info_dict_same_radec = self._createVisitInfoDict(ra_deg_start=25.0)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)

        task_config = ExposurePairerConfig(pointingThreshold=0.9 * 3600)
        task = ExposurePairer(config=task_config)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 0)

        visit_info_dict[2] = visit_info_dict_same_radec[0]
        visit_info_dict[3] = visit_info_dict_same_radec[1]

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)
        self.assertEqual(task_output[0].intra, 2)
        self.assertEqual(task_output[0].extra, 3)

    def testExposurePairerTimeThresh(self) -> None:
        task_config = ExposurePairerConfig(timeThreshold=65)
        task = ExposurePairer(config=task_config)

        visit_info_dict = self._createVisitInfoDict(time_sep_sec=61)
        visit_info_dict_same_time = self._createVisitInfoDict(time_mjd_start=60432.1)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)

        task_config = ExposurePairerConfig(timeThreshold=60)
        task = ExposurePairer(config=task_config)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 0)

        visit_info_dict[2] = visit_info_dict_same_time[0]
        visit_info_dict[3] = visit_info_dict_same_time[1]

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)
        self.assertEqual(task_output[0].intra, 2)
        self.assertEqual(task_output[0].extra, 3)

    def testExposurePairerRotationThresh(self) -> None:
        task_config = ExposurePairerConfig(rotationThreshold=2.0)
        task = ExposurePairer(config=task_config)

        visit_info_dict = self._createVisitInfoDict(rot_sep_deg=1.0)
        visit_info_dict_same_rot = self._createVisitInfoDict(rot_deg_start=25.0)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)

        task_config = ExposurePairerConfig(rotationThreshold=0.9)
        task = ExposurePairer(config=task_config)

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 0)

        visit_info_dict[2] = visit_info_dict_same_rot[0]
        visit_info_dict[3] = visit_info_dict_same_rot[1]

        task_output = task.run(visit_info_dict)

        self.assertEqual(len(task_output), 1)
        self.assertEqual(task_output[0].intra, 2)
        self.assertEqual(task_output[0].extra, 3)

        task_config = ExposurePairerConfig(ignoreThresholds=True)
        task = ExposurePairer(config=task_config)

        task_output = task.run(visit_info_dict)
        self.assertEqual(len(task_output), 2)

    def testExposurePairerForceUniquePairs(self) -> None:
        task_config = ExposurePairerConfig()
        task = ExposurePairer(config=task_config)

        visit_info_dict = self._createVisitInfoDict()

        visit_info_dict_multiple_intra = {}
        visit_info_dict_multiple_intra[0] = visit_info_dict[0]
        visit_info_dict_multiple_intra[1] = visit_info_dict[1]
        visit_info_dict_multiple_intra[2] = visit_info_dict[1]

        task_output = task.run(visit_info_dict_multiple_intra)

        self.assertEqual(len(task_output), 1)
        self.assertEqual(task_output[0].intra, 0)
        self.assertEqual(task_output[0].extra, 1)

        task_config = ExposurePairerConfig(forceUniquePairs=False)
        task = ExposurePairer(config=task_config)

        task_output = task.run(visit_info_dict_multiple_intra)

        self.assertEqual(len(task_output), 2)
        self.assertEqual(task_output[0].intra, 0)
        self.assertEqual(task_output[0].extra, 1)
        self.assertEqual(task_output[1].intra, 0)
        self.assertEqual(task_output[1].extra, 2)

    def testGroupPairerWrongNumberInGroup(self) -> None:
        task_config = GroupPairerConfig()
        task = GroupPairer(config=task_config)

        # Test 4,3,3,2 VisitInfos.  Should succeed if len is 2 or 3
        visit_info_dict = self._createVisitInfoDict()
        visit_info_dict.update(self._createVisitInfoDict(intra_id=2, extra_id=3))

        while visit_info_dict:
            if len(visit_info_dict) in (2, 3):
                with self.assertNoLogs(logger=task.log.name, level="WARNING"):
                    task.run(visit_info_dict)
            else:
                with self.assertLogs(logger=task.log.name, level="WARNING"):
                    task.run(visit_info_dict)
            visit_info_dict.popitem()

    def testGroupPairerIgnoreThresholds(self) -> None:
        task_config = GroupPairerConfig(ignoreThresholds=False)
        with self.assertRaises(pexConfig.FieldValidationError):
            GroupPairer(config=task_config)
