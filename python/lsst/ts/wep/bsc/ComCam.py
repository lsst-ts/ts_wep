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

from lsst.obs.lsst import LsstCamMapper
from lsst.afw.cameraGeom import DetectorType

from lsst.ts.wep.bsc.CameraData import CameraData


class ComCam(CameraData):
    def __init__(self):
        """Initialize the commissioning camera class."""

        # The comcam's configuration here is approximated by taking the central
        # raft of lsst camera.
        super(ComCam, self).__init__(LsstCamMapper().camera)
        self._initDetectors(DetectorType.SCIENCE)

        # Remove the ccd data that are not belong to ComCam
        detectorList = [
            "R22_S02",
            "R22_S12",
            "R22_S22",
            "R22_S01",
            "R22_S11",
            "R22_S21",
            "R22_S00",
            "R22_S10",
            "R22_S20",
        ]

        self.setWfsCcdList(detectorList)

        wfsCorners = dict()
        ccdDims = dict()
        for detector in detectorList:
            wfsCorners[detector] = self.getWfsCorner(detector)
            ccdDims[detector] = self.getCcdDim(detector)

        self.setWfsCorners(wfsCorners)
        self.setCcdDims(ccdDims)


if __name__ == "__main__":
    pass
