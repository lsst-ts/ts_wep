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

import lsst.obs.lsst as obs_lsst
import numpy as np
from lsst.afw.cameraGeom import FIELD_ANGLE


def getMappingFromFieldXY(fieldXY: np.ndarray) -> dict:
    """Map the sensor and field index based on the distance between the
    positions of sensor and field.

    Parameters
    ----------
    fieldXY : numpy.ndarray
        Field (x, y) in degree. This is a nx2 matrix that the first column
        is the x position and the second column is the y position.

    Returns
    -------
    dict
        Mapping data. The key is the sensor name and the value is the field
        index.
    """

    camera = obs_lsst.LsstCam().getCamera()

    # Get detector information
    nameList = []
    xList = []
    yList = []
    for det in camera:
        nameList.append(det.getName())
        centerPt = det.getCenter(FIELD_ANGLE)
        # Transposed to match output to original output
        xList.append(np.degrees(centerPt[1]))
        yList.append(np.degrees(centerPt[0]))
    sensorXY = np.array([xList, yList]).T

    # Calculate the distance matrix (sensor by field)
    fieldX = fieldXY[:, 0]
    disM = np.zeros((len(xList), len(fieldX)))
    for ii in range(len(fieldX)):
        vector = sensorXY - fieldXY[ii, :]
        dis = np.linalg.norm(vector, axis=1)
        disM[:, ii] = dis

    # Find the minimun distance for each sensor and assign the field index
    idxList = np.zeros(len(nameList), dtype="int")
    for ii in range(len(idxList)):
        idxList[ii] = np.argmin(disM[ii, :])

    # Collect the information
    mapping = dict()
    for ii in range(len(idxList)):
        mapping[nameList[ii]] = idxList[ii]

    return mapping


if __name__ == "__main__":
    # Set the 35 field points
    nArm = 6
    armLen = [0.379, 0.841, 1.237, 1.535, 1.708]
    fieldWFSx = [1.176, -1.176, -1.176, 1.176]
    fieldWFSy = [1.176, 1.176, -1.176, -1.176]
    pointAngle = np.arange(nArm) * (2 * np.pi) / nArm
    fieldX = np.concatenate([np.zeros(1), np.kron(armLen, np.cos(pointAngle)), fieldWFSx])
    fieldY = np.concatenate([np.zeros(1), np.kron(armLen, np.sin(pointAngle)), fieldWFSy])

    fieldXY = np.array([fieldX, fieldY]).T

    # Do the mapping
    mapping = getMappingFromFieldXY(fieldXY)

    # Print the mapping
    for sensorName, idxField in mapping.items():
        print(sensorName, idxField)
