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

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


def plotZernike(zkIdx, zk, unit, saveFilePath=None):
    """Plot the Zernike polynomials (zk).

    Parameters
    ----------
    zkIdx : list[int] or numpy.array[int]
        Index of zk.
    zk : numpy.array
        Zernike polynomials.
    unit : str
        Unit of Zernike polynomials.
    saveFilePath : str, optional
        File path to save the image. (the default is None.)
    """

    plt.plot(zkIdx, zk, marker="o", color="r", markersize=10)

    plt.xlabel("Zernike Index")
    plt.ylabel("Zernike coefficient (%s)" % unit)
    plt.grid()

    if saveFilePath is None:
        plt.show()
    else:
        plt.savefig(saveFilePath, bbox_inches="tight")
        plt.close()
