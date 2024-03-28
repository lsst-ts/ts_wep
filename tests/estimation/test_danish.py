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

import numpy as np
from lsst.ts.wep import Image, ImageMapper
from lsst.ts.wep.estimation import DanishAlgorithm
from scipy.ndimage import gaussian_filter


class TestDanishAlgorithm(unittest.TestCase):
    """Test DanishAlgorithm."""

    @staticmethod
    def _createData(seed: int = 1234):
        # Create some random Zernikes
        rng = np.random.default_rng(seed)
        zkTrue = rng.normal(0, 1e-5 / np.arange(1, 20) ** 2, size=19)
        zkTrue = np.clip(zkTrue, -1e-6, +1e-6)

        # Sample a random seeing
        seeing = rng.uniform(0.1, 1)  # arcseconds
        seeing /= 0.5  # arcseconds -> pixels

        # Create a pair of images
        mapper = ImageMapper()

        intraStamp = mapper.mapPupilToImage(
            Image(
                np.zeros((180, 180)),
                (0, -1),
                "intra",
                "r",
                blendOffsets=[[70, 85]],
            ),
            zkTrue,
        )
        intraStamp.image *= rng.uniform(50, 200)
        intraStamp.image = gaussian_filter(intraStamp.image, seeing)
        intraStamp.image += rng.normal(scale=np.sqrt(intraStamp.image))
        intraStamp.image += rng.normal(scale=10, size=intraStamp.image.shape)

        extraStamp = mapper.mapPupilToImage(
            Image(
                np.zeros((180, 180)),
                (0, -1),
                "extra",
                "r",
            ),
            zkTrue,
        )
        extraStamp.image *= rng.uniform(50, 200)
        extraStamp.image = gaussian_filter(extraStamp.image, seeing)
        extraStamp.image += rng.normal(scale=np.sqrt(extraStamp.image))
        extraStamp.image += rng.normal(scale=15, size=extraStamp.image.shape)

        # Return the Zernikes and both images
        return zkTrue, intraStamp, extraStamp

    def testBadLstsqKwargs(self):
        for kwarg in ["fun", "x0", "jac", "args"]:
            with self.assertRaises(KeyError):
                DanishAlgorithm(lstsqKwargs={kwarg: None})

    def testGoodLstsqKwargs(self):
        # Create estimator
        dan = DanishAlgorithm(lstsqKwargs={"max_nfev": 1})

        # Create some data
        zkTrue, intra, extra = self._createData()

        # Estimate Zernikes
        dan.estimateZk(intra, extra, saveHistory=True)

        # Check that nfev in the algorithm history equals 1
        for key, hist in dan.history.items():
            if key != "zk":
                self.assertEqual(hist["lstsqResult"]["nfev"], 1)

    def testAccuracy(self):
        # Create estimator
        dan = DanishAlgorithm()

        # Try several different random seeds
        for seed in [12345, 23451, 34512, 45123, 51234]:
            # Get the test data
            zkTrue, intra, extra = self._createData(seed)

            # Test estimation with pairs and single donuts:
            for images in [[intra, extra], [intra], [extra]]:
                # Estimate Zernikes (in meters)
                zkEst = dan.estimateZk(*images)

                # Check that results are fairly accurate
                self.assertLess(np.sqrt(np.sum((zkEst - zkTrue) ** 2)), 0.35e-6)
