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

from lsst.ts.wep.utils.testUtils import enforce_single_threading

enforce_single_threading()

from lsst.ts.wep.estimation import AiDonutAlgorithm  # noqa: E402
from lsst.ts.wep.utils.modelUtils import forwardModelPair  # noqa: E402


class TestAiDonutAlgorithm(unittest.TestCase):
    """Test DanishAlgorithm."""

    def testBadDevice(self) -> None:
        """Test that bad device raises error."""
        with self.assertRaises(ValueError):
            AiDonutAlgorithm(modelPath="model.pt", device="invalid_device")

    def testBadModelPath(self) -> None:
        """Test that bad model path raises error."""
        with self.assertRaises(FileNotFoundError):
            AiDonutAlgorithm(modelPath="non_existent_model.pt", device="cpu")

    def testBadNollIndices(self) -> None:
        """Test that bad noll indices raise error."""
        algo = AiDonutAlgorithm()
        _, intra, extra = forwardModelPair(seed=1234)
        with self.assertRaises(ValueError):
            algo.estimateZk(intra, extra, nollIndices=[4, 5, 6, 123, 124])
