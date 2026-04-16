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

    def testBadTemperature(self) -> None:
        """Test that bad temperature raises error."""
        with self.assertRaises(ValueError):
            AiDonutAlgorithm(temperature=-1.0)
        with self.assertRaises(ValueError):
            AiDonutAlgorithm(temperature=0.0)
        with self.assertRaises(TypeError):
            AiDonutAlgorithm(temperature="invalid")

    def testBadNollIndices(self) -> None:
        """Test that bad noll indices raise error."""
        algo = AiDonutAlgorithm()
        _, intra, extra = forwardModelPair(seed=1234)
        with self.assertRaises(ValueError):
            algo.estimateZk(intra, extra, nollIndices=[4, 5, 6, 123, 124])

    def testOutputs(self) -> None:
        """Test the algorithm returns expected outputs."""
        algo = AiDonutAlgorithm()
        _, intra, extra = forwardModelPair(seed=1234)
        nollIndices = [4, 5, 6]
        zk, zkMeta = algo.estimateZk(intra, extra, nollIndices=nollIndices)
        self.assertEqual(zk.shape[0], len(nollIndices))
        self.assertIn("fwhm", zkMeta)
        self.assertIn("weight", zkMeta)

    def testHistory(self) -> None:
        """Test that history is populated correctly."""
        algo = AiDonutAlgorithm()
        _, intra, extra = forwardModelPair(seed=1234)
        nollIndices = [4, 5, 6]
        algo.estimateZk(intra, extra, nollIndices=nollIndices, saveHistory=True)
        self.assertIn("modelPath", algo.history)
        self.assertIn("device", algo.history)
        self.assertIn("modelNollIndices", algo.history)
        for defocalType in [intra.defocalType.value, extra.defocalType.value]:
            self.assertIn(defocalType, algo.history)
            self.assertIn("zk", algo.history[defocalType])
            self.assertIn("zkScore", algo.history[defocalType])
            self.assertIn("fwhm", algo.history[defocalType])
        self.assertIn("nollIndices", algo.history)
        self.assertIn("zk", algo.history)
        self.assertIn("fwhm", algo.history)
        self.assertIn("weight", algo.history)
