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

from lsst.ts.wep.estimation import DanishAlgorithm, TieAlgorithm, WfAlgorithmFactory
from lsst.ts.wep.utils import WfAlgorithmName


class TestWfAlgorithmFactory(unittest.TestCase):
    """Test the WfAlgorithmFactory class."""

    def testCreateTieAlgorithm(self) -> None:
        # Make sure it returns the correct type
        self.assertIsInstance(WfAlgorithmFactory.createWfAlgorithm("tie"), TieAlgorithm)
        self.assertIsInstance(
            WfAlgorithmFactory.createWfAlgorithm(WfAlgorithmName.TIE),
            TieAlgorithm,
        )

        # Make sure config parameters are propagated
        algo = WfAlgorithmFactory.createWfAlgorithm("tie", {"maxIter": 30})
        self.assertEqual(algo.maxIter, 30)

    def testCreateDanishAlgorithm(self) -> None:
        # Make sure it returns the correct type
        self.assertIsInstance(WfAlgorithmFactory.createWfAlgorithm("danish"), DanishAlgorithm)
        self.assertIsInstance(
            WfAlgorithmFactory.createWfAlgorithm(WfAlgorithmName.Danish),
            DanishAlgorithm,
        )

        # Make sure config parameters are propagated
        algo = WfAlgorithmFactory.createWfAlgorithm("danish", {"lstsqKwargs": dict(ftol=1e-5)})
        self.assertEqual(algo.lstsqKwargs["ftol"], 1e-5)

    def testBadAlgoName(self) -> None:
        with self.assertRaises(ValueError):
            WfAlgorithmFactory.createWfAlgorithm("fake")


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
