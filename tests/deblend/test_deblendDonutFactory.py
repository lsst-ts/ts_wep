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

from lsst.ts.wep.deblend.deblendAdapt import DeblendAdapt
from lsst.ts.wep.deblend.deblendDonutFactory import DeblendDonutFactory
from lsst.ts.wep.utils import DeblendDonutType


class TestDeblendDonutFactory(unittest.TestCase):
    """Test the DeblendDonutFactory class."""

    def testCreateDeblendAdapt(self) -> None:
        deblendDonut = DeblendDonutFactory.createDeblendDonut(DeblendDonutType.Adapt)
        self.assertTrue(isinstance(deblendDonut, DeblendAdapt))

    def testCreateDeblendDonutWrongType(self) -> None:
        self.assertRaises(ValueError, DeblendDonutFactory.createDeblendDonut, "wrongType")


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
