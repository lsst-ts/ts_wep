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

from lsst.ts.wep.Utility import CentroidFindType
from lsst.ts.wep.cwfs.CentroidFindFactory import CentroidFindFactory
from lsst.ts.wep.cwfs.CentroidRandomWalk import CentroidRandomWalk
from lsst.ts.wep.cwfs.CentroidOtsu import CentroidOtsu
from lsst.ts.wep.cwfs.CentroidConvolveTemplate import CentroidConvolveTemplate


class TestCentroidFindFactory(unittest.TestCase):
    """Test the CentroidFindFactory class."""

    def testCreateCentroidFindRandomWalk(self):

        centroidFind = CentroidFindFactory.createCentroidFind(
            CentroidFindType.RandomWalk
        )
        self.assertTrue(isinstance(centroidFind, CentroidRandomWalk))

    def testCreateCentroidFindOtsu(self):

        centroidFind = CentroidFindFactory.createCentroidFind(CentroidFindType.Otsu)
        self.assertTrue(isinstance(centroidFind, CentroidOtsu))

    def testCreateCentroidFindConvolveTemplate(self):

        centroidFind = CentroidFindFactory.createCentroidFind(
            CentroidFindType.ConvolveTemplate
        )
        self.assertTrue(isinstance(centroidFind, CentroidConvolveTemplate))

    def testCreateCentroidFindWrongType(self):

        self.assertRaises(
            ValueError, CentroidFindFactory.createCentroidFind, "wrongType"
        )


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
