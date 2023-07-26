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

from lsst.ts.wep.cwfs.donutTemplateFactory import DonutTemplateFactory
from lsst.ts.wep.cwfs.donutTemplateModel import DonutTemplateModel
from lsst.ts.wep.cwfs.donutTemplatePhosim import DonutTemplatePhosim
from lsst.ts.wep.utility import DonutTemplateType


class TestTemplateMakerFactory(unittest.TestCase):
    """Test the TemplateMakerFactory class."""

    def testCreateTemplateModel(self):
        donutTemplate = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Model
        )
        self.assertTrue(isinstance(donutTemplate, DonutTemplateModel))

    def testCreateTemplatePhosim(self):
        donutTemplate = DonutTemplateFactory.createDonutTemplate(
            DonutTemplateType.Phosim
        )
        self.assertTrue(isinstance(donutTemplate, DonutTemplatePhosim))

    def testCreateCentroidFindWrongType(self):
        self.assertRaises(
            ValueError, DonutTemplateFactory.createDonutTemplate, "wrongType"
        )


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
