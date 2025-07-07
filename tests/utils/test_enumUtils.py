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

from lsst.ts.wep.utils.enumUtils import BandLabel, EnumDict


class TestEnumUtils(unittest.TestCase):
    """Test the Enum utils."""

    def testEnumDict(self) -> None:
        # First create a dictionary with BandLabel enums as keys
        bandDict = {BandLabel(band): i for i, band in enumerate("ugrizy")}

        # We cannot access these using the band strings
        self.assertTrue(bandDict[BandLabel.LSST_R] == 2)
        with self.assertRaises(KeyError):
            bandDict["r"]

        # Convert this dictionary to an EnumDict
        bandDict2 = EnumDict(BandLabel, bandDict)
        self.assertTrue(bandDict2["r"] == 2)

        # Finally, make sense that we can create an empty EnumDict
        # and fill it using the string values
        bandDict3 = EnumDict(BandLabel)
        self.assertTrue(len(bandDict3) == 0)
        for key, val in bandDict.items():
            bandDict3[key.value] = val
        self.assertTrue(len(bandDict3) == 6)


if __name__ == "__main__":
    # Run the unit test
    unittest.main()
