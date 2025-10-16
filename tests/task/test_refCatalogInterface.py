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

import os
import unittest

from lsst.daf.butler import Butler
from lsst.ts.wep.task.refCatalogInterface import RefCatalogInterface
from lsst.ts.wep.utils import getModulePath


class TestRefCatalogInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.boresightRa = 0.03
        self.boresightDec = -0.02
        self.boresightRotAng = 90.0
        self.refCatInterface = RefCatalogInterface(self.boresightRa, self.boresightDec, self.boresightRotAng)

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.butler = Butler.from_config(self.repoDir)

    def testGetHtmIds(self) -> None:
        """Test that the correct htmIds are returned."""

        # Test default radius
        self.assertEqual(len(self.refCatInterface.getHtmIds()), 56)

        # Test smaller radius
        smallRadIds = [131072, 188416, 196608, 253952]
        self.assertCountEqual(self.refCatInterface.getHtmIds(radius=0.2), smallRadIds)

    def testGetDataRefs(self) -> None:
        """Test that the dataRefs are gathered correctly."""

        htmIds = self.refCatInterface.getHtmIds()
        catalogName = "cal_ref_cat"
        collectionName = "refcats/gen2"
        dataRefs, dataIds = self.refCatInterface.getDataRefs(htmIds, self.butler, catalogName, collectionName)

        self.assertEqual(len(dataRefs), 7)
        self.assertEqual(len(dataIds), 7)

    def testGetDetectorWcs(self) -> None:
        """Test setting up a WCS for the pointing."""

        camera = self.butler.get("camera", instrument="LSSTCam", collections=["LSSTCam/calib/unbounded"])

        detector = camera["R22_S11"]
        detWcs = self.refCatInterface.getDetectorWcs(detector)
        wcsOrigin = detWcs.getSkyOrigin()

        self.assertAlmostEqual(wcsOrigin.getRa().asDegrees(), self.boresightRa)
        self.assertAlmostEqual(wcsOrigin.getDec().asDegrees(), self.boresightDec)
