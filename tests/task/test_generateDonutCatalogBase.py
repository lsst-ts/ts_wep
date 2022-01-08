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
import numpy as np

import lsst.geom
from lsst.daf import butler as dafButler
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.GenerateDonutCatalogBase import (
    GenerateDonutCatalogBaseTask,
    GenerateDonutCatalogBaseConfig,
)


class TestGenerateDonutCatalogBase(unittest.TestCase):
    def setUp(self):

        self.config = GenerateDonutCatalogBaseConfig()
        self.task = GenerateDonutCatalogBaseTask(config=self.config, name="Base Task")

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.centerRaft = ["R22_S10", "R22_S11"]

        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

    def _getRefCat(self):

        refCatList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats"]
        ).expanded()
        for ref in datasetGenerator:
            refCatList.append(self.butler.getDeferred(ref, collections=["refcats"]))

        return refCatList

    def validateConfigs(self):

        self.config.boresightRa = 0.03
        self.config.boresightDec = -0.02
        self.config.boresightRotAng = 90.0
        self.config.filterName = "r"
        self.task = GenerateDonutCatalogBaseTask(config=self.config)

        self.assertEqual(self.task.boresightRa, 0.03)
        self.assertEqual(self.task.boresightDec, -0.02)
        self.assertEqual(self.task.boresightRotAng, 90.0)
        self.assertEqual(self.task.filterName, "r")

    def testFilterResults(self):

        refCatList = self._getRefCat()
        refCat = self.butler.get(
            "cal_ref_cat", dataId=refCatList[0].dataId, collections=["refcats"]
        )
        testDataFrame = refCat.asAstropy().to_pandas()
        filteredDataFrame = self.task.filterResults(testDataFrame)
        np.testing.assert_array_equal(filteredDataFrame, testDataFrame)

    def testGetRefObjLoader(self):

        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        # Check that our refObjLoader loads the available objects
        # within a given search radius
        donutCatSmall = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(0.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatSmall.refCat), 8)

        donutCatFull = refObjLoader.loadSkyCircle(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            lsst.geom.Angle(2.5, lsst.geom.degrees),
            filterName="g",
        )
        self.assertEqual(len(donutCatFull.refCat), 24)

    def testDonutCatalogListToDataFrame(self):

        refCatList = self._getRefCat()
        refObjLoader = self.task.getRefObjLoader(refCatList)

        # Check that our refObjLoader loads the available objects
        # within a given footprint from a sample exposure
        testDataId = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
        }
        testExposure = self.butler.get(
            "raw", dataId=testDataId, collections="LSSTCam/raw/all"
        )
        donutCatSmall = refObjLoader.loadPixelBox(
            testExposure.getBBox(),
            testExposure.getWcs(),
            testExposure.getFilter().getName(),
        )
        fieldObjects = self.task.donutCatalogListToDataFrame(
            [donutCatSmall.refCat, donutCatSmall.refCat], ["R22_S99", "R22_S99"]
        )
        self.assertEqual(len(fieldObjects), 8)
        self.assertCountEqual(
            fieldObjects.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "source_flux",
                "detector",
            ],
        )
