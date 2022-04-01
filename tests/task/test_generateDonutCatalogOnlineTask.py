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
from astropy import units

from lsst.daf import butler as dafButler
from lsst.ts.wep.Utility import getModulePath
from lsst.ts.wep.task.RefCatalogInterface import RefCatalogInterface
from lsst.ts.wep.task.GenerateDonutCatalogOnlineTask import (
    GenerateDonutCatalogOnlineTask,
    GenerateDonutCatalogOnlineTaskConfig,
)


class TestGenerateDonutCatalogOnlineTask(unittest.TestCase):
    def setUp(self):

        boresightRa = 0.0
        boresightDec = 0.0
        boresightRotAng = 0.0
        self.refCatInterface = RefCatalogInterface(
            boresightRa, boresightDec, boresightRotAng
        )

        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.butler = dafButler.Butler(self.repoDir)
        self.registry = self.butler.registry

        shardIds = self.refCatInterface.getHtmIds()
        self.catalogName = "cal_ref_cat"
        self.collections = ["refcats/gen2"]
        dataRefs, dataIds = self.refCatInterface.getDataRefs(
            shardIds, self.butler, self.catalogName, self.collections
        )
        self.dataRefs = dataRefs
        self.dataIds = dataIds

        self.config = GenerateDonutCatalogOnlineTaskConfig()

        self.camera = self.butler.get(
            "camera", instrument="LSSTCam", collections=["LSSTCam/calib/unbounded"]
        )

    def testValidateConfigs(self):

        self.config.filterName = "r"
        self.config.doReferenceSelection = False
        self.config.doDonutSelection = False
        task = GenerateDonutCatalogOnlineTask(
            self.dataIds[0], self.dataRefs[0], config=self.config
        )

        self.assertEqual(task.filterName, "r")
        self.assertEqual(task.config.doReferenceSelection, False)
        self.assertEqual(task.config.doDonutSelection, False)

    def testFormatCatalog(self):

        detectorName = "R22_S01"
        detector = self.camera[detectorName]
        detWcs = self.refCatInterface.getDetectorWcs(detector)
        dataRefs, dataIds = self.refCatInterface.getDataRefs(
            [131072], self.butler, self.catalogName, self.collections
        )
        task = GenerateDonutCatalogOnlineTask(
            dataIds=dataIds, refCats=dataRefs, config=self.config
        )
        cat = task.refObjLoader.loadPixelBox(
            detector.getBBox(), detWcs, filterName=task.filterName
        )
        pandasRefCat = task._formatCatalog(cat.refCat, detector)

        self.assertEqual(len(cat.refCat), 2)
        self.assertEqual(len(pandasRefCat), 2)
        self.assertCountEqual(
            pandasRefCat.columns,
            ["coord_ra", "coord_dec", "centroid_x", "centroid_y", "source_flux"],
        )
        self.assertCountEqual(pandasRefCat["coord_ra"], cat.refCat["coord_ra"])
        self.assertCountEqual(pandasRefCat["coord_dec"], cat.refCat["coord_dec"])
        self.assertCountEqual(pandasRefCat["centroid_x"], cat.refCat["centroid_x"])
        self.assertCountEqual(pandasRefCat["centroid_y"], cat.refCat["centroid_y"])
        refFluxes = 15.0 * units.ABmag
        np.testing.assert_almost_equal(
            pandasRefCat["source_flux"], [refFluxes.to_value(units.nJy)] * 2
        )

    def testTaskRun(self):

        self.config.doDonutSelection = False
        task = GenerateDonutCatalogOnlineTask(
            self.dataIds, self.dataRefs, config=self.config
        )

        detectorName = "R22_S01"
        detector = self.camera[detectorName]
        detWcs = self.refCatInterface.getDetectorWcs(detector)
        dataIds = [131072, 188416]
        cat0 = self.butler.get(
            self.catalogName, dataId={"htm7": dataIds[0]}, collections=self.collections
        )
        cat1 = self.butler.get(
            self.catalogName, dataId={"htm7": dataIds[1]}, collections=self.collections
        )[2:]
        taskCat = task.run(detector, detWcs)
        donutCatalog = taskCat.donutCatalog

        self.assertEqual(len(donutCatalog), 4)
        self.assertCountEqual(
            donutCatalog.columns,
            ["coord_ra", "coord_dec", "centroid_x", "centroid_y", "source_flux"],
        )
        self.assertCountEqual(
            donutCatalog["coord_ra"], np.ravel([cat0["coord_ra"], cat1["coord_ra"]])
        )
        self.assertCountEqual(
            donutCatalog["coord_dec"], np.ravel([cat0["coord_dec"], cat1["coord_dec"]])
        )
        refFluxes = (np.array([15.0, 15.0, 15.0, 17.5]) * units.ABmag).to_value(
            units.nJy
        )
        np.testing.assert_almost_equal(
            donutCatalog["source_flux"], refFluxes, decimal=5
        )
