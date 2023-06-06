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

import lsst.geom
from lsst.daf import butler as dafButler
from lsst.ts.wep.utility import getModulePath
from lsst.ts.wep.task.refCatalogInterface import RefCatalogInterface
from lsst.ts.wep.task.generateDonutCatalogOnlineTask import (
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

        self.config = GenerateDonutCatalogOnlineTaskConfig()
        self.task = GenerateDonutCatalogOnlineTask(config=self.config)

        self.camera = self.butler.get(
            "camera", instrument="LSSTCam", collections=["LSSTCam/calib/unbounded"]
        )

    def _getRefCat(self):
        refCatList = []
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            refCatList.append(
                self.butler.getDeferred(ref, collections=["refcats/gen2"])
            )

        return refCatList

    def testValidateConfigs(self):
        self.config.filterName = "r"
        self.config.doDonutSelection = False
        task = GenerateDonutCatalogOnlineTask(config=self.config)

        self.assertEqual(task.config.filterName, "r")
        self.assertEqual(task.config.doDonutSelection, False)

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

    def testTaskRun(self):
        self.config.doDonutSelection = False
        task = GenerateDonutCatalogOnlineTask(config=self.config)

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
        taskCat = task.run(self.dataRefs, detector, detWcs)
        donutCatalog = taskCat.donutCatalog

        self.assertEqual(len(donutCatalog), 4)
        self.assertCountEqual(
            donutCatalog.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "source_flux",
                "blend_centroid_x",
                "blend_centroid_y",
            ],
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
        np.testing.assert_array_equal(
            [[]] * 4, list(donutCatalog["blend_centroid_x"].values)
        )
        np.testing.assert_array_equal(
            [[]] * 4, list(donutCatalog["blend_centroid_y"].values)
        )
