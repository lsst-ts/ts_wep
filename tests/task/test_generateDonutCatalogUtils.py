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

import lsst.geom
from astropy.table import QTable
from lsst.daf.butler import Butler
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.ts.wep.task import DonutSourceSelectorTask, DonutSourceSelectorTaskConfig
from lsst.ts.wep.task.generateDonutCatalogUtils import (
    addVisitInfoToCatTable,
    donutCatalogToAstropy,
    runSelection,
)
from lsst.ts.wep.utils import getModulePath


class TestGenerateDonutCatalogUtils(unittest.TestCase):
    def setUp(self) -> None:
        moduleDir = getModulePath()
        self.testDataDir = os.path.join(moduleDir, "tests", "testData")
        self.repoDir = os.path.join(self.testDataDir, "gen3TestRepo")
        self.centerRaft = ["R22_S10", "R22_S11"]

        self.butler = Butler.from_config(self.repoDir)
        self.registry = self.butler.registry

    def _getRefCat(self) -> list:
        refCatList = list()
        datasetGenerator = self.registry.queryDatasets(
            datasetType="cal_ref_cat", collections=["refcats/gen2"]
        ).expanded()
        for ref in datasetGenerator:
            refCatList.append(self.butler.getDeferred(ref, collections=["refcats/gen2"]))

        return refCatList

    def _createRefObjLoader(self) -> ReferenceObjectLoader:
        refCatalogList = self._getRefCat()
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.dataId for ref in refCatalogList],
            refCats=refCatalogList,
        )
        return refObjLoader

    def _createTestDonutCat(self, returnExposure: bool = False) -> QTable:
        refObjLoader = self._createRefObjLoader()

        # Check that our refObjLoader loads the available objects
        # within a given footprint from a sample exposure
        testDataId = {
            "instrument": "LSSTCam",
            "detector": 94,
            "exposure": 4021123106001,
        }
        testExposure = self.butler.get("raw", dataId=testDataId, collections="LSSTCam/raw/all")
        # From the test data provided this will create
        # a catalog of 4 objects.
        donutCatSmall = refObjLoader.loadPixelBox(
            testExposure.getBBox(),
            testExposure.getWcs(),
            testExposure.filter.bandLabel,
        )

        if returnExposure is False:
            return donutCatSmall.refCat
        else:
            return donutCatSmall.refCat, testExposure

    def testRunSelection(self) -> None:
        refObjLoader = self._createRefObjLoader()
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        detector = camera["R22_S11"]
        donutSelectorConfig = DonutSourceSelectorTaskConfig()
        donutSelectorConfig.useCustomMagLimit = True
        donutSelectorConfig.magMax = 17.0
        donutSelectorTask = DonutSourceSelectorTask(config=donutSelectorConfig)

        wcs = createInitialSkyWcsFromBoresight(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            90.0 * lsst.geom.degrees,
            detector,
            flipX=False,
        )
        # If we have a magLimit at 17 we should cut out
        # the one source at 17.5.
        edgeMargin = 60
        donutCatBrighterThan17, blendX, blendY = runSelection(
            refObjLoader, detector, wcs, "g", donutSelectorTask, edgeMargin
        )
        self.assertEqual(len(donutCatBrighterThan17), 3)

        # If we increase the mag limit to 18 we should
        # get all the sources in the catalog if we also
        # change unblendedSeparation so no stars are blended
        donutSelectorConfig.magMax = 18.0
        donutSelectorConfig.unblendedSeparation = 1
        donutSelectorTask = DonutSourceSelectorTask(config=donutSelectorConfig)
        donutCatFull, blendX, blendY = runSelection(
            refObjLoader, detector, wcs, "g", donutSelectorTask, edgeMargin
        )
        self.assertEqual(len(donutCatFull), 4)

        # If we set the margin to a large value we will exclude
        # most sources from the catalog
        edgeMargin = 1000
        donutCatSmall, blendX, blendY = runSelection(
            refObjLoader, detector, wcs, "g", donutSelectorTask, edgeMargin
        )
        self.assertEqual(len(donutCatSmall), 1)

        # This will happen even if we turn off donut selection
        edgeMargin = 1000
        donutCatSmall, blendX, blendY = runSelection(refObjLoader, detector, wcs, "g", None, edgeMargin)
        self.assertEqual(len(donutCatSmall), 1)

    def testRunSelectionNoTask(self) -> None:
        refObjLoader = self._createRefObjLoader()
        camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )
        detector = camera["R22_S11"]
        wcs = createInitialSkyWcsFromBoresight(
            lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            90.0 * lsst.geom.degrees,
            detector,
            flipX=False,
        )

        # When passing None instead of a DonutSourceSelectorTask
        # we should get the full catalog without cuts.
        unchangedCat, blendX, blendY = runSelection(refObjLoader, detector, wcs, "g", None, 60)
        self.assertEqual(len(unchangedCat), 4)
        self.assertEqual(blendX, [[]] * 4)
        self.assertEqual(blendY, [[]] * 4)

    def testDonutCatalogToAstropy(self) -> None:
        donutCatSmall = self._createTestDonutCat()

        fieldObjects = donutCatalogToAstropy(donutCatSmall, ["g"])
        self.assertEqual(len(fieldObjects), 4)
        self.assertCountEqual(
            fieldObjects.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "g_flux",
            ],
        )
        self.assertCountEqual(
            fieldObjects.meta.keys(),
            [
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        # Test that behavior is the same when filterName is a str
        fieldObjects = donutCatalogToAstropy(donutCatSmall, "g")
        self.assertEqual(len(fieldObjects), 4)
        self.assertCountEqual(
            fieldObjects.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "g_flux",
            ],
        )
        self.assertCountEqual(
            fieldObjects.meta.keys(),
            [
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        # Test that None returns an empty QTable
        fieldObjectsNone = donutCatalogToAstropy(None, "g")
        self.assertEqual(len(fieldObjectsNone), 0)
        self.assertCountEqual(
            fieldObjectsNone.columns,
            [
                "coord_ra",
                "coord_dec",
                "centroid_x",
                "centroid_y",
                "g_flux",
            ],
        )
        self.assertCountEqual(
            fieldObjectsNone.meta.keys(),
            [
                "blend_centroid_x",
                "blend_centroid_y",
            ],
        )

        # Test that blendCentersX and blendCentersY
        # get assigned correctly.
        fieldObjectsBlends = donutCatalogToAstropy(
            donutCatSmall,
            "g",
        )
        fieldObjectsBlends.meta["blend_centroid_x"][1].append(5)
        fieldObjectsBlends.meta["blend_centroid_y"][1].append(3)
        self.assertListEqual(fieldObjectsBlends.meta["blend_centroid_x"], [[], [5], [], []])
        self.assertListEqual(fieldObjectsBlends.meta["blend_centroid_y"], [[], [3], [], []])

    def testDonutCatalogToAstropyWithBlendCenters(self) -> None:
        donutCatSmall = self._createTestDonutCat()
        # Brightest three fluxes are all the same.
        # Change two slightly so that we get a consistent order
        # after sorting the catalog by brightness.
        donutCatSmall["g_flux"][1] -= 0.1
        donutCatSmall["g_flux"][2] -= 0.02
        blendCentersX: list = [list() for _ in range(len(donutCatSmall))]
        blendCentersY: list = [list() for _ in range(len(donutCatSmall))]

        fieldObjects = donutCatalogToAstropy(
            donutCatSmall, "g", blendCentersX=blendCentersX, blendCentersY=blendCentersY
        )
        self.assertListEqual(list(fieldObjects.meta["blend_centroid_x"]), [list()] * len(donutCatSmall))
        self.assertListEqual(list(fieldObjects.meta["blend_centroid_y"]), [list()] * len(donutCatSmall))

        blendValsX = [
            donutCatSmall["centroid_x"][0] + 10.0,
            donutCatSmall["centroid_x"][1] + 15.0,
        ]
        blendValsY = [
            donutCatSmall["centroid_y"][0] + 10.0,
            donutCatSmall["centroid_y"][1] + 15.0,
        ]
        blendCentersX[0].append(blendValsX[0])
        blendCentersY[0].append(blendValsY[0])
        fieldObjectsOneBlend = donutCatalogToAstropy(
            donutCatSmall, "g", blendCentersX=blendCentersX, blendCentersY=blendCentersY
        )
        self.assertListEqual(
            list(fieldObjectsOneBlend.meta["blend_centroid_x"]),
            [[blendValsX[0]], [], [], []],
        )
        self.assertListEqual(
            list(fieldObjectsOneBlend.meta["blend_centroid_y"]),
            [[blendValsY[0]], [], [], []],
        )

        blendCentersX[1].append(blendValsX[1])
        blendCentersY[1].append(blendValsY[1])
        fieldObjectsTwoBlends = donutCatalogToAstropy(
            donutCatSmall, "g", blendCentersX=blendCentersX, blendCentersY=blendCentersY
        )
        self.assertListEqual(
            list(fieldObjectsTwoBlends.meta["blend_centroid_x"]),
            [[blendValsX[0]], [], [blendValsX[1]], []],
        )
        self.assertListEqual(
            list(fieldObjectsTwoBlends.meta["blend_centroid_y"]),
            [[blendValsY[0]], [], [blendValsY[1]], []],
        )

    def testDonutCatalogToAstropyErrors(self) -> None:
        columnList = [
            "coord_ra",
            "coord_dec",
            "centroid_x",
            "centroid_y",
            "g_flux",
            "blend_centroid_x",
            "blend_centroid_y",
        ]
        donutCatZero = QTable(names=columnList)

        # Test blendCenters are both supplied or both left as None
        blendErrMsg = "blendCentersX and blendCentersY must be" + " both be None or both be a list."
        with self.assertRaises(ValueError) as context:
            donutCatalogToAstropy(donutCatZero, "g", blendCentersX=[])
        self.assertTrue(blendErrMsg in str(context.exception))
        with self.assertRaises(ValueError) as context:
            donutCatalogToAstropy(donutCatZero, "g", blendCentersY=[])
        self.assertTrue(blendErrMsg in str(context.exception))

        # Test blendCenters must be same length as donutCat
        lengthErrMsg = "blendCentersX and blendCentersY must be" + " both be None or both be a list."
        with self.assertRaises(ValueError) as context:
            donutCatalogToAstropy(donutCatZero, "g", blendCentersX=[[], []])
        self.assertTrue(lengthErrMsg in str(context.exception))
        with self.assertRaises(ValueError) as context:
            donutCatalogToAstropy(donutCatZero, "g", blendCentersY=[[], []])
        self.assertTrue(lengthErrMsg in str(context.exception))

        donutCatSmall = self._createTestDonutCat()

        # Test that each list within blendCentersX
        # has the same length as the list at the same
        # index within blendCentersY.
        xyMismatchErrMsg = (
            "Each list in blendCentersX must have the same "
            + "length as the list in blendCentersY at the "
            + "same index."
        )
        with self.assertRaises(ValueError) as context:
            donutCatalogToAstropy(
                donutCatSmall,
                "g",
                blendCentersX=[[1], [0], [], []],
                blendCentersY=[[4], [], [], []],
            )
        self.assertTrue(xyMismatchErrMsg in str(context.exception))

    def testAddVisitInfoToCatTable(self) -> None:
        donutCatSmall, testExposure = self._createTestDonutCat(returnExposure=True)
        fieldObjects = donutCatalogToAstropy(donutCatSmall, "g")
        catTableWithMeta = addVisitInfoToCatTable(testExposure, fieldObjects)
        visitInfoKeys = [
            "boresight_ra",
            "boresight_dec",
            "boresight_rot_angle",
            "boresight_par_angle",
            "mjd",
            "visit_id",
            "boresight_alt",
            "boresight_az",
            "rot_type_name",
            "rot_type_value",
            "focus_z",
            "instrument_label",
            "observatory_elevation",
            "observatory_latitude",
            "observatory_longitude",
            "ERA",
            "exposure_time",
            "donut_radius",
        ]

        self.assertTrue(isinstance(catTableWithMeta.meta["visit_info"], dict))
        # Test columns are all present
        self.assertCountEqual(visitInfoKeys, catTableWithMeta.meta["visit_info"].keys())
