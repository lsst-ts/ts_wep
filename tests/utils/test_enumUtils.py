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
from enum import IntEnum

from lsst.afw.cameraGeom import DetectorType
from lsst.ts.wep.utils import (
    BscDbType,
    CamType,
    CentroidFindType,
    DeblendDonutType,
    DonutTemplateType,
    FilterType,
    ImageType,
    getBscDbType,
    getCamNameFromCamType,
    getCamType,
    getCamTypeFromButlerName,
    getCentroidFindType,
    getDeblendDonutType,
    getDonutTemplateType,
    getFilterTypeFromBandLabel,
    getImageType,
    mapFilterRefToG,
)


class TestEnumUtils(unittest.TestCase):
    """Test the Enum utils."""

    def testGetFilterTypeFromBandLabel(self):
        # Test allowable filter band labels
        self.assertEqual(getFilterTypeFromBandLabel("u"), FilterType.LSST_U)
        self.assertEqual(getFilterTypeFromBandLabel("g"), FilterType.LSST_G)
        self.assertEqual(getFilterTypeFromBandLabel("r"), FilterType.LSST_R)
        self.assertEqual(getFilterTypeFromBandLabel("i"), FilterType.LSST_I)
        self.assertEqual(getFilterTypeFromBandLabel("z"), FilterType.LSST_Z)
        self.assertEqual(getFilterTypeFromBandLabel("y"), FilterType.LSST_Y)
        self.assertEqual(getFilterTypeFromBandLabel("SomeFilter"), FilterType.REF)

    def testMapFilterRefToG(self):
        mappedFilterType = mapFilterRefToG(FilterType.REF)
        self.assertEqual(mappedFilterType, FilterType.LSST_G)

    def testMapFilterRefToGForFilterU(self):
        mappedFilterType = mapFilterRefToG(FilterType.LSST_U)
        self.assertEqual(mappedFilterType, FilterType.LSST_U)

    def testGetCamType(self):
        self.assertEqual(getCamType("lsst"), CamType.LsstCam)
        self.assertEqual(getCamType("lsstfam"), CamType.LsstFamCam)
        self.assertEqual(getCamType("comcam"), CamType.ComCam)
        self.assertEqual(getCamType("auxTel"), CamType.AuxTel)
        instName = "telescope"
        assertMsg = f"Instrument name ({instName}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamType(instName)
        self.assertTrue(assertMsg in str(context.exception))

    def testGetCamNameFromCamType(self):
        # Test allowable CamType values
        self.assertEqual(getCamNameFromCamType(CamType.LsstCam), "lsst")
        self.assertEqual(getCamNameFromCamType(CamType.LsstFamCam), "lsstfam")
        self.assertEqual(getCamNameFromCamType(CamType.ComCam), "comcam")
        self.assertEqual(getCamNameFromCamType(CamType.AuxTel), "auxTel")
        self.assertEqual(getCamNameFromCamType(CamType.AuxTelZWO), "auxTelZWO")

        # Create a TestCamType that has a value greater than the last CamType
        class TestCamType(IntEnum):
            BadInst = len(CamType) + 1

        # Test the error is raised correctly with an incorrect CamType.
        badCamType = TestCamType.BadInst
        errMsg = f"CamType ({badCamType}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamNameFromCamType(badCamType)
        self.assertEqual(str(context.exception), errMsg)

    def testGetCamTypeFromButlerName(self):
        self.assertEqual(
            getCamTypeFromButlerName("LSSTCam", DetectorType.WAVEFRONT), CamType.LsstCam
        )
        instName = "LSSTComCam"
        wfAssertMsg = (
            f"Wavefront sensors for instrument name ({instName}) are not supported."
        )
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, DetectorType.WAVEFRONT)
        self.assertTrue(wfAssertMsg in str(context.exception))

        self.assertEqual(
            getCamTypeFromButlerName("LSSTCam", DetectorType.SCIENCE),
            CamType.LsstFamCam,
        )
        self.assertEqual(
            getCamTypeFromButlerName("LSSTComCam", DetectorType.SCIENCE), CamType.ComCam
        )
        self.assertEqual(
            getCamTypeFromButlerName("LATISS", DetectorType.SCIENCE), CamType.AuxTel
        )
        instName = "telescope"
        sciAssertMsg = (
            f"Science sensors for instrument name ({instName}) are not supported."
        )
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, DetectorType.SCIENCE)
        self.assertTrue(sciAssertMsg in str(context.exception))

        detType = DetectorType.GUIDER
        detAssertMsg = f"Detector Type ({detType.name}) is not supported."
        with self.assertRaises(ValueError) as context:
            getCamTypeFromButlerName(instName, detType)
        self.assertTrue(detAssertMsg in str(context.exception))

    def testGetBscDbType(self):
        self.assertEqual(getBscDbType("localDb"), BscDbType.LocalDb)
        self.assertEqual(getBscDbType("file"), BscDbType.LocalDbForStarFile)

    def testGetBscDbTypeWithWrongInput(self):
        self.assertRaises(ValueError, getBscDbType, "wrongType")

    def testGetImageType(self):
        self.assertEqual(getImageType("amp"), ImageType.Amp)
        self.assertEqual(getImageType("eimage"), ImageType.Eimg)

    def testGetImageTypeWithWrongInput(self):
        self.assertRaises(ValueError, getImageType, "wrongType")

    def testGetCentroidFindType(self):
        self.assertEqual(getCentroidFindType("randomWalk"), CentroidFindType.RandomWalk)
        self.assertEqual(getCentroidFindType("otsu"), CentroidFindType.Otsu)
        self.assertEqual(
            getCentroidFindType("convolveTemplate"), CentroidFindType.ConvolveTemplate
        )

    def testGetCentroidFindTypeWithWrongInput(self):
        self.assertRaises(ValueError, getCentroidFindType, "wrongType")

    def testGetDonutTemplateType(self):
        self.assertEqual(getDonutTemplateType("model"), DonutTemplateType.Model)
        self.assertEqual(getDonutTemplateType("phosim"), DonutTemplateType.Phosim)

    def testGetDonutTemplateTypeWithWrongInput(self):
        self.assertRaises(ValueError, getDonutTemplateType, "wrongType")

    def testGetDeblendDonutType(self):
        self.assertEqual(getDeblendDonutType("adapt"), DeblendDonutType.Adapt)

    def testGetDeblendDonutTypeWithWrongInput(self):
        self.assertRaises(ValueError, getDeblendDonutType, "wrongType")


if __name__ == "__main__":
    # Run the unit test
    unittest.main()
