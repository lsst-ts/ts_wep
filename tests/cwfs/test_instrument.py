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
import numpy as np
import unittest

from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.Utility import getConfigDir, CamType, getModulePath


class TestInstrument(unittest.TestCase):
    """Test the Instrument class."""

    def setUp(self):
        self.instConfigDir = os.path.join(getConfigDir(), "cwfs", "instData")
        self.instConfigFile = os.path.join(
            self.instConfigDir, "lsst", "instParamPipeConfig.yaml"
        )
        self.maskConfigFile = os.path.join(
            self.instConfigDir, "lsst", "maskMigrate.yaml"
        )
        self.instConfigDict = {
            # Obscuration (inner_radius / outer_radius of M1M3)
            "obscuration": 0.61,
            # Focal length in m
            "focalLength": 10.312,
            # Aperture diameter in m
            "apertureDiameter": 8.36,
            # Defocal distance offset in mm
            "offset": 1.5,
            # Camera pixel size in m
            "pixelSize": 10.0e-6,
        }

        self.dimOfDonutOnSensor = 120
        self.inst = Instrument()
        self.inst.configFromDict(
            self.instConfigDict,
            self.dimOfDonutOnSensor,
            CamType.LsstCam,
            self.maskConfigFile,
        )

    def testConfigFromFileDefault(self):
        newInst = Instrument()
        newInst.configFromFile(self.dimOfDonutOnSensor, CamType.LsstFamCam)
        self.assertDictEqual(self.inst.instParams, newInst.instParams)

    def testConfigFromDict(self):
        newInst = Instrument()
        newInst.configFromDict(
            self.inst.instParams, self.dimOfDonutOnSensor, CamType.LsstCam
        )
        self.assertDictEqual(self.inst.instParams, newInst.instParams)

    def testConfigFromDictWithIncorrectDictKeys(self):
        # Check that error is raised when configDict keys are incorrect
        newInst = Instrument()
        instParamsList = list(self.inst.instParams.items())
        badInstParams = {key: value for key, value in instParamsList[:4]}
        with self.assertRaises(AssertionError) as context:
            newInst.configFromDict(
                badInstParams, self.dimOfDonutOnSensor, CamType.LsstCam
            )
        errMsg = f"Config Dict Keys: {badInstParams.keys()} do not match required \
            instParamKeys: {self.inst.instParams.keys()}"
        self.assertEqual(str(context.exception), errMsg)

    def testConfigFromFileWithIncorrectInstConfigFilePath(self):
        badFilePath = "NoFile"
        with self.assertRaises(ValueError) as context:
            self.inst.configFromFile(120, CamType.LsstCam, badFilePath)
        self.assertEqual(
            str(context.exception),
            f"Instrument configuration file at {badFilePath} does not exist.",
        )

    def testConfigFromFileWithIncorrectInstConfigFormat(self):
        badFilePath = os.path.join(
            getModulePath(),
            "tests",
            "testData",
            "pipelineConfigs",
            "testBasePipeline.yaml",
        )
        with self.assertRaises(ValueError) as context:
            self.inst.configFromFile(120, CamType.LsstCam, badFilePath)
        errMsg = "Instrument configuration file does not have expected format. "
        errMsg += "See examples in policy/cwfs/instData."
        self.assertEqual(str(context.exception), errMsg)

    def testConfigFromFileWithIncorrectMaskConfigFilePath(self):
        badMaskFilePath = "NoMaskFile"
        with self.assertRaises(ValueError) as context:
            self.inst.configFromFile(
                120, CamType.LsstCam, maskConfigFile=badMaskFilePath
            )
        self.assertEqual(
            str(context.exception),
            f"Mask migrate file at {badMaskFilePath} does not exist.",
        )

    def testConfigFromDictWithIncorrectMaskConfigFilePath(self):
        badMaskFilePath = "NoMaskFile"
        with self.assertRaises(ValueError) as context:
            self.inst.configFromDict(
                self.inst.instParams,
                self.dimOfDonutOnSensor,
                CamType.LsstCam,
                maskConfigFile=badMaskFilePath,
            )
        self.assertEqual(
            str(context.exception),
            f"Mask migrate file at {badMaskFilePath} does not exist.",
        )

    def testSetDefaultMaskParams(self):
        newInst = Instrument()
        self.assertEqual(newInst.maskOffAxisCorr, [])

        # AuxTel has no default parameters available
        newInst.setDefaultMaskParams(CamType.AuxTel)
        self.assertEqual(newInst.maskOffAxisCorr, [])
        newInst.setDefaultMaskParams(CamType.AuxTelZWO)
        self.assertEqual(newInst.maskOffAxisCorr, [])

        # Test set correctly with valid camera
        newInst.setDefaultMaskParams(CamType.LsstCam)
        self.assertEqual(newInst.maskOffAxisCorr.shape, (9, 5))
        self.assertEqual(newInst.maskOffAxisCorr[0, 0], 1.07)
        self.assertEqual(newInst.maskOffAxisCorr[2, 3], -0.090100858)

    def testGetMaskOffAxisCorr(self):
        self.assertEqual(self.inst.maskOffAxisCorr.shape, (9, 5))
        self.assertEqual(self.inst.maskOffAxisCorr[0, 0], 1.07)
        self.assertEqual(self.inst.maskOffAxisCorr[2, 3], -0.090100858)

    def testGetDimOfDonutImg(self):
        dimOfDonutOnSensor = self.inst.dimOfDonutImg
        self.assertEqual(dimOfDonutOnSensor, self.dimOfDonutOnSensor)

    def testGetObscuration(self):
        obscuration = self.inst.obscuration
        self.assertEqual(obscuration, 0.61)

    def testGetFocalLength(self):
        focalLength = self.inst.focalLength
        self.assertEqual(focalLength, 10.312)

    def testGetApertureDiameter(self):
        apertureDiameter = self.inst.apertureDiameter
        self.assertEqual(apertureDiameter, 8.36)

    def testGetDefocalDisOffseInM(self):
        defocalDisInM = self.inst.defocalDisOffsetInM

        # The answer is 1.5 mm
        self.assertEqual(defocalDisInM * 1e3, 1.5)

    def testGetPixelSize(self):
        camPixelSizeInM = self.inst.pixelSize

        # The answer is 10 um
        self.assertEqual(camPixelSizeInM * 1e6, 10)

    def testGetMarginalFocalLength(self):
        marginalFL = self.inst.getMarginalFocalLength()
        self.assertAlmostEqual(marginalFL, 9.4268, places=4)

    def testGetSensorFactor(self):
        sensorFactor = self.inst.getSensorFactor()
        self.assertAlmostEqual(sensorFactor, 0.98679, places=5)

    def testGetSensorCoor(self):
        xSensor, ySensor = self.inst.getSensorCoor()
        self.assertEqual(
            xSensor.shape, (self.dimOfDonutOnSensor, self.dimOfDonutOnSensor)
        )
        self.assertAlmostEqual(xSensor[0, 0], -0.97857, places=5)
        self.assertAlmostEqual(xSensor[0, 1], -0.96212, places=5)

        self.assertEqual(
            ySensor.shape, (self.dimOfDonutOnSensor, self.dimOfDonutOnSensor)
        )
        self.assertAlmostEqual(ySensor[0, 0], -0.97857, places=5)
        self.assertAlmostEqual(ySensor[1, 0], -0.96212, places=5)

    def testGetSensorCoorAnnular(self):
        xoSensor, yoSensor = self.inst.getSensorCoorAnnular()
        self.assertEqual(
            xoSensor.shape, (self.dimOfDonutOnSensor, self.dimOfDonutOnSensor)
        )
        self.assertTrue(np.isnan(xoSensor[0, 0]))
        self.assertTrue(np.isnan(xoSensor[60, 60]))

        self.assertEqual(
            yoSensor.shape, (self.dimOfDonutOnSensor, self.dimOfDonutOnSensor)
        )
        self.assertTrue(np.isnan(yoSensor[0, 0]))
        self.assertTrue(np.isnan(yoSensor[60, 60]))

    def testCalcSizeOfDonutExpected(self):
        self.assertAlmostEqual(
            self.inst.calcSizeOfDonutExpected(), 121.60589604, places=7
        )

    def testDataAuxTel(self):
        auxTelConfigFile = os.path.join(
            self.instConfigDir, "auxTel", "instParamPipeConfig.yaml"
        )
        inst = Instrument()
        inst.configFromFile(160, CamType.AuxTel, auxTelConfigFile)

        self.assertEqual(inst.obscuration, 0.3525)
        self.assertEqual(inst.focalLength, 21.6)
        self.assertEqual(inst.apertureDiameter, 1.2)
        self.assertAlmostEqual(inst.defocalDisOffsetInM, 0.041 * 0.8)
        self.assertEqual(inst.pixelSize, 10.0e-6)
        self.assertAlmostEqual(inst.calcSizeOfDonutExpected(), 182.2222222, places=7)

    def testDataAuxTelZWO(self):
        auxTelZWOConfigFile = os.path.join(
            self.instConfigDir, "auxTelZWO", "instParamPipeConfig.yaml"
        )
        inst = Instrument()
        inst.configFromFile(160, CamType.AuxTelZWO, auxTelZWOConfigFile)

        self.assertEqual(inst.obscuration, 0.3525)
        self.assertEqual(inst.focalLength, 21.6)
        self.assertEqual(inst.apertureDiameter, 1.2)
        self.assertEqual(inst.defocalDisOffsetInM, 0.0205)
        self.assertEqual(inst.pixelSize, 15.2e-6)
        self.assertAlmostEqual(inst.calcSizeOfDonutExpected(), 74.92690058, places=7)


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
