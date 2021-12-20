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
from lsst.ts.wep.Utility import getConfigDir, CamType


class TestInstrument(unittest.TestCase):
    """Test the Instrument class."""

    def setUp(self):

        self.instDir = os.path.join(getConfigDir(), "cwfs", "instData")

        self.inst = Instrument(self.instDir)
        self.dimOfDonutOnSensor = 120

        self.inst.config(
            CamType.LsstCam, self.dimOfDonutOnSensor, announcedDefocalDisInMm=1.5
        )

    def testConfigWithUnsupportedCamType(self):

        self.assertRaises(ValueError, self.inst.config, "NoThisCamType", 120)

    def testGetInstFileDir(self):

        instFileDir = self.inst.getInstFileDir()

        ansInstFileDir = os.path.join(self.instDir, "lsst")
        self.assertEqual(instFileDir, ansInstFileDir)

    def testGetAnnDefocalDisInMm(self):

        annDefocalDisInMm = self.inst.getAnnDefocalDisInMm()
        self.assertEqual(annDefocalDisInMm, 1.5)

    def testSetAnnDefocalDisInMm(self):

        annDefocalDisInMm = 2.0
        self.inst.setAnnDefocalDisInMm(annDefocalDisInMm)

        self.assertEqual(self.inst.getAnnDefocalDisInMm(), annDefocalDisInMm)

    def testGetInstFilePath(self):

        instFilePath = self.inst.getInstFilePath()
        self.assertTrue(os.path.exists(instFilePath))
        self.assertEqual(os.path.basename(instFilePath), "instParam.yaml")

    def testGetMaskOffAxisCorr(self):

        maskOffAxisCorr = self.inst.getMaskOffAxisCorr()
        self.assertEqual(maskOffAxisCorr.shape, (9, 5))
        self.assertEqual(maskOffAxisCorr[0, 0], 1.07)
        self.assertEqual(maskOffAxisCorr[2, 3], -0.090100858)

    def testGetDimOfDonutOnSensor(self):

        dimOfDonutOnSensor = self.inst.getDimOfDonutOnSensor()
        self.assertEqual(dimOfDonutOnSensor, self.dimOfDonutOnSensor)

    def testGetObscuration(self):

        obscuration = self.inst.getObscuration()
        self.assertEqual(obscuration, 0.61)

    def testGetFocalLength(self):

        focalLength = self.inst.getFocalLength()
        self.assertEqual(focalLength, 10.312)

    def testGetApertureDiameter(self):

        apertureDiameter = self.inst.getApertureDiameter()
        self.assertEqual(apertureDiameter, 8.36)

    def testGetDefocalDisOffset(self):

        defocalDisInM = self.inst.getDefocalDisOffset()

        # The answer is 1.5 mm
        self.assertEqual(defocalDisInM * 1e3, 1.5)

    def testGetCamPixelSize(self):

        camPixelSizeInM = self.inst.getCamPixelSize()

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

        inst = Instrument(self.instDir)
        inst.config(CamType.AuxTel, 160, announcedDefocalDisInMm=0.8)

        self.assertEqual(inst.getObscuration(), 0.3525)
        self.assertEqual(inst.getFocalLength(), 21.6)
        self.assertEqual(inst.getApertureDiameter(), 1.2)
        self.assertEqual(inst.getDefocalDisOffset(), 0.041*0.8)
        self.assertEqual(inst.getCamPixelSize(), 10.0e-6)
        self.assertAlmostEqual(inst.calcSizeOfDonutExpected(), 182.2222222, places=7)

    def testDataAuxTelZWO(self):

        inst = Instrument(self.instDir)
        inst.config(CamType.AuxTelZWO, 160, announcedDefocalDisInMm=0.5)

        self.assertEqual(inst.getObscuration(), 0.3525)
        self.assertEqual(inst.getFocalLength(), 21.6)
        self.assertEqual(inst.getApertureDiameter(), 1.2)
        self.assertEqual(inst.getDefocalDisOffset(), 0.0205)
        self.assertEqual(inst.getCamPixelSize(), 14.4e-6)
        self.assertAlmostEqual(inst.calcSizeOfDonutExpected(), 79.08950617, places=7)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
