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

from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.Utility import getModulePath, getConfigDir, DefocalType, CamType


class TestWfEstimator(unittest.TestCase):
    """Test the wavefront estimator class."""

    def setUp(self):

        self.instParams = {
            "obscuration": 0.61,
            "focalLength": 10.312,
            "apertureDiameter": 8.36,
            "offset": 1.0,
            "pixelSize": 10.0e-6,
        }

        cwfsConfigDir = os.path.join(getConfigDir(), "cwfs")
        algoDir = os.path.join(cwfsConfigDir, "algo")
        self.wfsEst = WfEstimator(algoDir)

        # Define the image folder and image names
        # It is noted that image.readFile inuts is based on the txt file.
        self.modulePath = getModulePath()
        imageFolderPath = os.path.join(
            self.modulePath, "tests", "testData", "testImages", "LSST_NE_SN25"
        )
        intra_image_name = "z11_0.25_intra.txt"
        extra_image_name = "z11_0.25_extra.txt"

        # Path to image files
        self.intraImgFile = os.path.join(imageFolderPath, intra_image_name)
        self.extraImgFile = os.path.join(imageFolderPath, extra_image_name)

        # Field XY position
        self.fieldXY = (1.185, 1.185)

    def testCalWfsErrOfExp(self):

        # Setup the configuration
        # If the configuration is reset, the images are needed to be set again.
        self.wfsEst.config(
            self.instParams,
            solver="exp",
            camType=CamType.LsstCam,
            opticalModel="offAxis",
            sizeInPix=120,
            debugLevel=0,
        )

        # Setup the images
        self.wfsEst.setImg(self.fieldXY, DefocalType.Intra, imageFile=self.intraImgFile)
        self.wfsEst.setImg(self.fieldXY, DefocalType.Extra, imageFile=self.extraImgFile)

        # Test the images are set.
        self.assertEqual(self.wfsEst.getIntraImg().getDefocalType(), DefocalType.Intra)
        self.assertEqual(self.wfsEst.getExtraImg().getDefocalType(), DefocalType.Extra)

        # Evaluate the wavefront error
        wfsError = [
            2.593,
            14.102,
            -8.470,
            3.676,
            1.467,
            -9.724,
            8.207,
            -192.839,
            0.978,
            1.568,
            4.197,
            -0.391,
            1.551,
            1.235,
            -1.699,
            2.140,
            -0.296,
            -2.113,
            1.188,
        ]
        zer4UpNm = self.wfsEst.calWfsErr()
        self.assertAlmostEqual(
            np.sum(np.abs(zer4UpNm - np.array(wfsError))), 0.66984026, places=7
        )

    def testCalWfsErrOfFft(self):

        # Change the algorithm to fft
        self.wfsEst.config(
            self.instParams,
            solver="fft",
            camType=CamType.LsstCam,
            opticalModel="offAxis",
            sizeInPix=120,
            debugLevel=0,
        )

        # Reset the wavefront images
        self.wfsEst.setImg(self.fieldXY, DefocalType.Intra, imageFile=self.intraImgFile)
        self.wfsEst.setImg(self.fieldXY, DefocalType.Extra, imageFile=self.extraImgFile)

        # Evaluate the wavefront error
        wfsError = [
            12.484,
            10.358,
            -6.674,
            -0.043,
            -1.768,
            -15.593,
            12.511,
            -192.382,
            0.195,
            4.074,
            9.577,
            -1.930,
            3.538,
            3.420,
            -3.610,
            3.547,
            -0.679,
            -2.943,
            1.101,
        ]
        zer4UpNm = self.wfsEst.calWfsErr()
        self.assertAlmostEqual(
            np.sum(np.abs(zer4UpNm - np.array(wfsError))), 6.95092306, places=7
        )

        # Test to reset the data
        self.wfsEst.reset()
        self.assertEqual(np.sum(self.wfsEst.getAlgo().getZer4UpInNm()), 0)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
