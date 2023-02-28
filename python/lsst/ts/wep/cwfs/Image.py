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

__all__ = ["Image"]

import os
import numpy as np
from astropy.io import fits
import warnings

from lsst.ts.wep.Utility import CentroidFindType
from lsst.ts.wep.cwfs.CentroidFindFactory import CentroidFindFactory


class Image(object):
    """Image class for wavefront estimation.

    Parameters
    ----------
    centroidFindType : enum 'CentroidFindType', optional
        Algorithm to find the centroid of donut. (the default is
        CentroidFindType.RandomWalk.)
    """

    def __init__(self, centroidFindType=CentroidFindType.RandomWalk):
        self.image = np.array([])
        self.imageFilePath = ""

        self._centroidFind = CentroidFindFactory.createCentroidFind(centroidFindType)

    def getCentroidFind(self):
        """Get the centroid find object.

        Returns
        -------
        CentroidRandomWalk, CentroidOtsu
            Centroid find object.
        """

        return self._centroidFind

    def getImg(self):
        """Get the image.

        Returns
        -------
        numpy.ndarray
            Image.
        """

        return self.image

    def getImgFilePath(self):
        """Get the image file path.

        Returns
        -------
        str
            Image file path.
        """

        return self.imageFilePath

    def setImg(self, image=None, imageFile=None):
        """Set the wavefront image.

        Parameters
        ----------
        image : numpy.ndarray, optional
            Array of image. (the default is None.)
        imageFile : str, optional
            Path of image file. (the default is None.)
        """

        # Read the file if there is no input image
        if image is not None:
            self.image = image
            self.imageFilePath = ""
        else:
            if imageFile is not None:
                self.image = self._readImgFile(imageFile)
                self.imageFilePath = imageFile

    def _readImgFile(self, imageFile):
        """Read the donut image.

        Parameters
        ----------
        imageFile : str
            Path of image file.

        Returns
        -------
        numpy.ndarray
            Image data.
        """

        if os.path.isfile(imageFile):
            if imageFile.endswith((".fits", ".fits.gz")):
                image = fits.getdata(imageFile)
            else:
                image = np.loadtxt(imageFile)
                # This assumes this "txt" file is in the format
                # I[0,0]   I[0,1]
                # I[1,0]   I[1,1]
                image = image[::-1, :]
        else:
            image = np.array([])

        return image

    def updateImage(self, image):
        """Update the image of donut.

        Parameters
        ----------
        image : numpy.ndarray
            Donut image.
        """

        # Update the image
        if len(self.image) == 0:
            warnings.warn(
                "No 'image' is hold. Use setImg() instead.", category=UserWarning
            )
        else:
            self.image = image

    def getCenterAndR(self, image=None):
        """Get the centroid data.

        Parameters
        ----------
        image : numpy.ndarray, optional
            Image to do the analysis. (the default is None.)

        Returns
        -------
        float
            Centroid x.
        float
            Centroid y.
        float
            Effective weighting radius.
        """

        if image is not None:
            imgDonut = image
        else:
            imgDonut = self.image

        return self._centroidFind.getCenterAndR(imgDonut)

    def getSNR(self):
        """Get the signal to noise ratio of donut.

        Returns
        -------
        float
            Signal to noise ratio.
        """

        # Get the signal binary image
        imgBinary = self._centroidFind.getImgBinary(self.image)
        realcx, realcy, realR = self._centroidFind.getCenterAndRfromImgBinary(imgBinary)

        # Get the background binary img
        bgBinary = 1 - imgBinary

        # Get the donut image signal and calculate the intensity
        signal = self.image * imgBinary
        signal = np.mean(signal[signal != 0])

        # Get the background signal
        bg = self.image * bgBinary
        bg = bg[bg != 0]

        # Calculate the noise
        noise = np.std(bg - np.mean(bg))

        # Calculate SNR
        snr = signal / noise

        return snr
