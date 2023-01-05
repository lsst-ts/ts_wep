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

__all__ = ["MachineLearningAlgorithm"]

import numpy as np
import torch
from scipy.ndimage import zoom

from lsst.ts.wep.ParamReader import ParamReader


class MachineLearningAlgorithm(object):
    """Initialize the MachineLearningAlgorithm class.

    Algorithm that uses a convolutional neural net (CNN) to estimate
    Zernike coefficients from individual AOS images. This algorithm
    is an alternative to the TIE solvers in ts.wep.cwfs.Algorithm.

    Parameters
    ----------
    algoDir : str
        Algorithm configuration directory.
    """

    def __init__(self, algoDir):

        self.algoDir = algoDir
        self.algoParamFile = ParamReader()

        # create a model placeholder
        self._model = None

        # Show the calculation message based on this value
        # 0 means no message will be showed
        self.debugLevel = 0

        # Create an attribute to store the algorithm history
        self._history = dict()

    def config(self, mlFile=None, mlReshape="pad", debugLevel=0):
        """Configure the algorithm to estimate Zernikes.

        Parameters
        ----------
        mlFile : str, optional
            Path where the machine learning model is saved. The model must be
            saved in a torchscript file. TODO: ADD DEFAULT
        mlReshape : str, optional
            How to handle incompatible shapes when reshaping images for the
            machine learning algorithm. Options are:
            - "pad" : the image is symmetrically zero-padded to expand the
                        image to the correct shape. This assumes the input
                        image is smaller than the requested shape.
            - "crop" : the image is cropped to fit inside the requested shape,
                        then any empty space is filled with zeros to achieve
                        the requested shape.
            - "zoomMin" : the image is zoomed to the maximum extent that still
                        fits inside the requested shape. Any empty space is
                        then filled with zeros to achieve the requested shape.
                        This is only different from "zoomMax" if either the
                        image or the requested shape is rectangular.
            - "zoomMax" : the image is zoomed to the minimum extent that fills
                        the entire requested shape. Any part of the image that
                        lies outside the requested shape is cropped. This is
                        only different from "zoomMin" if either the image or
                        the requested shape is rectangular.
            (the default is "pad")
        debugLevel : int, optional
            Determines whether to save or print information during running.
            If 0, nothing is done; if 1, zernikes are saved in the history;
            if 2, zernikes are printed to the terminal. (default is 0.)

        Raises
        ------
        AttributeError
            The model must have an inputShape attribute.
        ValueError
            mlReshape value is invalid.
        """

        # Load the model
        model = torch.jit.load(mlFile)
        model.eval()

        # Check that the model has the inputShape attribute
        if not hasattr(model, "inputShape"):
            raise AttributeError("The model must have an inputShape attribute.")

        # Set the model
        self._model = model

        # Set the reshape mode
        allowed_mlReshape = ["pad", "crop", "zoomMin", "zoomMax"]
        if mlReshape not in allowed_mlReshape:
            raise ValueError(
                f"mlReshape must be one of {', '.join(allowed_mlReshape)}."
            )
        self.mlReshape = mlReshape

        # Set the debug level
        self.debugLevel = debugLevel

    def _recordItem(self, item, itemName, debugLevel=0):
        """Record the item in the algorithm history.

        If you use this method to store new items in the algorithm history, you
        should update the docstring of the history property below, which
        describes the (potential) contents of the algorithm history.

        Parameters
        ----------
        item : Any
            The item to record in the history.
        itemName : str
            The name of the item in the history.
        debugLevel : int, optional
            The debugLevel at which this item should be recorded.
            (the default is 0.)
        """

        # If debug level too low, do nothing
        if self.debugLevel < debugLevel:
            return

        # Record the item
        self._history[itemName] = item

    @property
    def history(self):
        """The algorithm history, which is created when debugLevel>=1.

        The algorithm history is a dictionary that (potentially) contains
        the following entries:
            - I1 - the initial I1 image
            - I1reshaped - the reshaped I1 image
            - zk1 - the Zernikes for I1 (in nm)
            - I2 - the initial I2 image
            - I2reshaped - the reshaped I2 image
            - zk2 - the Zernikes for I2 (in nm)
            - zk - the average of zk1 and zk2

        If only I1 or I2 is provided to runIt(), then only those entries will
        be present.
        """
        return self._history

    @staticmethod
    def _padImage(img, shape):
        """Symmetrically zero-pad the image so that it has the desired shape.

        Parameters
        ----------
        img : np.ndarray
            2D numpy array representing the image to be padded.
        shape : tuple
            Tuple of length 2, specifying the output shape of the padded image.

        Returns
        -------
        np.ndarray
            The padded 2D image.
        """

        # Determine how much to pad each axis
        padWidth = np.subtract(shape[-2:], img.shape)

        # Split these widths into front and back halves
        # If a width is odd, the front half is one less than the back half
        padWidth = np.repeat(padWidth, 2).reshape(2, 2)
        padWidth[:, 0] = np.floor(padWidth[:, 0] / 2)
        padWidth[:, 1] = np.ceil(padWidth[:, 1] / 2)

        # Pad the array
        imgPadded = np.pad(img, padWidth)

        return imgPadded

    @staticmethod
    def _cropImage(img, shape):
        """Crop the image so that it fits inside the desired shape.

        Parameters
        ----------
        img : np.ndarray
            2D numpy array representing the image to be cropped.
        shape : tuple
            Tuple of length 2, specifying the crop shape. Empty spaces are not
            filled, so for example, if an image of shape (6, 6) is cropped to
            shape (4, 8), the resulting shape is (4, 6).

        Returns
        -------
        np.ndarray
            The cropped 2D image.
        """

        # Determine how much to crop off each axis
        cropWidth = np.subtract(img.shape, shape)

        # Split these widths into front and back halves
        # If a width is odd, the front half is one less than the back half
        cropIdx = np.repeat(cropWidth, 2).reshape(2, 2)
        cropIdx[:, 0] = np.floor(cropIdx[:, 0] / 2)
        cropIdx[:, 1] = np.subtract(img.shape, np.ceil(cropIdx[:, 1] / 2))
        cropIdx = np.clip(cropIdx, 0, None)

        # crop the image
        imgCropped = img[
            cropIdx[0, 0] : cropIdx[0, 1],
            cropIdx[1, 0] : cropIdx[1, 1],
        ]

        return imgCropped

    def _reshapeImage(self, img, shape, mlReshape="pad"):
        """Reshape the image.

        This method does not assume the image is the correct size to be naively
        reshaped. In other words, this function works when img.reshape(shape)
        does not. It achieves this either by padding, cropping, or zooming.
        See `mlReshape` below.

        Parameters
        ----------
        img : np.ndarray
            2D numpy array representing the image to be reshaped.
        shape : tuple
            Tuple of arbitrary length, specifying the output shape of the
            image. The tuple must have the form (..., H, W), where H and W
            are the height and width of the image, respectively. The preceding
            values in the tuple ("..." above) can be batches, channels, etc.
        mlReshape : str, optional
            How to handle incompatible sizes in the last two image dimensions.
            - "pad" : the image is symmetrically zero-padded to expand the
                        image to the correct shape. This assumes the input
                        image is smaller than the requested shape.
            - "crop" : the image is cropped to fit inside the requested shape,
                        then any empty space is filled with zeros to achieve
                        the requested shape.
            - "zoomMin" : the image is zoomed to the maximum extent that still
                        fits inside the requested shape. Any empty space is
                        then filled with zeros to achieve the requested shape.
                        This is only different from "zoomMax" if either the
                        image or the requested shape is rectangular.
            - "zoomMax" : the image is zoomed to the minimum extent that fills
                        the entire requested shape. Any part of the image that
                        lies outside the requested shape is cropped. This is
                        only different from "zoomMin" if either the image or
                        the requested shape is rectangular.
            (the default is "pad")

        Raises
        ------
        ValueError
            mlReshape="pad" and the image does not fit inside requested shape.
        ValueError
            mlReshape value is invalid.
        """

        # first we will worry about reshaping the 2D image that corresponds
        # to the final two dimensions in shape
        if mlReshape == "pad":
            # check that the input image isn't too large
            if np.greater(img.shape, shape[-2:]).any():
                raise ValueError(
                    f"The image has shape {img.shape}, which cannot fit "
                    f"inside the requested input shape {shape[-2:]}. "
                    "Either use a smaller image, a different model, "
                    "or set mlReshape='crop', 'zoomMin', or 'zoomMax'."
                )

            # pad the image
            imgReshaped = self._padImage(img, shape[-2:])

        elif mlReshape == "crop":
            # crop the image
            imgReshaped = self._cropImage(img, shape[-2:])

            # fill empty space with zeros
            imgReshaped = self._padImage(imgReshaped, shape[-2:])

        elif mlReshape == "zoomMin":
            # zoom to the maximum extent that still fits inside the shape
            zoom_factor = np.divide(shape[-2:], img.shape).min()
            imgReshaped = zoom(img, zoom_factor)

            # fill empty space with zeros
            imgReshaped = self._padImage(imgReshaped, shape[-2:])

        elif mlReshape == "zoomMax":
            # zoom to the minimum extent that fills the entire shape
            zoom_factor = np.divide(shape[-2:], img.shape).max()
            imgReshaped = zoom(img, zoom_factor)

            # crop zoomed image to fit the shape
            imgReshaped = self._cropImage(imgReshaped, shape[-2:])

        else:
            raise ValueError(
                "mlReshape must be one of 'pad', 'crop', 'zoomMin' or 'zoomMax'."
            )

        # now we can call numpy.reshape to add any batch or channel dimensions
        imgReshaped = imgReshaped.reshape(shape)

        return imgReshaped

    def _predict(self, img):
        """Predict Zernikes using the model.

        Parameters
        ----------
        img : CompensableImage
            Intra- or extra-focal image.

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in nm.
        """
        with torch.no_grad():
            zk = self._model(img)
        return zk.numpy().squeeze()

    def _runSingleImg(self, img, number):
        """Reshape, predict, and log for a single image.

        Parameters
        ----------
        img : CompensableImage
            Intra- or extra-focal image.
        number : int
            The image number.

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in nm.
        """
        # Save the initial image
        imgInit = img.getImg()
        self._recordItem(imgInit, f"I{number}", 1)

        # Reshape the image
        imgReshaped = self._reshapeImage(
            imgInit, self._model.input_shape, self.mlReshape
        )
        img.updateImage(imgReshaped)
        self._recordItem(imgReshaped, f"I{number}reshaped", 1)

        # Estimate Zernikes
        zk = self._predict(img)
        self._recordItem(zk, f"zk{number}", 1)
        if self.debugLevel >= 2:
            print(f"Zernikes_I{number} (nm) = {zk:.3f}")

        # Reset the CompensableImage
        img.updateImage(imgInit)

        return zk

    def runIt(self, I1=None, I2=None):
        """Estimate Zernike coefficients in nm using the ML model.

        This method differs from the method in the TIE Algorithm class (i.e.
        in ts.wep.cwfs.Algorithm) in that it can estimate Zernikes for an
        individual image. Therefore, you need only provide either I1 or I2.
        If both I1 and I2 are provided, then average of the Zernikes are
        returned.

        Parameters
        ----------
        I1 : CompensableImage, optional
            Intra- or extra-focal image.
        I2 : CompensableImage, optional
            Intra- or extra-focal image.

        Returns
        -------
        numpy.ndarray
            Coefficients of Zernike polynomials (z4 - z22), in nm.

        Raises
        ------
        ValueError
            If neither I1 nor I2 are provided, or if I1 and I2 are on the same
            side of focus.
        """

        if I1 is None and I2 is None:
            raise ValueError("You must provide either I1 or I2.")
        if I1 is not None and I1 is not None and I1.defocalType == I2.defocalType:
            raise ValueError(
                "I1 and I2 must be on opposite sides of focus. "
                f"Currently, they are both {I2.defocalType.name}focal."
            )

        # Reset the algorithm history
        self._history = dict()

        # Estimate Zernikes
        zk1 = None if I1 is None else self._runSingleImg(I1, 1)
        zk2 = None if I2 is None else self._runSingleImg(I2, 2)

        # Determine which zernikes to return
        if zk1 is not None and zk2 is not None:
            # Average zk1 and zk2
            zk = np.mean([zk1, zk2], axis=0)

            # Record zk
            self._recordItem(zk, "zk", 1)
            if self.debugLevel >= 2:
                print(f"Zernikes_Avg (nm) = {zk:.3f}")
        else:
            # Return whichever is not None
            zk = zk1 or zk2

        return zk
