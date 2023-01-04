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

from lsst.ts.wep.ParamReader import ParamReader
import numpy as np
import torch


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

    def config(self, mlFile, debugLevel=0):
        """Configure the algorithm to estimate Zernikes.

        Parameters
        ----------
        mlFile : str
            Path where the machine learning model is saved.
        debugLevel : int, optional
            Determines whether to save or print information during running.
            If 0, nothing is done; if 1, zernikes are saved in the history;
            if 2, zernikes are printed to the terminal. (default is 0.)
        """

        # Load the model
        model = torch.jit.load(mlFile)
        model.eval()
        self._model = model

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
        """The algorithm history.

        The algorithm history is a dictionary with the entries:
            - zk1 - the Zernikes for I1, if I1 is not None
            - zk2 - the Zernikes for I2, if I2 is not None
            - zk - the average of zk1 and zk2
        """
        return self._history

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

        if I1 is not None:
            # Estimate Zernikes for I1
            zk1 = self._predict(I1)

            # Record zk1
            self._recordItem(zk1, "zk1", 1)
            if self.debugLevel >= 2:
                print(f"Zernikes_I1 (nm) = {zk1:.3f}")

        if I2 is not None:
            # Estimate Zernikes for I2
            zk2 = self._predict(I2)

            # Record zk2
            self._recordItem(zk2, "zk2", 1)
            if self.debugLevel >= 2:
                print(f"Zernikes_I2 (nm) = {zk2:.3f}")

        # Determine which zernikes to return
        if I1 is not None and I2 is not None:
            # Average zk1 and zk2
            zk = np.mean([zk1, zk2], axis=1)

            # Record zk
            self._recordItem(zk, "zk", 1)
            if self.debugLevel >= 2 and I2 is not None:
                print(f"Zernikes_I2 (nm) = {zk2:.3f}")

        elif I1 is not None:
            zk = zk1

        else:
            zk = zk2

        return zk
