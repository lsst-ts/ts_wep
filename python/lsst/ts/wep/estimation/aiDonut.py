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

import numpy as np
import torch

from lsst.ts.wep import Image, Instrument
from lsst.ts.wep.estimation.wfAlgorithm import WfAlgorithm
from lsst.ts.wep.utils import getModulePath, makeDense, makeSparse

__all__ = ["AiDonutAlgorithm"]

DEFAULT_MODEL_PATH = getModulePath() + "/tests/testData/testAiModels/test_aidonut_model_file.pt"


class AiDonutAlgorithm(WfAlgorithm):
    """Wavefront estimation using a PyTorch model.

    Parameters
    ----------
    modelPath : str
        Path to the torchscript model file. See notes below for
        model requirements. Default is a test model included with ts_wep.
    device : str, optional
        Device to load the model on ('cpu' or 'cuda'). Default is 'cpu'.

    Model Requirements
    ------------------
    - must be saved in TorchScript format
    - must accept following inputs:
        - img - batch of images with shape (N, 1, H, W), where N is 1 or 2.
        - fx - field angle in x (degrees)
        - fy - field angle in y (degrees)
        - focalFlag - 1 for intra-focal, 0 for extra-focal
        - band - integer 0-5 indicating filter band (ugrizy)
    - must output a tensor of shape (N, n_zernikes), where n_zernikes is the
    number of Zernike coefficients predicted, starting with Noll index 4.
    - Zernikes must be returned in meters.
    - must have a `nollIndices` attribute that lists the Noll indices
    corresponding to the output coefficients.
    """

    def __init__(
        self,
        modelPath: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.modelPath = modelPath

    @property
    def requiresPairs(self) -> bool:
        """This algorithm does not require a pair of images."""
        return False

    @property
    def device(self) -> str:
        """Device used for inference ('cpu' or 'cuda')."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """Set the device used for inference.

        Parameters
        ----------
        value : str
            Device to load the model on ('cpu' or 'cuda').
        """
        if value not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        self._device = value

    @property
    def modelPath(self) -> str:
        """Path to the PyTorch model file."""
        return self._modelPath

    @modelPath.setter
    def modelPath(self, value: str) -> None:
        """Set the path to the PyTorch model file and load the model.

        Parameters
        ----------
        value : str
            Path to the PyTorch model file.
        """
        # Load the model
        try:
            self.model = torch.load(value, map_location=self.device, weights_only=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {value}") from e
        self.model.eval()  # Put in evaluation mode

        # If loading the model succeeded, save the path
        self._modelPath = value

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary with the following entries:
        - "modelPath" - path to the PyTorch model file
        - "device" - device used for inference
        - "modelNollIndices" - Noll indices the model predicts
        - "intra" and/or "extra" - All Zernikes returned by model for images
        - "nollIndices" - Noll indices for which Zernikes are returned
        - "zk" - the final, averaged zernike estimate

        Note the units for all Zernikes are in meters, and all Zernikes start
        with Noll index 4.
        """
        return super().history

    def _estimateZk(
        self,
        I1: Image,
        I2: Image | None,
        zkStartI1: np.ndarray,
        zkStartI2: np.ndarray | None,
        nollIndices: np.ndarray,
        instrument: Instrument,
        saveHistory: bool,
    ) -> tuple[np.ndarray, dict]:
        """
        Estimate Zernike coefficients using a PyTorch model.

        Parameters
        ----------
        I1 : Image
            An Image object containing an intra- or extra-focal donut image.
        I2 : Image or None
            A second image, on the opposite side of focus from I1. Can be None.
        zkStartI1 : np.ndarray
            Starting Zernikes for I1 (unused; exists for compatibility).
        zkStartI2 : np.ndarray or None
            Starting Zernikes for I2 (unused; exists for compatibility).
        nollIndices : np.ndarray
            Noll indices for which you wish to estimate Zernike coefficients.
        instrument : Instrument
            Instrument object (unused; exists for compatibility).
        saveHistory : bool
            Whether to save the algorithm history in the self.history
            attribute. If True, then self.history contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        np.ndarray
            Zernike coefficients for the provided Noll indices, estimated from
            the images, in meters.
        dict
            Empty dictionary (exists for compatibility).
        """
        # First, let's make sure we haven't requested any Zernikes
        # the model doesn't predict
        if not set(nollIndices).issubset(set(self.model.nollIndices.tolist())):
            raise ValueError(
                f"Requested Noll indices {nollIndices.tolist()} are not "
                "supported by the model, which can only predict some subset of "
                f"{self.model.nollIndices.tolist()}."
            )

        # Stack inputs into a batch
        imgs = [I1.image]
        fxs, fys = [I1.fieldAngle[0]], [I1.fieldAngle[1]]
        focalFlags = [1 if I1.defocalType.value == "intra" else 0]
        bands = ["ugrizy".index(I1.bandLabel.value)]
        if I2 is not None:
            imgs.append(I2.image)
            fxs.append(I2.fieldAngle[0])
            fys.append(I2.fieldAngle[1])
            focalFlags.append(1 if I2.defocalType.value == "intra" else 0)
            bands.append("ugrizy".index(I2.bandLabel.value))

        # Stack arrays
        imgs_np = np.stack(imgs, axis=0)
        fxs_np = np.array(fxs).reshape(-1, 1)
        fys_np = np.array(fys).reshape(-1, 1)
        focalFlags_np = np.array(focalFlags).reshape(-1, 1)
        bands_np = np.array(bands).reshape(-1, 1)

        # Convert to torch
        imgs_tch = torch.from_numpy(imgs_np).float().to(self.device)
        fxs_tch = torch.from_numpy(fxs_np).float().to(self.device)
        fys_tch = torch.from_numpy(fys_np).float().to(self.device)
        focalFlags_tch = torch.from_numpy(focalFlags_np).float().to(self.device)
        bands_tch = torch.from_numpy(bands_np).float().to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(
                imgs_tch,
                fxs_tch,
                fys_tch,
                focalFlags_tch,
                bands_tch,
            )
        outputs = outputs.cpu().numpy()  # shape: (2, n_zernikes)

        # Average the two estimates
        zk = outputs.mean(axis=0)

        # Only return requested nollIndices
        zk = makeDense(zk, self.model.nollIndices)
        zk = makeSparse(zk, nollIndices)

        # Save history if requested
        if saveHistory:
            # Metadata and I1 Zernikes
            self._history = {
                "modelPath": self.modelPath,
                "device": self.device,
                "modelNollIndices": self.model.nollIndices,
                I1.defocalType.value: outputs[0],
            }
            # Save Zernikes for I2, if present
            if I2 is not None:
                self._history |= {I2.defocalType.value: outputs[1]}
            # Average Zernikes
            self._history |= {
                "nollIndices": nollIndices,
                "zk": zk,
            }

        return zk, {}
