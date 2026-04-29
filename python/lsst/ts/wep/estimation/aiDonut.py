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
    temperature : float, optional
        Temperature parameter for softmax weighting of predictions
        when both intra- and extra-focal images are provided. Must be
        a positive float. Lower values place more weight on the prediction
        with lower estimated error. Default is 0.005.

    Notes
    -----
    Model Requirements:

    - Must be saved in TorchScript format.
    - Must accept the following inputs:

        - img - batch of images with shape (N, 1, H, W), where N is 1 or 2.
        - fx - field angle in x (degrees)
        - fy - field angle in y (degrees)
        - focalFlag - 1 for intra-focal, 0 for extra-focal
        - band - integer 0-5 indicating filter band (ugrizy)

    - Must output a tensor of shape (N, n_zernikes), where n_zernikes is the
      number of Zernike coefficients predicted, starting with Noll index 4.
    - Zernikes must be returned in meters.
    - Must have a ``nollIndices`` attribute listing the Noll indices
      corresponding to the output coefficients.
    """

    def __init__(
        self,
        modelPath: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        temperature: float = 0.005,
    ) -> None:
        self.device = device
        self.temperature = temperature
        self.modelPath = os.path.expandvars(modelPath)

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
    def temperature(self) -> float:
        """Temperature parameter for softmax weighting of predictions."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature parameter.

        Parameters
        ----------
        value : float
            Temperature parameter. Must be a positive float.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Temperature must be a number.")
        if value <= 0:
            raise ValueError("Temperature must be positive.")
        self._temperature = float(value)

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
        try:
            self.model = torch.load(value, map_location=self.device, weights_only=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {value}") from e
        self.model.eval()  # Put model in evaluation mode
        self._modelPath = value  # If loading model succeeded, save path

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary with the following entries:

        - "modelPath" - path to the PyTorch model file
        - "device" - device used for inference
        - "temperature" - temperature used for weighted averaging
        - "modelNollIndices" - Noll indices the model predicts
        - "intra" and/or "extra" - all Zernikes returned by the model
        - "nollIndices" - Noll indices for which Zernikes are returned
        - "zk" - the final, averaged Zernike estimate

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
        """Estimate Zernike coefficients using a PyTorch model.

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
            Noll indices for which to estimate Zernike coefficients.
        instrument : Instrument
            Instrument object (unused; exists for compatibility).
        saveHistory : bool
            Whether to save the algorithm history in the ``self.history``
            attribute. If True, then ``self.history`` contains information
            about the most recent time the algorithm was run.

        Returns
        -------
        zk : np.ndarray
            Zernike coefficients for the provided Noll indices, estimated from
            the images, in meters.
        zkMeta : dict
            Dictionary containing metadata such as FWHM estimates and weights.
        """
        # Verify the model supports all requested Noll indices.
        if not set(nollIndices).issubset(set(self.model.nollIndices.tolist())):
            raise ValueError(
                f"Requested Noll indices {nollIndices.tolist()} are not "
                "supported by the model, which can only predict some subset of "
                f"{self.model.nollIndices.tolist()}."
            )

        # Stack inputs into a batch.
        imgs = [I1.image]
        fxs = [I1.fieldAngle[0]]
        fys = [I1.fieldAngle[1]]
        focalFlags = [1 if I1.defocalType.value == "intra" else 0]
        bands = ["ugrizy".index(I1.bandLabel.value)]
        if I2 is not None:
            imgs.append(I2.image)
            fxs.append(I2.fieldAngle[0])
            fys.append(I2.fieldAngle[1])
            focalFlags.append(1 if I2.defocalType.value == "intra" else 0)
            bands.append("ugrizy".index(I2.bandLabel.value))

        # Convert to numpy arrays.
        imgsNp = np.stack(imgs, axis=0)
        fxsNp = np.array(fxs).reshape(-1, 1)
        fysNp = np.array(fys).reshape(-1, 1)
        focalFlagsNp = np.array(focalFlags).reshape(-1, 1)
        bandsNp = np.array(bands).reshape(-1, 1)

        # Convert to torch tensors.
        imgsTch = torch.from_numpy(imgsNp).float().to(self.device)
        fxsTch = torch.from_numpy(fxsNp).float().to(self.device)
        fysTch = torch.from_numpy(fysNp).float().to(self.device)
        focalFlagsTch = torch.from_numpy(focalFlagsNp).float().to(self.device)
        bandsTch = torch.from_numpy(bandsNp).float().to(self.device)

        # Run the model.
        with torch.no_grad():
            outputs = self.model(
                imgsTch,
                fxsTch,
                fysTch,
                focalFlagsTch,
                bandsTch,
            )

        # Split outputs. Models may return:
        #   - a single tensor (zk only)
        #   - a 2-tuple (zk, zkScore)
        #   - a 3-tuple (zk, zkScore, fwhm)
        if isinstance(outputs, tuple):
            outZk = outputs[0].cpu().numpy()
            outZkScore = outputs[1].cpu().numpy() if len(outputs) > 1 else np.full_like(outZk, np.nan)
            outFwhm = outputs[2].cpu().numpy() if len(outputs) > 2 else np.full((len(imgs), 2), np.nan)
        else:
            outZk = outputs.cpu().numpy()
            outZkScore = np.full_like(outZk, np.nan)
            outFwhm = np.full((len(imgs), 2), np.nan)

        # Zero out entire stamp if any of its scores are NaN — a NaN score
        # for one Zernike indicates the whole stamp prediction is unreliable.
        # This allows the other stamp in the pair to still contribute normally.
        finite_mask = np.isfinite(outZkScore).all(axis=1)  # (N,) per stamp
        if finite_mask.any():
            rawWeights = np.where(
                finite_mask[:, None],
                np.exp(-outZkScore / self.temperature),
                0.0,
            )
            pairWeight = float(rawWeights.sum())
            col_sums = rawWeights.sum(axis=0, keepdims=True)
            weights = np.where(col_sums > 0, rawWeights / col_sums, 1.0 / finite_mask.sum())
            zk = (outZk * weights).sum(axis=0)
        else:
            pairWeight = 1.0
            zk = outZk.mean(axis=0)

        zkMeta = {"fwhm": outFwhm.mean(axis=0), "weight": pairWeight}

        # Sparsify zk to the requested Noll indices.
        zk = makeDense(zk, self.model.nollIndices)
        zk = makeSparse(zk, nollIndices)

        # Save history if requested.
        if saveHistory:
            self._history = {
                "modelPath": self.modelPath,
                "device": self.device,
                "temperature": self.temperature,
                "modelNollIndices": self.model.nollIndices,
                I1.defocalType.value: {
                    "zk": outZk[0],
                    "zkScore": outZkScore[0],
                    "fwhm": outFwhm[0],
                },
            }
            if I2 is not None:
                self._history |= {
                    I2.defocalType.value: {
                        "zk": outZk[1],
                        "zkScore": outZkScore[1],
                        "fwhm": outFwhm[1],
                    }
                }
            self._history |= {
                "nollIndices": nollIndices,
                "zk": zk,
                "fwhm": zkMeta["fwhm"],
                "weight": zkMeta["weight"],
            }

        return zk, zkMeta
