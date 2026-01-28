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

__all__ = ["DonutStampSelectorTaskConfig", "DonutStampSelectorTask"]

from typing import Any

import numpy as np
from astropy.table import QTable

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ts.wep.task.donutStamps import DonutStamps
from lsst.ts.wep.utils import readConfigYaml
from lsst.utils.timer import timeMethod


class DonutStampSelectorTaskConfig(pexConfig.Config):
    maxSelect: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=5,
        doc="Maximum number of donut stamps to select. If -1, all are selected.",
    )
    selectWithEntropy: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use entropy in deciding to use the donut.",
    )
    selectWithSignalToNoise: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to use signal to noise ratio in deciding to use the donut. "
        + "By default the values from snLimitStar.yaml config file are used.",
    )
    selectWithFracBadPixels: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to use fraction of bad pixels in deciding to use the donut. "
        + "Bad pixels correspond to mask values of 'SAT', 'BAD', 'NO_DATA'.",
    )
    selectWithMaxPowerGrad: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to use the max of the gradient of the stamp power spectrum "
        + "(at k < 10) to select donuts. By setting this to a low positive "
        + "value, this ensures the stamp power spectrum is not monotonically "
        + "decreasing within the tolerace specified by maxPowerGradThresh. This "
        + "makes sure we select stamps whose power isn't all at large scales, "
        + "as these stamps lack sharp(-ish) donut edges. This is designed to "
        + "reject galaxy-donuts which are very blurry and therefore have most "
        + "of their power at low k.",
    )
    useCustomSnLimit: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply user-defined signal to noise minimum cutoff? If this is False then the code"
        + " will default to use the minimum values in snLimitStar.yaml.",
    )
    minSignalToNoise: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=600,
        doc=str(
            "The minimum signal to noise threshold to use (keep donuts only above the value)."
            + " This is used only if useCustomSnLimit is True."
            + " If used, it overrides values from snLimitStar.yaml."
        ),
    )
    maxEntropy: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=3.5,
        doc=str("The entropy threshold to use (keep donuts only below the threshold)."),
    )
    maxFracBadPixels: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc=str("Maximum fraction of bad pixels in selected donuts."),
    )
    maxPowerGradThresh: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=1e-4,
        doc=str(
            "Min of the gradient of the stamp power spectrum (at k < 10). "
            + "the stamp's MAX_POWER_GRAD must be above this minimum value "
            + "to be selected."
        ),
    )
    doSelection: pexConfig.Field = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Do any donut stamp selection? If this is False then we return the donut quality table,"
        + " but allow all donut stamps to be selected.",
    )


class DonutStampSelectorTask(pipeBase.Task):
    """
    Donut Stamp Selector uses information about donut stamp calculated at
    the stamp cutting out stage to select those that specified criteria.
    """

    ConfigClass = DonutStampSelectorTaskConfig
    _DefaultName = "donutStampSelectorTask"
    config: DonutStampSelectorTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def run(self, donutStamps: DonutStamps) -> pipeBase.Struct:
        """Select good stamps and return them together with quality table.
        By default all stamps are selected.

        Parameters
        ----------
        donutStamps : `lsst.ts.wep.task.donutStamps.DonutStamps`
            Donut postage stamps.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - donutStampsSelect :
                    `lsst.ts.wep.task.donutStamps.DonutStamps`
                    Selected donut postage stamps.
                - selected : `numpy.ndarray` of `bool`
                    Boolean array of stamps that were selected, same length as
                    donutStamps.
                - donutsQuality : `astropy.table.QTable`
                    A table with calculated signal to noise measure, entropy
                    value per donut, and fraction of bad pixels, together with
                    selection outcome for all input donuts.

        """
        result = self.selectStamps(donutStamps)

        selectedStamps = DonutStamps([donutStamps[i] for i in range(len(donutStamps)) if result.selected[i]])
        selectedStamps._refresh_metadata()
        # Need to copy a few other fields by hand
        for k in ["SN", "ENTROPY", "FRAC_BAD_PIX", "MAX_POWER_GRAD"]:
            if k in donutStamps.metadata:
                selectedStamps.metadata[k] = np.array(
                    [
                        donutStamps.metadata.getArray(k)[i]
                        for i in range(len(donutStamps))
                        if result.selected[i]
                    ]
                )
        for key, val in donutStamps.metadata.items():
            if key.startswith("BORESIGHT") or key in [
                "MJD",
                "VISIT",
                "DFC_DIST",
                "DET_NAME",
                "BANDPASS",
            ]:
                selectedStamps.metadata[key] = val
            else:
                continue

        return pipeBase.Struct(
            donutStampsSelect=selectedStamps,
            selected=result.selected,
            donutsQuality=result.donutsQuality,
        )

    @timeMethod
    def selectStamps(self, donutStamps: DonutStamps) -> pipeBase.Struct:
        """
        Run the stamp selection algorithm and return the indices
        of donut stamps that that fulfill the selection criteria,
        as well as the table of calculated signal to noise measure
        and entropy value per donut. By default all stamps are
        selected.


        Parameters
        ----------
        donutStamps : `lsst.ts.wep.task.donutStamps.DonutStamps`
            Donut postage stamps.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            The struct contains the following data:
                - selected : `numpy.ndarray` of `bool`
                    Boolean array of stamps that were selected, same length as
                    donutStamps.
                - donutsQuality : `astropy.table.QTable`
                    A table with calculated signal to noise measure and entropy
                    value per donut, together with selection outcome for all
                    input donuts.
        """
        # Which donuts to use for Zernike estimation
        # initiate these by selecting all donuts
        entropySelect = np.ones(len(donutStamps), dtype="bool")

        # Collect donut id values
        if "DONUT_ID" in list(donutStamps.metadata):
            donutId = donutStamps.metadata.getArray("DONUT_ID")
        else:
            donutId = np.zeros(len(donutStamps), dtype="str")

        # Collect the entropy information if available
        entropyValue = np.full(len(donutStamps), np.nan)
        if "ENTROPY" in list(donutStamps.metadata):
            fillVals = np.asarray(donutStamps.metadata.getArray("ENTROPY"))
            entropyValue[: len(fillVals)] = fillVals
            if self.config.selectWithEntropy:
                entropySelect = entropyValue < self.config.maxEntropy
                self.log.info(
                    f"{sum(entropySelect)} of {len(entropySelect)} donuts passed entropy selection."
                )
        elif self.config.selectWithEntropy:
            self.log.warning("selectWithEntropy==True but ENTROPY not in stamp metadata.")

        # Collect the donut radius metric from stamps metadata
        if "RADIUS" in list(donutStamps.metadata):
            donutRadii = np.ones(len(donutStamps)) * donutStamps.metadata["RADIUS"]
        else:
            donutRadii = np.zeros(len(donutStamps))

        # By default select all donuts,  only overwritten
        # if selectWithSignalToNoise is True
        snSelect = np.ones(len(donutStamps), dtype="bool")

        # collect the SN information if available
        snValue = np.full(len(donutStamps), np.nan)
        if "SN" in list(donutStamps.metadata):
            fillVals = np.asarray(donutStamps.metadata.getArray("SN"))
            snValue[: len(fillVals)] = fillVals
            if self.config.selectWithSignalToNoise:
                # Use user defined SN cutoff or the filter-dependent
                # defaults, depending on useCustomSnLimit
                if self.config.useCustomSnLimit:
                    snThreshold = self.config.minSignalToNoise
                else:
                    snPolicyDefaults = readConfigYaml("policy:snLimitStar.yaml")
                    filterName = donutStamps[0].bandpass
                    filterKey = f"filter{filterName.upper()}"
                    snThreshold = snPolicyDefaults[filterKey]

                # Select using the given threshold
                snSelect = snThreshold < snValue
                self.log.info(f"{sum(snSelect)} of {len(snSelect)} donuts passed SNR selection.")

        elif self.config.selectWithSignalToNoise:
            self.log.warning("selectWithSignalToNoise==True but SN not in stamp metadata.")

        # By default select all donuts,  only overwritten
        # if selectWithFracBadPixels is True
        fracBadPixSelect = np.ones(len(donutStamps), dtype="bool")

        # collect fraction-of-bad-pixels information if available
        fracBadPix = np.full(len(donutStamps), np.nan)
        if "FRAC_BAD_PIX" in list(donutStamps.metadata):
            fillVals = np.asarray(donutStamps.metadata.getArray("FRAC_BAD_PIX"))
            fracBadPix[: len(fillVals)] = fillVals
            if self.config.selectWithFracBadPixels:
                fracBadPixSelect = fracBadPix <= self.config.maxFracBadPixels
                self.log.info(
                    f"{sum(fracBadPixSelect)} of {len(fracBadPixSelect)} donuts passed bad pixel selection."
                )
        elif self.config.selectWithFracBadPixels:
            self.log.warning("selectWithFracBadPixels==True but FRAC_BAD_PIX not in stamp metadata.")

        # By default select all donuts,  only overwritten
        # if selectWithMaxPowerGrad is True
        maxPowerGradSelect = np.ones(len(donutStamps), dtype="bool")

        # collect fraction-of-bad-pixels information if available
        maxPowerGrad = np.full(len(donutStamps), np.nan)
        if "MAX_POWER_GRAD" in list(donutStamps.metadata):
            fillVals = np.asarray(donutStamps.metadata.getArray("MAX_POWER_GRAD"))
            maxPowerGrad[: len(fillVals)] = fillVals
            if self.config.selectWithMaxPowerGrad:
                maxPowerGradSelect = maxPowerGrad > self.config.maxPowerGradThresh
                self.log.info(
                    f"{sum(maxPowerGradSelect)} of {len(maxPowerGradSelect)} "
                    "donuts passed power spectrum selection."
                )
        elif self.config.selectWithMaxPowerGrad:
            self.log.warning("selectWithMaxPowerGrad==True but MAX_POWER_GRAD not in stamp metadata.")

        # choose only donuts that satisfy all selected conditions
        if self.config.doSelection:
            selected = entropySelect * snSelect * fracBadPixSelect * maxPowerGradSelect
            self.log.info(f"{sum(selected)} of {len(selected)} donuts passed combined selection criteria.")
            # make sure we don't select more than maxSelect
            if self.config.maxSelect != -1:
                selected[np.cumsum(selected) > self.config.maxSelect] = False

        else:
            selected = np.ones(len(donutStamps), dtype="bool")
            self.log.info("Donut stamp selector is off.")

        # store information about which donuts were selected
        # use QTable even though no units at the moment in
        # case we end up adding more later we don't have to change
        # data storage type in butler and rename
        donutsQuality = QTable(
            data=[
                snValue,
                entropyValue,
                fracBadPix,
                maxPowerGrad,
                snSelect,
                entropySelect,
                fracBadPixSelect,
                maxPowerGradSelect,
                donutRadii,
                selected,
                donutId,
            ],
            names=[
                "SN",
                "ENTROPY",
                "FRAC_BAD_PIX",
                "MAX_POWER_GRAD",
                "SN_SELECT",
                "ENTROPY_SELECT",
                "FRAC_BAD_PIX_SELECT",
                "MAX_POWER_GRAD_SELECT",
                "RADIUS",
                "FINAL_SELECT",
                "DONUT_ID",
            ],
        )
        # Add all configuration used for selection criteria as metadata;
        # that way it doesn't have to be pulled from calcZernikes config
        donutsQuality.meta = {"DonutStampSelectorTaskConfig": self.config.toDict()}
        self.log.info("Selected %d/%d donut stamps", selected.sum(), len(donutStamps))

        return pipeBase.Struct(selected=selected, donutsQuality=donutsQuality)
