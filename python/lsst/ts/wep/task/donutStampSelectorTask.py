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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from astropy.table import Table
from lsst.ts.wep.utils import readConfigYaml
from lsst.utils.timer import timeMethod


class DonutStampSelectorTaskConfig(pexConfig.Config):
    selectWithEntropy = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use entropy in deciding to use the donut.",
    )

    selectWithSignalToNoise = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use signal to noise ratio in deciding to use the donut."
        + "By default the values from snLimitStar.yaml config file are used.",
    )

    useCustomSnLimit = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply user-defined signal to noise minimum cutoff? If this is False then the code"
        + " will default to use the minimum values in snLimitStar.yaml.",
    )

    minSignalToNoise = pexConfig.Field(
        dtype=float,
        default=600,
        doc=str(
            "The minimum signal to noise threshold to use (keep donuts only above the value)."
            + " This is used only if useCustomSnLimit is True."
            + " If used, it overrides values from snLimitStar.yaml."
        ),
    )
    maxEntropy = pexConfig.Field(
        dtype=float,
        default=3.5,
        doc=str("The entropy threshold to use (keep donuts only below the threshold)."),
    )


class DonutStampSelectorTask(pipeBase.Task):
    """
    Donut Stamp Selector uses information about donut stamp calculated at
    the stamp cutting out stage to select those that fulfill entropy
    and/or signal-to-noise criteria.
    """

    ConfigClass = DonutStampSelectorTaskConfig
    _DefaultName = "donutStampSelectorTask"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

    def run(self, donutStamps):
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
                - donutsQuality : `pandas.DataFrame`
                    A table with calculated signal to noise measure and entropy
                    value per donut, together with selection outcome for all
                    input donuts.

        """
        result = self.selectStamps(donutStamps)

        return pipeBase.Struct(
            donutStampsSelect=np.array(donutStamps)[result.selected],
            selected=result.selected,
            donutsQuality=result.donutsQuality,
        )

    @timeMethod
    def selectStamps(self, donutStamps):
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
                - donutsQuality : `pandas.DataFrame`
                    A table with calculated signal to noise measure and entropy
                    value per donut, together with selection outcome for all
                    input donuts.
        """

        # Which donuts to use for Zernike estimation
        # initiate these by selecting all donuts
        entropySelect = np.ones(len(donutStamps), dtype="bool")

        # Collect the entropy information if available
        if "ENTROPY" in list(donutStamps.metadata):
            entropyValue = np.asarray(donutStamps.metadata.getArray("ENTROPY"))
            if self.config.selectWithEntropy:
                entropySelect = entropyValue < self.config.maxEntropy
        else:
            self.log.warning("Entropy not in donut stamps metadata. Using all donuts.")
        # By default select all donuts,  only overwritten
        # if selectWithSignalToNoise is True
        snSelect = np.ones(len(donutStamps), dtype="bool")

        # collect the SN information if available
        if "SN" in list(donutStamps.metadata):
            snValue = np.asarray(donutStamps.metadata.getArray("SN"))

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
        else:
            self.log.warning("SN not in donut stamps metadata. Using all donuts.")
        # AND condition : if both selectWithEntropy
        # and selectWithSignalToNoise, then
        # only donuts that pass with SN criterion as well
        # as entropy criterion are selected
        selected = entropySelect * snSelect

        # store information about which donuts were selected
        donutsQuality = Table(
            data=[
                snValue,
                entropyValue,
                snSelect,
                entropySelect,
                selected,
            ],
            names=["SN", "ENTROPY", "SN_SELECT", "ENTROPY_SELECT", "FINAL_SELECT"],
        ).to_pandas()

        self.log.info("Selected %d/%d donut stamps", selected.sum(), len(donutStamps))

        return pipeBase.Struct(selected=selected, donutsQuality=donutsQuality)