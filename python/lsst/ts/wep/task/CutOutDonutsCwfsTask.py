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

__all__ = ["CutOutDonutsCwfsTaskConfig", "CutOutDonutsCwfsTask"]

import typing
import numpy as np
import pandas as pd

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.obs.lsst as obs_lsst
from lsst.utils.timer import timeMethod

from lsst.ts.wep.Utility import DefocalType
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.task.CutOutDonutsBase import (
    CutOutDonutsBaseTaskConnections,
    CutOutDonutsBaseTaskConfig,
    CutOutDonutsBaseTask,
)


class CutOutDonutsCwfsTaskConfig(
    CutOutDonutsBaseTaskConfig,
    pipelineConnections=CutOutDonutsBaseTaskConnections,
):
    pass


class CutOutDonutsCwfsTask(CutOutDonutsBaseTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = CutOutDonutsCwfsTaskConfig
    _DefaultName = "CutOutDonutsCwfsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set size (in pixels) of donut template image used for
        # final centroiding by convolution of initial cutout with template
        self.donutTemplateSize = self.config.donutTemplateSize
        # Set final size (in pixels) of postage stamp images returned as
        # DonutStamp objects
        self.donutStampSize = self.config.donutStampSize
        # Add this many pixels onto each side of initial
        # cutout stamp beyond the size specified
        # in self.donutStampSize. This makes sure that
        # after recentroiding the donut from the catalog
        # position by convolving a template on the initial
        # cutout stamp we will still have a postage stamp
        # of size self.donutStampSize.
        self.initialCutoutPadding = self.config.initialCutoutPadding
        # Specify optical model
        self.opticalModel = self.config.opticalModel

        # Set which sensors are extra and intra focal
        # See LCA-13381 for definition
        self.extraFocalNames = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
        self.intraFocalNames = ["R00_SW1", "R04_SW1", "R40_SW1", "R44_SW1"]

    def runQuantum(
        self,
        butlerQC: pipeBase.ButlerQuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ):
        """
        We need to be able to take pairs of detectors from the full
        set of detector exposures and run the task. Then we need to put
        the outputs back into the butler repository with
        the appropriate butler dataIds.

        For the `outputZernikesRaw` and `outputZernikesAvg`
        we only have one set of values per pair of wavefront detectors
        so we put this in the dataId associated with the
        extra-focal detector.
        """

        camera = butlerQC.get(inputRefs.camera)

        # Get the detector IDs for the wavefront sensors so
        # that we can appropriately match up pairs of detectors
        if camera.getName() == "LSSTCam":
            detectorMap = (
                obs_lsst.translators.lsstCam.LsstCamTranslator.detector_mapping()
            )
        else:
            raise ValueError(f"{camera.getName()} is not a valid camera name.")

        extraFocalIds = [detectorMap[detName][0] for detName in self.extraFocalNames]
        intraFocalIds = [detectorMap[detName][0] for detName in self.intraFocalNames]

        detectorIdArr = np.array(
            [exp.dataId["detector"] for exp in inputRefs.exposures]
        )
        donutCatIdArr = np.array(
            [dCat.dataId["detector"] for dCat in inputRefs.donutCatalog]
        )

        # Find cwfs detectors in the list of detectors being processed
        runExtraIds = list(set(detectorIdArr).intersection(extraFocalIds))
        runExtraIds.sort()
        runIntraIds = list(set(detectorIdArr).intersection(intraFocalIds))
        runIntraIds.sort()
        if len(runExtraIds) != len(runIntraIds):
            raise ValueError("Unequal number of intra and extra focal detectors.")

        for extraId, intraId in zip(runExtraIds, runIntraIds):
            if abs(extraId - intraId) != 1:
                raise ValueError("Intra and extra focal detectors not adjacent.")
            extraListIdx = np.where(detectorIdArr == extraId)[0][0]
            intraListIdx = np.where(detectorIdArr == intraId)[0][0]
            dCatExtraIdx = np.where(donutCatIdArr == extraId)[0][0]
            dCatIntraIdx = np.where(donutCatIdArr == intraId)[0][0]
            expInputs = butlerQC.get(
                [inputRefs.exposures[extraListIdx], inputRefs.exposures[intraListIdx]]
            )
            dCatInputs = butlerQC.get(
                [
                    inputRefs.donutCatalog[dCatExtraIdx],
                    inputRefs.donutCatalog[dCatIntraIdx],
                ]
            )
            outputs = self.run(expInputs, dCatInputs, camera)

            butlerQC.put(
                outputs.donutStampsExtra, outputRefs.donutStampsExtra[extraListIdx]
            )
            # Assign both outputs to the same dataId so that we can run
            # Zernike estimation fully in parallel through the dataIds
            # of the extra-focal detectors using CalcZernikesTask.
            butlerQC.put(
                outputs.donutStampsIntra, outputRefs.donutStampsIntra[extraListIdx]
            )

    @timeMethod
    def run(
        self,
        exposures: typing.List[afwImage.Exposure],
        donutCatalogs: typing.List[pd.DataFrame],
        camera: lsst.afw.cameraGeom.Camera,
    ) -> pipeBase.Struct:

        cameraName = camera.getName()
        extraCatalog, intraCatalog = donutCatalogs

        # Get the donut stamps from extra and intra focal images
        donutStampsExtra = DonutStamps([])
        donutStampsIntra = DonutStamps([])

        for exposure in exposures:
            if self.config.instDefocalOffset is None:
                self.instParams["offset"] = exposure.visitInfo.focusZ
            else:
                self.instParams["offset"] = self.config.instDefocalOffset
            detectorName = exposure.getDetector().getName()
            if detectorName in self.extraFocalNames:
                # LSST extrafocal chips are offset -1.5 mm
                # when LSST camera defocus is at 0.
                self.instParams["offset"] = np.abs(self.instParams["offset"] - 1.5)
                donutStampsExtraExp = self.cutOutStamps(
                    exposure, extraCatalog, DefocalType.Extra, cameraName
                )
                donutStampsExtra.extend([stamp for stamp in donutStampsExtraExp])
            elif detectorName in self.intraFocalNames:
                # LSST intrafocal chips are offset +1.5 mm
                # when LSST camera defocus is at 0.
                self.instParams["offset"] = np.abs(self.instParams["offset"] + 1.5)
                donutStampsIntraExp = self.cutOutStamps(
                    exposure, intraCatalog, DefocalType.Intra, cameraName
                )
                donutStampsIntra.extend([stamp for stamp in donutStampsIntraExp])
            else:
                continue

        # If no donuts are in the donutCatalog for a set of exposures
        # then return the Zernike coefficients as nan.
        if (len(donutStampsExtra) == 0) or (len(donutStampsIntra) == 0):
            return pipeBase.Struct(
                donutStampsExtra=DonutStamps([]),
                donutStampsIntra=DonutStamps([]),
            )

        # Return extra-focal DonutStamps, intra-focal DonutStamps and
        # Zernike coefficient numpy array as Struct that can be saved to
        # Gen 3 repository all with the same dataId.
        return pipeBase.Struct(
            donutStampsExtra=donutStampsExtra,
            donutStampsIntra=donutStampsIntra,
        )
