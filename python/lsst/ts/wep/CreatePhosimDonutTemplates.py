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
import shutil
import numpy as np
from lsst.ts.wep.Utility import getConfigDir, runProgram, DefocalType, CentroidFindType
from lsst.ts.wep.CamIsrWrapper import CamIsrWrapper
from lsst.ts.wep.CamDataCollector import CamDataCollector
from lsst.ts.wep.cwfs.Image import Image
from lsst.ts.wep.cwfs.CentroidFindFactory import CentroidFindFactory
from lsst.ts.wep.ctrlIntf.MapSensorNameAndId import MapSensorNameAndId


class CreatePhosimDonutTemplates(object):
    def __init__(self):
        """
        Class to create donut templates from Phosim. The templates
        will be saved to `policy/cwfs/donutTemplateData/phosimTemplates`.
        Requires path to Phosim to be set as environment variable
        `PHOSIMPATH`.
        """

        # Location of the data needed for Phosim to generate the templates
        self.templateDataPath = os.path.join(
            getConfigDir(), "cwfs", "donutTemplateData"
        )
        # Set up the temporary work directory for Phosim output
        self.tempWorkPath = os.path.join(self.templateDataPath, "tempDir")
        # Specify the location of the butler repo for ingestion of
        # Phosim output and to store the postISR images
        self.repoDir = os.path.join(self.tempWorkPath, "input")
        # The location where we will store the final templates
        self.templateDestDir = os.path.join(self.templateDataPath, "phosimTemplates")

    def setTempWorkPaths(self, newBasePath):
        """
        Point to a new temporary work directory. Update work repo as well.

        Parameters
        ----------
        newBasePath : str
            New location of temporary work directories.
        """

        self.tempWorkPath = newBasePath
        self.repoDir = os.path.join(self.tempWorkPath, "input")

    def createWorkDirectories(self):
        """
        Create the final template directory as well as
        temporary work directories.
        """

        print("Making temporary work directories")
        # First clean up previous work if it exists
        if os.path.exists(self.tempWorkPath):
            self.cleanUpWorkDirs()

        os.makedirs(os.path.join(self.tempWorkPath, "phosimOutput", "extra"))
        os.mkdir(os.path.join(self.tempWorkPath, "phosimOutput", "intra"))
        os.mkdir(os.path.join(self.tempWorkPath, "phosimWorkDir"))
        os.mkdir(os.path.join(self.tempWorkPath, "input"))
        os.mkdir(os.path.join(self.tempWorkPath, "raw"))
        os.mkdir(os.path.join(self.tempWorkPath, "calibs"))

    def createDetectorLists(self, detectorStr=""):
        """
        Create the list of detectors for Phosim and generating flats.

        Parameters
        ----------
        detectorStr : str, optional
            String specifying a set of detectors to generate phosim templates.
            A space is required between each detector name
            (Example: "R22_S11 R22_S10"). If the str is "" then it will
            generate a template for every detector in the focal plane.
            (The default is "".)

        Returns
        -------
        str
            String with the detector names specified in format required
            for Phosim input on the command line.
        str
            String with the detector names specified in format required
            to make flats.
        """

        if detectorStr == "":
            # Load default sensor file for ts_wep
            sensorNameFile = MapSensorNameAndId()._sensorNameToIdFile
            # Get all sensors in file
            sensorNameList = list(sensorNameFile.getContent().keys())
        else:
            sensorNameList = detectorStr.split(" ")

        detectorStrPhosim = "|".join(sensorNameList)
        detectorStrFlats = " ".join(sensorNameList)

        return detectorStrPhosim, detectorStrFlats

    def generateDefocalImages(self, detectorStrPhosim, numOfProc):
        """
        Run Phosim to generate the defocal images using
        the provided instance catalogs which generate
        a single donut at the center of each detector.

        Parameters
        ----------
        detectorStrPhosim : str
            String with the detector names specified in format required
            for Phosim input on the command line. Example: "R22_S11|R22_S10"
        numOfProc : int
            Number of processors to use with phosim
        """

        print(f"Running phosim with {numOfProc} processors")

        phosimPath = os.getenv("PHOSIMPATH")
        runPhosimCmd = f"python {phosimPath}phosim.py"
        runPhosimArgs = f"-w {self.tempWorkPath}/phosimWorkDir "
        runPhosimArgs += f'-s "{detectorStrPhosim}" '
        runPhosimArgs += f"-p {numOfProc} "
        runPhosimArgs += "-i lsst "
        runPhosimArgs += "-e 1 "
        runPhosimArgs += f"-c {self.templateDataPath}/star.cmd "

        runPhosimArgsExtra = f"{self.templateDataPath}/starExtra.inst "
        runPhosimArgsExtra += runPhosimArgs
        runPhosimArgsExtra += f"-o {self.tempWorkPath}/phosimOutput/extra"

        runPhosimArgsIntra = f"{self.templateDataPath}/starIntra.inst "
        runPhosimArgsIntra += runPhosimArgs
        runPhosimArgsIntra += f"-o {self.tempWorkPath}/phosimOutput/intra"

        # Generate Defocal Images with Phosim
        runProgram(runPhosimCmd, argstring=runPhosimArgsExtra)
        runProgram(runPhosimCmd, argstring=runPhosimArgsIntra)

    def repackagePhosimImages(self):
        """
        Run the phosim repackager.
        """

        print("Repackaging phosim output")

        argString = f"--out_dir {self.tempWorkPath}/raw "
        argStringExtra = argString + f"{self.tempWorkPath}/phosimOutput/extra"
        argStringIntra = argString + f"{self.tempWorkPath}/phosimOutput/intra"

        # Run repackager
        runProgram("phosim_repackager.py", argstring=argStringExtra)
        runProgram("phosim_repackager.py", argstring=argStringIntra)

    def ingestImages(self):
        """
        Ingest the raw extrafocal images.
        """

        print("Ingest images")

        dataCollector = CamDataCollector(f"{self.tempWorkPath}/input")
        # Create mapper file
        dataCollector.genLsstCamMapper()
        # Run Ingestion
        dataCollector.ingestImages(f"{self.tempWorkPath}/raw/*.fits")

    def makeFlats(self, detectorStrFlats):
        """
        Make flats for ISR.
        """

        print("Making flats")

        # Change to flats directory and make flats. Then change back to CWD.
        cwd = os.getcwd()
        os.chdir(f"{self.tempWorkPath}/calibs")
        runProgram("makeGainImages.py", argstring=f"--detector_list {detectorStrFlats}")
        os.chdir(cwd)

    def ingestCalibs(self):
        """
        Ingest flats for ISR.
        """

        print("Ingest Flats")

        dataCollector = CamDataCollector(f"{self.tempWorkPath}/input")
        dataCollector.ingestCalibs(f"{self.tempWorkPath}/calibs/*.fits")

    def runISR(self):
        """
        Run ISR on extrafocal images.
        """

        print("Running ISR")

        camIsrObj = CamIsrWrapper(self.repoDir)
        camIsrObj.config(doFlat=True, doOverscan=True)
        camIsrObj.doISR(self.repoDir)

    def cutOutIntraExtraTemplates(self, templateWidth, intraVisitId, extraVisitId):
        """
        Cut out the donut templates from the larger extrafocal and intrafocal
        Phosim images for every detector simulated.

        Parameter
        ---------
        templateWidth : int
            Width of square template image in pixels.
        intraVisitId : int
            Visit Id of the intrafocal images from Phosim.
        extraVisitId : int
            Visit Id of the extrafocal images from Phosim.
        """

        postIsrDir = os.path.join(self.repoDir, "rerun", "run1", "postISRCCD",)
        expIds = os.listdir(postIsrDir)

        intraSuffix = str(intraVisitId)[-5:]
        extraSuffix = str(extraVisitId)[-5:]

        #  Always generate both extra- and intra- focal images
        #  so that both  Ids exist
        for expId in expIds:

            if intraSuffix in expId:
                intraExpId = expId
            elif extraSuffix in expId:
                extraExpId = expId

        intraDir = os.path.join(postIsrDir, intraExpId)
        extraDir = os.path.join(postIsrDir, extraExpId)

        print("Generating intra templates")
        self.cutOutTemplatesAndSave(
            intraDir, templateWidth, DefocalType.Intra, intraVisitId
        )

        print("Generating extra templates")
        self.cutOutTemplatesAndSave(
            extraDir, templateWidth, DefocalType.Extra, extraVisitId
        )

    def cutOutTemplatesAndSave(
        self, phosimImageDir, templateWidth, defocalType, visitId
    ):
        """
        Loop through all detectors in folder and cut out square region
        around the donut from a CCD image to use
        as the donut template for that detector. Saves the cutout to file.

        Parameter
        ---------
        phosimImageDir : str
            Directory where the visit ISR output is located. Inside should
            be folders for each raft.
        templateWidth : int
            Width of square template image in pixels.
        defocalType : enum 'DefocalType'
            Defocal type.
        visitId : int
            Visit Id of the defocal image from Phosim.
        """
        if defocalType == DefocalType.Intra:
            defocalLabel = "intra"
        else:
            defocalLabel = "extra"

        # Set output path
        phosimTemplateDir = os.path.join(self.templateDataPath, "phosimTemplates")
        # Set path to centroid file
        phosimCentroidDir = os.path.join(
            self.tempWorkPath, "phosimOutput", defocalLabel
        )

        stampHalfWidth = int(templateWidth / 2)

        centroidFind = CentroidFindFactory.createCentroidFind(
            CentroidFindType.RandomWalk
        )

        for sensorDir in os.listdir(phosimImageDir):
            sensorPath = os.path.join(phosimImageDir, sensorDir)
            for templateFile in os.listdir(sensorPath):
                # Open postISR File
                imgObj = Image()
                imgObj.setImg(imageFile=os.path.join(sensorPath, templateFile))
                imgData = imgObj.getImg()
                splitName = templateFile.split("-")
                templateName = (
                    f"{defocalLabel}_template-{splitName[2]}_{splitName[3]}.txt"
                )

                # Pick an area around the phosim centroid as the template
                centroidFilename = f"centroid_lsst_e_{visitId}_f1_{splitName[2]}_{splitName[3]}_E000.txt"
                centroidData = np.genfromtxt(
                    os.path.join(phosimCentroidDir, centroidFilename),
                    unpack=True,
                    skip_header=1,
                )
                centroidX = int(centroidData[2])
                centroidY = int(centroidData[3])
                templateStamp = imgData[
                    centroidX - stampHalfWidth : centroidX + stampHalfWidth,
                    centroidY - stampHalfWidth : centroidY + stampHalfWidth,
                ]
                # Reduce background noise
                templateStampBinary = centroidFind.getImgBinary(templateStamp)
                # Save to file
                np.savetxt(
                    os.path.join(phosimTemplateDir, "%s" % templateName),
                    templateStampBinary,
                    fmt="%i",
                )

    def cleanUpWorkDirs(self):
        """
        Clean up all temporary work directories.
        """

        shutil.rmtree(self.tempWorkPath)
