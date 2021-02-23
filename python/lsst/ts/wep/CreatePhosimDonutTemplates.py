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
import yaml
import shutil
import numpy as np
from astropy.io import fits
from lsst.ts.wep.Utility import getConfigDir, runProgram, DefocalType
from lsst.ts.wep.CamIsrWrapper import CamIsrWrapper
from lsst.ts.wep.cwfs.CentroidRandomWalk import CentroidRandomWalk


class CreatePhosimDonutTemplates(object):
    def __init__(self, templateDestDir=""):
        """
        Class to create donut templates from Phosim.

        Parameters
        ----------
        templateDestDir : str, optional
            The destination directory for the Phosim donut templates.
            If the string is "" then it will put the final templates in
            ts_wep/policy/cwfs/donutTemplateData/phosimTemplates.
            (The default is "".)
        """

        self.policyDir = getConfigDir()
        self.templateDataPath = os.path.join(
            self.policyDir, "cwfs", "donutTemplateData"
        )
        self.tempWorkPath = os.path.join(self.templateDataPath, "tempDir")
        self.repoDir = os.path.join(self.tempWorkPath, "input")

        if templateDestDir == "":
            self.templateDestDir = os.path.join(
                self.templateDataPath, "phosimTemplates"
            )
        else:
            self.templateDestDir = templateDestDir

    def createDirectories(self):
        """
        Create the final template directory as well as
        temporary work directories.
        """

        print("Making template directory")
        os.makedirs(self.templateDestDir, exist_ok=True)

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
            (Example: "R22_S11 R22_S10"). If the str is "" then it will generate
            a template for every detector in the focal plane. (The default is "".)

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

            sensorNameFile = os.path.join(self.policyDir, "sensorNameToId.yaml")

            with open(sensorNameFile, "r") as f:
                sensorData = yaml.load(f, Loader=yaml.FullLoader)

            sensorNameList = list(sensorData.keys())

        else:

            sensorNameList = detectorStr.split(" ")

        detectorStrPhosim = ""
        detectorStrFlats = ""

        for sensorName in sensorNameList:
            detectorStrPhosim += f"{sensorName}|"
            detectorStrFlats += f"{sensorName} "

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
        runPhosimCmd = f"python {phosimPath}/phosim.py"
        runPhosimArgs = f"-w {self.tempWorkPath}/phosimWorkDir "
        runPhosimArgs += f'-s "{detectorStrPhosim[:-1]}" '
        runPhosimArgs += f"-p {numOfProc} "
        runPhosimArgs += f"-i lsst "
        runPhosimArgs += f"-e 1 "
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

        print(f"Repackaging phosim output")

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

        # Create mapper file
        cmdString = f"echo lsst.obs.lsst.phosim.PhosimMapper > {self.tempWorkPath}/input/_mapper"
        runProgram(cmdString)

        # Run Ingestion
        ampImg = f"{self.tempWorkPath}/raw/*.fits"
        runProgram("ingestImages.py", argstring=f"{self.repoDir} {ampImg}")

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

        argString = f"{self.repoDir} "
        argString += f"{self.tempWorkPath}/calibs/*.fits "
        argString += f"--validity 99999 --output {self.repoDir}"
        runProgram("ingestCalibs.py", argstring=argString)

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

        intraDir = os.path.join(
            self.repoDir, "rerun", "run1", "postISRCCD", f"{intraVisitId:#08}-g"
        )
        extraDir = os.path.join(
            self.repoDir, "rerun", "run1", "postISRCCD", f"{extraVisitId:#08}-g"
        )

        print("Generating intra templates")
        self.cutOutTemplatesAndSave(
            intraDir, templateWidth, DefocalType.Intra, intraVisitId
        )

        print("Generating extra templates")
        self.cutOutTemplatesAndSave(
            extraDir, templateWidth, DefocalType.Extra, extraVisitId
        )

    def cutOutTemplatesAndSave(
        self,
        phosimImageDir,
        templateWidth,
        defocalType,
        visitId,
        phosimTemplateDir="",
        phosimCentroidDir="",
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

        phosimTemplateDir : str, optional
            Directory where templates will be stored. If it is an empty string
            then it will assign it to
            ts_wep/policy/cwfs/donutTemplateData/phosimTemplates.
            (The default is "".)

        phosimCentroidDir : str, optional
            The path to the phosim centroid file for the image. If it is
            an empty string then it will look in temporary phosimOutput
            directory. (The default is "".)
        """
        if defocalType == DefocalType.Intra:
            defocalLabel = "intra"
        else:
            defocalLabel = "extra"

        # Set default paths
        if phosimTemplateDir == "":
            phosimTemplateDir = os.path.join(self.templateDataPath, "phosimTemplates")
        if phosimCentroidDir == "":
            phosimCentroidDir = os.path.join(
                self.tempWorkPath, "phosimOutput", defocalLabel
            )

        stampHalfWidth = int(templateWidth / 2)

        centroidRW = CentroidRandomWalk()

        for sensorDir in os.listdir(phosimImageDir):
            sensorPath = os.path.join(phosimImageDir, sensorDir)
            for templateFile in os.listdir(sensorPath):
                # Open ISR File
                testHdu = fits.open(os.path.join(sensorPath, templateFile))
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
                templateStamp = testHdu[1].data[
                    centroidX - stampHalfWidth : centroidX + stampHalfWidth,
                    centroidY - stampHalfWidth : centroidY + stampHalfWidth,
                ]
                # Reduce background noise
                templateStampBinary = centroidRW.getImgBinary(templateStamp)
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
