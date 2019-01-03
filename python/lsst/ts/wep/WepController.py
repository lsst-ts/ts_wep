import os
import re
import time
import numpy as np
from astropy.io import fits

from lsst.ts.wep.ButlerWrapper import ButlerWrapper
from lsst.ts.wep.DefocalImage import DefocalImage
from lsst.ts.wep.DonutImage import DonutImage
from lsst.ts.wep.Utility import getModulePath, abbrevDectectorName, \
                                searchDonutPos


class WepController(object):

    def __init__(self, dataCollector, isrWrapper, sourSelc, sourProc, wfsEsti):
        """Initialize the wavefront estimation pipeline (WEP) controller class.

        Parameters
        ----------
        dataCollector : CamDataCollector
            Camera data collector.
        isrWrapper : CamIsrWrapper
            Instrument signature removal (ISR) wrapper.
        sourSelc : SourceSelector
            Source selector.
        sourProc : SourceProcessor
            Source processor.
        wfsEsti : WfEstimator
            Wavefront estimator.
        """

        self.dataCollector = dataCollector
        self.isrWrapper = isrWrapper
        self.sourSelc = sourSelc
        self.sourProc = sourProc
        self.wfsEsti = wfsEsti

        self.butlerWrapper = None

    def getWfsList(self):
        """
        
        Get the corner wavefront sensor (WFS) list in the canonical form.
        
        Returns:
            [list] -- WFS name list.
        """

        wfsList = ["R:0,0 S:2,2,A", "R:0,0 S:2,2,B", "R:0,4 S:2,0,A", "R:0,4 S:2,0,B", 
                   "R:4,0 S:0,2,A", "R:4,0 S:0,2,B", "R:4,4 S:0,0,A", "R:4,4 S:0,0,B"]

        return wfsList

    def getTargetStarByFile(self, dbAdress, skyInfoFilePath, pointing, cameraRotation, 
                            orientation=None, tableName="TempTable"):
        """
        
        Get the target stars by querying the file.
        
        Arguments:
            dbAdress {[str]} -- Local database address.
            skyInfoFilePath {[str]} -- File path of sky information.
            pointing {[tuple]} -- Camera boresight (RA, Decl) in degree.
            cameraRotation {[float]} -- Camera rotation angle in degree.
        
        Keyword Arguments:
            orientation {[str]} -- Orientation of wavefront sensor(s) on camera. (default: {None})
            tableName {[str]} -- Table name. (default: {None})
        
        Returns:
            {[dict]} -- Information of neighboring stars and candidate stars with the name of 
                        sensor as a dictionary.
            {[dict]} -- Information of stars with the name of sensor as a dictionary.
            {[dict]} -- Corners of sensor with the name of sensor as a dictionary.
        """

        # Check the database name is local database
        if (self.sourSelc.name != self.sourSelc.LocalDb):
            raise TypeError("The database type is not LocalDatabaseDecorator.")

        # Get the filter type
        aFilter = self.sourSelc.getFilter()

        # Connect the database
        self.sourSelc.connect(dbAdress)

        # Create the table
        self.sourSelc.db.createTable(aFilter)

        # Insert the sky data
        self.sourSelc.db.insertDataByFile(skyInfoFilePath, aFilter, skiprows=1)
        
        # Do the query and analysis
        neighborStarMap, starMap, wavefrontSensors = self.sourSelc.getTargetStar(pointing, cameraRotation, 
                                                                orientation=orientation, tableName=tableName)

        neighborStarMap, starMap, wavefrontSensors = self.__analyzeStarMap(neighborStarMap, starMap, 
                                                                                    wavefrontSensors)

        # Delete the table
        self.sourSelc.db.deleteTable(aFilter)
        
        # Disconnect the database
        self.sourSelc.disconnect()

        return neighborStarMap, starMap, wavefrontSensors

    def __analyzeStarMap(self, neighborStarMap, starMap, wavefrontSensors):
        """
        
        Analyze the star map and remove the sensor without bright stars.
        
        Arguments:
            neighborStarMap {[dict]} -- Information of neighboring stars and candidate stars with 
                                        the name of sensor as a dictionary.
            starMap {[dict]} -- Information of stars with the name of sensor as a dictionary.
            wavefrontSensors {[dict]} -- Corners of sensor with the name of sensor as a dictionary.
        
        Returns:
            {[dict]} -- Information of neighboring stars and candidate stars with the name 
                        of sensor as a dictionary.
            {[dict]} -- Information of stars with the name of sensor as a dictionary.
            {[dict]} -- Corners of sensor with the name of sensor as a dictionary.
        """

        # Collect the sensor list without the bright star
        noStarSensorList = []
        for aKey, aItem in starMap.items():
            if len(aItem.RA) == 0:
                noStarSensorList.append(aKey)

        # Remove the keys in map
        for aKey in noStarSensorList:
            neighborStarMap.pop(aKey)
            starMap.pop(aKey)
            wavefrontSensors.pop(aKey)
        
        return neighborStarMap, starMap, wavefrontSensors

    def __searchFileName(self, fileList, matchName, snap=0):
        """
        
        Search the file name in list.
        
        Arguments:
            fileList {[list]} -- File name list.
            matchName {[str]} -- Match name.

        Keyword Arguments:
            snap {int} -- Snap number (default: {0})
        
        Returns:
            [str] -- Matched file name.
        """

        matchFileName = None
        for fileName in fileList:
            m = re.match(r"\S*%s_E00%d\S*" % (matchName, snap), fileName)

            if (m is not None):
                matchFileName = m.group()
                break

        return matchFileName

    def getPostISRDefocalImgMap(self, sensorNameList, obsIdList=None, wfsDir=None, snap=0, expInDmCoor=False):
        """
        
        Get the post-ISR defocal image map.
        
        Arguments:
            sensorNameList {[list]} -- List of sensor name which is in the canonical form.
        
        Keyword Arguments:
            obsIdList {[list]} -- Observation Id list in [intraObsId, extraObsId]. (default: {None})
            wfsDir {[str]} -- Directory to wavefront sensor image data. (default: {None})
            snap {[int]} -- Snap number (default: {0})
            expInDmCoor {[bool]} -- Exposure image is in DM coordinate system. If True, this function will 
                                    rotate the exposure image to camera coordinate. This only works for 
                                    LSST FAM at this moment.
        
        Returns:
            [dict] -- Post-ISR image map.
        """

        # Construct the dictionary 
        wfsImgMap = {}

        # Get the file list
        if (wfsDir is not None):
            # Get the file list
            wfsFileList = self.__getRawFileList(wfsDir)

        # Get the waveront image map
        for sensorName in sensorNameList:

            # Get the sensor name information
            raft, sensor, channel = self.__getSensorInfo(sensorName)

            # The intra/ extra defocal images are decided by obsId
            if (obsIdList is not None):

                imgList = []
                for ii in range(2):
                    dataId = dict(visit=obsIdList[ii], snap=snap, raft=raft, sensor=sensor)
                    img = self.isrWrapper.butler.get(datasetType="postISRCCD", dataId=dataId, 
                                                     immediate=True)

                    # Get the exposure image in ndarray
                    img = self.butlerWrapper.getImageData(img)

                    # Change the image to camera coordinate
                    if (expInDmCoor):
                        img = np.rot90(img.copy(), k=3)

                    imgList.append(img)

                wfsImgMap[sensorName] = DefocalImage(intraImg=imgList[0], extraImg=imgList[1])

            # The intra/ extra defocal images are decided by physical configuration
            # C0: intra, C1: extra
            if (wfsDir is not None):
                
                # Get the abbreviated name
                abbrevName = abbrevDectectorName(sensorName)

                # Search for the file name
                matchFileName = self.__searchFileName(wfsFileList, abbrevName, snap=snap)
                
                if (matchFileName is not None):
                    
                    # Get the file name
                    fitsFilsPath = os.path.join(self.dataCollector.pathOfRawData, wfsDir, 
                                                matchFileName)
                    wfsImg = fits.getdata(fitsFilsPath)

                    # Add image to map
                    wfsImgMap[sensorName] = DefocalImage()

                    # "C0" = "A" = "Intra-focal image"
                    if (channel=="A"):
                        wfsImgMap[sensorName].setImg(intraImg=wfsImg)
                    # "C1" = "B" = "extra-focal image"
                    elif (channel=="B"):
                        wfsImgMap[sensorName].setImg(extraImg=wfsImg)

        return wfsImgMap

    def __getRawFileList(self, dataDir):
        """
        
        Get the raw file list in the directory.
        
        Arguments:
            dataDir {[str]} -- Data directory.
        
        Returns:
            [list] -- File list.
        """

        # Get all files in the directory
        fullDataDir = os.path.join(self.dataCollector.pathOfRawData, dataDir)
        fileList = [f for f in os.listdir(fullDataDir) if os.path.isfile(os.path.join(fullDataDir, f))]

        return fileList

    def __getSensorInfo(self, sensorName):
        """
        
        Get the sensor information.
        
        Arguments:
            sensorName {[str]} -- Sensor name (e.g. "R:2,2 S:1,1" or "R:0,0 S:2,2,A")
        
        Returns:
            [str] -- Raft.
            [str] -- Sensor.
            [str] -- Channel.
        """

        raft = sensor = channel = None
        
        # Use the regular expression to analyze the input name
        m = re.match(r"R:(\d,\d) S:(\d,\d)(?:,([A,B]))?$", sensorName)
        if (m is not None):
            raft, sensor, channel = m.groups()[0:3]

        return raft, sensor, channel

    def __searchDonutListId(self, donutList, starId):
        """
        
        Search the bright star ID in the donut list.
        
        Arguments:
            donutList {[list]} -- List of DonutImage object.
            starId {[int]} -- Star ID.
        
        Returns:
            [int] -- Index of donut image object with specific starId.
        """

        index = -1
        for ii in range(len(donutList)):
            if (donutList[ii].starId == int(starId)):
                index = ii
                break

        return index

    def getDonutMap(self, neighborStarMap, wfsImgMap, aFilter, doDeblending=False, sglDonutOnly=False):
        """
        
        Get the donut map on each wavefront sensor (WFS).
        
        Arguments:
            neighborStarMap {[dict]} -- Information of neighboring stars and candidate stars with 
                                        the name of sensor as a dictionary.
            wfsImgMap {[dict]} --  Post-ISR image map.
            aFilter {[str]} -- Active filter type ("u", "g", "r", "i", "z", "y").
        
        Keyword Arguments:
            doDeblending {[bool]} -- Do the deblending or not. (default: {False})
            sglDonutOnly{[bool]} -- Only consider the single donut based on the bright star catalog. 
                                    (default: {False})
        
        Returns:
            [dict] -- Donut image map.
        """

        # Get the corner wavefront sensor names
        wfsList = self.getWfsList()
        
        # Collect the donut images and put into the map/ dictionary
        donutMap = {}
        for sensorName in wfsImgMap.keys():

            # Get the abbraviated sensor name
            abbrevName = abbrevDectectorName(sensorName)

            # Configure the source processor
            self.sourProc.config(sensorName=abbrevName)

            # Get the bright star id list on specific sensor
            simobjIdList = list(neighborStarMap[sensorName].SimobjID.keys())

            # Get the defocal images: [intra, extra]
            defocalImgList = [wfsImgMap[sensorName].intraImg, wfsImgMap[sensorName].extraImg]

            for ii in range(len(simobjIdList)):
                
                # Get the single star map
                for jj in range(2):

                    ccdImg = defocalImgList[jj]

                    # Get the segment of image
                    if (ccdImg is not None):
                        singleSciNeiImg, allStarPosX, allStarPosY, magRatio, offsetX, offsetY = \
                                                        self.sourProc.getSingleTargetImage(ccdImg, 
                                                            neighborStarMap[sensorName], ii, aFilter)

                        # Check the single donut or not based on the bright star catalog only
                        # This method should be updated in the future
                        if (sglDonutOnly):
                            if (len(magRatio) != 1):
                                continue

                        # Get the single donut/ deblended image
                        if (len(magRatio) == 1) or (not doDeblending):
                            imgDeblend = singleSciNeiImg

                            if (len(magRatio) == 1):
                                realcx, realcy = searchDonutPos(imgDeblend)
                            else:                               
                                realcx = allStarPosX[-1]
                                realcy = allStarPosY[-1]
                        # Do the deblending or not
                        elif (len(magRatio) == 2 and doDeblending):
                            imgDeblend, realcx, realcy = self.sourProc.doDeblending(singleSciNeiImg, 
                                                                  allStarPosX, allStarPosY, magRatio)
                            # Update the magnitude ratio
                            magRatio = [1]

                        else:
                            continue

                        # Extract the image
                        if (len(magRatio) == 1):
                            x0 = np.floor(realcx-self.wfsEsti.sizeInPix/2).astype("int")
                            y0 = np.floor(realcy-self.wfsEsti.sizeInPix/2).astype("int")
                            imgDeblend = imgDeblend[y0:y0+self.wfsEsti.sizeInPix, 
                                                    x0:x0+self.wfsEsti.sizeInPix]

                        # Rotate the image if the sensor is the corner wavefront sensor
                        if sensorName in wfsList:

                            # Get the Euler angle
                            eulerZangle = round(self.sourProc.getEulerZinDeg(abbrevName))
                            
                            # Change the sign if the angle < 0
                            while (eulerZangle < 0):
                                eulerZangle += 360

                            # Do the rotation of matrix
                            numOfRot90 = eulerZangle//90
                            imgDeblend = np.flipud(np.rot90(np.flipud(imgDeblend), numOfRot90))

                        # Put the deblended image into the donut map
                        if sensorName not in donutMap.keys():
                            donutMap[sensorName] = []

                        # Check the donut exists in the list or not
                        starId = simobjIdList[ii]
                        donutIndex = self.__searchDonutListId(donutMap[sensorName], starId)                             

                        # Create the donut object and put into the list if it is needed
                        if (donutIndex < 0):

                            # Calculate the field X, Y
                            pixelX = realcx+offsetX
                            pixelY = realcy+offsetY
                            fieldX, fieldY = self.sourProc.camXYtoFieldXY(pixelX, pixelY)

                            # Instantiate the DonutImage class
                            donutImg = DonutImage(starId, pixelX, pixelY, fieldX, fieldY)
                            donutMap[sensorName].append(donutImg)

                            # Search for the donut index again
                            donutIndex = self.__searchDonutListId(donutMap[sensorName], starId)

                        # Get the donut image list
                        donutList = donutMap[sensorName]

                        # Take the absolute value for images, which might contain the 
                        # negative value after the ISR correction. This happens for the 
                        # amplifier images.
                        imgDeblend = np.abs(imgDeblend)
                    
                        # Set the intra focal image
                        if (jj == 0):
                            donutList[donutIndex].setImg(intraImg=imgDeblend)
                        # Set the extra focal image
                        elif (jj == 1):
                            donutList[donutIndex].setImg(extraImg=imgDeblend)

        return donutMap

    def genMasterImgSglCcd(self, sensorName, donutImgList, zcCol=np.zeros(22)):
        """
        
        Generate the master donut image on signle CCD.
        
        Arguments:
            sensorName {[str]} -- Sensor name.
            donutImgList {[list]} -- List of donut images.
        
        Keyword Arguments:
            zcCol {[ndarray]} -- Coefficients of wavefront (z1-z22). (default: {np.zeros(22)})
        
        Returns:
            [DefocalImage] -- Master donut image.
        """

        # Configure the source processor
        abbrevName = abbrevDectectorName(sensorName)
        self.sourProc.config(sensorName=abbrevName)

        intraProjImgList = []
        extraProjImgList = []

        for donutImg in donutImgList:

            # Get the field x, y
            pixelX = donutImg.pixelX
            pixelY = donutImg.pixelY

            fieldX, fieldY = self.sourProc.camXYtoFieldXY(pixelX, pixelY)
            fieldXY = (fieldX, fieldY)

            # Set the image
            if (donutImg.intraImg is not None):

                # Get the projected image
                projImg = self.__getProjImg(fieldXY, donutImg.intraImg, 
                                            self.wfsEsti.ImgIntra.INTRA, zcCol)

                # Collect the projected donut
                intraProjImgList.append(projImg)

            if (donutImg.extraImg is not None):
                
                # Get the projected image
                projImg = self.__getProjImg(fieldXY, donutImg.extraImg, 
                                            self.wfsEsti.ImgExtra.EXTRA, zcCol)

                # Collect the projected donut
                extraProjImgList.append(projImg)

        # Generate the master donut
        stackIntraImg = self.__stackImg(intraProjImgList)
        stackExtraImg = self.__stackImg(extraProjImgList)

        # Put the master donut to donut map
        masterDonut = DonutImage(0, None, None, 0, 0, intraImg=stackIntraImg, 
                                    extraImg=stackExtraImg)

        return masterDonut

    def generateMasterImg(self, donutMap, zcCol=np.zeros(22)):
        """
        
        Generate the master donut image map.
        
        Arguments:
            donutMap {[dict]} -- Donut image map.
        
        Keyword Arguments:
            defocalDisInMm {float} -- Defocal distance in mm. (default: {1})
            zcCol {[ndarray]} -- Coefficients of wavefront (z1-z22) in m. 
                                 (default: {np.zeros(22)})
        
        Returns:
            [dict] -- Master donut image map.
        """

        masterDonutMap = {}
        for sensorName, donutImgList in donutMap.items():

            # Get the master donut on single CCD
            masterDonut = self.genMasterImgSglCcd(sensorName, donutImgList, zcCol=zcCol)

            # Put the master donut to donut map
            masterDonutMap[sensorName] = [masterDonut]

        return masterDonutMap

    def __stackImg(self, imgList):
        """
        
        Stack the images to generate the master image.
        
        Arguments:
            imgList {[list]} -- Image list.
        
        Returns:
            [ndarray] -- Stacked image.
        """

        if (len(imgList) == 0):
            stackImg = None
        else:
            # Get the minimun image dimension
            dimXlist = []
            dimYlist = []
            for img in imgList:
                dy, dx = img.shape
                dimXlist.append(dx)
                dimYlist.append(dy)

            dimX = np.min(dimXlist)
            dimY = np.min(dimYlist)

            deltaX = dimX//2
            deltaY = dimY//2

            # Stack the image by summation directly
            stackImg = np.zeros([dimY, dimX])
            for img in imgList:
                dy, dx = img.shape
                cy = int(dy/2)
                cx = int(dx/2)
                stackImg += img[cy-deltaY:cy+deltaY, cx-deltaX:cx+deltaX]

        return stackImg

    def __getProjImg(self, fieldXY, defocalImg, aType, zcCol):
        """
        
        Get the projected image on the pupil.
        
        Arguments:
            fieldXY {[tuple]} -- Position of donut on the focal plane in degree for intra- and 
                                 extra-focal images.
            defocalImg {[ndarray]} -- Defocal image.
            aType {[str]} -- Defocal type.
            zcCol {[ndarray]} -- Coefficients of wavefront (z1-z22).
        
        Returns:
            [ndarray] -- Projected image.
        """
        
        # Set the image
        self.wfsEsti.setImg(fieldXY, image=defocalImg, defocalType=aType)

        # Get the distortion correction (offaxis)
        offAxisCorrOrder = self.wfsEsti.algo.parameter["offAxisPolyOrder"]
        instDir = os.path.dirname(self.wfsEsti.inst.filename)
        if (aType == self.wfsEsti.ImgIntra.INTRA):
            img = self.wfsEsti.ImgIntra
        elif (aType == self.wfsEsti.ImgExtra.EXTRA):
            img = self.wfsEsti.ImgExtra
        img.getOffAxisCorr(instDir, offAxisCorrOrder)

        # Do the image cocenter
        img.imageCoCenter(self.wfsEsti.inst)

        # Do the compensation/ projection
        img.compensate(self.wfsEsti.inst, self.wfsEsti.algo, zcCol, self.wfsEsti.opticalModel)

        # Return the projected image
        return img.image

    def calcWfErr(self, donutMap):
        """
        
        Calculate the wavefront error in annular Zernike polynomials (z4-z22).
        
        Arguments:
            donutMap {[dict]} -- Donut image map.
        
        Returns:
            [dict] -- Donut image map with calculated wavefront error.
        """

        for sensorName, donutList in donutMap.items():

            for ii in range(len(donutList)):

                # Get the intra- and extra-focal donut images

                # Check the sensor is the corner WFS or not. Only consider "A"
                # Intra: C0 -> A; Extra: C1 -> B

                # Look for the intra-focal image
                if sensorName.endswith("A"):
                    intraDonut = donutList[ii]

                    # Get the extra-focal sensor name
                    extraFocalSensorName = sensorName.replace("A", "B")

                    # Get the donut list of extra-focal sensor
                    extraDonutList = donutMap[extraFocalSensorName]
                    if (ii < len(extraDonutList)):
                        extraDonut = extraDonutList[ii]
                    else:
                        continue

                # Pass the extra-focal image
                elif sensorName.endswith("B"):
                    continue
                # Scientific sensor
                else:
                    intraDonut = extraDonut = donutList[ii]

                # Calculate the wavefront error

                # Get the field X, Y position
                intraFieldXY = (intraDonut.fieldX, intraDonut.fieldY)
                extraFieldXY = (extraDonut.fieldX, extraDonut.fieldY)

                # Get the defocal images
                intraImg = intraDonut.intraImg
                extraImg = extraDonut.extraImg

                # Calculate the wavefront error
                zer4UpNm = self.calcSglWfErr(intraImg, extraImg, intraFieldXY, extraFieldXY)

                # Put the value to the donut image
                intraDonut.setWfErr(zer4UpNm)
                extraDonut.setWfErr(zer4UpNm)

        return donutMap

    def calcSglWfErr(self, intraImg, extraImg, intraFieldXY, extraFieldXY):
        """
        
        Calculate the wavefront error in annular Zernike polynomials (z4-z22) for 
        single donut.
        
        Arguments:
            intraImg {[ndarray]} -- Intra-focal donut image.
            extraImg {[ndarray]} -- Extra-focal donut image.
            intraFieldXY {[tuple]} -- Field x, y in degree of intra-focal donut image.
            extraFieldXY {[tuple]} -- Field x, y in degree of extra-focal donut image.
        
        Returns:
            [ndarray] -- Coefficients of Zernike polynomials (z4 - z22) in nm.
        """

        # Set the images
        self.wfsEsti.setImg(intraFieldXY, image=intraImg, 
                            defocalType=self.wfsEsti.ImgIntra.INTRA)
        self.wfsEsti.setImg(extraFieldXY, image=extraImg, 
                            defocalType=self.wfsEsti.ImgExtra.EXTRA)

        # Reset the wavefront estimator
        self.wfsEsti.reset()

        # Calculate the wavefront error
        zer4UpNm = self.wfsEsti.calWfsErr()

        return zer4UpNm

    def calcSglAvgWfErr(self, donutImgList):
        """
        
        Calculate the average of wavefront error on single CCD.
        
        Arguments:
            donutImgList {[list]} -- List of donut images.
        
        Returns:
            [ndarray] -- Average of wavefront error in nm.
        """

        # Calculate the weighting of donut image
        weightingRatio = self.calcWeiRatio(donutImgList)

        # Calculate the mean wavefront error (z4 - z22)
        numOfZk = self.wfsEsti.algo.parameter["numTerms"] - 3
        avgErr = np.zeros(numOfZk)
        for ii in range(len(donutImgList)):
            donutImg = donutImgList[ii]

            # Get the zer4UpNm
            if (donutImg.zer4UpNm is not None):
                zer4UpNm = donutImg.zer4UpNm
            else:
                zer4UpNm = 0

            avgErr = avgErr + weightingRatio[ii]*zer4UpNm

        return avgErr

    def calcWeiRatio(self, donutImgList):
        """
        
        Calculate the weighting ratio of donut image in the list.
        
        Arguments:
            donutImgList {[list]} -- List of donut images.
        
        Returns:
            [ndarray] -- Array of Weighting ratio of image donuts.
        """

        # Weighting of donut image. Use the simple average at this moment.
        # Need to consider the S/N and other factors in the future

        # Check the available zk and give the ratio
        weightingRatio = []
        for donutImg in donutImgList:
            if (donutImg.zer4UpNm is not None):
                weightingRatio.append(1)
            else:
                weightingRatio.append(0)

        # Do the normalization
        weightingRatio = np.array(weightingRatio)
        weightingRatio = weightingRatio/np.sum(weightingRatio)

        return weightingRatio


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
