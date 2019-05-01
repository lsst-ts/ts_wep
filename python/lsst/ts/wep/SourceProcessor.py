import os
import numpy as np

from lsst.ts.wep.deblend.BlendedImageDecorator import BlendedImageDecorator
from lsst.ts.wep.Utility import readPhoSimSettingData, mapFilterRefToG, \
    getConfigDir
from lsst.ts.wep.ParamReader import ParamReader


class SourceProcessor(object):

    def __init__(self, settingFileName="default.yaml",
                 focalPlaneFileName="focalplanelayout.txt"):
        """Initialize the SourceProcessor class.

        Parameters
        ----------
        settingFileName : str, optional
            Setting file name. (the default is "default.yaml".)
        focalPlaneFileName : str, optional
            Focal plane file name used in the PhoSim instrument directory. (the
            default is "focalplanelayout.txt".)
        """

        self.sensorName = ""
        self.blendedImageDecorator = BlendedImageDecorator()

        configDir = getConfigDir()
        settingFilePath = os.path.join(configDir, settingFileName)
        self.settingFile = ParamReader(filePath=settingFilePath)

        self.sensorFocaPlaneInDeg = dict()
        self.sensorFocaPlaneInUm = dict()
        self.sensorDimList = dict()
        self.sensorEulerRot = dict()
        self._readFocalPlane(configDir, focalPlaneFileName)

    def _readFocalPlane(self, folderPath, focalPlaneFileName):
        """Read the focal plane data used in PhoSim to get the ccd dimension
        and fieldXY in chip center.

        Parameters
        ----------
        folderPath : str
            Directory of focal plane file.
        focalPlaneFileName : str
            Focal plane file name used in the PhoSim instrument directory.
        """

        # Read the focal plane data by the delegation
        ccdData = readPhoSimSettingData(folderPath, focalPlaneFileName,
                                        "fieldCenter")

        # Collect the focal plane data
        sensorFocaPlaneInDeg = dict()
        sensorFocaPlaneInUm = dict()
        sensorDimList = dict()
        for sensorName, data in ccdData.items():

            # Consider the x-translation in corner wavefront sensors
            self._shiftCenterWfs(sensorName, data)

            # Change the unit from um to degree
            xInUm = float(data[0])
            yInUm = float(data[1])
            pixelSizeInUm = float(data[2])
            sizeXinPixel = int(data[3])
            sizeYinPixel = int(data[4])

            # 1 degree = 3600 arcsec
            pixelToArcsec = self.settingFile.getSetting("pixelToArcsec")
            fieldX = xInUm / pixelSizeInUm * pixelToArcsec / 3600
            fieldY = yInUm / pixelSizeInUm * pixelToArcsec / 3600

            # Get the data
            sensorFocaPlaneInDeg[sensorName] = (fieldX, fieldY)
            sensorFocaPlaneInUm[sensorName] = (xInUm, yInUm)
            sensorDimList[sensorName] = (sizeXinPixel, sizeYinPixel)

        # Assign the values
        self.sensorDimList = sensorDimList
        self.sensorFocaPlaneInDeg = sensorFocaPlaneInDeg
        self.sensorFocaPlaneInUm = sensorFocaPlaneInUm
        self.sensorEulerRot = readPhoSimSettingData(
            folderPath, focalPlaneFileName, "eulerRot")

    def _shiftCenterWfs(self, sensorName, focalPlaneData):
        """Shift the fieldXY of center of wavefront sensors.

        The input data is the center of combined chips (C0+C1). The input data
        will be updated directly.

        Parameters
        ----------
        sensorName : str
            Abbreviated sensor name.
        focalPlaneData : list
            Data of focal plane: [x position (microns), y position (microns),
            pixel size (microns), number of x pixels, number of y pixels].
        """

        # The layout is shown in the following:

        # R04_S20              R44_S00
        # --------           -----------       /\ +y
        # |  C0  |           |    |    |        |
        # |------|           | C1 | C0 |        |
        # |  C1  |           |    |    |        |
        # --------           -----------        -----> +x

        # R00_S22              R40_S02
        # -----------          --------
        # |    |    |          |  C1  |
        # | C0 | C1 |          |------|
        # |    |    |          |  C0  |
        # -----------          --------

        xInUm = float(focalPlaneData[0])
        yInUm = float(focalPlaneData[1])
        pixelSizeInUm = float(focalPlaneData[2])
        sizeXinPixel = float(focalPlaneData[3])

        # Consider the x-translation in corner wavefront sensors
        tempX = None
        tempY = None

        if sensorName in ("R44_S00_C0", "R00_S22_C1"):
            # Shift center to +x direction
            tempX = xInUm + sizeXinPixel / 2 * pixelSizeInUm
        elif sensorName in ("R44_S00_C1", "R00_S22_C0"):
            # Shift center to -x direction
            tempX = xInUm - sizeXinPixel / 2 * pixelSizeInUm
        elif sensorName in ("R04_S20_C1", "R40_S02_C0"):
            # Shift center to -y direction
            tempY = yInUm - sizeXinPixel / 2 * pixelSizeInUm
        elif sensorName in ("R04_S20_C0", "R40_S02_C1"):
            # Shift center to +y direction
            tempY = yInUm + sizeXinPixel / 2 * pixelSizeInUm

        # Replace the value by the shifted one
        if (tempX is not None):
            focalPlaneData[0] = str(tempX)
        elif (tempY is not None):
            focalPlaneData[1] = str(tempY)

    def config(self, sensorName=None):
        """Do the configuration.

        Parameters
        ----------
        sensorName : str, optional
            Abbreviated sensor name. (the default is None.)
        """

        if (sensorName is not None):
            self.sensorName = sensorName

    def getEulerZinDeg(self, sensorName):
        """Get the Euler Z angle of sensor in degree.

        Parameters
        ----------
        sensorName : str
            Abbreviated sensor name.

        Returns
        -------
        float
            Euler Z angle in degree.
        """

        return float(self.sensorEulerRot[sensorName][0])

    def camXYtoFieldXY(self, pixelX, pixelY):
        """Get the field X, Y from the pixel x, y position on CCD.

        Parameters
        ----------
        pixelX : float
            Pixel x on camera coordinate.
        pixelY : float
            Pixel y on camera coordinate.

        Returns
        -------
        float
            Field x in degree.
        float
            Field y in degree.
        """

        # The wavefront sensors will do the counter-clockwise rotation as the
        # following based on the euler angle:

        # R04_S20              R44_S00
        # O-------           -----O----O       /\ +y
        # |  C0  |           |    |    |        |
        # O------|           | C1 | C0 |        |
        # |  C1  |           |    |    |        |
        # --------           -----------        O----> +x

        # R00_S22              R40_S02
        # -----------          --------
        # |    |    |          |  C1  |
        # | C0 | C1 |          |------O
        # |    |    |          |  C0  |
        # O----O-----          -------O

        # Get the field X, Y of sensor's center
        fieldXc, fieldYc = self.sensorFocaPlaneInDeg[self.sensorName]

        # Get the center pixel position
        pixelXc, pixelYc = self.sensorDimList[self.sensorName]
        pixelXc = pixelXc / 2
        pixelYc = pixelYc / 2

        # Calculate the delta x and y in degree
        # 1 degree = 3600 arcsec
        pixelToArcsec = self.settingFile.getSetting("pixelToArcsec")
        deltaX = (pixelX - pixelXc) * pixelToArcsec / 3600.0
        deltaY = (pixelY - pixelYc) * pixelToArcsec / 3600.0

        # Calculate the transformed coordinate in degree.
        fieldX, fieldY = self._rotCam2FocalPlane(
            self.sensorName, fieldXc, fieldYc, deltaX, deltaY)

        return fieldX, fieldY

    def _rotCam2FocalPlane(self, sensorName, centerX, centerY, deltaX, deltaY,
                           clockWise=False):
        """Do the rotation from camera coordinate to focal plane coordinate or
        vice versa.

        Parameters
        ----------
        sensorName : str
            Abbreviated sensor name.
        centerX : float
            CCD center x.
        centerY : float
            CCD center y.
        deltaX : float
            Delta x from the CCD's center.
        deltaY : float
            Delta y from the CCD's center.
        clockWise : bool, optional
            Rotation direction (True: clockwise, False: counter-clockwise).
            (the default is False.)

        Returns
        -------
        float
            Transformed x position.
        float
            Transformed y position.
        """

        # Get the euler angle in z direction (only consider the z rotatioin at
        # this moment)
        eulerZ = round(self.getEulerZinDeg(sensorName))
        eulerZinRad = np.deg2rad(eulerZ)

        # Counter-clockwise or clockwise rotation
        if (clockWise):
            eulerZinRad = -eulerZinRad

        # Calculate the new x, y by the rotation. This is important for
        # wavefront sensor.
        newX = centerX + np.cos(eulerZinRad) * deltaX - \
            np.sin(eulerZinRad) * deltaY
        newY = centerY + np.sin(eulerZinRad) * deltaX + \
            np.cos(eulerZinRad)*deltaY

        return newX, newY

    def dmXY2CamXY(self, pixelDmX, pixelDmY):
        """Transform the pixel x, y from DM library to camera to use.

        Parameters
        ----------
        pixelDmX : float
            Pixel x defined in DM coordinate.
        pixelDmY : float
            Pixel y defined in DM coordinate.

        Returns
        -------
        float
            Pixel x defined in camera coordinate based on LCA-13381.
        float
            Pixel y defined in camera coordinate based on LCA-13381.
        """

        # Camera coordinate is defined in LCA-13381. Define camera coordinate
        # (x, y) and DM coordinate (x', y'), then the relation is:
        # Camera team +y = DM team +x'
        # Camera team +x = DM team -y'

        #  O---->y
        #  |
        #  |   ----------------------
        #  \/ |                      |   (x', y') = (200, 500) =>
        #  x  |                      |   (x, y) = (-500, 200) -> (3500, 200)
        #     |4000                  |
        # y'  |                      |
        #  /\ |       4072           |
        #  |  |----------------------
        #  |
        #  O-----> x'

        # Get the CCD dimension
        dimX, dimY = self.sensorDimList[self.sensorName]

        # Calculate the transformed coordinate
        pixelCamX = dimX - pixelDmY
        pixelCamY = pixelDmX

        return pixelCamX, pixelCamY

    def camXY2DmXY(self, pixelCamX, pixelCamY):
        """Transform the pixel x, y from camera coordinate to DM coordinate.

        Parameters
        ----------
        pixelCamX : float
            Pixel x defined in Camera coordinate based on LCA-13381.
        pixelCamY : float
            Pixel y defined in Camera coordinate based on LCA-13381.

        Returns
        -------
        float
            Pixel x defined in DM coordinate.
        float
            Pixel y defined in DM coordinate.
        """

        # Check the comment in dmXY2CamXY() for the details of coordinate
        # systems of DM and camera teams.

        # Get the CCD dimension
        dimX, dimY = self.sensorDimList[self.sensorName]

        # Calculate the transformed coordinate
        pixelDmX = pixelCamY
        pixelDmY = dimX - pixelCamX

        return pixelDmX, pixelDmY

    def isVignette(self, fieldX, fieldY):
        """The donut is vignetted or not by calculating the donut's distance to
        center.

        Parameters
        ----------
        fieldX : float
            Field x in degree.
        fieldY : float
            Field y in degree.

        Returns
        -------
        bool
            True if the donut is vignette.
        """

        # Calculate the distance to center in degree to judge the donut is
        # vignetted or not.
        fldr = np.sqrt(fieldX**2 + fieldY**2)
        distVignette = self.settingFile.getSetting("distVignette")
        if (fldr >= distVignette):
            return True
        else:
            return False

    def simulateImg(self, imageFolderPath, defocalDis, nbrStar, filterType,
                    noiseRatio=0.01):
        """Simulate the defocal CCD images with the neighboring star map.

        This function is only for the test use.

        Parameters
        ----------
        imageFolderPath : str
            Path to image directory.
        defocalDis : float
            Defocal distance in mm.
        nbrStar : NbrStar
            Neighboring star on single detector.
        filterType : FilterType
            Filter type.
        noiseRatio : float, optional
            The noise ratio. (the default is 0.01.)

        Returns
        -------
        numpy.ndarray
            Simulated intra-focal images.
        numpy.ndarray
            Simulated extra-focal images.

        Raises
        -------
        ValueError
            No available donut images.
        ValueError
            The numbers of intra- and extra-focal images are different.
        """

        # Generate the intra- and extra-focal ccd images
        d1, d2 = self.sensorDimList[self.sensorName]
        ccdImgIntra = np.random.random([d2, d1])*noiseRatio
        ccdImgExtra = ccdImgIntra.copy()

        # Get all files in the image directory in a sorted order
        fileList = sorted(os.listdir(imageFolderPath))

        # Redefine the format of defocal distance
        defocalDis = "%.2f" % defocalDis

        # Get the available donut files
        intraFileList = []
        extraFileList = []
        for afile in fileList:

            # Get the file name
            fileName, fileExtension = os.path.splitext(afile)

            # Split the file name for the analysis
            fileNameStr = fileName.split("_")

            # Find the file name with the correct defocal distance
            if (len(fileNameStr) == 3 and fileNameStr[1] == defocalDis):

                # Collect the file name based on the defocal type
                if (fileNameStr[-1] == "intra"):
                    intraFileList.append(afile)
                elif (fileNameStr[-1] == "extra"):
                    extraFileList.append(afile)

        # Get the number of available files
        numFile = len(intraFileList)
        if (numFile == 0):
            raise ValueError("No available donut images.")

        # Check the numbers of intra- and extra-focal images should be the same
        if (numFile != len(extraFileList)):
            raise ValueError(
                "The numbers of intra- and extra-focal images are different.")

        # Get the magnitude of stars
        mappedFilterType = mapFilterRefToG(filterType)
        starMag = nbrStar.getMag(mappedFilterType)

        # Based on the nbrStar to reconstruct the image
        for brightStar, neighboringStar in nbrStar.getId().items():

            # Generate a random number
            randNum = np.random.randint(0, high=numFile)

            # Choose a random donut image from the file
            donutImageIntra = self._getDonutImgFromFile(
                imageFolderPath, intraFileList[randNum])
            donutImageExtra = self._getDonutImgFromFile(
                imageFolderPath, extraFileList[randNum])

            # Get the bright star magnitude
            magBS = starMag[brightStar]

            # Combine the bright star and neighboring stars. Put the bright
            # star in the first one.
            allStars = neighboringStar[:]
            allStars.insert(0, brightStar)

            # Add the donut image
            for star in allStars:

                # Get the brigtstar pixel x, y
                starX, starY = nbrStar.getRaDeclInPixel()[star]
                magStar = starMag[star]

                # Transform the coordiante from DM team to camera team
                starX, starY = self.dmXY2CamXY(starX, starY)

                # Ratio of magnitude between donuts (If the magnitudes of stars
                # differs by 5, the brightness differs by 100.)
                # (Magnitude difference shoulbe be >= 1.)
                magDiff = magStar - magBS
                magRatio = 1 / 100 ** (magDiff / 5.0)

                # Add the donut image
                self._addDonutImage(magRatio * donutImageIntra, starX, starY,
                                    ccdImgIntra)
                self._addDonutImage(magRatio * donutImageExtra, starX, starY,
                                    ccdImgExtra)

        return ccdImgIntra, ccdImgExtra

    def _getDonutImgFromFile(self, imageFolderPath, fileName):
        """Read the donut image from the file.

        Parameters
        ----------
        imageFolderPath : str
            Path to image directory.
        fileName : str
            File name.

        Returns
        -------
        numpy.ndarray
            Donut image.
        """

        # Get the donut image from the file by the delegation
        self.blendedImageDecorator.setImg(
            imageFile=os.path.join(imageFolderPath, fileName))

        return self.blendedImageDecorator.getImg().copy()

    def _addDonutImage(self, donutImage, starX, starY, ccdImg):
        """Add the donut image to simulated CCD image frame.

        Parameters
        ----------
        donutImage : numpy.ndarray
            Donut image.
        starX : float
            Star position in pixel x.
        starY : float
            Star position in pixel y.
        ccdImg : numpy.ndarray
            CCD image.
        """

        # Get the dimension of donut image
        d1, d2 = donutImage.shape

        # Get the interger of position to use as the index
        y = int(starY)
        x = int(starX)

        # Add the donut image on the CCD image
        ccdImg[y-int(d1/2):y-int(d1/2)+d1, x-int(d2/2):x-int(d2/2)+d2] += \
            donutImage

    def getSingleTargetImage(self, ccdImg, nbrStar, index, filterType):
        """Get the image of single scientific target and related neighboring
        stars.

        Parameters
        ----------
        ccdImg : numpy.ndarray
            CCD image.
        nbrStar : NbrStar
            Neighboring star on single detector.
        index : int
            Index of science target star in neighboring star.
        filterType : FilterType
            Filter type.

        Returns
        -------
        numpy.ndarray
            Ccd image of target stars.
        numpy.ndarray
            Star x-positions. The arange is [neighboring stars, bright star].
        numpy.ndarray
            Star y-positions. The arange is [neighboring stars, bright star].
        numpy.ndarray
            Star magnitude ratio compared with the bright star. The arange is
            [neighboring stars, bright star].
        float
            Offset x from the origin of target star image to the origin of CCD
            image.
        float
            Offset y from the origin of target star image to the origin of CCD
            image.

        Raises
        ------
        ValueError
            Index is higher than the length of star map.
        """

        # Get the target star position
        nbrStarId = nbrStar.getId()
        if (index >= len(nbrStarId)):
            raise ValueError("Index is higher than the length of star map.")

        # Get the star SimobjID
        brightStar = list(nbrStarId)[index]
        neighboringStar = nbrStarId[brightStar]

        # Get all star SimobjID list
        allStar = neighboringStar[:]
        allStar.append(brightStar)

        # Get the pixel positions
        raDeclInPixel = nbrStar.getRaDeclInPixel()
        allStarPosX = []
        allStarPosY = []
        for star in allStar:

            # Get the star pixel position
            starX, starY = raDeclInPixel[star]

            # Transform the coordiante from DM team to camera team
            starX, starY = self.dmXY2CamXY(starX, starY)

            allStarPosX.append(starX)
            allStarPosY.append(starY)

        # Check the ccd image dimenstion
        ccdD1, ccdD2 = ccdImg.shape

        # Define the range of image
        # Get min/ max of x, y
        minX = int(min(allStarPosX))
        maxX = int(max(allStarPosX))

        minY = int(min(allStarPosY))
        maxY = int(max(allStarPosY))

        # Get the central point
        cenX = int(np.mean([minX, maxX]))
        cenY = int(np.mean([minY, maxY]))

        # Get the image dimension
        starRadiusInPixel = self.settingFile.getSetting("starRadiusInPixel")
        d1 = (maxY - minY) + 4 * starRadiusInPixel
        d2 = (maxX - minX) + 4 * starRadiusInPixel

        # Make d1 and d2 to be symmetric and even
        d = max(d1, d2)
        if (d%2 == 1):
            # Use d-1 instead of d+1 to avoid the boundary touch
            d = d-1

        # If central x or y plus d/2 will over the boundary, shift the
        # central x, y values
        cenY = self._shiftCenter(cenY, ccdD1, d / 2)
        cenY = self._shiftCenter(cenY, 0, d / 2)

        cenX = self._shiftCenter(cenX, ccdD2, d / 2)
        cenX = self._shiftCenter(cenX, 0, d / 2)

        # Get the bright star and neighboring stas image
        offsetX = cenX - d / 2
        offsetY = cenY - d / 2
        singleSciNeiImg = \
            ccdImg[int(offsetY):int(cenY + d / 2), int(offsetX):int(cenX + d / 2)]

        # Get the stars position in the new coordinate system
        # The final one is the bright star
        allStarPosX = np.array(allStarPosX) - offsetX
        allStarPosY = np.array(allStarPosY) - offsetY

        # Get the star magnitude
        mappedFilterType = mapFilterRefToG(filterType)
        magList = nbrStar.getMag(mappedFilterType)

        # Get the list of magnitude
        magRatio = np.array([])
        for star in allStar:
            neiMag = magList[star]
            magRatio = np.append(magRatio, neiMag)

        # Calculate the magnitude ratio
        magRatio = 1 / 100 ** ((magRatio - magRatio[-1]) / 5.0)

        return singleSciNeiImg, allStarPosX, allStarPosY, magRatio, offsetX, offsetY

    def _shiftCenter(self, center, boundary, distance):
        """Shift the center if its distance to boundary is less than required.

        Parameters
        ----------
        center : float
            Center point.
        boundary : float
            Boundary point.
        distance : float
            Required distance.

        Returns
        -------
        float
            Shifted center.
        """

        # Distance between the center and boundary
        delta = boundary - center

        # Shift the center if needed
        if (abs(delta) < distance):
            return boundary - np.sign(delta)*distance
        else:
            return center

    def doDeblending(self, blendedImg, allStarPosX, allStarPosY, magRatio):
        """Do the deblending.

        It is noted that the algorithm now is only for one bright star and one
        neighboring star.

        Parameters
        ----------
        blendedImg : numpy.ndarray
            Blended image.
        allStarPosX : list or numpy.ndarray
            Star's position x in pixel. The arange is [neighboring star,
            bright star].
        allStarPosY : list or numpy.ndarray
            Star's position y in pixel. The arange is [neighboring star,
            bright star].
        magRatio : list or numpy.ndarray
            Star magnitude ratio compared with the bright star. The arange is
            [neighboring stars, bright star].

        Returns
        -------
        numpy.ndarray
            Deblended image.
        float
            Pixel x of bright star.
        float
            Pixel y of bright star.

        Raises
        ------
        ValueError
            Only one neighboring star allowed.
        """

        # Check there is only one bright star and one neighboring star.
        # This is the limit of deblending algorithm now.
        if (len(magRatio) != 2):
            raise ValueError("Only one neighboring star allowed.")

        # Set the image for the deblending
        self.blendedImageDecorator.setImg(image=blendedImg)

        # Do the deblending
        imgDeblend, realcx, realcy = \
            self.blendedImageDecorator.deblendDonut((allStarPosX[0],
                                                     allStarPosY[0]))

        return imgDeblend, realcx, realcy


if __name__ == "__main__":
    pass
