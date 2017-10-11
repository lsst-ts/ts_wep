import os
import numpy as np

from deblend.BlendedImageDecorator import BlendedImageDecorator

from isr.WfsIsrTask import poltExposureImage, plotHist
from isr.changePhoSimInstrument import readData

from SourceSelector import SourceSelector

from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pylab as plt

from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils.CameraUtils import focalPlaneCoordsFromRaDec, chipNameFromRaDec
from lsst.obs.lsstSim import LsstSimMapper


class SourceProcessor(object):

	def __init__(self, sensorName):

		self.sensorName = sensorName
		self.donutRadiusInPixel = None

		self.sensorFocaPlaneInDeg = None
		self.sensorFocaPlaneInUm = None
		self.sensorDimList = None
		self.sensorEulerRot = None

		self.blendedImageDecorator = BlendedImageDecorator()

	def config(self, sensorName=None, donutRadiusInPixel=None):

		# Give the sensor name
		if (sensorName is not None):
			self.sensorName = sensorName

		# Give the donut radius in pixel
		if (donutRadiusInPixel is not None):
			self.donutRadiusInPixel = donutRadiusInPixel

	def readFocalPlane(self, folderPath, fileName="focalplanelayout.txt", pixel2Arcsec=0.2):
		"""

		Read the focal plane data used in PhoSim to get the ccd dimension and fieldXY in chip center.

		Arguments:
			folderPath {[str]} -- Directory of focal plane file.

		Keyword Arguments:
			fileName {[str]} -- Filename of focal plane. (default: {"focalplanelayout.txt"})
			pixel2Arcsec {number} -- Pixel to arcsec. (default: {0.2})
		"""

		# Read the focal plane data by the delegation
		ccdData = readData(folderPath, fileName, "fieldCenter")

		# Collect the focal plane data
		sensorFocaPlaneInDeg = {}
		sensorFocaPlaneInUm = {}
		sensorDimList = {}
		for akey, aitem in ccdData.items():

			# Consider the x-translation in corner wavefront sensors
			aitem = self.__shiftCenterWfs(akey, aitem)

			# Change the unit from um to degree
			fieldX = float(aitem[0])/float(aitem[2])*pixel2Arcsec/3600
			fieldY = float(aitem[1])/float(aitem[2])*pixel2Arcsec/3600

			# Get the data
			sensorFocaPlaneInDeg.update({akey: (fieldX, fieldY)})
			sensorFocaPlaneInUm.update({akey: (float(aitem[0]), float(aitem[1]))})
			sensorDimList.update({akey: (int(aitem[3]), int(aitem[4]))})

		# Assign the values
		self.sensorDimList = sensorDimList
		self.sensorFocaPlaneInDeg = sensorFocaPlaneInDeg
		self.sensorFocaPlaneInUm = sensorFocaPlaneInUm
		self.sensorEulerRot = readData(folderPath, fileName, "eulerRot")

	def __shiftCenterWfs(self, sensorName, focalPlaneData):
		"""

		Get the fieldXY of center of wavefront sensors. The input data is the center of combined chips (C0+C1).
		The layout is shown in the following:

		R04_S20              R44_S00
		--------           -----------       /\ +y
		|  C0  |           |    |    |        |
		|------|           | C1 | C0 |		  |
		|  C1  |           |    |    |		  |
		--------           -----------        -----> +x

		R00_S22              R40_S02
		-----------          --------
		|    |    |          |  C1  |
		| C0 | C1 |		     |------|
		|    |    |          |  C0  |
		-----------			 --------

		Arguments:
			sensorName {[str]} -- Sensor name.
			focalPlaneData {[list]} -- Data of focal plane: x position (microns), y position (microns),
									   pixel size (microns), number of x pixels, number of y pixels.
		Returns:
			[list] -- Updated focal plane data.
		"""

		# Consider the x-translation in corner wavefront sensors
		tempX = None
		tempY = None

		if sensorName in ("R44_S00_C0", "R00_S22_C1"):
			# Shift center to +x direction
			tempX = float(focalPlaneData[0]) + float(focalPlaneData[3])/2*float(focalPlaneData[2])
		elif sensorName in ("R44_S00_C1", "R00_S22_C0"):
			# Shift center to -x direction
			tempX = float(focalPlaneData[0]) - float(focalPlaneData[3])/2*float(focalPlaneData[2])
		elif sensorName in ("R04_S20_C1", "R40_S02_C0"):
			# Shift center to -y direction
			tempY = float(focalPlaneData[1]) - float(focalPlaneData[3])/2*float(focalPlaneData[2])
		elif sensorName in ("R04_S20_C0", "R40_S02_C1"):
			# Shift center to +y direction
			tempY = float(focalPlaneData[1]) + float(focalPlaneData[3])/2*float(focalPlaneData[2])

		# Replace the value by the shifted one
		if (tempX is not None):
			focalPlaneData[0] = str(tempX)
		elif (tempY is not None):
			focalPlaneData[1] = str(tempY)

		# Return the center position of wave front sensor
		return focalPlaneData

	def getFieldXY(self, sensorName, pixelX, pixelY, pixel2Arcsec=0.2):
		"""

		Get the field X, Y of the pixel postion in CCD. It is noted that the wavefront sensors
		will do the counter-clockwise rotation as the following based on the euler angle:

		R04_S20              R44_S00
		O-------           -----O----O       /\ +y
		|  C0  |           |    |    |        |
		O------|           | C1 | C0 |		  |
		|  C1  |           |    |    |		  |
		--------           -----------        O----> +x

		R00_S22              R40_S02
		-----------          --------
		|    |    |          |  C1  |
		| C0 | C1 |		     |------O
		|    |    |          |  C0  |
		O----O-----			 -------O

		Arguments:
			sensorName {[str]} -- Sensor name.
			pixelX {[float]} -- Pixel x on camera coordinate.
			pixelY {[float]} -- Pixel y on camera coordinate.

		Keyword Arguments:
			pixel2Arcsec {float} -- Pixel to arcsec. (default: {0.2})

		Returns:
			[float] -- Field X, Y in degree.
		"""

		# Get the field X, Y of sensor's center
		fieldXc, fieldYc = self.sensorFocaPlaneInDeg[sensorName]

		# Get the center pixel position
		pixelXc, pixelYc = self.sensorDimList[sensorName]
		pixelXc = pixelXc/2
		pixelYc = pixelYc/2

		# Calculate the delta x and y in degree
		deltaX = (pixelX-pixelXc)*pixel2Arcsec/3600.0
		deltaY = (pixelY-pixelYc)*pixel2Arcsec/3600.0

		# Calculate the transformed coordinate in degree.
		fieldX, fieldY = self.__rotCam2FocalPlane(sensorName, fieldXc, fieldYc, deltaX, deltaY)

		return fieldX, fieldY

	def focalPlaneXY2CamXY(self, sensorName, xInUm, yInUm, pixel2um=10.0):
		"""
		
		Get the x, y position on camera plane from the focal plane position.
		
		Arguments:
			sensorName {[str]} -- Sensor name.
			xInUm {[float]} -- Position x on focal plane in um.
			yInUm {[float]} -- Position y on focal plane in um.
		
		Keyword Arguments:
			pixel2um {float} -- Pixel to um. (default: {10.0})
		
		Returns:
			[float] -- Pixel x, y position on camera plane.
		"""

		# Get the central position of sensor in um
		xc, yc = self.sensorFocaPlaneInUm[sensorName]

		# Get the center pixel position
		pixelXc, pixelYc = self.sensorDimList[sensorName]
		pixelXc = pixelXc/2
		pixelYc = pixelYc/2

		# Calculate the delta x and y in pixel
		deltaX = (xInUm-xc)/pixel2um
		deltaY = (yInUm-yc)/pixel2um

		# Calculate the transformed coordinate
		pixelX, pixelY = self.__rotCam2FocalPlane(sensorName, pixelXc, pixelYc, deltaX, deltaY, counterClockWise=False)

		return pixelX, pixelY

	def __rotCam2FocalPlane(self, sensorName, centerX, centerY, deltaX, deltaY, counterClockWise=True):
		"""
		
		Do the rotation from camera coordinate to focal plane coordinate or vice versa.
		
		Arguments:
			sensorName {[str]} -- Sensor name.
			centerX {[float]} -- CCD center X.
			centerY {[float]} -- CCD center Y.
			deltaX {[float]} -- Delta X from the CCD's center.
			deltaY {[float]} -- Delta Y from the CCD's center.
		
		Keyword Arguments:
			counterClockWise {bool} -- Direction of rotation: counter-clockwise or clockwise. (default: {True})
		
		Returns:
			[float] -- Transformed X, Y position.
		"""

		# Get the euler angle in z direction (only consider the z rotatioin at this moment)
		eulerZ = round(float(self.sensorEulerRot[sensorName][0]))

		# Change the unit to radian
		eulerZ = eulerZ/180.0*np.pi

		# Counter-clockwise or clockwise rotation
		if (not counterClockWise):
			eulerZ = -eulerZ

		# Calculate the new x, y by the rotation. This is important for wavefront sensor.
		newX = centerX + np.cos(eulerZ)*deltaX - np.sin(eulerZ)*deltaY
		newY = centerY + np.sin(eulerZ)*deltaX + np.cos(eulerZ)*deltaY

		return newX, newY

	def dmXY2CamXY(self, sensorName, pixelDmX, pixelDmY):
		"""

		Transform the pixel x, y from DM library to camera to use. Camera coordinate is defined
		in LCA-13381. Define camera coordinate (x, y) and DM coordinate (x', y'), then the relation
		is dx' = -dy, dy' = dx.

		 O---->y
		 |
		 |   ----------------------
		 \/ | 					   |   (x', y') = (200, 500) => (x, y) = (-500, 200) -> (3500, 200)
		 x  |					   |
			|4000				   |
		y'  |					   |
		 /\ | 		4072		   |
		 |  |----------------------
		 |
		 O-----> x'

		Arguments:
			sensorName {[str]} -- Sensor name.
			pixelDmX {[float]} -- Pixel x defined in DM coordinate.
			pixelDmY {[float]} -- Pixel y defined in DM coordinate.

		Returns:
			[float] -- Pixel x, y defined in camera coordinate based on LCA-13381.
		"""

		# Get the CCD dimension
		dimX, dimY = self.sensorDimList[sensorName]

		# Calculate the transformed coordinate
		pixelCamX = dimX-pixelDmY
		pixelCamY = pixelDmX

		return pixelCamX, pixelCamY

	def removeBg(self):
		# Remove the backgrond noise.
		pass

	def evalSNR(self):
		# Evaluate the SNR of donut.
		# Put the responsibility of evaluating the SNR in this high level class.
		pass

	def analDonutImgQual(self):
		# Analyze the donut image quality.
		pass

	def evalDonutQuality(self):
		# Evaluate the donut image quality
		pass

	def evalVignette(self, fieldX, fieldY, distanceToVignette):
		# Correct the vignette of donut images (or skip them?).
		pass

	def getSingleTargetImage(self, ccdImg, neighboringStarMapOnSingleSensor, index):
		"""

		Get the image of single scientific target and related neighboring stars.

		Arguments:
			ccdImg {[float]} -- Ccd image.
			neighboringStarMapOnSingleSensor {[dict]} -- Neighboring star map.
			index {[int]} -- Index of science target star in neighboring star map.

		Returns:
			[float] -- Ccd image of target stars.
			[float] -- Star positions in x, y.

		Raises:
			ValueError -- Science star index is out of the neighboring star map.
		"""

		# Get the target star position
		if (index >= len(neighboringStarMapOnSingleSensor.SimobjID)):
			raise ValueError("Index is higher than the length of star map.")

		# Get the star SimobjID
		brightStar = neighboringStarMapOnSingleSensor.SimobjID.keys()[index]
		neighboringStar = neighboringStarMapOnSingleSensor.SimobjID.values()[index]

		# Get all star SimobjID list
		allStar = neighboringStar[:]
		allStar.append(brightStar)

		# Get the pixel positions
		allStarPosX = []
		allStarPosY = []
		for star in allStar:
			pixelXY = neighboringStarMapOnSingleSensor.RaDeclInPixel[star]
			allStarPosX.append(pixelXY[0])
			allStarPosY.append(pixelXY[1])

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
		d1 = (maxY-minY) + 4*self.donutRadiusInPixel
		d2 = (maxX-minX) + 4*self.donutRadiusInPixel

		# Make d1 and d2 to be symmetric and even
		d = max(d1, d2)
		if (d%2 == 1):
			# Use d-1 instead of d+1 to avoid the boundary touch
			d = d-1

		# Compare the distances from the central point to four boundaries of ccd image
		cenYup = ccdD1 - cenY
		cenXright = ccdD2 - cenX

		# If central x or y plus d/2 will over the boundary, shift the central x, y values
		cenY = self.__shiftCenter(cenY, ccdD1, d/2)
		cenY = self.__shiftCenter(cenY, 0, d/2)

		cenX = self.__shiftCenter(cenX, ccdD2, d/2)
		cenX = self.__shiftCenter(cenX, 0, d/2)

		# Get the bright star and neighboring stas image
		singleSciNeiImg = ccdImg[cenY-d/2:cenY+d/2, cenX-d/2:cenX+d/2]

		# Get the stars position in the new coordinate system
		# The final one is the bright star
		allStarPosX = np.array(allStarPosX)-cenX+d/2
		allStarPosY = np.array(allStarPosY)-cenY+d/2

		return singleSciNeiImg, allStarPosX, allStarPosY

	def __shiftCenter(self, center, boundary, distance):
		"""

		Shift the center if its distance to boundary is less than required.

		Arguments:
			center {[float]} -- Center point.
			boundary {[float]} -- Boundary point.
			distance {[float]} -- Required distance.

		Returns:
			[float] -- Shifted center.
		"""

		# Distance between the center and boundary
		delta = boundary - center

		# Shift the center if needed
		if (abs(delta) < distance):
			center = boundary - np.sign(delta)*distance

		return center

	def simulateImg(self, imageFolderPath, defocalDis, neighboringStarMapOnSingleSensor, aFilterType):
		"""

		Simulate the defocal CCD images with the neighboring star map.

		Arguments:
			imageFolderPath {[str]} -- Path to image directory.
			defocalDis {[float]} -- Defocal distance in mm.
			neighboringStarMapOnSingleSensor {[dict]} -- Neighboring star map.
			aFilterType {[string]} -- Active filter type.

		Returns:
			[float] -- Simulated intra- and extra-focal images.

		Raises:
			ValueError -- No intra-focal image files.
			ValueError -- Numbers of intra- and extra-focal image files are different.
		"""

		# Generate the intra- and extra-focal ccd images
		ccdImgIntra = np.zeros(self.sensorDimList[self.sensorName])
		ccdImgExtra = ccdImgIntra.copy()

		# Redefine the format of defocal distance
		defocalDis = "%.2f" % defocalDis

		# Get all files in the image directory in a sorted order
		fileList = sorted(os.listdir(imageFolderPath))

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
			raise ValueError("The numbers of intra- and extra-focal images are different.")

		# Get the magnitude of stars
		nameOfMagAttribute = "LSSTMag" + aFilterType.upper()
		starMag = getattr(neighboringStarMapOnSingleSensor, nameOfMagAttribute)

		# Based on the neighboringStarMapOnSingleSensor to reconstruct the image
		for brightStar, neighboringStar in neighboringStarMapOnSingleSensor.SimobjID.items():

			# Generate a random number
			randNum = np.random.randint(0, high=numFile)

			# Choose a random donut image from the file
			donutImageIntra = self.__getDonutImgFromFile(imageFolderPath, intraFileList[randNum])
			donutImageExtra = self.__getDonutImgFromFile(imageFolderPath, extraFileList[randNum])

			# Get the bright star magnitude
			magBS = starMag[brightStar]

			# Combine the bright star and neighboring stars. Put the bright star in the first one.
			allStars = neighboringStar[:]
			allStars.insert(0, brightStar)

			# Add the donut image
			for star in allStars:

				# Get the brigtstar pixel x, y
				starX, starY = neighboringStarMapOnSingleSensor.RaDeclInPixel[star]
				magStar = starMag[star]

				# Ratio of magnitude between donuts (If the magnitudes of stars differs by 5,
				# the brightness differs by 100.)
				# (Magnitude difference shoulbe be >= 1.)
				magDiff = magStar-magBS
				magRatio = 1/100**(magDiff/5.0)

				# Add the donut image
				self.__addDonutImage(magRatio*donutImageIntra, starX, starY, ccdImgIntra)
				self.__addDonutImage(magRatio*donutImageExtra, starX, starY, ccdImgExtra)

		return ccdImgIntra, ccdImgExtra

	def __getDonutImgFromFile(self, imageFolderPath, fileName):
		"""

		Read the donut image from the file.

		Arguments:
			imageFolderPath {[str]} -- Path to image directory.
			fileName {[str]} -- File name.

		Returns:
			[float] -- Image in numpy array.
		"""

		# Get the donut image from the file by the delegation
		self.blendedImageDecorator.setImg(imageFile=os.path.join(imageFolderPath, fileName))

		return self.blendedImageDecorator.image.copy()


	def __addDonutImage(self, donutImage, starX, starY, ccdImg):
		"""

		Add the donut image to simulated CCD image frame.

		Arguments:
			donutImage {[float]} -- Image in numpy array.
			starX {[float]} -- Star position in pixel x.
			starY {[float]} -- Star position in pixel y.
			ccdImg {[float]} -- CCD image in numpy array.
		"""

		# Get the dimension of donut image
		d1, d2 = donutImage.shape

		# Get the interger of position to use as the index
		y = int(starY)
		x = int(starX)

		# Add the donut image on the CCD image
		ccdImg[y-int(d1/2):y-int(d1/2)+d1, x-int(d2/2):x-int(d2/2)+d2] += donutImage

if __name__ == '__main__':

	# Define the database and get the neighboring star map
	# Address of local database
	dbAdress = "/Users/Wolf/bsc.db3"

	# Boresight (RA, Dec) (unit: degree) (0 <= RA <= 360, -90 <= Dec <= 90)
	pointing = (20.0, 30.0)

	# Camera rotation
	cameraRotation = 0.0

	# Active filter type
	aFilterType = "u"

	# Camera type: "lsst" or "comcam"
	cameraType = "lsst"

	# Set the camera MJD
	cameraMJD = 59580.0

	# Camera orientation for ComCam ("center" or "corner" or "all")
	# Camera orientation for LSSTcam ("corner" or "all")
	orientation = "corner"

	# Maximum distance in units of radius one donut must be considered as a neighbor.
	spacingCoefficient = 2.5

	# For the defocus = 1.5 mm, the star's radius is 63 pixel.
	starRadiusInPixel = 63

	# Get the neighboring star map
	localDb = SourceSelector("LocalDb", cameraType)
	localDb.connect(dbAdress)
	localDb.config(starRadiusInPixel, spacingCoefficient, maxNeighboringStar=1)
	localDb.setFilter(aFilterType)
	neighborStarMapLocal, starMapLocal, wavefrontSensorsLocal = localDb.getTargetStar(pointing,
																		cameraRotation, orientation=orientation)
	localDb.disconnect()

	# Get the sensor list
	sensorList = wavefrontSensorsLocal.keys()

	# Donut image folder
	imageFolder = "/Users/Wolf/Documents/stash/cwfs_test_images"
	donutImageFolder = "LSST_C_SN26"
	fieldXY = [0, 0]

	# Instantiate a source processor
	sourProc = SourceProcessor("R00_S22_C0")

	# Give the path to the image folder
	imageFolderPath = os.path.join(imageFolder, donutImageFolder)
	sourProc.config(donutRadiusInPixel=starRadiusInPixel)

	# CCD focal plane file
	ccdFocalPlaneFolder = "/Users/Wolf/Documents/bitbucket/phosim_syseng2/data/lsst/"

	# Read the CCD focal plane data
	sourProc.readFocalPlane(ccdFocalPlaneFolder)

	# Need to do the pixel transformation for the neighboring star map


	# Generate the simulated image
	defocalDis = 1.5
	ccdImgIntra, ccdImgExtra = sourProc.simulateImg(imageFolderPath, defocalDis, neighborStarMapLocal[sensorList[0]], aFilterType)

	# Get the images of one bright star map
	singleSciNeiImg, allStarPosX, allStarPosY = sourProc.getSingleTargetImage(ccdImgIntra, neighborStarMapLocal[sensorList[0]], 0)

	# Plot the ccd image
	# poltExposureImage(ccdImgIntra, name="Intra focal image", scale="log", cmap=None)
	# poltExposureImage(ccdImgExtra, name="Extra focal image", scale="log", cmap=None)

	# poltExposureImage(ccdImgIntra, name="Intra focal image", scale="linear", cmap=None)
	# poltExposureImage(ccdImgExtra, name="Extra focal image", scale="linear", cmap=None)

	# poltExposureImage(singleSciNeiImg, name="", scale="linear", cmap=None)
	# poltExposureImage(singleSciNeiImg, name="", scale="log", cmap=None)

	# Plot the center
	posX = []
	posY = []
	for akeys, aitem in sourProc.sensorFocaPlaneInDeg.items():
		posX.append(aitem[0])
		posY.append(aitem[1])

	plt.figure()
	plt.plot(posX, posY, "bo")
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.show()

	# Get the field X, Y of donut
	pixelX = [0, 0, 2000, 2000]
	pixelY = [0, 4072, 0, 4072]
	# pixelX = [0]
	# pixelY = [0]
	# stars = neighborStarMapLocal[sensorList[0]]
	# pixelX = []
	# pixelY = []
	# for akey, aitem in stars.RaDeclInPixel.items():
	# 	pixelX.append(aitem[0])
	# 	pixelY.append(aitem[1])

	fieldX, fieldY = sourProc.getFieldXY("R00_S22_C0", np.array(pixelX), np.array(pixelY))
	plt.plot(fieldX, fieldY, "rx")
	plt.show()

	# Test the coordinate transformation
	pixelCamX, pixelCamY = sourProc.dmXY2CamXY("R00_S22_C0", 4072, 2000)
	print pixelCamX, pixelCamY

	# Test to get the focal plane position
	# When writing the test cases, need to add four corners
	camera = LsstSimMapper().camera
	obs = ObservationMetaData(pointingRA=pointing[0], pointingDec=pointing[1], rotSkyPos=cameraRotation, mjd=cameraMJD)

	stars = neighborStarMapLocal[sensorList[7]]

	bscId = stars.RaDecl.keys()[0]

	focalX, focalY = focalPlaneCoordsFromRaDec(stars.RaDecl[bscId][0], stars.RaDecl[bscId][1], obs_metadata=obs, camera=camera)
	print sourProc.focalPlaneXY2CamXY("R00_S22_C0", focalX*1000, focalY*1000)
	print sourProc.dmXY2CamXY("R00_S22_C0", stars.RaDeclInPixel[bscId][0], stars.RaDeclInPixel[bscId][1])















