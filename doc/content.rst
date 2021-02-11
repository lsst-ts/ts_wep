.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-modules:

##########
Modules
##########

The classes and files for each module are listed below.

.. _lsst.ts.wep-modules_wep:

-------------
wep
-------------

This module is a high-level module to use other modules.

.. uml:: uml/wepClass.uml
    :caption: Class diagram of wep

* **ButlerWrapper**: Wrapper of DM butler class to get the raw and post-ISR CCD image.
* **CamDataCollector**: Ingest the amplifier images and calibration products based on the DM command line task.
* **CamIsrWrapper**: Do the ISR and assemble the CCD images based on the DM command line task.
* **SourceSelector**: Query the bright star catalog (BSC) to select the available target to calculate the wavefront error.
* **SourceProcessor**: Process the post-ISR images to get the clean star images with measured optical field position. The deblending algorithm is used to get the single target bright star image if the neighboring star exists.
* **WfEstimator**: Calculate the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal donut images.
* **DefocalImage**: Defocal image class that provides the accessor methods.
* **DonutImage**: Donut image class that provides the accessor methods.
* **WepController**: High level class to use the WEP package.
* **Utility**: Utility functions used in WEP.
* **PlotUtil**: Plot utility functions used in WEP.
* **ParamReader**: Parameter reader class to read the yaml configuration files used in the calculation.
* **DonutImageCheck**: Donut image check class to judge the donut image is effective or not.

.. _lsst.ts.wep-modules_wep_bsc:

-------------
wep.bsc
-------------

This module queries the bright star catalog and gets the scientific target.

.. uml:: uml/bscClass.uml
    :caption: Class diagram of wep.bsc

* **CamFactory**: Camera factory to create the concrete camera object.
* **CameraData**: Camera data class as the parent of specific camera child class.
* **ComCam**: Commissioning camera class. The parent class is the CameraData class.
* **LsstCam**: LSST camera class to use the corner wavefront sensor. The parent class is the CameraData class.
* **LsstFamCam**: Lsst camera class to use the full-array mode (FAM). The wavefront sensor is the scientific sensor. The parent class is the CameraData class.
* **DatabaseFactory**: Database factory to create the concrete database object.
* **DefaultDatabase**: Default database class as the parent of specific database child class.
* **LocalDatabase**: Local database class. The parent class is the DefaultDatabase class.
* **LocalDatabaseForStarFile**: Local database class to read the star file. The parent class is the LocalDatabase class.
* **StarData**: Star data class for the scientific target star.
* **NbrStar**: Neighboring star class to have the bright star and the related neighboring stars.
* **Filter**: Filter class to provide the scientific target star magnitude boundary.
* **WcsSol**: Wavefront coordinate system (WCS) solution class to map the sky position to camera pixel position and vice versa.
* **PlotStarFunc**: Plot funtions used in this bsc module.
* **BaseBscTestCase**: Base class for the bright star catalog (BSC) tests.

.. _lsst.ts.wep-modules_wep_ctrlIntf:

-------------
wep.ctrlIntf
-------------

This module provides the interface classes to the main telescope active optics system (MTAOS). The factory pattern is applied to support the multiple instruments.

.. uml:: uml/ctrlIntfClass.uml
    :caption: Class diagram of wep.ctrlIntf

* **WEPCalculationFactory**: Factory for creating the correct WEP calculation based off the camera type currently being used.
* **WEPCalculation**: Base class for converting the wavefront images into wavefront errors.
* **WEPCalculationOfPiston**: The child class of WEPCalculation that gets the defocal images by the camera piston.
* **WEPCalculationOfLsstCam**: The concrete child class of WEPCalculation of the LSST camera (corner wavefront sensor).
* **WEPCalculationOfComCam**: The concrete child class of WEPCalculationOfPiston of the commionning camera (ComCam).
* **WEPCalculationOfLsstFamCam**: The concrete child class of WEPCalculationOfPiston of the LSST full-array mode (FAM) camera.
* **SensorWavefrontError**: Sensor wavefront error class. This class contains the information of sensor Id and related wavefront error.
* **SensorWavefrontData**: Sensor wavefront data class that has the information of sensor Id, list of donut, master donut, and wavefront error. This is the child class of SensorWavefrontError class.
* **WcsData**: Contains the world coordinate system (WCS) data of a camera.
* **AstWcsSol**: AST world coordinate system (WCS) solution provided by DM team.
* **RawExpData**: Raw exposure data class to populate the information of visit, snap, and data directory.
* **MapSensorNameAndId**: Map the sensor name and Id class to transform the name and Id with each other.

.. _lsst.ts.wep-modules_wep_cwfs:

-------------
wep.cwfs
-------------

This module calculates the wavefront error by solving the TIE.

.. uml:: uml/cwfsClass.uml
    :caption: Class diagram of wep.cwfs

* **Algorithm**: Algorithm class to solve the TIE to get the wavefront error.
* **CompensableImage**: Compensable image class to project the donut image from the image plane to the pupil plane.
* **Image**: Image class to have the function to get the donut center.
* **Instrument**: Instrument class to have the instrument information used in the Algorithm class to solve the TIE.
* **Tool**: Annular Zernike polynomials related functions.
* **CentroidFindFactory**: Factory for creating the centroid find object to calculate the centroid of donut.
* **CentroidDefault**: Default centroid class.
* **CentroidRandomWalk**: CentroidDefault child class to get the centroid of donut by the random walk model.
* **CentroidOtsu**: CentroidDefault child class to get the centroid of donut by the Otsu's method.
* **CentroidConvolveTemplate**: CentroidDefault child class to get the centroids of one or more donuts in an image by convolution with a template donut.
* **BaseCwfsTestCase**: Base class for CWFS tests.
* **DonutTemplateFactory**: Factory for creating donut template objects used by CentroidConvolveTemplate.
* **DonutTemplateDefault**: Default donut template class.
* **DonutTemplateModel**: DonutTemplateDefault child class to make donut templates using an Instrument model.
* **DonutTemplatePhosim**: DonutTemplateDefault child class to make donut templates from templates created with Phosim.

.. _lsst.ts.wep-modules_wep_deblend:

-------------
wep.deblend
-------------

This module does the image deblending.

.. uml:: uml/deblendClass.uml
    :caption: Class diagram of wep.deblend

* **DeblendDonutFactory**: Factory for creating the deblend donut object to deblend the bright star donut from neighboring stars.
* **DeblendDefault**: Default deblend class.
* **DeblendAdapt**: DeblendDefault child class to do the deblending by the adaptive threshold method.
* **nelderMeadModify**: Do the numerical optimation according to the Nelder-Mead algorithm.
