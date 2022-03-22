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

* **WfEstimator**: Calculate the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal donut images.
* **Utility**: Utility functions used in WEP.
* **PlotUtil**: Plot utility functions used in WEP.
* **ParamReader**: Parameter reader class to read the yaml configuration files used in the calculation.
* **DonutImageCheck**: Donut image check class to judge the donut image is effective or not.
* **DonutDetector**: Detect donuts directly from an out of focus image by convolution with a template image.

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
* **DonutTemplatePhosim**: DonutTemplateDefault child class to make donut templates from templates created with Phosim. See :doc:`here <phosimDonutTemplates>` for more information on creating and using Phosim donut templates.

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

.. _lsst.ts.wep-modules_wep_task:

-------------
wep.task
-------------

This module has the tasks to run WEP as a pipeline with Gen 3 LSST DM middleware.

.. uml:: uml/taskClass.uml
    :caption: Class diagram of wep.task
* **GenerateDonutDirectDetectTaskConnections**: Connections setup for GenerateDonutDirectDetectTask to run in a pipeline with Gen 3 middleware.
* **GenerateDonutDirectDetectTaskConfig**: Configuration setup for GenerateDonutDirectDetectTask.
* **GenerateDonutDirectDetectTask**: Gen 3 middleware task to convolve the defocal postISRCCD exposure with a donut template and and create a catalog of donut sources for that exposure.
* **GenerateDonutCatalogOnlineTaskConfig**: Configuration setup for GenerateDonutCatalogOnlineTask.
* **GenerateDonutCatalogOnlineTask**: Task to take pointing information and create a catalog of donut sources in that pointing. Not a pipeline task.
* **GenerateDonutCatalogWcsTaskConnections**: Connections setup for GenerateDonutCatalogWcsTask to run in a pipeline with Gen 3 middleware.
* **GenerateDonutCatalogWcsTaskConfig**: Configuration setup for GenerateDonutCatalogWcsTask.
* **GenerateDonutCatalogWcsTask**: Gen 3 middleware task to take the WCS from each detector in a postISRCCD exposure and create a catalog of donut sources for that exposure.
* **DonutSourceSelectorTaskConfig**: Configuration setup for DonutSourceSelectorTask.
* **DonutSourceSelectorTask**: Filter a reference catalog according to parameters specified in DonutSourceSelectorTaskConfig to create a catalog of donut sources acceptable for EstimateZernikes...Task.
* **DonutStamp**: Storage class for a single donut postage stamp and associated metadata.
* **DonutStamps**: Gen 3 Butler readable storage class for a list of DonutStamp objects with helper functions to get metadata and to save DonutStamps object as FITS file.
* **EstimateZernikesBaseTaskConnections**: Base connections class for EstimateZernikes tasks.
* **EstimateZernikesBaseTaskConfig**: Base configuration class for EstimateZernikes tasks.
* **EstimateZernikesBaseTask**: Base class for EstimateZernikes tasks.
* **EstimateZernikesScienceSensorTaskConnections**: Connections setup for EstimateZernikesScienceSensorTask to run in a pipeline with Gen 3 middleware.
* **EstimateZernikesScienceSensorTaskConfig**: Configuration setup for EstimateZernikesScienceSensorTask.
* **EstimateZernikesScienceSensorTask**: Gen 3 middleware task to take exposures and donut source catalogs and calculate Zenikes coefficients for each CCD when running LSSTCam in full-array mode (FAM) or LSSTComCam. Saves Zernike coefficients and associated DonutStamps to Gen 3 repository.
* **EstimateZernikesLatissTaskConnections**: Connections setup for EstimateZernikesLatissTask to run in a pipeline with Gen 3 middleware.
* **EstimateZernikesLatissTaskConfig**: Configuration setup for EstimateZernikesLatissTask.
* **EstimateZernikesLatissTask**: Gen 3 middleware task to take exposures and donut source catalogs and calculate Zenikes coefficients for each CCD when running LATISS (auxiliary telescope). Saves Zernike coefficients and associated DonutStamps to Gen 3 repository.
* **EstimateZernikesCwfsTaskConnections**: Connections setup for EstimateZernikesCwfsTask to run in a pipeline with Gen 3 middleware.
* **EstimateZernikesCwfsTaskConfig**: Configuration setup for EstimateZernikesCwfsTask.
* **EstimateZernikesCwfsTask**: Gen 3 middleware task to take exposures and donut source catalogs and calculate Zenikes coefficients for each CCD when running on corner wave front sensors (CWFS). Saves Zernike coefficients and associated DonutStamps to Gen 3 repository.
* **RefCatalogInterface**: Tools to pick out the pieces of a reference catalog in the Gen3 Butler that cover the sky area of a pointing.
* **CombineZernikesBaseTask**: Base class for CombineZernikes tasks that combine the Zernike coefficients from multiple donuts on a detector into a single set of coefficients for the detector.
* **CombineZernikesBaseConfig**: Configuration setup for CombineZernikesBaseTask.
* **CombineZernikesMeanTask**: Gen 3 middleware task to combine the Zernike coefficients using an unweighted mean of coefficients from all donut pairs.
* **CombineZernikesSigmaClipTask**: Gen 3 middleware task to combine the Zernike coefficients with a sigma clipping method that will remove donuts with outlier Zernike values from the final averaging of donut pairs.
* **CombineZernikesSigmaClipTaskConfig**: Configuration setup for CombineZernikesSigmaClipTask.
