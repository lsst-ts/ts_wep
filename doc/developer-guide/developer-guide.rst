.. _Developer_Guide:

###################
WEP Developer Guide
###################

Wavefront estimation pipeline (WEP) calculates the wavefront error in annular Zernike polynomials up to 22 terms based on the intra- and extra-focal donut images.
The wavefront error is determined by solving the transport of intensity equation (TIE) that approximates the change of intensity mainly comes from the wavefront error.
For an in-depth explanation of the algorithm that solves the TIE equation see the following technote: `SITCOMTN-046: AOS Algorithm for Wavefront Estimation <https://sitcomtn-046.lsst.io/>`_.

.. _WEP_Main_Classes:

Main Classes
============

.. toctree::
   :maxdepth: 1

Important classes:

* `WfEstimator` calculates the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal donut images.

Important Pipeline Tasks:

* `task.GenerateDonutCatalogWcsTask` creates a donut source catalog from available reference catalogs.
* `task.CutOutDonutsScienceSensorTask` cuts out donuts stamps in image from LSSTFam or ComCam sensors when provided input exposures and source catalog.
* `task.CutOutDonutsCwfsTask` cuts out donuts stamps in image from LSSTCam corner wavefront sensors when provided input exposures and source catalog.
* `task.CalcZernikesTask` calculates the Zernikes polynomials from donut stamps already stored in a butler repository.

Important enums:

* `FilterType` defines the type of active filter.
* `CamType` defines the type of camera.
* `BscDbType` defines the type of bright star catalog database.

.. _WEP_Modules:

Modules
=======

The classes and files for each module are listed below.

.. _WEP_modules_wep:

wep
-------------

This module is a high-level module to use other modules.

.. uml:: ../uml/wepClass.uml
    :caption: Class diagram of wep

* **WfEstimator**: Calculate the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal donut images.
* **Utility**: Utility functions used in WEP.
* **PlotUtil**: Plot utility functions used in WEP.
* **ParamReader**: Parameter reader class to read the yaml configuration files used in the calculation.
* **DonutImageCheck**: Donut image check class to judge the donut image is effective or not.
* **DonutDetector**: Detect donuts directly from an out of focus image by convolution with a template image.

.. _WEP_modules_wep_cwfs:

wep.cwfs
-------------

This module calculates the wavefront error by solving the TIE.

.. uml:: ../uml/cwfsClass.uml
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
* **DonutTemplatePhosim**: DonutTemplateDefault child class to make donut templates from templates created with Phosim. See :doc:`here <../phosimDonutTemplates>` for more information on creating and using Phosim donut templates.

.. _WEP_modules_wep_deblend:

wep.deblend
-------------

This module does the image deblending.

.. uml:: ../uml/deblendClass.uml
    :caption: Class diagram of wep.deblend

* **DeblendDonutFactory**: Factory for creating the deblend donut object to deblend the bright star donut from neighboring stars.
* **DeblendDefault**: Default deblend class.
* **DeblendAdapt**: DeblendDefault child class to do the deblending by the adaptive threshold method.
* **nelderMeadModify**: Do the numerical optimation according to the Nelder-Mead algorithm.

.. _WEP_modules_wep_task:

wep.task
-------------

This module has the tasks to run WEP as a pipeline with Gen 3 LSST DM middleware.

.. uml:: ../uml/taskClass.uml
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
* **DonutSourceSelectorTask**: Filter a reference catalog according to parameters specified in DonutSourceSelectorTaskConfig to create a catalog of donut sources acceptable for Zernike estimation.
* **DonutQuickMeasurementTaskConfig** Configuration setup for DonutQuickMeasurementTask.
* **DonutQuickMeasurementTask**: Run quick donut detection and measurement on exposures.
* **DonutStamp**: Storage class for a single donut postage stamp and associated metadata.
* **DonutStamps**: Gen 3 Butler readable storage class for a list of DonutStamp objects with helper functions to get metadata and to save DonutStamps object as FITS file.
* **RefCatalogInterface**: Tools to pick out the pieces of a reference catalog in the Gen3 Butler that cover the sky area of a pointing.
* **CombineZernikesBaseTask**: Base class for CombineZernikes tasks that combine the Zernike coefficients from multiple donuts on a detector into a single set of coefficients for the detector.
* **CombineZernikesBaseConfig**: Configuration setup for CombineZernikesBaseTask.
* **CombineZernikesMeanTask**: Gen 3 middleware task to combine the Zernike coefficients using an unweighted mean of coefficients from all donut pairs.
* **CombineZernikesSigmaClipTask**: Gen 3 middleware task to combine the Zernike coefficients with a sigma clipping method that will remove donuts with outlier Zernike values from the final averaging of donut pairs.
* **CombineZernikesSigmaClipTaskConfig**: Configuration setup for CombineZernikesSigmaClipTask.
* **CalcZernikesTask**: Gen 3 middleware task to calculate the zernikes from donut stamps that already exist in the butler.
* **CalcZernikesTaskConnections**: Connections setup for CalcZernikesTask.
* **CalcZernikesTaskConfig**: Configuration setup for CalcZernikesTask.
* **CutOutDonutsBaseTask**: Base class for CutOutDonuts tasks.
* **CutOutDonutsBaseTaskConnections**: Base connections class for CutOutDonuts tasks.
* **CutOutDonutsBaseTaskConfig**: Base configuration class for CutOutDonuts tasks.
* **CutOutDonutsCwfsTask**: Gen 3 middleware task to cut out donut stamps on LSST Corner Wavefront Sensors from donut catalogs produced by GenerateDonutCatalogs tasks.
* **CutOutDonutsCwfsTaskConfig**: Configuration setup for CutOutDonutsCwfsTask.
* **CutOutDonutsScienceSensorTask**: Gen 3 middleware task to cut out donut stamps on science sensors from donut catalogs produced by GenerateDonutCatalogs tasks.
* **CutOutDonutsScienceSensorTaskConnections**: Connections setup for CutOutDonutsScienceSensorTask.
* **CutOutDonutsScienceSensorTaskConfig**: Configuration setup for CutOutDonutsScienceSensorTask.
* **FitWcsFromDetectedTask**: Optional pipeline task to take a catalog of detected donuts from GenerateDonutDirectDetectTask and fit a WCS to the input exposure and return a donut catalog using the new WCS and a reference catalog.
* **FitWcsFromDetectedTaskConnections**: Connections setup for FitWcsFromDetectedTask.
* **FitWcsFromDetectedTaskConfig**: Configuration setup for FitWcsFromDetectedTask.

.. _WEP_API:

Python API reference
====================

This section is autogenerated from docstrings.

.. automodapi:: lsst.ts.wep
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.cwfs
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.deblend
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.task
    :no-inheritance-diagram:

.. _WEP_contributing:

Contributing
============

To contribute, please start a new pull request on `GitHub <https://github.com/lsst-ts/ts_wep>`_.
Feature requests shall be filled in JIRA with *ts_aos* as the component.
In all cases, reaching out to the :ref:`contacts for this module <Contact_Personnel>` is recommended.
