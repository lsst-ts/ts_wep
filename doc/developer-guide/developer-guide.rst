.. _Developer_Guide:

###################
WEP Developer Guide
###################

Wavefront estimation pipeline (WEP) calculates the wavefront error in annular Zernike polynomials based on the intra- and extra-focal donut images.

.. _WEP_Main_Classes:

Main Classes
============

.. toctree::
   :maxdepth: 1

Important classes:

* `WfEstimator` calculates the wavefront error in annular Zernike polynomials based on the defocal donut images.

Important Pipeline Tasks:

* `task.GenerateDonutCatalogWcsTask` creates a donut source catalog from available reference catalogs.
* `task.CutOutDonutsScienceSensorTask` cuts out donuts stamps in image from LSSTFam or ComCam sensors when provided input exposures and source catalog.
* `task.CutOutDonutsCwfsTask` cuts out donuts stamps in image from LSSTCam corner wavefront sensors when provided input exposures and source catalog.
* `task.CalcZernikesTask` calculates the Zernikes polynomials from donut stamps already stored in a butler repository.
* `blitz.DonutBlitzCornerTask` runs the full corner-wavefront-sensor pipeline -- detection, selection, cutout, and Zernike fitting -- as a single visit-level task.
* `blitz.DonutBlitzFamTask` runs the same end-to-end pipeline for the full-array-mode science sensors over an intra/extra-focal exposure pair.

Important enums:

* `BandLabel` defines the type of active filter.
* `BscDbType` defines the type of bright star catalog database.

.. _WEP_Modules:

Modules
=======

The classes and files for each module are listed below.

.. _WEP_modules_wep:

wep
-------------

This module is a high-level module to use other modules.

.. mermaid:: ../uml/wepClass.mmd
    :caption: Class diagram of wep

* **Instrument**: Class that defines the geometry, mask model, Batoid model, etc. for different telescopes.
* **Image**: Image class to have the function to get the donut center.
* **ImageMapper**: Class that maps images between the pupil and image planes, and creates masks.
* **DonutImageCheck**: Donut image check class to judge the donut image is effective or not.
* **DonutDetector**: Detect donuts directly from an out of focus image by convolution with a template image.

.. _WEP_modules_wep_centroid:

wep.centroid
-------------

.. mermaid:: ../uml/centroidClass.mmd
    :caption: Class diagram of wep.centroid

This module finds the centroids of donuts.

* **CentroidFindFactory**: Factory for creating the centroid find object to calculate the centroid of donut.
* **CentroidDefault**: Default centroid class.
* **CentroidRandomWalk**: CentroidDefault child class to get the centroid of donut by the random walk model.
* **CentroidOtsu**: CentroidDefault child class to get the centroid of donut by Otsu's method.
* **CentroidConvolveTemplate**: CentroidDefault child class to get the centroids of one or more donuts in an image by convolution with a template donut.

.. _WEP_modules_wep_estimation:

wep.estimation
-------------

.. mermaid:: ../uml/estimationClass.mmd
    :caption: Class diagram of wep.estimation

This module estimates the wavefront error.

* **WfEstimator**: Calculate the wavefront error in annular Zernike polynomials based on the defocal donut images.
* **WfAlgorithm**: Base class for algorithms that estimate Zernikes from donut images.
* **WfAlgorithmFactory**: Factory for creating the algorithm classes
* **TieAlgorithm**: Class that estimates Zernikes by solving the transport of intensity equation.

.. _WEP_modules_wep_deblend:

wep.deblend
-------------

This module does the image deblending.

.. mermaid:: ../uml/deblendClass.mmd
    :caption: Class diagram of wep.deblend

* **DeblendDonutFactory**: Factory for creating the deblend donut object to deblend the bright star donut from neighboring stars.
* **DeblendDefault**: Default deblend class.
* **DeblendAdapt**: DeblendDefault child class to do the deblending by the adaptive threshold method.
* **nelderMeadModify**: Do the numerical optimation according to the Nelder-Mead algorithm.

.. _WEP_modules_wep_task:

wep.task
-------------

This module has the tasks to run WEP as a pipeline with Gen 3 LSST DM middleware.

.. mermaid:: ../uml/taskClass.mmd
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
* **EstimateZernikesBaseTask**: Base class for EstimateZernikes subtasks that estimate the Zernike coefficients from donut images.
* **EstimateZernikesBaseConfig**: Configuration for EstimateZernikesBase.
* **EstimateZernikesTieTask**: Subtask that estimates Zernikes from stamps using the TIE algorithm.
* **EstimateZernikesTieConfig**: Configuration for EstimateZernikesTieTask.
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
* **GenerateDonutFromRefitWcsTask**: Optional pipeline task to take a catalog of detected donuts from GenerateDonutDirectDetectTask and fit a WCS to the input exposure and return a donut catalog using the new WCS and a reference catalog.
* **GenerateDonutFromRefitWcsTaskConnections**: Connections setup for GenerateDonutFromRefitWcsTask.
* **GenerateDonutFromRefitWcsTaskConfig**: Configuration setup for GenerateDonutFromRefitWcsTask.
* **GenerateDonutCatalogUtils**: Common utility functions for the GenerateDonutCatalog...Tasks.
* **ExposurePairer**: Subtask to pair intra- and extra-focal exposures heuristically.
* **TablePairer**: Subtask to pair intra- and extra-focal exposures manually from a table.  Use generatePairTable.py script as a useful starting point.
* **ReassignCwfsCutoutsTask**: PipelineTask to pair intra and extra-focal donut stamps assigned to the same extra-focal dataId.
* **ReassignCwfsCutoutsTaskConfig**: Configuration for ReassignCwfsCutoutsTask.
* **ReassignCwfsCutoutsTaskConnections**: Connections for ReassignCwfsCutoutsTask.

.. _WEP_modules_wep_blitz:

wep.blitz
-------------

.. mermaid:: ../uml/blitzClass.mmd
    :caption: Class diagram of wep.blitz

This module is a self-contained, LSSTCam-only wavefront estimation pipeline that fuses donut detection, source selection, stamp cutting, and Danish wavefront fitting into a single Gen 3 middleware task per operating mode.
Unlike the staged ``wep.task`` pipeline, which hands intermediate data products (catalogs, stamps) through the butler between separate tasks, blitz runs the whole chain in one quantum and parallelizes the per-detector and per-fit-group work across a fork-based process pool.
It emits a single per-donut catalog carrying selection metrics, fit results, Zernikes, stamp/model images, and all metadata needed to regenerate diagnostic plots.

Top-level pipeline tasks:

* **DonutBlitzCornerTask**: Visit-level task that runs the end-to-end pipeline on whichever of the 8 corner wavefront sensors are present. All butler I/O happens in the parent process, and the loaded calibrations and reference catalogs are inherited by forked workers via copy-on-write.
* **DonutBlitzCornerConfig** / **DonutBlitzCornerConnections**: Configuration and connections setup for ``DonutBlitzCornerTask``.
* **DonutBlitzFamTask**: Full-array-mode counterpart that processes the 189 science detectors of an intra/extra-focal exposure pair, one quantum per exposure ``group``. To bound memory, the parent resolves only deferred butler handles and each forked worker does its own I/O, fits, and frees the pixels.
* **DonutBlitzFamConfig** / **DonutBlitzFamConnections**: Configuration and connections setup for ``DonutBlitzFamTask``.
* **DonutBlitzPlotTask**: Subtask that draws the donut, wavefront, and focal-plane diagnostic plots, either inline during a run or standalone from a persisted ``donutBlitzCornerResults`` catalog.
* **DonutBlitzPlotConfig** / **DonutBlitzPlotConnections** / **DonutBlitzPlotOutputConnections**: Configuration and connections setup for ``DonutBlitzPlotTask``.

Pipeline stages (run as subtasks within the top-level tasks):

* **BlitzDetectTask**: Template cross-correlation donut detection on a post-ISR exposure.
* **BlitzDetectConfig**: Configuration setup for ``BlitzDetectTask``.
* **MeasureDonutCandidatesTask**: Aperture photometry and quality metrics for candidate donuts.
* **MeasureDonutCandidatesConfig**: Configuration setup for ``MeasureDonutCandidatesTask``.
* **CutDonutStampsTask**: Cuts donut stamps and evaluates saturation, flux-fraction, and SNR rejection criteria.
* **CutDonutStampsConfig**: Configuration setup for ``CutDonutStampsTask``.
* **WavefrontFittingTask**: Danish wavefront fitting, including donut grouping into joint fits and the per-group fitting worker.
* **WavefrontFittingConfig**: Configuration setup for ``WavefrontFittingTask``.

Data structures passed between stages (in ``dataStructures``):

* **Donut**: One cut donut stamp with its selection and quality metrics.
* **CutoutResult**: One exposure of one detector, as the cutout pipeline returns it.
* **DetectorView** / **SourceSet**: One detector's binned image and the per-stage overlay sources for the focal-plane selection plot.
* **WfDonutResult** / **WfGroupResult**: The per-donut and per-group halves of a wavefront fit.
* **FamDetectorResult**: The full-array per-detector work-unit envelope, bundling the cutout results, fit results, and timings a worker returns.

Internal support modules:

* **catalogBuilder**: Assembles the per-donut output catalog and the focal-plane image and overlay tables shared by both modes.
* **cutoutPipeline**: The per-detector cutout pipeline (ISR, background subtraction, detection, WCS refit, selection, stamp cutting) run inside each worker.
* **famPipeline**: The full-array worker and the donut-pairing/grouping logic it uses.
* **zernikesTable**: Rebuilds the legacy per-corner ``zernikes`` table from blitz's per-donut catalog for consumers of the non-blitz corner pipeline.
* **forkPool**: A fork-per-work-unit process pool plus the hang watchdog that guards it, surviving workers killed outright (e.g. by an OOM cgroup kill).
* **lsstCam**: A frozen inventory of the LSSTCam geometry and model facts blitz depends on; imports nothing from blitz.
* **utils**: Constants and helpers shared by more than one blitz module, including the copy-on-write store that passes preloaded inputs to forked workers.

.. _WEP_modules_wep_utils:

wep.utils
-------------

This module contains utility functions that are used elsewhere in WEP.

* **enumUtils**: Enum definitions and related functions.
* **ioUtils**: Functions for reading and writing files.
* **maskUtils**: Functions for generating a mask model for an instrument.
* **taskUtils**: Functions for running command line tasks from a python script.
* **zernikeUtils**: Functions for evaluating and fitting Zernike polynomials.
* **plotUtils**: Functions for plotting results.
* **miscUtils**: Miscellaneous utility functions.
* **testUtils**: Functions for testing.

.. _WEP_API:

Python API reference
====================

This section is autogenerated from docstrings.

.. automodapi:: lsst.ts.wep
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.centroid
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.estimation
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.deblend
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.task
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.blitz
    :no-inheritance-diagram:

.. automodapi:: lsst.ts.wep.utils
    :no-inheritance-diagram:

.. _WEP_contributing:

Contributing
============

To contribute, please start a new pull request on `GitHub <https://github.com/lsst-ts/ts_wep>`_.
Feature requests shall be filled in JIRA with *ts_aos* as the component.
In all cases, reaching out to the :ref:`contacts for this module <Contact_Personnel>` is recommended.
