.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-version_history:

##################
Version History
##################

.. _lsst.ts.wep-6.2.0:

-------------
6.2.0
-------------

* Add optional pipeline task to fit WCS from direct detect donut catalogs and generate new donut catalog from reference catalogs with fit WCS.

.. _lsst.ts.wep-6.1.3:

-------------
6.1.3
-------------

* Add license information to test_generateDonutDirectDetectTask.

.. _lsst.ts.wep-6.1.3:

-------------
6.1.3
-------------

* Add license information to test_generateDonutDirectDetectTask.

.. _lsst.ts.wep-6.1.2:

-------------
6.1.2
-------------

* Import MaskedImage directly from afw.image to fix errors from change in w_2023_16.

.. _lsst.ts.wep-6.1.1:

-------------
6.1.1
-------------

* Fix blend_centroid_x and blend_centroid_y to only return donuts bright enough to count as blended when maxBlended is greater than the total number of donuts.

.. _lsst.ts.wep-6.1.0:

-------------
6.1.0
-------------

* Add optional ability to specify filter in GenerateDonutCatalogWcsTask.

.. _lsst.ts.wep-6.0.2:

-------------
6.0.2
-------------

* Fix assignment of blend_centroid_x and blend_centroid_y in donut catalogs.

.. _lsst.ts.wep-6.0.1:

-------------
6.0.1
-------------

* Redesign and enhance documentation to match style and detail of other TS repositories.

.. _lsst.ts.wep-6.0.0:

-------------
6.0.0
-------------

* Rename all modules to start with lowercase in order to align with DM convention.
* Add information into documentation about how this new change breaks repositories with data processed with older versions of ts_wep and how to fix it.

.. _lsst.ts.wep-5.1.0:

-------------
5.1.0
-------------

* Add bandpass information into DonutStamp objects.

.. _lsst.ts.wep-5.0.1:

-------------
5.0.1
-------------

* Run black v23.1.0.

.. _lsst.ts.wep-5.0.0:

-------------
5.0.0
-------------

* Remove deprecated keywords in cwfs/Tool.py and deprecated function in cwfs/CompensableImage.py.
* Remove deprecated EstimateZernikes Tasks.
* Remove deprecated pipelineConfig test files.

.. _lsst.ts.wep-4.2.3:

-------------
4.2.3
-------------

* Add transposeImages as optional config to CalcZernikesTask.

.. _lsst.ts.wep-4.2.2:

-------------
4.2.2
-------------

* Add flux sort into GenerateDonutDirectDetectTask to make it consistent with other catalog generation tasks.

.. _lsst.ts.wep-4.2.1:

-------------
4.2.1
-------------

* Use nan string instead of None so we can convert to float and use writeFits method in DonutStamps successfully and save in butler.

.. _lsst.ts.wep-4.2.0:

-------------
4.2.0
-------------

* Add DonutQuickMeasurementTask.py to incorporate quick donut detection and measurement using LSST Science Pipelines.
* Integrate DonutQuickMeasurementTask into GenerateDonutDirectDetectTask to speed up direct detection catalog generation.

.. _lsst.ts.wep-4.1.0:

-------------
4.1.0
-------------

* GenerateDonutCatalogWcsTask takes filter information from exposures automatically.
* DonutSourceSelectorTask uses policy/task/magLimitStar.yaml for default magnitude limits.

.. _lsst.ts.wep-4.0.4:

-------------
4.0.4
-------------

* Update calls to deprecated LoadIndexedReferenceObjectsTask to use LoadReferenceObjectsTask.

.. _lsst.ts.wep-4.0.3:

-------------
4.0.3
-------------

* Add blend_centroid_x and blend_centroid_y to GenerateDonutDirectDetectTask catalogs.

.. _lsst.ts.wep-4.0.2:

-------------
4.0.2
-------------

* Fix test_estimateZernikesCwfsTask call to ButlerQuantumContext.

.. _lsst.ts.wep-4.0.1:

-------------
4.0.1
-------------

* Remove Gen2 daf_persistence from UPS table.

.. _lsst.ts.wep-4.0.0:

-------------
4.0.0
-------------

* Add masked deblending to CompensableImage and pipeline tasks.
* Change how DonutSourceSelectorTask works by adding minBlendedSeparation parameter and changing DonutRadius to unblendedSeparation parameter.

.. _lsst.ts.wep-3.2.0:

-------------
3.2.0
-------------

* Port Latiss functionality from EstimateZernikesLatissTask into CutOutDonutsScienceSensorTask + CalcZernikesTask pipeline.
* Deprecate EstimateZernikes family of tasks. These tasks will no longer be updated and will be removed after January 2023.

.. _lsst.ts.wep-3.1.5:

-------------
3.1.5
-------------

* Throw exception when auxTel is trying to use offAxis model.

.. _lsst.ts.wep-3.1.4:

-------------
3.1.4
-------------

* Remove imageCoCenter step from Algorithm.
* Add DeprecationWarning that imageCoCenter function in CompensableImage will be removed after January 2023.

.. _lsst.ts.wep-3.1.3:

-------------
3.1.3
-------------

* Added default value to DonutStamp for DFC_DIST to allow the butler to read DonutStamp from repositories created with older versions of ts_wep.

.. _lsst.ts.wep-3.1.2:

-------------
3.1.2
-------------

* Update phosimOutput corner sensors test files.

.. _lsst.ts.wep-3.1.1:

-------------
3.1.1
-------------

* Fix tests pipeline yaml files updating the ISR setting to use 'MEDIAN' for overscan fit type.
* Remove obsolete _generateTestExposures.
* Fix `test_generateDonutDirectDetectTask.py`

.. _lsst.ts.wep-3.1.0:

-------------
3.1.0
-------------

* Added a history to the Algorithm class that stores intermediate products of the algorithm (see `Algorithm.getHistory()`).
* Fixed the algorithm so that it is once again symmetric with respect to I1 and I2.
  This involved simplifying the way that mask and image orientation are handled for the extrafocal image (see below).
* Added the option to create masks in the orientation of the original images by setting `compensated=False` in `CompensableImage.makeMask()`.

.. _lsst.ts.wep-3.0.1:

-------------
3.0.1
-------------

* Fix ``test_generateDonutCatalogWcsTask.py`` to work with more recent versions of the DM stack.

.. _lsst.ts.wep-3.0.0:

-------------
3.0.0
-------------

* Refactor tasks to directly accept instrument parameters in their configuration.

.. _lsst.ts.wep-2.7.0:

-------------
2.7.0
-------------

* Remove dictionary defining allowable offsets in Instrument.py and replace with settable parameter.
* Allow Instrument.py to be configured directly from dictionary of instrument parameters in addition to policy file.

.. _lsst.ts.wep-2.6.0:

-------------
2.6.0
-------------

* Replace getters and setters in Instrument.py with properties to make more pythonic.
* Update Algorithm, CompensableImage and DonutTemplateModel with new Instrument.py design.

.. _lsst.ts.wep-2.5.8:

-------------
2.5.8
-------------

* Change focusZ in headers of repackaged phosim data to be in mm instead of microns after phosim_utils update.

.. _lsst.ts.wep-2.5.7:

-------------
2.5.7
-------------

* Add defocal distance into DonutStamp.

.. _lsst.ts.wep-2.5.6:

-------------
2.5.6
-------------

* Fix task input order in test_estimateZernikes... tests.

.. _lsst.ts.wep-2.5.5:

-------------
2.5.5
-------------

* Change default maxFieldDistance in DonutSourceSelectorTask.py to 1.813 degrees based upon results from DM-33180.
* Fix test in test_calcZernikesTaskScienceSensor to use correct intraFocal dataId.

.. _lsst.ts.wep-2.5.4:

-------------
2.5.4
-------------

* Update science sensor and LATISS tasks to get focusZ from exposure visitInfo instead of metadata after update in DM-35186.

.. _lsst.ts.wep-2.5.3:

-------------
2.5.3
-------------

* Update tests and gen3TestRepo to work with latest version of the stack (w_2022_28).

.. _lsst.ts.wep-2.5.2:

-------------
2.5.2
-------------

* Add ComCam to donutTemplateModel.
* Add error message to donutTemplateModel for AuxTel if not run with 'onAxis' optical model.

.. _lsst.ts.wep-2.5.1:

-------------
2.5.1
-------------

* Correct orientation of masks in pipeline tasks.

.. _lsst.ts.wep-2.5.0:

-------------
2.5.0
-------------

* Update names of cMask to mask_comp (padded), pMask to mask_pupil (non-padded)
* Correct output of getPaddedMask to mask_comp, getNonPaddedMask to mask_pupil

.. _lsst.ts.wep-2.4.4:

-------------
2.4.4
-------------

* Added documentation link to the README.

.. _lsst.ts.wep-2.4.3:

-------------
2.4.3
-------------

* Fix online documentation build errors.

.. _lsst.ts.wep-2.4.2:

-------------
2.4.2
-------------

* Remove matplotlib backend switching in PlotUtil.py

.. _lsst.ts.wep-2.4.1:

-------------
2.4.1
-------------

* Add information on Jupyter Notebooks in ts_analysis_notebooks to README.

.. _lsst.ts.wep-2.4.0:

-------------
2.4.0
-------------

* Add CutOutDonuts tasks and CalcZernikesTask to separate cutting out donut stamps and calculating Zernikes from donut stamps as separate tasks.

.. _lsst.ts.wep-2.3.8:

-------------
2.3.8
-------------

* Remove phosim_utils dependency.

.. _lsst.ts.wep-2.3.7:

-------------
2.3.7
-------------

* Optimize CWFS algorithms.

.. _lsst.ts.wep-2.3.6:

-------------
2.3.6
-------------

* Fix rotation of sensors in EstimateZernikesBase.

.. _lsst.ts.wep-2.3.5:

-------------
2.3.5
-------------

* Update scipy.ndimage namespace to fix deprecation warnings.
* Run black v22.3.

.. _lsst.ts.wep-2.3.4:

-------------
2.3.4
-------------

* Fix test for `EstimateZernikesLatissTask`, to run for any user with /repo/main/ access.

.. _lsst.ts.wep-2.3.3:

-------------
2.3.3
-------------

* Add donut location configuration setting to `DonutSourceSelectorTask`.

.. _lsst.ts.wep-2.3.2:

-------------
2.3.2
-------------

* Change `CombineZernikesSigmaClip` to use the more robust `mad_std` standard deviation algorithm.
* Add `maxZernClip` configuration parameter to `CombineZernikesSigmaClip`.
* Change `CombineZernikes` metadata to use integer flags.

.. _lsst.ts.wep-2.3.1:

-------------
2.3.1
-------------

* Rely on GalSim for Zernike and Cartesian polynomial evaluation.

.. _lsst.ts.wep-2.3.0:

-------------
2.3.0
-------------

* Add `EstimateZernikesLatissTask` to process auxTel data
* Add `GenerateDonutDirectDetectTask` to find donuts with template fitting
* Add choices for binary image creation in `DonutDetector`
* Add `getCamType` and `getDefocalDisInMm` to `Utility`
* Add donut template for auxTel in  `DonutTemplateModel`

.. _lsst.ts.wep-2.2.4:

-------------
2.2.4
-------------

* Update Jenkinsfile to always pull the image before new builds and improve cleanup stages to make build more robust.

.. _lsst.ts.wep-2.2.3:

-------------
2.2.3
-------------

* Change `EstimateZernikesCwfsTask` to be able to accept only a single pair of wavefront sensors.
* Remove `runQuantum` function from `EstimateZernikesScienceSensorTask` since it does not add any functionality now that the task gets the camera from the butler.

.. _lsst.ts.wep-2.2.2:

-------------
2.2.2
-------------

* Update functions marked deprecated as of stack version `w_2022_06`.

.. _lsst.ts.wep-2.2.1:

-------------
2.2.1
-------------

* Distinguish AuxTel ZWO camera from LATISS

.. _lsst.ts.wep-2.2.0:

-------------
2.2.0
-------------

* Add CombineZernikes...Tasks that combine the Zernike coefficients from multiple donut pairs into a single set of coefficients.

.. _lsst.ts.wep-2.1.4:

-------------
2.1.4
-------------

* Remove `timeMethod` deprecation warnings and use static calibration camera.

.. _lsst.ts.wep-2.1.3:

-------------
2.1.3
-------------

* Fix maxBlended parameter in DonutSourceSelectorTask and improve tests to check this configuration setting.

.. _lsst.ts.wep-2.1.2:

-------------
2.1.2
-------------

* Make sure catalogs from GenerateDonutCatalog...Tasks have same columns.

.. _lsst.ts.wep-2.1.1:

-------------
2.1.1
-------------

* Get camera from the butler when running pipeline tasks.

.. _lsst.ts.wep-2.1.0:

-------------
2.1.0
-------------

* Refactor GenerateDonutCatalog*.py tasks.
* Update EstimateZernikes...Tasks after DonutCatalog refactor.

.. _lsst.ts.wep-2.0.4:

-------------
2.0.4
-------------

* Add DonutSourceSelectorTask to task module.

.. _lsst.ts.wep-2.0.3:

-------------
2.0.3
-------------

* Add RefCatalogInterface to task module.

.. _lsst.ts.wep-2.0.2:

-------------
2.0.2
-------------

* Patch to work with weekly `w_2022_2`:
    * `loadSkyCircle` no longer returns centroid column, use `loadPixelBox` instead.

.. _lsst.ts.wep-2.0.1:

-------------
2.0.1
-------------

* Patch to work with latest weekly.
* Update Jenkinsfile for CI job:
    * git command is no longer working after the latest update on our Jenkins server.
    * update path to plantuml.

.. _lsst.ts.wep-2.0.0:

-------------
2.0.0
-------------

* Removed code not used in Gen3 Pipelines.

.. _lsst.ts.wep-1.8.2:

-------------
1.8.2
-------------

* Removed CreatePhosimDonutTemplates.py and moved to `ts_phosim`.

.. _lsst.ts.wep-1.8.1:

-------------
1.8.1
-------------

* Get sensor orientation and field position directly from camera through new DonutStamp objects instead of using SourceProcessor.
* Fix rotation of postage stamps sent to WFEsti.

.. _lsst.ts.wep-1.8.0:

-------------
1.8.0
-------------

* Refactored DonutStamp.py and added ability to recreate masks as afwImage.Mask objects.

.. _lsst.ts.wep-1.7.10:

-------------
1.7.10
-------------

* Save outputZernikes for pairs of wavefront detectors not just a single output for all detectors.

.. _lsst.ts.wep-1.7.9:

-------------
1.7.9
-------------

* Remove _shiftCenterWfs from Source Processor.

.. _lsst.ts.wep-1.7.8:

-------------
1.7.8
-------------

* Update stamp rotations to work with CWFS.

.. _lsst.ts.wep-1.7.7:

-------------
1.7.7
-------------

* Update focalplanelayout.txt with new Euler angle for SW0 sensors.

.. _lsst.ts.wep-1.7.6:

-------------
1.7.6
-------------
* Update donutStamp with archive property.
* Add `LSSTCam/calib` to collections path in test Gen3 pipelines.

.. _lsst.ts.wep-1.7.5:

-------------
1.7.5
-------------

* Break generic pieces of GenerateDonutCatalogOnlineTask.py into GenerateDonutCatalogOnlineBase.py
* Add GenerateDonutCatalogWcsTask.py to calculate donut catalogs when WCS is available

.. _lsst.ts.wep-1.7.4:

-------------
1.7.4
-------------

* Remove old e-image corner wavefront sensor files.
* Add updated corner wavefront sensor test data.
* Add CWFS Zernikes code and tests.

.. _lsst.ts.wep-1.7.3:

-------------
1.7.3
-------------

* Break generic pieces of EstimateZernikesFamTask.py into EstimateZernikesBase.py

.. _lsst.ts.wep-1.7.2:

-------------
1.7.2
-------------

* Fix ``append`` and ``extend`` methods in ``DonutStamps.py``.
* Update tests in ``test_donutStamps.py`` to properly check ``append`` and ``extend`` methods.

.. _lsst.ts.wep-1.7.1:

-------------
1.7.1
-------------

* Update ``FOCUSZ`` parameter in test data.

.. _lsst.ts.wep-1.7.0:

-------------
1.7.0
-------------

* Replace ``WcsSol`` by DM's wcs code in ``GenerateDonutCatalogOnlineTask``.
* Fix intra/extra zernike selection.

.. _lsst.ts.wep-1.6.9:

-------------
1.6.9
-------------

* Add focusz as an argument to repackagePhosimImages in CreatePhosimDonutTemplates.py

.. _lsst.ts.wep-1.6.8:

-------------
1.6.8
-------------

* Return both raw and averaged Zernikes to Butler repository in EstimateZernikesFamTask.py.

.. _lsst.ts.wep-1.6.7:

-------------
1.6.7
-------------

* Fix flake error and update Jenkinsfile

.. _lsst.ts.wep-1.6.6:

-------------
1.6.6
-------------

* Remove 90 degree offset from WcsSol.py now that phosim headers are updated.

.. _lsst.ts.wep-1.6.5:

-------------
1.6.5
-------------

* Use `FOCUSZ` header information in EstimateZernikesFamTask.py.

.. _lsst.ts.wep-1.6.4:

-------------
1.6.4
-------------

* Add EstimateZernikesFamTask.py to calculate Zernike coefficients in full-array mode through a Gen 3 pipeline.

.. _lsst.ts.wep-1.6.3:

-------------
1.6.3
-------------

* Add DonutStamp and DonutStamps storage classes to hold postage stamps of donuts.

.. _lsst.ts.wep-1.6.2:

-------------
1.6.2
-------------

* Update ROTANG header in realComcam test files

.. _lsst.ts.wep-1.6.1:

-------------
1.6.1
-------------

* Update GenerateDonutCatalogOnlineTask.py to get instrument directly from pipeline configuration.
* Setup `ctrl_mpexec` package in Jenkinsfile so tests can run `pipetask` command.

.. _lsst.ts.wep-1.6.0:

-------------
1.6.0
-------------

* Create new task module
* Add GenerateDonutCatalogOnlineTask.py in task module
* Add `tests/testData/gen3TestRepo` as sample Gen 3 repo for testing

.. _lsst.ts.wep-1.5.9:

-------------
1.5.9
-------------

* Build and upload documentation as part of the CI job.
* Use develop-env image for the CI job, due to the need of java to build the documentation.
* Disable concurrent builds.
* Fix docstring in `SourceSelector.connect` method.

.. _lsst.ts.wep-1.5.8:

-------------
1.5.8
-------------

* Reformat the code by `black` v20.8b1.

.. _lsst.ts.wep-1.5.7:

-------------
1.5.7
-------------

* Update import of `DetectorType`.

.. _lsst.ts.wep-1.5.6:

-------------
1.5.6
-------------

* Reformat code with `black`.

.. _lsst.ts.wep-1.5.5:

-------------
1.5.5
-------------

* Add `DonutDetector` class.

.. _lsst.ts.wep-1.5.4:

-------------
1.5.4
-------------

* Update to using ``LsstCamMapper`` and new geometry, including ``focalplanelayout.txt``

.. _lsst.ts.wep-1.5.3:

-------------
1.5.3
-------------

* Add ``DonutTemplatePhosim`` class.
* Add ``CreatePhosimDonutTemplates`` class and add ``bin.src/runCreatePhosimDonutTemplates.py``

.. _lsst.ts.wep-1.5.2:

-------------
1.5.2
-------------

* Fix the ``ZernikeMaskedFit()`` when passing masked data

.. _lsst.ts.wep-1.5.1:

-------------
1.5.1
-------------

* Add donut template classes to make templates for ``CentroidConvolveTemplate``.
* Add ``DonutTemplateFactory``, ``DonutTemplateDefault``, and ``DonutTemplateModel``.

.. _lsst.ts.wep-1.5.0:

-------------
1.5.0
-------------

* Add ``CentroidConvolveTemplate`` as a new centroid finding method.

.. _lsst.ts.wep-1.4.9:

-------------
1.4.9
-------------

* Unify the line ending to LF.

.. _lsst.ts.wep-1.4.8:

-------------
1.4.8
-------------

* Remove the ``abbrevDectectorName()`` and ``expandDetectorName()``.
* Remove the unused arguments of ``epoch``, ``includeDistortion``, and ``mjd`` in WCS related functions.
* Fix the ``calcWfErr()`` for the **LsstCamMapper**.

.. _lsst.ts.wep-1.4.7:

-------------
1.4.7
-------------

* Remove ``sims`` and ``obs_lsstSim`` dependencies.
* Update WCS code to use ``obs_lsst``.

.. _lsst.ts.wep-1.4.6:

-------------
1.4.6
-------------

* Use the ``sims_w_2020_38``.

.. _lsst.ts.wep-1.4.5:

-------------
1.4.5
-------------

* Use the ``sims_w_2020_36``.
* Support the LSST full-array mode (FAM). Add the classes of **BaseCwfsTestCase** and **BaseBscTestCase**.
* Put the limits of star's magnitude into a configuration file.
* Remove the serialization functions in **FilterType** enum.

.. _lsst.ts.wep-1.4.4:

-------------
1.4.4
-------------

* Use the ``pybind11`` instead of ``cython``.
* Add the ``clang-format`` check to ``.githooks``.

.. _lsst.ts.wep-1.4.3:

-------------
1.4.3
-------------

* Reformat the code by ``black``.
* Add the ``black`` check to ``.githooks``.
* Ignore ``flake8`` check of E203 ans W503 for the ``black``.
* Use the ``sims_w_2020_21``.

.. _lsst.ts.wep-1.4.2:

-------------
1.4.2
-------------

* Improved handling of IO errors - catch more OS Errors instead of only file not exists.

.. _lsst.ts.wep-1.4.1:

-------------
1.4.1
-------------

* Add the function to recenter the donut image with the template.
* Add the instrument and test data of auxilirary telescope.

.. _lsst.ts.wep-1.4.0:

-------------
1.4.0
-------------

* Use the ``sims_w_2020_15``.
* Use the factory pattern for deblend module.

.. _lsst.ts.wep-1.3.9:

-------------
1.3.9
-------------

* Use the ``sims_w_2020_14``.

.. _lsst.ts.wep-1.3.8:

-------------
1.3.8
-------------

* Use the ``sims_w_2020_07``.

.. _lsst.ts.wep-1.3.7:

-------------
1.3.7
-------------

* Use the ``sims_w_2020_06``.
* Skip two tests in **test_butlerWrapper.py** and **test_camIsrWrapper.py** for the bugs in upstream.
* Feedback to DM team.

.. _lsst.ts.wep-1.3.6:

-------------
1.3.6
-------------

* Use the ``sims_w_2020_04``.

.. _lsst.ts.wep-1.3.5:

-------------
1.3.5
-------------

* Use the ``sims_w_2019_50``.

.. _lsst.ts.wep-1.3.4:

-------------
1.3.4
-------------

* Use the ``sims_w_2019_38``.

.. _lsst.ts.wep-1.3.3:

-------------
1.3.3
-------------

* Use the ``sims_w_2019_31``.
* Remove the ``conda`` package installation in **Jenkinsfile**.
* Update the permission of workspace after the unit test.

.. _lsst.ts.wep-1.3.2:

-------------
1.3.2
-------------

* Use the ``sims_w_2019_29``.
* Add the unit tests of ``cwfs`` module to check the outputs of cython related code.
* Move the ``plotImage()`` from **Tool.py** to **PlotUtil.py**.
* Install the ``ipython`` in **Jenkinsfile** to make the test environment to be consistent with the development.

.. _lsst.ts.wep-1.3.1:

-------------
1.3.1
-------------

* Use the factory pattern for centroid find algorithms.
* Move the **SensorWavefrontError** class of ``ts_ofc`` to here.

.. _lsst.ts.wep-1.3.0:

-------------
1.3.0
-------------

* Use ``sims_w_2019_24``.
* Support the eimage.
* Enable to update and save the setting file.

.. _lsst.ts.wep-1.2.9:

-------------
1.2.9
-------------

* Use ``sims_w_2019_22``.
* Adapt the new version of ``ip_isr`` that fixes the bug that can not do the ISR continuously.

.. _lsst.ts.wep-1.2.8:

-------------
1.2.8
-------------

* Use ``sims_w_2019_20``.

.. _lsst.ts.wep-1.2.7:

-------------
1.2.7
-------------

* Put the default BSC path and sky file path in default ``yaml`` file.
* Concrete **WEPCalculation** class will connect and disconnect the database at each query.
* Use ``sims_w_2019_18``.

.. _lsst.ts.wep-1.2.6:

-------------
1.2.6
-------------

* Utilize the interface classes to main telescope active optics system (MTAOS).
* Use ``sims_w_2019_17``.

.. _lsst.ts.wep-1.2.5:

-------------
1.2.5
-------------

* Support the ``documenteer``.

.. _lsst.ts.wep-1.2.4:

-------------
1.2.4
-------------

* Use the ``yaml`` format for configuration files of ``cwfs`` module.
* Use ``sims_w_2019_15``.

.. _lsst.ts.wep-1.2.3:

-------------
1.2.3
-------------

* Add the ``eups`` as the package manager.
* Use ``sims_w_2019_12``.

.. _lsst.ts.wep-1.2.2:

-------------
1.2.2
-------------

* Add the **RawExpData** class and update the related functions.

.. _lsst.ts.wep-1.2.1:

-------------
1.2.1
-------------

* Add the interface to **MTAOS** in ``ctrlIntf`` module.

.. _lsst.ts.wep-1.1.1:

-------------
1.1.1
-------------

* Updated to use the scientific pipeline of ``sims_w_2019_02``.
* Add the referece filter type.

.. _lsst.ts.wep-1.1.0:

-------------
1.1.0
-------------

* Updated the WEP to use the ``obs_lsst`` and scientific pipeline of ``sims_w_2018_47``.
* The ``phosim_utils`` is used to repackage the PhoSim output amplifer images to the format of multi-extention FITS.

.. _lsst.ts.wep-1.0.1:

-------------
1.0.1
-------------

* Updated the WEP to use the obs_lsst and scientific pipeline of ``sims_w_2018_47``.
* The phosim_utils is used to repackage the PhoSim output amplifer images to the format of multi-extention FITS.

.. _lsst.ts.wep-1.0.0:

-------------
1.0.0
-------------

* Finished the WEP in totally ideal condition with the scientific pipeline v.14.
