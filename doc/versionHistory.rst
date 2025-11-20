.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-version_history:

##################
Version History
##################

.. _lsst.ts.wep-16.0.2:

-------------
16.0.2
-------------

* Change intrinsic map interpolation to LinearNDInterpolator when handling irregular grid of intrinsic Zernike measurements that appear due to vignetting in the model that creates the maps.
* Add small test for intrinsic map creation in test_calcZernikesDanishTaskCwfs.py.

.. _lsst.ts.wep-16.0.1:

-------------
16.0.1
-------------

* Update astromTask defaults inside refitWcsTask to turn off new default centroid error limit added in DM stack w_2025_46.

.. _lsst.ts.wep-16.0.0:

-------------
16.0.0
-------------

* Added intrinsic tables to CalcZernikesTask. Intrinsics and wavefront deviations are now saved in the Zernikes table.
* CombineZernikesTask now operates on the Zernikes table, rather than on the old arrays.

.. _lsst.ts.wep-15.1.0:

-------------
15.1.0
-------------

* Add CalcZernikesNeuralTask for neural network-based Zernike estimation using TARTS
  (Triple-stage Alignment and Reconstruction using Transformer Systems for Active Optics).
* Add new config option `oodModelPath` to pass an Out of Distribution (OOD) model directory to TARTS
  (forwarded as `ood_model_path` to NeuralActiveOpticsSys).
* Add per-donut `ood_score` column to the `zernikes` table when available from
  TARTS; falls back to NaN when the OOD model is not loaded or the score is
  missing.

.. _lsst.ts.wep-15.0.2:

-------------
15.0.2
-------------

* Add model parameters to metadata in Danish estimator outputs.

.. _lsst.ts.wep-15.0.1:

-------------
15.0.1
-------------

* Fix donut mask selection in calculateSN of cutOutDonutsBase.

.. _lsst.ts.wep-15.0.0:

-------------
15.0.0
-------------

* Fix CalcZernikesUnpairedTask to handle new `numCores` parameter in estimateZernikes.
* Redefine Unpaired tasks to run on each detector independently.
* Remove cutOutDonutsUnpairedTask since it replicates code in cutOutDonutsCwfsTask.

.. _lsst.ts.wep-14.17.2:

-------------
 14.17.2
-------------

* Run ruff format after upgrade in lsst-ts/ts_pre_commit_conf.

.. _lsst.ts.wep-14.17.1:

-------------
 14.17.1
-------------

* Fix mypy issues in image.py and plotUtils.py

.. _lsst.ts.wep-14.17.0:

-------------
 14.17.0
-------------

* Add AiDonutAlgorithm for wavefront estimation using a PyTorch model.
* Add EstimateZernikesAiDonutTask to use AiDonutAlgorithm.

.. _lsst.ts.wep-14.16.1:

-------------
 14.16.1
-------------

* Fix missing logic in generateDonutFromRefitWcsTask that caused failures when scatter in WCS fit was too large.
* Return original post_isr_image with new WCS if WCS fit successful.

.. _lsst.ts.wep-14.16.0:

-------------
 14.16.0
-------------

* Add DonutSizeCorrelator to measure donut diameters at the exposure level.
* Updated handling of radius information to donut stamps metadata, and donut quality table.

.. _lsst.ts.wep-14.15.0:

-------------
 14.15.0
-------------

* Add background subtraction to generateDonutDirectDetectTask.

.. _lsst.ts.wep-14.14.1:

-------------
 14.14.1
-------------

* Add mypy as a dependency to ts_wep.

.. _lsst.ts.wep-14.14.0:

-------------
 14.14.0
-------------

* Mypy the entire ts_wep repository.

.. _lsst.ts.wep-14.13.1:

-------------
 14.13.1
-------------

* Minor changes to help danish run more robustly.

.. _lsst.ts.wep-14.13.0:

-------------
 14.13.0
-------------

* Fix fitDonutRadiusTask to use median of cuts at different angles.

.. _lsst.ts.wep-14.12.0:

-------------
 14.12.0
-------------

* Update default exposure connection to post_isr_image from postISRCCD in pipelines and tasks.

.. _lsst.ts.wep-14.11.0:

-------------
 14.11.0
-------------

* Fix behavior of calcZernikesTask when doDonutStampSelector is False.
* Add doSelection config to donutStampSelectorTaskConfig.

.. _lsst.ts.wep-14.10.1:

-------------
 14.10.1
-------------

* Fix adding new metadata in generateDonutFromRefitWcsTask
* Update cutOutDonutsBase test value due to ISR changes

.. _lsst.ts.wep-14.10.1:

-------------
 14.10.0
-------------

* Add donut radius fail flag to donut quality tables.

.. _lsst.ts.wep-14.10.0:

-------------
 14.10.0
-------------

* Add donut radius, left edge and right edge to metadata of donutStamps.
* Refactor fitDonutRadiusTask to be a regular task and take donutStamps as input.

.. _lsst.ts.wep-14.9.0:

-------------
 14.9.0
-------------

* Add metadata from estimator outputs to zernikes table metadata.

.. _lsst.ts.wep-14.8.0:

-------------
 14.8.0
-------------

* Change metadata in donut stamps to only accept one value of visit level metadata.
* Adjust metadata of other outputs.

.. _lsst.ts.wep-14.7.0:

-------------
 14.7.0
-------------

* Add global pytest pretests to run the CWFS and Science WEP pipelines only once before running the tests.

.. _lsst.ts.wep-14.6.0:

-------------
 14.6.0
-------------

* In CalcZernikeTask compute zernikes from donut radii fit if no donuts are good for zernikeEstimation.

.. _lsst.ts.wep-14.5.0:

-------------
 14.5.0
-------------

* Add CalcZernikesTask multiprocessing.

.. _lsst.ts.wep-14.4.1:

-------------
 14.4.1
-------------

* Fixed tests broken by last PR
* Updated linting to work with ts_pre_commit_conf v0.9.16.

.. _lsst.ts.wep-14.4.0:

-------------
 14.4.0
-------------

* Added selectWithMaxPowerGrad to DonutStampSelectorTask to reject galaxy-donuts.

.. _lsst.ts.wep-14.3.3:

-------------
 14.3.3
-------------

* Update .gitattributes to include all test data files.
* Revert ROTPA change in test data.

.. _lsst.ts.wep-14.3.2:

-------------
 14.3.2
-------------

* Update dataset type definitions for cutOutDonutsCwfsPairTask outputs in test butler.
* Move additional test butler files to git lfs.

.. _lsst.ts.wep-14.3.1:

-------------
 14.3.1
-------------

 * Fixed bugs associated with the defocal offset by removing functionality to infer defocal offsets from exposure metadata. Defocal offsets will now always use the value specified in the Instrument object used by the task.

.. _lsst.ts.wep-14.3.0:

-------------
14.3.0
-------------

* Remove unused test data from testData/gen3TestRepo.
* Remove old phosim data from testData.
* Gzip test data and unzip when actually running tests.

.. _lsst.ts.wep-14.2.1:

-------------
14.2.1
-------------

* Fix donutSourceSelector when no sources pass magnitude cut.
* Handle empty donutCatalogs with cutoutDonutsCwfsTask.
* Handle no donutStamps for one side of focus for fitDonutRadiusTask.

.. _lsst.ts.wep-14.2.0:

-------------
14.2.0
-------------

* Add fitDonutRadiusTask.
* Update ReassignCwfsCutoutsTask to work per detector pair instead of visit id.
* Update CutOutDonutsCwfsTask to output donutStampsOut per detector.

.. _lsst.ts.wep-14.1.2:

-------------
14.1.2
-------------

* Fix test data by adjusting ROTPA values in headers after DM-49838.
* Enforce additional possible single threading options.

.. _lsst.ts.wep-14.1.1:

-------------
14.1.1
-------------

* Move production pipelines to donut_viz to avoid circular imports.

.. _lsst.ts.wep-14.1.0:

-------------
14.1.0
-------------

* Fix hanging in Jenkins testing by easing convergence parameters in numpy optimization methods and specifying single threading.
* Update repository infrastructure to include pyproject.toml and setup.py in line with lsst-ts developer guide.
* Add test_utils.py for helpful functions that are used in unit tests.

.. _lsst.ts.wep-14.0.1:

-------------
14.0.1
-------------

* Update tests to be compatible with Python 3.12.

.. _lsst.ts.wep-14.0.0:

-------------
14.0.0
-------------

* Change CutOutDonutsCwfsTask to run on a single detector at a time.
* Add ReassignCwfsCutoutsTask to gather cwfs donutStamps and reassign pair of intra and extra-focal stamps to extra-focal ids.

.. _lsst.ts.wep-13.4.1:

-------------
13.4.1
-------------

* Cache Zernike bases to speed up TIE algorithm.

.. _lsst.ts.wep-13.4.0:

-------------
13.4.0
-------------

* Safeguard against edge donuts at detection stage with edgeMargin parameter.

.. _lsst.ts.wep-13.3.4:

-------------
13.3.4
-------------

* Speed up cutOutDonuts tasks.

.. _lsst.ts.wep-13.3.3:

-------------
13.3.3
-------------

* Clean up and reformat USDF pipelines.

.. _lsst.ts.wep-13.3.2:

-------------
13.3.2
-------------

* Fix generateDonutDirectDetect for null donut selection.

.. _lsst.ts.wep-13.3.1:

-------------
13.3.1
-------------

* Add isr configs back into default pipelines.

.. _lsst.ts.wep-13.3.0:

-------------
13.3.0
-------------

* Add donut quality tables to outputs even when there are no donuts that pass so that we can match it up to the donut stamps and understand rejections.
* Change default pipeline setting to false for rubinTV upload.

.. _lsst.ts.wep-13.2.0:

-------------
13.2.0
-------------

* Implemented joint-fitting of donut pairs with Danish.

.. _lsst.ts.wep-13.1.0:

-------------
13.1.0
-------------

* Set saveHistory=True and loosen convergence criteria in the Danish production pipeline
* Upgrades to the forward modeling util, including specifying flux ratios for blends, miscentering donuts, and simulating "flat" donuts without intensity patterns
* Fixed bug in forward modeling util when adding noise to large flux values

.. _lsst.ts.wep-13.0.4:

-------------
13.0.4
-------------

* Increased maxFracBadPixels in pipelines to 8 pixels per 200^2.
* Updated configs for TIE and Danish production pipelines to reflect current defaults

.. _lsst.ts.wep-13.0.3:

-------------
13.0.3
-------------

* Task plotPsfFromZern added in comCamRapidAnalysisPipeline and comCamRapidAnalysisDanishPipeline.

.. _lsst.ts.wep-13.0.2:

-------------
13.0.2
-------------

* Use _refresh_metadata in cutOutStamps function so DonutStamps have correct set of metadata when running cutOutDonuts tasks interactively.

.. _lsst.ts.wep-13.0.1:

-------------
13.0.1
-------------

* Reorganize pipelines and add daily processing and danish pipelines.

.. _lsst.ts.wep-13.0.0:

-------------
13.0.0
-------------

* enabled sparse Zernike estimation
* removed most jmax and return4Up configs in favor of nollIndices configs
* removed return4Up from estimator WfEstimator and WfAlgorithm
* added makeSparse and makeDense to Zernike utils

.. _lsst.ts.wep-12.7.0:

-------------
12.7.0
-------------

* Added requireConverge to TIE and defaulted to True in task
* Fixed bug with None types in EstimateZernikeTask metadata histories

.. _lsst.ts.wep-12.6.1:

-------------
12.6.2
-------------

* Update RA production pipeline to use group dimension in aggregate donut tables step.

.. _lsst.ts.wep-12.6.1:

-------------
12.6.1
-------------

* Added a unit test for specifying DonutStampSelector.config.maxSelect in a pipeline config yaml.

.. _lsst.ts.wep-12.6.0:

-------------
12.6.0
-------------

* Added maxSelect config to DonutStampSelector

.. _lsst.ts.wep-12.5.0:

-------------
12.5.0
-------------

* Enable CutOutDonutsScienceSensorTask to operate for a pair with same-sign focusZ.

.. _lsst.ts.wep-12.4.2:

-------------
12.4.2
-------------

* Increase stamp size in Rapid Analysis pipeline to avoid clipping donut edges.

.. _lsst.ts.wep-12.4.1:

-------------
12.4.1
-------------

* Fixed bug where CalcZernikesTask fails when the number of intra/extra stamps is not equal

.. _lsst.ts.wep-12.4.0:

-------------
12.4.0
-------------

* Added a threshold on fraction-of-bad-pixels to DonutStampSelectorTask
* Modified DonutStampSelectorTaskConfig so that, by default, selections are run on fraction-of-bad-pixels and signal-to-noise ratio.
* Modified CalcZernikesTask so that DonutStampSelectorTask is run by default
* Fixed bug where DM mask bits weren't persisting in DonutStamp

.. _lsst.ts.wep-12.3.0:

-------------
12.3.0
-------------

* Added CutOutDonutsUnpairedTask and CalcZernikesUnpairedTask

.. _lsst.ts.wep-12.2.0:

-------------
12.2.0
-------------

* Update pipelines to use zernikes table instead of separate raw, avg zernike arrays.
* Propogate visit info from donut table into donutStamps to avoid calling visitInfo from the butler.

.. _lsst.ts.wep-12.1.0:

-------------
12.1.0
-------------

* Change zernikes butler storage format to QTable.

.. _lsst.ts.wep-12.0.0:

-------------
12.0.0
-------------

* Change pandas.DataFrame outputs to Astropy Tables.

.. _lsst.ts.wep-11.5.2:

-------------
11.5.2
-------------

* Added a ComCamSim production pipeline for testing purposes.

.. _lsst.ts.wep-11.5.1:

-------------
11.5.1
-------------

* Fixed bug in donutSourceSelectorTask where the task set with maxBlended > 0 and sources with a number of overlapping donuts greater than maxBlended did not give correct blend centers in the final catalog.

.. _lsst.ts.wep-11.5.0:

-------------
11.5.0
-------------

* Add astropy table output to CalcZernikesTask.

.. _lsst.ts.wep-11.4.2:

-------------
11.4.2
-------------

* Add full comcam pipeline to pipelines folder including wep and donut_viz tasks.

.. _lsst.ts.wep-11.4.1:

-------------
11.4.1
-------------

* Fix treatment of binary dilation in calculateSN.
* Fix how calculateSN masks treat blended pixels.
* Make calculateSN formatting consistent with the rest of cutOutDonutsBaseTask.
* Add a test with a blended stamp for calculateSN.
* Make variance plane warning only appear once.
* Fix test values in test_donutStampSelectorTask due to changes to ISR in w_2024_38.

.. _lsst.ts.wep-11.4.0:

-------------
11.4.0
-------------

* Set default maxNollIndex to zk28 in estimateZernikesBase.

.. _lsst.ts.wep-11.3.0:

-------------
11.3.0
-------------

* Add option to bin donut stamps before estimating the wavefront.

.. _lsst.ts.wep-11.2.0:

-------------
11.2.0
-------------

* Change CalcZernikesTask output to be at least 2D for average as well as raw to make integration with MTAOS easier.

.. _lsst.ts.wep-11.1.0:

-------------
11.1.0
-------------

* Make maxRecenteringDistance cut more robust in cutOutDonutsBase by first subtracting median shift and then comparing shifts to maxRecenteringDistance.

.. _lsst.ts.wep-11.0.0:

-------------
11.0.0
-------------

* Add donut image quality checking.

.. _lsst.ts.wep-10.6.0:

-------------
10.6.0
-------------

* Update Image bandLabel setter to handle condition where the bandLabel is string but the string is not a valid BandLabel enumeration.

.. _lsst.ts.wep-10.5.0:

-------------
10.5.0
-------------

* Fix handling of empty exposures in generateDonutDirectDetect.

.. _lsst.ts.wep-10.4.2:

-------------
10.4.2
-------------

* Add pipelines directory to easily share pipeline templates.

.. _lsst.ts.wep-10.4.1:

-------------
10.4.1
-------------

* Add visit to donutStamps metadata.

.. _lsst.ts.wep-10.4.0:

-------------
10.4.0
-------------

* Added random field angles in lsst.ts.wep.utils.modelUtils.forwardModelPair
* Fixed two bugs related to the random number generator in lsst.ts.wep.utils.modelUtils.forwardModelPair
* Added tests for lsst.ts.wep.utils.modelUtils.forwardModelPair

.. _lsst.ts.wep-10.3.0:

-------------
10.3.0
-------------

* Added single-side-of-focus mode to the TIE.

.. _lsst.ts.wep-10.2.0:

-------------
10.2.0
-------------

* Add option to pair intra/extra focal exposures by group dimension.

.. _lsst.ts.wep-10.1.1:

-------------
10.1.1
-------------

* Separate recenterFlags in cutOutDonuts tasks metadata into recenterFlagsExtra and recenterFlagsIntra.

.. _lsst.ts.wep-10.1.0:

-------------
10.1.0
-------------

* Added lsst.ts.wep.utils.modelUtils.forwardModelPair to facilitate forward modeling donuts for testing and data exploration
* Added lsst.ts.wep.utils.plotUtils.plotTieConvergence to diagnose TIE convergence

.. _lsst.ts.wep-10.0.0:

-------------
10.0.0
-------------

* Removed Zernike units configuration from tasks so that tasks always return Zernikes in microns

.. _lsst.ts.wep-9.9.0:

-------------
9.9.0
-------------

* Add auto-dilation option to making blend masks in ImageMapper.
* Fixed bugs with blend offsets for extrafocal image masks.

.. _lsst.ts.wep-9.8.1:

-------------
9.8.1
-------------

* Fixed bug in convertMetadataToHistory that failed when array shape values were floats.

.. _lsst.ts.wep-9.8.0:

-------------
9.8.0
-------------

* Add maxRecenterDistance configuration option to cutOutDonutsBase.

.. _lsst.ts.wep-9.7.0:

-------------
9.7.0
-------------

* Change configuration options for GenerateDonutFromRefitWcsTask to specify filter for photometric catalog as well.

.. _lsst.ts.wep-9.6.0:

-------------
9.6.0
-------------

* Change CombineZernikesSigmaClipTask to use kwargs dict to set arguments in astropy.stats.sigma_clip.

.. _lsst.ts.wep-9.5.8:

-------------
9.5.8
-------------

* Update to use ts_jenkins_shared_library.

.. _lsst.ts.wep-9.5.7:

-------------
9.5.7
-------------

* Update default maxFieldDist in donutSourceSelectorTask.py after analysis in DM-42067 (see ts_analysis_notebooks/aos/vignetting).

.. _lsst.ts.wep-9.5.6:

-------------
9.5.6
-------------

* Move class diagrams to mermaid from plantUML.

.. _lsst.ts.wep-9.5.5:

-------------
9.5.5
-------------

* Correct indices used to calculate Zernike average.
* Update tests to discern whether flags and mean use the same indices.

.. _lsst.ts.wep-9.5.4:

-------------
9.5.4
-------------

* Fix blend centroid coordinates in donut stamp generation.

.. _lsst.ts.wep-9.5.3:

-------------
9.5.3
-------------

* Fixed bug where blended masks have sharp edges when using dilateBlends.

.. _lsst.ts.wep-9.5.2:

-------------
9.5.2
-------------

* Fix units in ExposurePairer and add tests.

.. _lsst.ts.wep-9.5.1:

-------------
9.5.1
-------------

* Fixed compatibility with Batoid 0.6.2

.. _lsst.ts.wep-9.5.0:

-------------
9.5.0
-------------

* Add exposure pairing for full array mode.

.. _lsst.ts.wep-9.4.0:

-------------
9.4.0
-------------

* Added the Danish wavefront estimation algorithm.

.. _lsst.ts.wep-9.3.1:

-------------
9.3.1
-------------

* Added conditional sigma clipping for averaging Zernike coefficients.

.. _lsst.ts.wep-9.3.0:

-------------
9.3.0
-------------

* Added a separate instrument for full-array mode
* Updated the ComCam mask model to match the bug fixes in Batoid

.. _lsst.ts.wep-9.2.1:

-------------
9.2.1
-------------

* Added unit test directly comparing ``ImageMapper`` optical models to Batoid raytracing.

.. _lsst.ts.wep-9.2.0:

-------------
9.2.0
-------------

* Add ``LSSTComCamSim`` as allowed camera type.

.. _lsst.ts.wep-9.1.1:

-------------
9.1.1
-------------

* Fix latiss tests by using getpass, and updating Zk values

.. _lsst.ts.wep-9.1.0:

-------------
9.1.0
-------------

* Added ``jmin`` arguments to Zernike utility functions.
* Added ``jmin`` and ``jmax`` value checks to the Zernike utility functions.

.. _lsst.ts.wep-9.0.0:

-------------
9.0.0
-------------

This is a big backwards-incompatible refactor of WEP. The major changes are:

* Split the ``cwfs`` modules into ``centroid``, and ``estimation``.
* Donut Images are now held by the ``Image`` class. This class is meant to hold information in the global camera coordinate system (CCS).
* A new ``Instrument`` class with new configurations in the ``policy/instruments`` directory. This class holds geometric information about the different telescopes and cameras, as well as interfaces with the Batoid models.
* The ``ImageMapper`` class maps ``Image`` objects between the image and pupil planes, and creates pupil and image masks. The "offAxis" model now uses a real-time band-dependent fit with Batoid. The "onAxis" and "paraxial" models work the same as before.
* The Zernike estimation classes have been generalized to allow different wavefront algorithm classes to plug into ``WfEstimator``.
* The TIE algorithm is implemented in ``estimation.TieAlgorithm``.
* There are new utilities in ``utils`` for fitting mask models and plotting mask models and the ``ImageMapper`` methods.
* ``Instrument`` configuration in tasks is now pulled from the default parameter files for each camera type. Overrides can be provided via the ``instConfigFile`` parameter. With the default instrument configurations, defocal offsets are pulled from the exposure metadata. If ``defocalOffset`` is explicitly set in the ``instConfigFile`` override, that defocal offset is used instead of the values from the exposure metadata.
* The ``donutTemplateSize`` config parameter has been removed from all the relevant tasks, as the new ``ImageMapper`` can predict the required template size. ``initialCutoutPadding`` provides padding beyond this predicted value.
* The ``multiplyMask`` and ``maskGrowthIter`` parameters have been removed from ``CutOutDonutsBase``. To mask blends during TIE fitting, instead use the ``maskKwargs`` parameter of the ``EstimateZernikesTieTask``.
* When estimating Zernikes, the maximum Noll index (jmax) is now a configurable parameter (``maxNollIndex`` in ``EstimateZernikesBaseConfig``). You can also toggle whether estimation starts from zero or from the telescope's instrinsic Zernikes. You can toggle whether the task returns the full optical path difference (OPD) or just the wavefront deviation (OPD - intrinsic Zernikes). You can toggle whether the returned Zernikes start with Noll index 4 (the previous standard), or with index 0 (matching the Galsim convention). You can also set the units of the returned Zernikes.
* The algorithm history can now be saved at the Task level using the ``saveHistory`` option in ``EstimateZernikesBaseConfig``. The history is saved in the task metadata in a json-compatible format. To convert the history back to the native format, use `utils.convertMetadataToHistory`.
* Changing from the native butler coordinate system (data visualization coordinate system with rotated wavefront sensors) to the WEP coordinate system (camera coordinate system with de-rotated wavefront sensors) now happens entirely in ``task.DonutStamp._setWepImage``. Furthermore, the ``defocal_distance`` saved in the stamp is now the detector offset (or equivalent detector offset) rather than the raw focusZ info.
* The AuxTel/LATISS unit tests have been fixed, and the LATISS Zernike calculation test has been explicitly switched to a regression test (rather than an accuracy test).
* Enum's now map to strings instead of integers. This natural Enum-string connection replaces the various utils that previously existed to map between Enums and strings.

.. _lsst.ts.wep-8.3.1:

-------------
8.3.1
-------------

* Update tests to be more robust to DM changes and fix failures after DM stack update to w_2024_08.
* Run black v24.2.

.. _lsst.ts.wep-8.3.0:

-------------
8.3.0
-------------

* Remove mask_comp and mask_pupil from DonutStamp since they don't persist and mask is already contained in MaskedImage stamp.

.. _lsst.ts.wep-8.2.0:

-------------
8.2.0
-------------

* Add background subtraction to cutOutDonutsBase.

.. _lsst.ts.wep-8.1.1:

-------------
8.1.1
-------------

* Replace calls to removed pipeBase.ButlerQuantumContext with pipeBase.QuantumContext.

.. _lsst.ts.wep-8.1.0:

-------------
8.1.0
-------------

* Remove Zemax Coordinate System (ZCS) conversions now that ts_ofc works exclusively in Camera Coordinate System (CCS).

.. _lsst.ts.wep-8.0.4:

-------------
8.0.4
-------------

* Update default config on GenerateDonutFromRefitWcsTask after updates in meas_astrom.

.. _lsst.ts.wep-8.0.3:

-------------
8.0.3
-------------

* Attach locally linear WCSs to DonutStamps.

.. _lsst.ts.wep-8.0.2:

-------------
8.0.2
-------------

* Adds support for MacOS.

.. _lsst.ts.wep-8.0.1:

-------------
8.0.1
-------------

* Add convertZernikesToPsfWidth to zernikeUtils.

.. _lsst.ts.wep-8.0.0:

-------------
8.0.0
-------------

* Save all DonutStamps with images aligned with focal plane science sensors.
* This version will break compatibility in the closed loop with Phosim and ts_phosim going forward.


.. _lsst.ts.wep-7.0.1:

-------------
7.0.1
-------------

* Fix generateDonutDirectDetect when doDonutSelection is not run.

.. _lsst.ts.wep-7.0.0:

-------------
7.0.0
-------------

* Organize all utility functions inside the ``utils`` module.

.. _lsst.ts.wep-6.4.12:

-------------
6.4.12
-------------

* Update ts_pre_commit_config with ruff.

.. _lsst.ts.wep-6.4.11:

-------------
6.4.11
-------------

* Fix GenerateDonutFromRefitWcsTask adding coord_raErr, coord_decErr fields.

.. _lsst.ts.wep-6.4.10:

-------------
6.4.10
-------------

* Update calcZernikesLatissPipeline yaml with instrument-specific setup for generateDonutDirectDetectTask.

.. _lsst.ts.wep-6.4.9:

-------------
6.4.9
-------------

* Replacing lookUpCalibrations function to use the one in lsst.fgcmcal.utilities

.. _lsst.ts.wep-6.4.8:

-------------
6.4.8
-------------

* Add github actions to check version history was updated and linting.
* Fix black and flake8 violations.
* Fix Jenkinfile.

.. _lsst.ts.wep-6.4.7:

-------------
6.4.7
-------------

* Set default optical model for comCam to onAxis.

.. _lsst.ts.wep-6.4.6:

-------------
6.4.6
-------------

* Fix tests that failed due to changes in numpy testing methods and WCS output.

.. _lsst.ts.wep-6.4.5:

-------------
6.4.5
-------------

* Update setup files with pre-commit hooks, run black and isort.

.. _lsst.ts.wep-6.4.4:

-------------
6.4.4
-------------

* In ``utility``, update ``getFilterTypeFromBandLabel`` to return ``FilterType.REF`` if the ``bandLabel`` is not recognized.

.. _lsst.ts.wep-6.4.3:

-------------
6.4.3
-------------

* Fix error in Jenkinsfile that caused git-lfs to fail when running on develop branch.

.. _lsst.ts.wep-6.4.2:

-------------
6.4.2
-------------

* Move fits files to git-lfs.

.. _lsst.ts.wep-6.4.1:

-------------
6.4.1
-------------

* Add documentation explaining how to run the WEP pipeline on the USDF batch system.

.. _lsst.ts.wep-6.4.0:

-------------
6.4.0
-------------

* Create generateDonutCatalogUtils to store common methods.
* Update generateDonutCatalogOnlineTask to match output of other generateDonutCatalog...Tasks.

.. _lsst.ts.wep-6.3.5:

-------------
6.3.5
-------------

* Make sure output from empty catalogs match that expected from catalogs with sources in donutSourceSelectorTask.
* Add tests for run method in donutSourceSelectorTask.

.. _lsst.ts.wep-6.3.4:

-------------
6.3.4
-------------

* Patch refCatalogInterface to eliminate warnings from latest version of daf_butler.

.. _lsst.ts.wep-6.3.3:

-------------
6.3.3
-------------

* Change filter name in testData/gen3TestRepo camera fits files to comply with new obs_lsst convention.

.. _lsst.ts.wep-6.3.2:

-------------
6.3.2
-------------

* Change CWFS pipeline configuration files to have 1.5mm offset included and to handle this properly in CWFS version of tasks.

.. _lsst.ts.wep-6.3.1:

-------------
6.3.1
-------------

* Directly calculate dI/dz in Algorithm, without the intermediate dI.
* Save dI/dz and I0 in Algorithm history when debugLevel>=1.

.. _lsst.ts.wep-6.3.0:

-------------
6.3.0
-------------

* Add filterLabel property to CompensableImage.

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
