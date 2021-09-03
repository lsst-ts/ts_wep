.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-version_history:

##################
Version History
##################

.. _lsst.ts.wep-1.8.0:

-------------
1.8.0
-------------

* Refactored DonutStamp.py and added ability to recreate masks as afwImage.MaskX objects.

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

---------
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
