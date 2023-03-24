.. _User_Guide:

################
WEP User Guide
################

WEP is designed to run as a set of pipeline tasks using the `LSST Science Pipelines <https://pipelines.lsst.io/>`_.
The full WEP pipeline takes in a pair of intra and extra-focal images and generates catalogs of acceptable sources for wavefront estimation, cuts postage stamps of the sources out of the images, and finally estimates the wavefront error in terms of Zernikes polynomials from pairs of intra and extra-focal postage stamps.
All these intermediate data products are stored in the data repository for later analysis.

.. _Installing_WEP:

Installing WEP
==============

Needed Packages
---------------

- ``lsst_distrib`` from LSST Science Pipelines (tag: ``w_latest``)
- `galsim <https://github.com/GalSim-developers/GalSim>`_ (version >= 2.3; should be available from science pipelines v2.0.0 and up)
- `black <https://github.com/psf/black>`_ (optional)
- `documenteer <https://github.com/lsst-sqre/documenteer>`_ (optional)
- `plantuml <http://plantuml.com>`_ (optional)
- `sphinxcontrib-plantuml <https://pypi.org/project/sphinxcontrib-plantuml/>`_ (optional)

Install the LSST Packages and ts_wep
------------------------------------

1. Setup the LSST environment by ``source $LSST_DIR/loadLSST.bash``.
   The ``LSST_DIR`` is the directory of scientific pipeline.
2. Install the ``lsst_distrib`` by ``eups distrib install lsst_distrib -t $TAG``.
   The ``TAG`` is the weekly built version such as ``w_latest``.
3. Fix the path by:

.. code:: bash

    curl -sSL https://raw.githubusercontent.com/lsst/shebangtron/master/shebangtron | python

4. Under the directory of ``ts_wep``, do:

.. code:: bash

    setup -k -r .
    scons

.. _Using WEP:

Using WEP
===========

1. Setup the LSST Science Pipelines environment.

.. code:: bash

    source $LSST_DIR/loadLSST.bash

2. Setup the WEP environment.

.. code:: bash

    cd $path_of_ts_wep
    setup -k -r .


Example Running the Pipeline Task Framework
===========================================

To run the pipeline with the LSST Science Pipelines you can use the test repository in ``tests/testData/gen3TestRepo``.
Here we show how to run the pipeline on the LSSTCam corner wavefront sensors.

1. The order of the tasks and the configuration overrides for the tasks are set in the pipeline definition ``yaml`` file.
   In this example we will use the `testCalcZernikesCwfsPipeline <https://www.github.com/lsst-ts/ts_wep/blob/develop/tests/testData/pipelineConfigs/testCalcZernikesCwfsPipeline.yaml>`_.
   For more information on how this pipeline configuration works see :doc:`../configuration/configuration`.

2. Run the `pipetask` from the command line:

.. code:: bash

    pipetask run -b $path_of_ts_wep/tests/testData/gen3TestRepo -i refcats/gen2,LSSTCam/raw/all,LSSTCam/calib --instrument lsst.obs.lsst.LsstCam --register-dataset-types --output-run run1 -p $path_of_ts_wep/tests/testData/pipelineConfigs/testCalcZernikesCwfsPipeline.yaml -d "exposure IN (4021123106000)"

The options used above are as follows:

- ``-b``: Specifies the location of the butler repository.
- ``-i``: The ``collections`` (data) needed for the tasks.
- ``--instrument``: Defines which camera we are using.
- ``--register-dataset-types``: Add the specific datasets types for our tasks to the registry.
  - Dataset Types that are original to this repository are the ``donutCatalog``, ``donutStampsExtra``, ``donutStampsIntra``, ``outputZernikesRaw`` and ``outputZernikesAvg``. These are defined in the tasks that create them in ``python/lsst/ts/wep/task``.
- ``--output-run``: Define the name of the new collection that will hold the output from the tasks.
- ``-p``: Specifies the location of the pipeline configuration file.
- ``-d``: Define a query to further specify which data in the ``collections`` we should use.

  - The query in the example above defines the exposure ID we want to use.
    Since we have a specific exposure in the gen3 test data that includes the wavefront sensors we specify that number ``4021123106000`` here in our query.

3. If the pipeline ran successfully you can run the following command to see that the name of the new output run is present in the list:

.. code:: bash

    butler query-collections $path_of_ts_wep/tests/testData/gen3TestRepo/

Example Jupyter Notebooks
=========================

For more examples see the Jupyter notebooks found in the AOS section of `ts_analysis_notebooks <https://github.com/lsst-ts/ts_analysis_notebooks/tree/develop/aos>`_.

.. _WEP_see_also:

See Also
========

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    ../phosimDonutTemplates.rst
