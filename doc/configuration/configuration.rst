.. _WEP_Configuration:

#######################
WEP Configuration
#######################

Configuration File
==================

The WEP pipeline uses ``yaml`` configuration files to define the tasks it will run on a set of intra and extra-focal images.
An example configuration that can be used to run a pair of intra and extra-focal images taken with the LSSTCam Corner Wavefront Sensors (CWFS) can be found `here <https://github.com/lsst-ts/ts_wep/blob/develop/tests/testData/pipelineConfigs/testCalcZernikesCwfsPipeline.yaml>`_.
The WEP pipeline itself begins after the Instrument Signature Removal (ISR) task and consists of three steps with different versions of each step available to configure your pipeline.
The three basic steps are:

1. Generate a donut source catalog for intra and extra-focal images.
2. Cut out postage stamps of the donut sources.
3. Pair up the intra and extra-focal sources and estimate wavefront error.

Under the task chosen for each step we can then configure the settings for that individual task based upon the instrument and the data.
Each task in the pipeline saves output into the data repository that is then accessible through the Butler.
In the next section we will outline the steps in the pipeline and the available tasks for each step.

WEP Pipeline Outline
====================
1. **Generate a donut source catalog.**

   Available tasks here are:

   - ``GenerateDonutCatalogsWcsTask``

     - This task creates the donut catalog from a specified reference catalog using the image's World Coordinate System (WCS) information.

   - ``GenerateDonutDirectDetectTask``

     - This task convolves the image with a model template donut to find donuts and estimate magnitudes on the image without a reference catalog.

   Butler Accessible Output:

   - ``donutCatalog``

     - A ``pandas dataframe`` with source locations and magnitudes.

2. **Cut out postage stamps.**

   Available tasks:

   - ``CutOutDonutsCwfsTask``

    - Use this task if you are analyzing data from the corner wavefront sensors.

   - ``CutOutDonutsScienceSensorTask``

     - Use this task if you are analyzing data from science sensors such as in LSSTCam Full Array Mode (FAM) or from the AuxTel.


   Butler Accessible Outputs:

   - ``donutStampsExtra``

     - Postage stamps of the extra-focal sources.

   - ``donutStampsIntra``

     - Postage stamps of the intra-focal sources.

3. **Pair up stamps and estimate wavefront error.**

   Available Task:

   - ``CalcZernikesTask``

     - This task takes in the ``donutStamps``, pairs them up and then calculates wavefront error in terms of Zernike polynomials.

   Butler Accessible Outputs:

   - ``zernikeEstimateRaw``

     - The Zernike coefficients for the estimated wavefront error for each donut pair in a pair of images.

   - ``zernikeEstimateAvg``

     - A single set of Zernike coefficients averaged over the set of coefficients from all donut pairs.
       This can be a true average over all sources but the default is a clipped average to remove outliers.
