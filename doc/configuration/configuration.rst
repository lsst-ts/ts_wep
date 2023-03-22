.. _WEP_Configuration:

#######################
WEP Configuration
#######################

Configuration File
==================

The WEP pipeline uses ``yaml`` configuration files to define the tasks it will run on a set of intra and extra-focal images.
The following is a sample configuration that can be used to run a pair of intra and extra-focal images taken with the LSSTCam Corner Wavefront Sensors (CWFS).

.. code:: yaml

    # This yaml file is used to define the tasks and configuration of
    # a Gen 3 pipeline used for testing in ts_wep.
    description: wep basic processing test pipeline
    # Here we specify the corresponding instrument for the data we
    # will be using.
    instrument: lsst.obs.lsst.LsstCam
    # Then we can specify each task in our pipeline by a name
    # and then specify the class name corresponding to that task
    tasks:
      isr:
        class: lsst.ip.isr.isrTask.IsrTask
        # Below we specify the configuration settings we want to use
        # when running the task in this pipeline. Since our data doesn't
        # include bias or flats we only want to use doApplyGains and
        # doOverscan in our isr task.
        config:
          connections.outputExposure: 'postISRCCD'
          doBias: False
          doVariance: False
          doLinearize: False
          doCrosstalk: False
          doDefect: False
          doNanMasking: False
          doInterpolate: False
          doBrighterFatter: False
          doDark: False
          doFlat: False
          doApplyGains: True
          doFringe: False
          doOverscan: True
          python: OverscanCorrectionTask.ConfigClass.fitType = 'MEDIAN'
      generateDonutCatalogWcsTask:
        class: lsst.ts.wep.task.generateDonutCatalogWcsTask.GenerateDonutCatalogWcsTask
        config:
          donutSelector.unblendedSeparation: 160
      cutOutDonutsCwfsTask:
        class: lsst.ts.wep.task.cutOutDonutsCwfsTask.CutOutDonutsCwfsTask
        config:
          # And here we specify the configuration settings originally defined in
          # CutOutDonutsCwfsTaskConfig.
          # Test with default instrument configuration parameters
          donutTemplateSize: 160
          donutStampSize: 160
          initialCutoutPadding: 40
          # Obscuration (inner_radius / outer_radius of M1M3)
          instObscuration: 0.61
          # Focal length in m
          instFocalLength: 10.312
          # Aperture diameter in m
          instApertureDiameter: 8.36
          # Defocal distance offset in mm
          # Set to 0.0 when running CWFS because
          # focal plane offset is 0.
          instDefocalOffset: 0.0
          # Camera pixel size in m
          instPixelSize: 10.0e-6
      calcZernikesTask:
        class: lsst.ts.wep.task.calcZernikesTask.CalcZernikesTask
        config:
          # Obscuration (inner_radius / outer_radius of M1M3)
          instObscuration: 0.61
          # Focal length in m
          instFocalLength: 10.312
          # Aperture diameter in m
          instApertureDiameter: 8.36
          # Defocal distance offset in mm
          # Set to 0.0 when running CWFS because
          # focal plane offset is 0.
          instDefocalOffset: 0.0
          # Camera pixel size in m
          instPixelSize: 10.0e-6

The WEP pipeline itself begins after the ISR and consists of three steps with different versions of each step available to configure your pipeline.
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

     - This task creates the donut catalog from a specified reference catalog using the image's WCS information.

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
