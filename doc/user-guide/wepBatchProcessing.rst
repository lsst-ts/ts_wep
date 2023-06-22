.. _wep_batch_processing:

#################################################
Running the WEP Pipeline on the USDF Batch System
#################################################

This guide shows how to run the WEP pipeline in batch mode on the Rubin US Data Facility (USDF).
Helpful sources to learn more about batch processing with the Rubin Science Pipelines and using the USDF nodes can be found in the following links:

- `Batch Processing on USDF <https://developer.lsst.io/usdf/batch.html>`_
- `LSST Science Pipelines Htcondor user guide <https://pipelines.lsst.io/modules/lsst.ctrl.bps.htcondor/userguide.html>`_

Steps
=====

1. Run ``allocateNodes.py`` to get "glide-in" spots in the batch processing system.
   An example command is like the following:

    .. code:: bash

        allocateNodes.py -v --dynamic -n 1 -c 2 -m 1-00:00:00 -q roma,milano -g 900 s3df

From the `USDF documentation <https://developer.lsst.io/usdf/batch.html>`_:

- ``s3df`` is specified as the target platform.
- The ``-q roma,milano`` option specifies that the glide-in jobs may run in either the roma or milano partition.
- The ``-n 1 -c 2`` options request 1 individual glide-in slot of size 2 cores (each glide-in is a Slurm job that obtains a partial node).
    - For a single pair of images on a single sensor like the Auxtel data in this example then 2 cores on a single node is as much as you need.
      For an exposure with all four wavefront sensor pairs, the maximum number of tasks that will run concurrently is 8 so there should be no need to request more than 8 cores on a node for CWFS visits.
      Furthermore, to minimize the amount of idle core time a ``-c`` value of 4 would be preferable since the later tasks in the pipeline run on pairs of wavefront sensors together.
      For a visit across the focal plane with a large number of science sensors then values of ``-c`` equal to the number of sensors up to a maximum value of 32 would be reasonable.
      Since these are requests for a partial node ``-c`` values more than 32 may be harder to schedule.
      If submitting multiple jobs to run multiple visits then it would make sense to add more nodes and increase the value of the ``-n`` parameter and procure more glide-in slots.
- The maximum possible time is set to 1 day via ``-m 1-00:00:00``.
  The glide-in Slurm jobs may not run for the full 4 days however, as the option ``-g 900`` specifies a condor glide-in shutdown time of 900 seconds or 15 minutes.
  This means that the htcondor daemons will shut themselves down after 15 minutes of inactivity (for example, after the workflow is complete), and the glide-in Slurm jobs will exit at that time to avoid wasting idle resources.
- The ``--dynamic`` option requests that the htcondor slots be dynamic, partionable slots; this is the recommended setting as it supports possible multi-core jobs in the workflow.

2. Now one can run the `ctrl_bps_htcondor <https://pipelines.lsst.io/modules/lsst.ctrl.bps.htcondor/userguide.html>`_ workflow.
   Make sure that the environment needed to run the WEP pipeline is set up in the current terminal session and run:

   .. code:: bash

       bps submit bps_wep_test.yaml

   where ``bps_wep_test.yaml`` looks like:

   .. code:: yaml

       # This is the WEP pipeline configuration file.
       pipelineYaml: "/sdf/home/b/brycek/u/dev-repos/observing/latissWepPipeline.yaml"

       # These are identifiers when looking at the bps report to check on status
       project: AOS
       campaign: DM-38273

       # Instructions to the butler
       payload:
         # This will be the name of the collection after the added prefix of 'u/$USER/'
         payloadName: latiss_wep_test
         # This is the location of the butler repository with the data you need
         butlerConfig: /sdf/group/rubin/repo/embargo/butler.yaml
         # These are the collections in that repository with the data you want
         inCollection: LATISS/calib/unbounded,LATISS/raw/all,LATISS/runs/quickLook
         # Specify a set of images with the query here (one intra-focal and one extra-focal per job)
         dataQuery: exposure in (2023020200434, 2023020200435)

   If you are unfamiliar with the WEP Pipeline Configuration files see :doc:`../configuration/configuration`.

Additional Notes
----------------

1. Check on job status with ``bps report``.
2. You can overwrite the data query in the batch ``yaml`` file from the command line with ``-d``.
   For example:

   .. code:: bash

    bps submit -d "exposure.observation_type='cwfs' and exposure.day_obs=20220912 and exposure.seq_num in (96..97)"  bps_wep_test.yaml

   The pipeline will run and the output can be found in the butler specified by ``butlerConfig`` under the collection specified under ``payloadName`` in ``bps_wep_test.yaml``.
   To overwrite the ``butlerConfig`` on the command line use the ``-b`` option and to overwrite the output collection use the ``-o`` option in the command line instruction for ``bps submit``.
   For more options and information on ``bps submit`` see `here <https://pipelines.lsst.io/v/weekly/modules/lsst.ctrl.bps/quickstart.html#submitting-a-run>`_.
