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

1. Run `allocateNodes.py` to get "glide-in" spots in the batch processing system.
   An example command is like the following:

    .. code:: bash

        allocateNodes.py -v --dynamic -n 20 -c 32 -m 4-00:00:00 -q roma,milano -g 900 s3df

From the `USDF documentation <https://developer.lsst.io/usdf/batch.html>`_:

    - `s3df` is specified as the target platform.
    - The `-q roma,milano` option specifies that the glide-in jobs may run in either the roma or milano partition.
    - The `-n 20 -c 32` options request 20 individual glide-in slots of size 32 cores each (each is a Slurm job that obtains a partial node).
    - The maximum possible time is set to 4 days via `-m 4-00:00:00`.
      The glide-in Slurm jobs may not run for the full 4 days however, as the option `-g 900` specifies a condor glide-in shutdown time of 900 seconds or 15 minutes.
      This means that the htcondor daemons will shut themselves down after 15 minutes of inactivity (for example, after the workflow is complete), and the glide-in Slurm jobs will exit at that time to avoid wasting idle resources.
    - The `--dynamic` option requests that the htcondor slots be dynamic, partionable slots; this is the recommended setting as it supports possible multi-core jobs in the workflow.

2. Now one can run the `ctrl_bps_htcondor <https://pipelines.lsst.io/modules/lsst.ctrl.bps.htcondor/userguide.html>` workflow
3. Make sure that the environment needed to run the WEP pipeline is set up in the current terminal session and run:

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
         # Specify a set of images with the query here
         dataQuery: exposure in (2023020200434, 2023020200435)

   If you are unfamiliar with the WEP Pipeline Configuration files see :doc:`../configuration/configuration`.

Additional Notes
----------------

1. Check on job status with `bps report`
2. You can overwrite the data query in the batch ``yaml`` file from the command line with ``-d``.
   For example:

   .. code:: bash

    bps submit -d "exposure.observation_type='cwfs' and exposure.day_obs=20220912 and exposure.seq_num in (96..97)"  bps_wep_test.yaml
