#!/bin/bash

butler create DATA
butler register-instrument DATA/ lsst.obs.lsst.LsstCamPhoSim
butler ingest-raws DATA /astro/store/epyc/users/suberlak/Commissioning/aos/ts_phosim/notebooks/analysis_scripts/baselineTest/iter0/img/extra/
butler ingest-raws DATA /astro/store/epyc/users/suberlak/Commissioning/aos/ts_phosim/notebooks/analysis_scripts/baselineTest/iter0/img/intra/
butler define-visits DATA/ lsst.obs.lsst.LsstCamPhoSim
butler write-curated-calibrations DATA/ lsst.obs.lsst.LsstCamPhoSim
pipetask run -j 9 -b DATA/ -i LSSTCam-PhoSim/raw/all,LSSTCam-PhoSim/calib \
    -p testPhosimPipeline.yaml --register-dataset-types --output-run run1
