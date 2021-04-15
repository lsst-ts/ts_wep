#!/bin/bash

# Make Gen 2 Repo for reference catalog
mkdir gen2TestRepo
echo "lsst.obs.lsst.LsstCamMapper" > gen2TestRepo/_mapper
ingestReferenceCatalog.py gen2TestRepo phosimOutput/realComCam/skyComCamInfoRefCatalog.txt --configfile gen3TestRepo/refCat.cfg

# Convert to Gen 3 (currently only way to get Gen3 ref cat)
butler convert --gen2root gen2TestRepo --config-file gen3TestRepo/convertRefCat.cfg gen3TestRepo

# Ingest Raws
butler ingest-raws -t relsymlink gen3TestRepo phosimOutput/realComCam/repackagedFiles/extra/*.fits
butler ingest-raws -t relsymlink gen3TestRepo phosimOutput/realComCam/repackagedFiles/intra/*.fits
butler define-visits gen3TestRepo lsst.obs.lsst.LsstCam

# Clean Up
rm -r gen2TestRepo
