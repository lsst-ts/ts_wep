#!/bin/bash

### This script along with generateTemplates.py
### is to generate the phosim templates contained in the
### policy/cwfs/templateData directory and to be used with
### donutTemplatePhosim.py

echo 'Making temporary work directory'
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/intra
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/input
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/calibs

echo 'Run phosim'

# The different commands expect the detector list in different formats
DETECTOR_LIST_DONUTS="R22_S00|R22_S01|R22_S02|R22_S10|R22_S11|R22_S12|R22_S20|R22_S21|R22_S22|"
DETECTOR_LIST_FLATS="R22_S00 R22_S01 R22_S02 R22_S10 R22_S11 R22_S12 R22_S20 R22_S21 R22_S22"

# Generate the extrafocal images
python $PHOSIMPATH/phosim.py -w $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir \
-s  $DETECTOR_LIST_DONUTS  \
$TS_WEP_DIR/policy/cwfs/templateData/starExtra.inst \
-i lsst -e 1 -c $TS_WEP_DIR/policy/cwfs/templateData/star.cmd \
-o $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra

# Repackage the phosim output
phosim_repackager.py --out_dir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw \
$TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra

# Generate the intrafocal images
python $PHOSIMPATH/phosim.py -w $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir \
-s  $DETECTOR_LIST_DONUTS  \
$TS_WEP_DIR/policy/cwfs/templateData/starIntra.inst \
-i lsst -e 1 -c $TS_WEP_DIR/policy/cwfs/templateData/star.cmd \
-o $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/intra

# Repackage the phosim output
phosim_repackager.py --out_dir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw \
$TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/intra

echo 'Ingest images'

# Create mapper file
echo lsst.obs.lsst.phosim.PhosimMapper > \
$TS_WEP_DIR/policy/cwfs/templateData/tempDir/input/_mapper

# Run ingestion
outputDir=$TS_WEP_DIR/policy/cwfs/templateData/tempDir/input
ampImg=$TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw/*.fits
ingestImages.py $outputDir $ampImg


echo 'Make flats'

# Generate flats
CWD=$PWD
cd $TS_WEP_DIR/policy/cwfs/templateData/tempDir/calibs
makeGainImages.py --detector_list $DETECTOR_LIST_FLATS
cd $CWD

echo 'Ingest flats'

# Ingest calibs
ingestCalibs.py $outputDir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/calibs/*.fits \
--validity 99999 --output $outputDir

echo 'Run ISR'

# Run ISR
runIsr.py $outputDir --rerun run1 --calib $outputDir --configfile $TS_WEP_DIR/policy/cwfs/templateData/isr_config.py --id visit=9006001^9006002

echo 'Cut Out Templates and save to ts_wep'

# Take center of each phosim image and only save the donut
python $TS_WEP_DIR/policy/cwfs/templateData/GenerateTemplates.py

echo 'Remove temporary work directories'

# Remove temporary work directories
rm -r $TS_WEP_DIR/policy/cwfs/templateData/tempDir