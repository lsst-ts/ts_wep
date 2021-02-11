#!/bin/bash

### This script along with generateTemplates.py
### is to generate the phosim templates contained in the
### policy/cwfs/templateData directory and to be used with
### donutTemplatePhosim.py

# Integer to specify number of processors to use in phosim
NUM_PROCS="${1:-1}"

TEMPLATE_CAMERA=PHOSIM_MAPPER

echo 'Making template directory'
mkdir $TS_WEP_DIR/policy/cwfs/templateData/phosimTemplates

echo 'Making temporary work directory'
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/intra
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/input
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw
mkdir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/calibs

echo 'Creating detector lists'
echo $TEMPLATE_CAMERA
if [ "$TEMPLATE_CAMERA" == "PHOSIM_MAPPER" ]
then
    file="$TS_WEP_DIR/policy/sensorNameToId.yaml"
    DETECTOR_LIST_DONUTS=""
    DETECTOR_LIST_FLATS=""
    i=1
    # Done this way to include last line in file that has no newline at end
    # See: https://stackoverflow.com/questions/12916352/shell-script-read-missing-last-line
    while read line || [ -n "$line" ]; do
        if ((i < 4))
            then
            i=$((i+1))
            else
            #Reading each line and add in sensor name
            SENSORNAME=${line:0:7}
            DETECTOR_LIST_DONUTS+="$SENSORNAME|"
            DETECTOR_LIST_FLATS+="$SENSORNAME "
            i=$((i+1))
        fi
    done < $file
else
    echo "Template Generation FAILED: Unidentified Camera"
    exit
fi

echo "Run phosim with $NUM_PROCS processors"

DETECTOR_LIST_DONUTS="R22_S11"
DETECTOR_LIST_FLATS="R22_S11"

# Generate the extrafocal images
python $PHOSIMPATH/phosim.py -w $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir \
-s $DETECTOR_LIST_DONUTS -p $NUM_PROCS \
$TS_WEP_DIR/policy/cwfs/templateData/starExtra.inst \
-i lsst -e 1 -c $TS_WEP_DIR/policy/cwfs/templateData/star.cmd \
-o $TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra

# Repackage the phosim output
phosim_repackager.py --out_dir $TS_WEP_DIR/policy/cwfs/templateData/tempDir/raw \
$TS_WEP_DIR/policy/cwfs/templateData/tempDir/phosimOutput/extra

# Generate the intrafocal images
python $PHOSIMPATH/phosim.py -w $TS_WEP_DIR/policy/cwfs/templateData/tempDir/workDir \
-s $DETECTOR_LIST_DONUTS  -p $NUM_PROCS \
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