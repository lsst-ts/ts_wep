[![docs](https://img.shields.io/badge/docs-ts--wep.lsst.io-brightgreen)](https://ts-wep.lsst.io/)

# Wavefront Estimation Pipeline (WEP)

This module calculates the wavefront error in annular Zernike polynomials based on the intra- and extra-focal donut images in the large synoptic survey telescope (LSST).

See the docs: <https://ts-wep.lsst.io/>

## Platform

- CentOS 7
- python: 3.7.8
- scientific pipeline (`newinstall.sh` from `master` branch)

## Needed Package

- lsst_distrib (tag: `w_latest`)
- [galsim](https://github.com/GalSim-developers/GalSim) (version >= 2.3; should be available from science pipelines v2.0.0 and up)
- [black](https://github.com/psf/black) (optional)
- [documenteer](https://github.com/lsst-sqre/documenteer) (optional)
- [plantuml](http://plantuml.com) (optional)
- [sphinxcontrib-plantuml](https://pypi.org/project/sphinxcontrib-plantuml/) (optional)

## Install the LSST Packages and ts_wep

1. Setup the LSST environment by `source $LSST_DIR/loadLSST.bash`. The `LSST_DIR` is the directory of scientific pipeline.
2. Install the `lsst_distrib` by `eups distrib install lsst_distrib -t $TAG`. The `TAG` is the weekly built version such as `w_2020_52`.
3. Fix the path by `curl -sSL https://raw.githubusercontent.com/lsst/shebangtron/master/shebangtron | python`. The [shebangtron repo](https://github.com/lsst/shebangtron) has the further discussion of this.
4. Under the directory of `ts_wep`, do:

```bash
setup -k -r .
scons
```

## Pull the Built Image from Docker Hub

Pull the built docker image by `docker pull lsstsqre/centos:w_latest`.
The scientific pipeline (`lsst_distrib`) is installed already.
For the details of scientic pipeline, please follow the [Index of /stack/src/tags](https://eups.lsst.codes/stack/src/tags/).

## Code Format

1. The Python code is automatically formatted by `black`.

To enable this with a git pre-commit hook:

- Install the `black` Python package.
- Run `git config core.hooksPath .githooks` once in this repository.

## Use of Module

1. Setup the DM environment.

```bash
source $path_of_lsst_scientific_pipeline/loadLSST.bash
```

2. Setup the WEP environment.

```bash
cd $path_of_ts_wep
setup -k -r .
```

## Example Script

- **mapSensorAndFieldIdx.py**: Map the sensor name to the field point index for LSST.
- **mapSensorAndFieldIdxLsstFam.py**: Map the sensor name to the field point for LSST full-array mode (FAM).

## Example Jupyter Notebooks

To learn more about how to run the pipeline tasks associated with WEP we have a series of Jupyter Notebooks available in the AOS section of the `ts_analysis_notebooks` [repository](https://github.com/lsst-ts/ts_analysis_notebooks).

## Test Gen 3 Repository

In the folder `tests/testData/` there is a test repository for tasks that run with the Gen 3 DM middleware.
This repository is the folder `tests/testData/gen3TestRepo` and was built using the files `tests/testData/createGen3TestRepo.sh`.
This script performs the following steps to create the Gen 3 repository:

1. Ingest a reference catalog with the Gen 2 middleware.
This reference catalog is the file `tests/testData/phosimOutput/realComCam/skyComCamInfoRefCatalog.txt` which contains the same information as `skyComCamInfo.txt` in the same folder but formatted to be read in by the Gen 2 middleware as a reference catalog.
A column of g magnitude error is added with a value of 0.1 to fit reference catalog format.
The format of the reference catalog is configured with the file `tests/testData/gen3TestRepo/refCat.cfg`.
2. Convert the Gen 2 repository with the reference catalog into a Gen 3 repository since this is currently the only way to create a Gen 3 reference catalog.
This requires the configuration file `tests/testData/gen3TestRepo/convertRefCat.cfg`.
3. Ingest the raw files in `tests/testData/phosimOutput/realComCam/repackagedFiles`.
4. Clean up the original Gen 2 repository.

Inside the Gen 3 repository there are files for the Gen 3 Butler configuration (`butler.yaml`) and the repository database (`gen3.sqlite3`).

## Example Running with Gen3 Middleware (Pipeline Task Framework)

To run the pipeline with the Gen3 DM Middleware you can use the test repository in `tests/testData/gen3TestRepo`.
Here we show how to run the pipeline on the LSSTCam corner wavefront sensors.

1. The order of the tasks and the configuration overrides for the tasks are set in the pipeline definition `yaml` file.
In this case that is found in `tests/testData/pipelineConfigs/testCalcZernikesCwfsPipeline.yaml`.
Looking at this file we see that the four tasks we will run are:

- `lsst.ip.isr.isrTask.IsrTask`
- `lsst.ts.wep.task.generateDonutCatalogWcsTask.GenerateDonutCatalogWcsTask`
- `lsst.ts.wep.task.cutOutDonutsCwfsTask.CutOutDonutsCwfsTask`
- `lsst.ts.wep.task.calcZernikesTask.CalcZernikesTask`

2. Run the `pipetask` from the command line:

```bash
pipetask run -b $path_of_ts_wep/tests/testData/gen3TestRepo -i refcats/gen2,LSSTCam/raw/all,LSSTCam/calib --instrument lsst.obs.lsst.LsstCam --register-dataset-types --output-run run1 -p $path_of_ts_wep/tests/testData/pipelineConfigs/testCalcZernikesCwfsPipeline.yaml -d "exposure IN (4021123106000)"
```

The options used above are as follows:

- `-b`: Specifies the location of the butler repository.
- `-i`: The `collections` (data) needed for the tasks.
- `--instrument`: Defines which camera we are using.
- `--register-dataset-types`: Add the specific datasets types for our tasks to the registry.
  - Dataset Types that are original to this repository are the `donutCatalog`, `donutStampsExtra`, `donutStampsIntra`, `outputZernikesRaw` and `outputZernikesAvg`. These are defined in the tasks that create them in `python/lsst/ts/wep/task`.
- `--output-run`: Define the name of the new collection that will hold the output from the tasks.
- `-p`: Specifies the location of the pipeline configuration file.
- `-d`: Define a query to further specify which data in the `collections` we should use.
  - The query in the example above defines the exposure ID we want to use. Since we have a specific exposure in the gen3 test data that includes the wavefront sensors we specify that number `4021123106000` here in our query.

3. If the pipeline ran successfully you can run the following command to see that the name of the new output run is present in the list:

```bash
butler query-collections $path_of_ts_wep/tests/testData/gen3TestRepo/
```

## Using butler to access `donutStamps` created with `ts_wep` versions < 6.0.0

In `ts_wep` release 6.0.0 we updated the module names to start with a lower case letter in order to bring them in line with LSST Data Management convention and enable batch processing of the WEP Gen 3 pipeline.
This will lead to errors when attempting to load `donutStamps` created with older versions of `ts_wep`.
However adding symlinks in the following way will enable loading the older `donutStamps` through the LSST Science Pipelines butler once again.

```bash
cd $path_of_ts_wep/python/lsst/ts/wep/task/
ln -s donutStamp.py DonutStamp.py
ln -s donutStamps.py DonutStamps.py
```

Note that adding these symbolic links might make the `git` workspace dirty.
Please be careful not to add these file locations when making commits.

## Adding WCS fitting into the WEP pipeline

When running the WEP pipeline and generating donut catalogs with `GenerateDonutDirectDetectTask` it is now possible to use the directly detected donut catalogs as input for WCS fitting and catalog generation from a reference catalog.
This allows the user to use donuts selected for Zernike estimation from available reference catalogs that are more complete and have accurate astrometry and photometry rather than the directly detected donut catalogs.
To add this functionality into the WEP pipeline the user can add `GenerateDonutFromRefitWcsTask` to their pipeline configuration after `GenerateDonutDirectDetectTask`.
Thus the pipeline steps shown in [Example Running with Gen3 Middleware (Pipeline Task Framework)](#example-running-with-gen3-middleware-pipeline-task-framework) require adding in the new task after direct detection of the donuts and the complete pipeline steps in this case would be:
- `lsst.ip.isr.isrTask.IsrTask`
- `lsst.ts.wep.task.generateDonutDirectDetectTask.GenerateDonutDirectDetectTask`
- `lsst.ts.wep.task.generateDonutFromRefitWcsTask.GenerateDonutFromRefitWcsTask`
- `lsst.ts.wep.task.cutOutDonutsCwfsTask.CutOutDonutsCwfsTask`
- `lsst.ts.wep.task.calcZernikesTask.CalcZernikesTask`

Note that because the `GenerateDonutFromRefitWcsTask` saves a new output catalog and exposure with a corrected WCS there is a required additional change in the names of connections between the tasks that can be specified in the pipeline configuration file.
For an example see the [pipeline configuration file](tests/testData/pipelineConfigs/testDonutFromRefitWcsPipeline.yaml) for the tests of `GenerateDonutFromRefitWcsTask`.

## Diagram of the Corner Wavefront Sensor Geometry

    # The wavefront sensors will do the rotation as following, based on the Euler angle.

    #    R04                 R44
    # O-------           ----------O        /\ +y (CCS)
    # |  SW1 |           |    |    |        |
    # |------|           |SW0 | SW1|        |
    # |  SW0 |           |    |    |        |
    # -------O           O----------        _
    #                                   +z (.) -----> +x
    #      R00                  R40
    # ------------O          O-------
    # |     |     |          |  SW0 |
    # | SW1 | SW0 |          |------|
    # |     |     |          |  SW1 |
    # O------------          -------O

## Verify the Calculated Wavefront Error

1. The user can use the `Algorithm.getWavefrontMapEsti()` and `Algorithm.getWavefrontMapResidual()` in `cwfs` module to judge the estimated wavefront error after the solve of transport of intensity equation (TIE). The residual of wavefront map should be low compared with the calculated wavefront map if most of low-order Zernike terms (z4 - z22) have been captured and compensated.
2. The idea of TIE is to compensate the defocal images back to the focal one (image on pupil). Therefore, in the ideal case, the compensated defocal images should be similar. After the solve of TIE, the user can use the `CompensableImage.getImg()` in `cwfs` module to compare the compensated defocal images.

## Note of Auxiliary Telescope Images

1. While testing with the sky images obtained with the auxiliary telescope ZWO high speed camera (located in `tests/testData/testImages/auxTelZWO`), this package and [cwfs](https://github.com/bxin/cwfs) show the similar result in "onAxis" optical model. However, for the "paraxial" optical model, the results of two packages are different.
2. The main difference comes from the strategy of compensation in the initial loop of solving the TIE. However, it is hard to have a conclusion at this moment because of the low singal-to-noise ratio in test images.

## Difference in Corner Wavefront Sensors Euler angles to Phosim values

The Euler angles used for the corner wavefront sensors in `ts_wep` come from the [`obs_lsst`](https://github.com/lsst/obs_lsst) package maintained by the Rubin DM team.
The rotations for the R04 and R40 wavefront sensors given by the `obs_lsst` package differ by 180 degrees from those used in `phosim_syseng4` and defined in that repository in the
file [`data/lsst/focalPlaneLayout.txt`](https://github.com/lsst-ts/phosim_syseng4/blob/aos/data/lsst/focalplanelayout.txt). This is due to a difference in the rotation direction
when phosim creates images and the 180 degree difference offsets that and creates the correct orientation with the rest of the focal plane.
This is shown in the notebook [here](https://github.com/suberlak/AOS/blob/main/AOS_DM-30367_summary.ipynb).

## Note on the orientation of images and masks

***Summary:***
***Image compensation puts both images into the same orientation.***
***Pupil masks are generated in the same orientation.***
***Therefore, we do not need to explicitly rotate images or masks inside the algorithm.***

The Zernike estimation algorithm uses donut images and masks to solve the transport of intensity equation (TIE; see [Xin 2015](https://ui.adsabs.harvard.edu/abs/2015ApOpt..54.9045X)).
A pair of intra- and extra-focal donuts is used to approximate the gradient of the donut intensity with respect to the focal direction (i.e. the left-hand side of [Xin 2015](https://ui.adsabs.harvard.edu/abs/2015ApOpt..54.9045X), Equation 1), while the average of the pair is used to approximate the in-focus image (i.e. $I$ on the right-hand side of [Xin 2015](https://ui.adsabs.harvard.edu/abs/2015ApOpt..54.9045X), Equation 1).

In order to do this, we must ensure that the donuts are in the same orientation, and apply a mask that corrects for potential vignetting of the intrafocal donut.
Since the two donuts in the pair are on opposite sides of focus, their orientations are initially rotated by 180 degrees with respect to each other.
However, before calculating the mean and difference intensities, the donuts are mapped to the pupil plane and "compensated" for the current best-guess of the optical aberrations ([Xin 2015](https://ui.adsabs.harvard.edu/abs/2015ApOpt..54.9045X), Equations 9-11).
This mapping to the pupil plane automatically corrects for the orientation difference.
Therefore, we do not need to explicitly rotate either image in the algorithm.

In addition, the pupil mask that is applied to both donut images is calculated in the same orientation.
Therefore, we do not need to rotate the masks either.

## Build the Document

To build project documentation, run `package-docs build` to build the documentation.
The packages of **documenteer**, **plantuml**, and **sphinxcontrib-plantuml** are needed.
The path of `plantuml.jar` in `doc/conf.py` needs to be updated to the correct path.
To clean the built documents, use `package-docs clean`.
See [Building single-package documentation locally](https://developer.lsst.io/stack/building-single-package-docs.html) for further details.

## Reference

1. For the parameters of donut image migration, please follow: [How we predict the shapes of donuts in the WFS devices](doc/ref/200313_mask_param.pdf).
