# Wavefront Estimation Pipeline (WEP)

This module calculates the wavefront error in annular Zernike polynomials up to 22 terms based on the intra- and extra-focal donut images in the large synoptic survey telescope (LSST).

## Platform

- CentOS 7
- python: 3.7.8
- scientific pipeline (`newinstall.sh` from `master` branch)

## Needed Package

- lsst_distrib (tag: `w_latest`)
- [phosim_utils](https://github.com/lsst-dm/phosim_utils)
- clang-format (optional)
- [black](https://github.com/psf/black) (optional)
- [documenteer](https://github.com/lsst-sqre/documenteer) (optional)
- [plantuml](http://plantuml.com) (optional)
- [sphinxcontrib-plantuml](https://pypi.org/project/sphinxcontrib-plantuml/) (optional)

## Install the LSST Packages, phosim_utils, and ts_wep

1. Setup the LSST environment by `source $LSST_DIR/loadLSST.bash`. The `LSST_DIR` is the directory of scientific pipeline.
2. Install the `lsst_distrib` by `eups distrib install lsst_distrib -t $TAG`. The `TAG` is the weekly built version such as `w_2020_52`.
3. Fix the path by `curl -sSL https://raw.githubusercontent.com/lsst/shebangtron/master/shebangtron | python`. The [shebangtron repo](https://github.com/lsst/shebangtron) has the further discussion of this.
4. Clone the repository of `phosim_utils` to some other directory. Under the `phosim_utils/` directory, use `setup -k -r . -t current` to setup the package in `eups` and use `scons` to build the module. The tag of `current` means the current weekly built version. It is noted that the build process is only needed for the first time.
5. Under the directory of `ts_wep`, do:

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
2. The C++ code is automatically formatted by `clang-format`.

To enable this with a git pre-commit hook:

- Install the `black` Python package.
- Install the `clang-format` C++ package.
- Run `git config core.hooksPath .githooks` once in this repository.

## DM Command Line Task (obs_lsst and phosim_utils)

1. Make the faked flat images. Flats only need to be made once. They can then be shared between repos. The flats can be faked with (1) all sensors, (2) corner wavefront sensors, or (3) user-specified sensor list.

```bash
cd $work_dir
mkdir fake_flats
cd fake_flats/
makeGainImages.py
cd ..
```

2. Repackage the PhoSim output amplifiers. The data needs to be put in single 16 extension MEFs (Multi-Extension FITS) for processing.

```bash
phosim_repackager.py $phosim_amp_dir --out_dir=repackaged_files
```

3. Make the repository for butler to use, ingest the images, and ingest the calibration products.

```bash
mkdir input
echo lsst.obs.lsst.phosim.PhosimMapper > input/_mapper
ingestImages.py input repackaged_files/*.fits
ingestCalibs.py input fake_flats/* --validity 99999 --output input
```

4. Make the config override file to turn only flat field on.

```bash
echo "config.isr.doBias=False
config.isr.doDark=False
config.isr.doFlat=True
config.isr.doFringe=False
config.isr.doDefect=False" >isr_config.py
```

5. Run the instrument signature removal (ISR).

```bash
runIsr.py input --id --rerun=run1 --configfile isr_config.py
```

## Use of Module

1. Setup the DM environment.

```bash
source $path_of_lsst_scientific_pipeline/loadLSST.bash
cd $path_of_phosim_utils
setup -k -r . -t current
```

2. Setup the WEP environment.

```bash
cd $path_of_ts_wep
setup -k -r .
```

## Example Script

- **mapSensorAndFieldIdx.py**: Map the sensor name to the field point index for LSST.
- **mapSensorAndFieldIdxLsstFam.py**: Map the sensor name to the field point for LSST full-array mode (FAM).

## Verify the Calculated Wavefront Error

1. The user can use the `Algorithm.getWavefrontMapEsti()` and `Algorithm.getWavefrontMapResidual()` in `cwfs` module to judge the estimated wavefront error after the solve of transport of intensity equation (TIE). The residual of wavefront map should be low compared with the calculated wavefront map if most of low-order Zernike terms (z4 - z22) have been captured and compensated.
2. The idea of TIE is to compensate the defocal images back to the focal one (image on pupil). Therefore, in the ideal case, the compensated defocal images should be similar. After the solve of TIE, the user can use the `CompensableImage.getImg()` in `cwfs` module to compare the compensated defocal images.

## Note of Auxiliary Telescope Images

1. While testing with the sky images obtained with the auxiliary telescope (localed in `tests/testData/testImages/auxTel`), this package and [cwfs](https://github.com/bxin/cwfs) show the similar result in "onAxis" optical model. However, for the "paraxial" optical model, the results of two packages are different.
2. The main difference comes from the stratege of compensation in the initial loop of solving the TIE. However, it is hard to have a conclusion at this moment because of the low singal-to-noise ratio in test images.

## Test Gen 3 Repository

In the folder `tests/testData/` there is a test repository for tasks that run with the Gen 3 DM middleware. This repository is the folder `tests/testData/gen3TestRepo` and was built using the
files `tests/testData/createGen3TestRepo.sh`. This script performs the following steps to create the Gen 3 repository:

  1. Ingest a reference catalog with the Gen 2 middleware. This reference catalog is the file `tests/testData/phosimOutput/realComCam/skyComCamInfoRefCatalog.txt` which contains the same information as
  `skyComCamInfo.txt` in the same folder but formatted to be read in by the Gen 2 middleware as a reference catalog. A column of g magnitude error is added with a value of 0.1 to fit reference catalog format.
  The format of the reference catalog is configured with the file `tests/testData/gen3TestRepo/refCat.cfg`.
  2. Convert the Gen 2 repository with the reference catalog into a Gen 3 repository since this is currently the only way to create a Gen 3 reference catalog. This requires the configuration file
  `tests/testData/gen3TestRepo/convertRefCat.cfg`.
  3. Ingest the raw files in `tests/testData/phosimOutput/realComCam/repackagedFiles`.
  4. Clean up the original Gen 2 repository.

Inside the Gen 3 repository there are files for the Gen 3 Butler configuration (`butler.yaml`) and the repository database (`gen3.sqlite3`).

## Build the Document

To build project documentation, run `package-docs build` to build the documentation.
The packages of **documenteer**, **plantuml**, and **sphinxcontrib-plantuml** are needed.
The path of `plantuml.jar` in `doc/conf.py` needs to be updated to the correct path.
To clean the built documents, use `package-docs clean`.
See [Building single-package documentation locally](https://developer.lsst.io/stack/building-single-package-docs.html) for further details.

## Reference

1. For the parameters of donut image migration, please follow: [How we predict the shapes of donuts in the WFS devices](doc/ref/200313_mask_param.pdf).
