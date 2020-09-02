import os
import numpy as np
import pandas as pd
import lsst.daf.persistence as dafPersist
import lsst.geom
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.base as measBase
import astropy.units as u
from copy import deepcopy
from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask, MagnitudeLimit
from lsst.meas.astrom import AstrometryTask
from lsst.afw.image.utils import defineFilter
from lsst.ts.wep.bsc.DonutDetector import DonutDetector
from lsst.ts.wep.bsc.LocalDatabaseFromImage import LocalDatabaseFromImage
from lsst.ts.wep.cwfs.TemplateUtils import createTemplateImage
from lsst.ts.wep.Utility import abbrevDetectorName


class LocalDatabaseFromRefCat(LocalDatabaseFromImage):

    PRE_TABLE_NAME = "StarTable"

    def insertDataFromRefCat(self, butlerRootPath, settingFileInst,
                            visitList, defocalState,
                            filterType, camera,
                            skiprows=1, keepFile=True,
                            fileOut='foundDonuts.txt'):

        self.expWcs = settingFileInst.getSetting("expWcs")
        centroidTemplateType = settingFileInst.getSetting("centroidTemplateType")
        donutImgSize = settingFileInst.getSetting("donutImgSizeInPixel")
        overlapDistance = settingFileInst.getSetting("minUnblendedDistance")
        maxSensorStars = settingFileInst.getSetting("maxSensorStars")
        doDeblending = settingFileInst.getSetting("doDeblending")
        blendMagDiff = settingFileInst.getSetting("blendMagDiff")
        refCatDir = settingFileInst.getSetting("refCatDir")
        pix2arcsec = settingFileInst.getSetting("pixelToArcsec")
        refButler = dafPersist.Butler(refCatDir)
        self.refObjLoader = LoadIndexedReferenceObjectsTask(butler=refButler)
        skyDf = self.identifyDonuts(butlerRootPath, visitList, filterType,
                                    defocalState, camera, pix2arcsec,
                                    centroidTemplateType, donutImgSize,
                                    overlapDistance, doDeblending,
                                    blendMagDiff, maxSensorStars)
        self.writeSkyFile(skyDf, fileOut)
        self.insertDataByFile(fileOut, filterType, skiprows=1)

        return

    def identifyDonuts(self, butlerRootPath, visitList, filterType,
                       defocalState, camera, pix2arcsec,
                       templateType, donutImgSize, overlapDistance,
                       doDeblending, blendMagDiffLimit, maxSensorStars=None):

        butler = dafPersist.Butler(butlerRootPath)
        sensorList = butler.queryMetadata('postISRCCD', 'detectorName')
        visitOn = visitList[0]
        full_ref_cat_df = None
        for detector in camera.getWfsCcdList():

            raft, sensor = abbrevDetectorName(detector).split('_')

            if sensor not in sensorList:
                continue

            data_id = {'visit': visitOn, 'filter': filterType.toString(),
                       'raftName': raft, 'detectorName': sensor}
            print(data_id)

            # TODO: Rename this to reflect this is postISR not raw image.
            raw = butler.get('postISRCCD', **data_id)

            template = createTemplateImage(defocalState,
                                           detector, pix2arcsec,
                                           templateType, donutImgSize)
            donut_detect = DonutDetector(template)
            # min_overlap_distance = 10. # Use for detecting for ref_cat matching
            donut_df = donut_detect.detectDonuts(raw, overlapDistance)
            source_cat = self.makeSourceCat(donut_df)
            wcs_solver_result, new_wcs = self.solveWCS(raw, source_cat)

            # Update WCS if using exposure WCS for source selection
            if self.expWcs is True:
                camera._wcs.wcsData[detector] = new_wcs

            ref_cat = wcs_solver_result.refCat
            ref_cat_df = ref_cat.asAstropy().to_pandas()
            x_lim, y_lim = list(raw.getDimensions())
            ref_cat_df = ref_cat_df.query(str('centroid_x < %i and centroid_y < %i and ' %
                                              (x_lim - donutImgSize, y_lim - donutImgSize) +
                                              'centroid_x > %i and centroid_y > %i' %
                                              (donutImgSize, donutImgSize)))

            ranked_ref_cat_df = ref_cat_df.sort_values(['phot_g_mean_flux'],
                                                       ascending=False)
            ranked_ref_cat_df = ranked_ref_cat_df.reset_index(drop=True)
            ranked_ref_cat_df = donut_detect.labelUnblended(ranked_ref_cat_df,
                                                            overlapDistance,
                                                            'centroid_x',
                                                            'centroid_y')

            # Convert nJy to mags
            mag_list = (ranked_ref_cat_df['phot_g_mean_flux'].values * u.nJy).to(u.ABmag)
            ranked_ref_cat_df['mag'] = np.array(mag_list)

            if doDeblending is False:
                ranked_ref_cat_df = ranked_ref_cat_df.query('blended == False').reset_index(drop=True)
            else:
                single_blends_df = ranked_ref_cat_df.query('num_blended_neighbors == 1')
                new_df = pd.DataFrame(ranked_ref_cat_df.query('blended == False'))
                keep_sys_index = []
                for keep_sys_on in single_blends_df.index:
                    blend_index = ranked_ref_cat_df.iloc[keep_sys_on]['blended_with'][0]
                    mag_keep = ranked_ref_cat_df.iloc[keep_sys_on]['mag']
                    mag_blend = ranked_ref_cat_df.iloc[blend_index]['mag']
                    blend_mag_diff = mag_blend - mag_keep
                    if np.abs(blend_mag_diff) > blendMagDiffLimit:
                        keep_sys_index.append(keep_sys_on)
                if len(keep_sys_index) > 0:
                    new_df = pd.concat([new_df, ranked_ref_cat_df.iloc[keep_sys_index]])

                ranked_ref_cat_df = new_df.reset_index(drop=True)


            ranked_ref_cat_df['ra'] = np.degrees(ranked_ref_cat_df['coord_ra'])
            ranked_ref_cat_df['dec'] = np.degrees(ranked_ref_cat_df['coord_dec'])
            ranked_ref_cat_df['raft'] = raft
            ranked_ref_cat_df['sensor'] = sensor

            # Get magnitudes
            # Code commented out below will be useful if we get a flux
            # off the image for the objects
            # photo_calib = raw.getPhotoCalib()
            # mag_list = []
            # for mean_flux in ranked_ref_cat_df['phot_g_mean_flux'].values:
            #     mag_list.append(photo_calib.instFluxToMagnitude(mean_flux))

            # bright_mag_limit = 11.1
            # ranked_ref_cat_df = \
            #     ranked_ref_cat_df.query('mag > %f' % bright_mag_limit).reset_index(drop=True)

            # Make coordinate change appropriate to sourProc.dmXY2CamXY
            # FIXME: This is a temporary workaround
            if self.expWcs is False:
                # Transpose because wepcntl. _transImgDmCoorToCamCoor
                dimY, dimX = list(raw.getDimensions())
                pixelCamX = ranked_ref_cat_df['centroid_x'].values
                pixelCamY = dimX - ranked_ref_cat_df['centroid_y'].values
                ranked_ref_cat_df['x_center'] = pixelCamX
                ranked_ref_cat_df['y_center'] = pixelCamY
            else:
                ranked_ref_cat_df['x_center'] = ranked_ref_cat_df['centroid_x']
                ranked_ref_cat_df['y_center'] = ranked_ref_cat_df['centroid_y']

            ra, dec = camera._wcs.raDecFromPixelCoords(
                ranked_ref_cat_df['x_center'].values,
                ranked_ref_cat_df['y_center'].values,
                # pixelCamX, pixelCamY,
                detector, epoch=2000.0, includeDistortion=True
            )
            print(ranked_ref_cat_df['ra'])
            ranked_ref_cat_df['ra'] = ra
            ranked_ref_cat_df['dec'] = dec
            print(ranked_ref_cat_df['ra'])

            if full_ref_cat_df is None:
                full_ref_cat_df = ranked_ref_cat_df.copy(deep=True)
            else:
                full_ref_cat_df = full_ref_cat_df.append(
                    ranked_ref_cat_df)

        full_ref_cat_df = full_ref_cat_df.reset_index(drop=True)

        # TODO: Comment out when not debugging
        full_ref_cat_df.to_csv('image_donut_df.csv')

        return full_ref_cat_df

    def makeSourceCat(self, donutDf):
        """Make a source catalog by reading the position reference stars and distorting the positions
        """

        sourceSchema = afwTable.SourceTable.makeMinimalSchema()
        measBase.SingleFrameMeasurementTask(schema=sourceSchema)

        sourceCat = afwTable.SourceCatalog(sourceSchema)
        sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
        sourceIdKey = sourceSchema["id"].asKey()
        sourceInstFluxKey = sourceSchema["slot_ApFlux_instFlux"].asKey()
        sourceInstFluxErrKey = sourceSchema["slot_ApFlux_instFluxErr"].asKey()

        #populate source catalog with objects from reference catalog
        sourceCat.reserve(len(donutDf['x_center']))

        for i,(_x,_y) in enumerate(zip(donutDf['x_center'], donutDf['y_center'])):
            src = sourceCat.addNew()
            src.set(sourceIdKey, i)

            src.set(sourceCentroidKey, lsst.geom.Point2D(_x, _y))
            src.set(sourceInstFluxKey, 15.)
            src.set(sourceInstFluxErrKey, 15./100)

        return sourceCat

    def solveWCS(self, raw, sourceCat):

        exposure = deepcopy(raw)

        astromConfig = AstrometryTask.ConfigClass()

        magLimit = MagnitudeLimit()
        magLimit.minimum = 11
        magLimit.maximum = 14
        astromConfig.referenceSelector.magLimit = magLimit
        astromConfig.referenceSelector.magLimit.fluxField = "phot_rp_mean_flux"
        astromConfig.matcher.minMatchedPairs = 4
        astromConfig.matcher.maxRotationDeg = 2.99
        astromConfig.matcher.maxOffsetPix = 40
        astromConfig.wcsFitter.order = 3
        astromConfig.wcsFitter.numRejIter = 0
        astromConfig.wcsFitter.maxScatterArcsec = 15

        # this is a bit sleazy (as RHL would say) but I'm just forcing the exposure
        # to have the same name as the one in the Gaia catalog for now
        referenceFilterName = 'phot_rp_mean'
        defineFilter(referenceFilterName, 656.28)
        referenceFilter = afwImage.filter.Filter(referenceFilterName)
        exposure.setFilter(referenceFilter)

        sourceSchema = sourceCat.getSchema()
        solver = AstrometryTask(config=astromConfig, refObjLoader=self.refObjLoader, schema=sourceSchema,)
        results = solver.run(sourceCat=sourceCat, exposure=exposure,)

        return results, exposure.getWcs()