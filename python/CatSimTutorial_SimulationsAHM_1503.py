import numpy
from lsst.sims.catalogs.definitions import InstanceCatalog

class simpleStarCatalog(InstanceCatalog):
    column_outputs = ['raJ2000', 'decJ2000', 'sedFilename']
    
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

from lsst.sims.utils import ObservationMetaData

myObsMetadata = ObservationMetaData(pointingRA=45.0, pointingDec=-10.0,
                                    boundType='circle', boundLength=0.02)

from lsst.sims.catUtils.baseCatalogModels import StarObj

starTableConnection = StarObj()

myCatalog = simpleStarCatalog(starTableConnection,
                              obs_metadata = myObsMetadata)

myCatalog.write_catalog('test_catalog.txt')

readCatalog = open('test_catalog.txt', 'r').readlines()

get_ipython().system('cat test_catalog.txt')

starTableConnection.show_mapped_columns()

class demoProperMotionCatalog(InstanceCatalog):
    column_outputs = ['raJ2000', 'decJ2000', 'correctedRA', 'correctedDec']

    transformations = {'raJ2000':numpy.degrees,
                       'decJ2000':numpy.degrees,
                       'correctedRA':numpy.degrees,
                       'correctedDec':numpy.degrees}
    
    def get_correctedRA(self):
        dt = self.obs_metadata.mjd.TAI - 51544.0
        ra = self.column_by_name('raJ2000')
        speed = self.column_by_name('properMotionRa')
        return ra + speed*dt
    
    def get_correctedDec(self):
        dt = self.obs_metadata.mjd.TAI - 51544.0
        dec = self.column_by_name('decJ2000')
        speed = self.column_by_name('properMotionDec')
        return dec + speed*dt

myObsMetadata = ObservationMetaData(pointingRA=45.0, pointingDec=-10.0,
                                    boundType='circle', boundLength=0.02,
                                    mjd=57098.0)

myProperMotionCat = demoProperMotionCatalog(starTableConnection,
                                            obs_metadata=myObsMetadata)

myProperMotionCat.write_catalog('proper_motion_example.txt')
get_ipython().system('cat proper_motion_example.txt')

from lsst.sims.catalogs.decorators import compound

class demoProperMotionCatalog2(InstanceCatalog):
    column_outputs = ['raJ2000', 'decJ2000', 'correctedRA', 'correctedDec']

    transformations = {'raJ2000':numpy.degrees,
                       'decJ2000':numpy.degrees,
                       'correctedRA':numpy.degrees,
                       'correctedDec':numpy.degrees}
    
    @compound('correctedRA', 'correctedDec')
    def get_correctedCoords(self):
        dt = self.obs_metadata.mjd.TAI - 51544.0
        ra = self.column_by_name('raJ2000')
        speedRa = self.column_by_name('properMotionRa')
        dec = self.column_by_name('decJ2000')
        speedDec = self.column_by_name('properMotionDec')
        
        #The new columns must be returned as rows of a numpy array
        #in the order that they were specified to the @compound getter
        return numpy.array([ra + speedRa*dt, dec + speedDec*dt])

myObsMetadata = ObservationMetaData(pointingRA=45.0, pointingDec=-10.0,
                                    boundType='circle', boundLength=0.02,
                                    mjd=57098.0)

myProperMotionCat = demoProperMotionCatalog2(starTableConnection,
                                            obs_metadata=myObsMetadata)

myProperMotionCat.write_catalog('proper_motion_example.txt')
get_ipython().system('cat proper_motion_example.txt')

from lsst.sims.catUtils.mixins import AstrometryStars

class astrometricCatalog(InstanceCatalog, AstrometryStars):
    column_outputs = ['raJ2000', 'decJ2000', 'raObserved', 'decObserved']
    transformations = {'raJ2000':numpy.degrees,
                       'decJ2000':numpy.degrees,
                       'raObserved':numpy.degrees,
                       'decObserved':numpy.degrees}

myObsMetadata = ObservationMetaData(pointingRA=45.0, pointingDec=-10.0,
                                    boundType='circle', boundLength=0.02,
                                    mjd=57098.0)

myAstrometricCat = astrometricCatalog(starTableConnection,
                                            obs_metadata=myObsMetadata)

myAstrometricCat.write_catalog('astrometry_example.txt')
get_ipython().system('cat astrometry_example.txt')

class defaultColumnExampleCatalog(InstanceCatalog):
    column_outputs = ['raJ2000', 'decJ2000', 'fudgeFactor1', 'fudgeFactor2']
    
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}
    
    default_columns = [('fudgeFactor1', 1.1, float),
                       ('fudgeFactor2', 'hello', (str,5))]

fudgeCat = defaultColumnExampleCatalog(starTableConnection,
                                       obs_metadata=myObsMetadata)

fudgeCat.write_catalog('default_example.txt')
get_ipython().system('cat default_example.txt')

import os
import eups
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator

opsimPath = os.path.join(eups.productDir('sims_data'),'OpSimData')
opsimDB = os.path.join(opsimPath,'opsimblitz1_1133_sqlite.db')

#you need to provide ObservationMetaDataGenerator with the connection
#string to an OpSim output database.  This is the connection string
#to a test database that comes when you install CatSim.
obs_generator = ObservationMetaDataGenerator(database=opsimDB, driver='sqlite')

obsMetaDataResults = obs_generator.getObservationMetaData(limit=10, fieldRA=(5.0, 8.0))

for obs_metadata in obsMetaDataResults:    
    print obs_metadata.pointingRA

help(ObservationMetaDataGenerator.getObservationMetaData)

from __future__ import with_statement
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catalogs.definitions import InstanceCatalog
from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.catUtils.exampleCatalogDefinitions import         (PhoSimCatalogPoint, PhoSimCatalogSersic2D, PhoSimCatalogZPoint,
         DefaultPhoSimHeaderMap)

from lsst.sims.catUtils.baseCatalogModels import *

obs_metadata_list = obs_generator.getObservationMetaData(obsHistID=10)
obs_metadata = obs_metadata_list[0]

starObjNames = ['msstars', 'bhbstars', 'wdstars', 'rrlystars', 'cepheidstars']

doHeader= True
for starName in starObjNames:
    stars = CatalogDBObject.from_objid(starName)
    star_phoSim=PhoSimCatalogPoint(stars,obs_metadata=obs_metadata) #the class for phoSim input files
                                                                #containing point sources
    if (doHeader):
        star_phoSim.phoSimHeaderMap = DefaultPhoSimHeaderMap
        with open("phoSim_example.txt","w") as fh:
            star_phoSim.write_header(fh)
        doHeader = False

    #below, write_header=False prevents the code from overwriting the header just written
    #write_mode = 'a' allows the code to append the new objects to the output file, rather
    #than overwriting the file for each different class of object.
    star_phoSim.write_catalog("phoSim_example.txt",write_mode='a',write_header=False,chunk_size=20000)

gals = CatalogDBObject.from_objid('galaxyBulge')

#now append a bunch of objects with 2D sersic profiles to our output file
galaxy_phoSim = PhoSimCatalogSersic2D(gals, obs_metadata=obs_metadata)
galaxy_phoSim.write_catalog("phoSim_example.txt",write_mode='a',write_header=False,chunk_size=20000)

gals = CatalogDBObject.from_objid('galaxyDisk')
galaxy_phoSim = PhoSimCatalogSersic2D(gals, obs_metadata=obs_metadata)
galaxy_phoSim.write_catalog("phoSim_example.txt",write_mode='a',write_header=False,chunk_size=20000)

gals = CatalogDBObject.from_objid('galaxyAgn')

#PhoSimCatalogZPoint is the phoSim input class for extragalactic point sources (there will be no parallax
#or proper motion)
galaxy_phoSim = PhoSimCatalogZPoint(gals, obs_metadata=obs_metadata)
galaxy_phoSim.write_catalog("phoSim_example.txt",write_mode='a',write_header=False,chunk_size=20000)

from lsst.sims.GalSimInterface import GalSimStars, DoubleGaussianPSF
from lsst.obs.lsstSim import LsstSimMapper #this is to model the LSST camera

class testGalSimStars(GalSimStars):
    bandpassNames = ['u', 'r']
    camera = LsstSimMapper().camera
    PSF = DoubleGaussianPSF()

#use our ObservationMetaDataGenerator from above
obsMetaDataResults = generator.getObservationMetaData(limit=1, fieldRA = (-5.0, 5.0),
                                                      boundType = 'circle', boundLength = 0.01)

myTestCat = testGalSimStars(starTableConnection, obs_metadata=obsMetaDataResults[0])
myTestCat.write_catalog('galsim_test_catalog.txt')
myTestCat.write_images(nameRoot = 'testImage')

from lsst.sims.GalSimInterface import PSFbase

help(PSFbase)

from lsst.sims.GalSimInterface import ExampleCCDNoise

help(ExampleCCDNoise)

