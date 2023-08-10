from lsst.obs.lsstSim import LsstSimMapper
camera = LsstSimMapper().camera

import os
import eups
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator

#the code below just points to an OpSim output database that
#is carried around with the Simulations stack for testing purposes
opSimDbName = 'opsimblitz1_1133_sqlite.db'
fullName = os.path.join(eups.productDir('sims_data'),'OpSimData',opSimDbName)

obsMD_generator = ObservationMetaDataGenerator(database=fullName, driver='sqlite')

boundLength=3.0 #the radius of our field of view in degrees
obs_metadata = obsMD_generator.getObservationMetaData(fieldRA=(24.0,100.0),
                                                      limit=1, boundLength=boundLength)
print obs_metadata[0].pointingRA, obs_metadata[0].rotSkyPos

import numpy
epoch = 2000.0
nsamples = 10
numpy.random.seed(42)
radius = boundLength*numpy.random.sample(nsamples)
theta = 2.0*numpy.pi*numpy.random.sample(nsamples)

raRaw = obs_metadata[0].pointingRA + radius*numpy.cos(theta)
decRaw = obs_metadata[0].pointingDec + radius*numpy.sin(theta)

from lsst.sims.coordUtils import chipNameFromRaDec

chipNames = chipNameFromRaDec(ra=raRaw, dec=decRaw,
                              camera=camera, epoch=epoch,
                              obs_metadata=obs_metadata[0])

print chipNames

from lsst.sims.coordUtils import pixelCoordsFromRaDec

pixelCoords = pixelCoordsFromRaDec(ra=raRaw, dec=decRaw,
                                   camera=camera, epoch=epoch,
                                   obs_metadata=obs_metadata[0])

for name, x, y in zip(chipNames, pixelCoords[0], pixelCoords[1]):
    print name, x, y

from lsst.sims.utils import pupilCoordsFromRaDec

help(pupilCoordsFromRaDec)

xPup, yPup = pupilCoordsFromRaDec(raRaw, decRaw,
                                  obs_metadata=obs_metadata[0], epoch=epoch)

for x,y in zip(xPup, yPup):
    print x, y

from lsst.sims.utils import observedFromICRS

help(observedFromICRS)

from lsst.sims.utils import appGeoFromICRS

help(appGeoFromICRS)

from lsst.sims.utils import observedFromAppGeo

help(observedFromAppGeo)

from lsst.sims.catUtils.mixins import AstrometryBase
for methodName in dir(AstrometryBase):
    if 'get_' in methodName:
        print methodName

from lsst.sims.catUtils.mixins import AstrometryStars
for methodName in dir(AstrometryStars):
    if 'get_' in methodName and methodName not in dir(AstrometryBase):
        print methodName

from lsst.sims.catUtils.mixins import AstrometryGalaxies
for methodName in dir(AstrometryGalaxies):
    if 'get_' in methodName and methodName not in dir(AstrometryBase):
        print methodName

from lsst.sims.catUtils.mixins import CameraCoords
for methodName in dir(CameraCoords):
    if 'get_' in methodName and methodName not in dir(AstrometryBase):
        print methodName

from lsst.sims.catalogs.measures.instance import InstanceCatalog
from lsst.sims.catUtils.mixins import AstrometryStars

class chipNameCatalog(InstanceCatalog, AstrometryStars, CameraCoords):
    column_outputs = ['raJ2000', 'decJ2000', 'raObserved', 'decObserved', 
                      'chipName', 'xPix', 'yPix']

    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees,
                       'raObserved':numpy.degrees, 'decObserved':numpy.degrees}
    
    camera = LsstSimMapper().camera

from lsst.sims.catUtils.baseCatalogModels import WdStarObj

#define a smaller ObservationMetaData so that we don't create an over large catalog
obs_metadata = obsMD_generator.getObservationMetaData(fieldRA=(24.0, 100.0),
                                                      limit=1, boundLength=0.5)

#again, use the white dwarf database table so that we don't get too many objects
#in this small example
starDB = WdStarObj()

testCat = chipNameCatalog(starDB, obs_metadata=obs_metadata[0])

catName = 'test_cat.txt'

if os.path.exists(catName):
    os.unlink(catName)
    
testCat.write_catalog(catName)

get_ipython().system('cat test_cat.txt')

