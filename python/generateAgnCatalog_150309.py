from lsst.sims.utils import ObservationMetaData

print help(ObservationMetaData)

#create a circular field of view centered on RA = 25 degrees, Dec = 30 degrees
#with a radius of 0.1 degrees
circleObsMetadata = ObservationMetaData(pointingRA=25.0, pointingDec=30.0,
                                        boundType='circle', boundLength=0.1)

#create a square field of view centered on RA = 25 degrees, Dec = 30 degrees
#with a side-length of 0.2 degrees (in this case boundLength is half the length of a side)
squareObsMetadata = ObservationMetaData(pointingRA=25.0, pointingDec=30.0,
                                        boundType='box', boundLength=0.1)

#create a rectangular field of view centered on RA = 25 degrees, Dec = 30 degrees
#with an RA side length of 0.2 degrees and a Dec side length of 0.1 degrees
rectangleObsMetadata = ObservationMetaData(pointingRA=25.0, pointingDec=30.0,
                                           boundType='box', boundLength=(0.1, 0.05))

from lsst.sims.catUtils.baseCatalogModels import GalaxyTileObj

galaxyDB = GalaxyTileObj()

galaxyDB.show_mapped_columns()

import numpy
from lsst.sims.catalogs.measures.instance import InstanceCatalog

class basicAgnCatalog(InstanceCatalog):
    
    #list defining the columns we want output to our catalog
    column_outputs = ['raJ2000', 'decJ2000',
                     'sedFilenameAgn', 'magNormAgn']
    
    #dict performing any unit transformations upon output
    #(note that angles are stored in radians by default;
    #must explicitly be transformed to degrees on output)
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}
    
    #This lists all of the columns whose values cannot be Nan, None, NULL etc.
    #In this case, we are filtering out all of the galaxies without AGN by
    #ignoring galaxies that do not have an SED assigned for an AGN
    cannot_be_null = ['sedFilenameAgn']

agnCircle = basicAgnCatalog(galaxyDB, obs_metadata=circleObsMetadata)
agnCircle.write_catalog('agn_circle.txt')

agnSquare = basicAgnCatalog(galaxyDB, obs_metadata=squareObsMetadata)
agnSquare.write_catalog('agn_square.txt')

agnRectangle = basicAgnCatalog(galaxyDB, obs_metadata=rectangleObsMetadata)
agnRectangle.write_catalog('agn_rectangle.txt')

import matplotlib
dtype = numpy.dtype([('raJ2000',float),('decJ2000',float),('sedFilenameAgn',(str,40)),('magNormAgn',float)])

circleData = numpy.loadtxt('agn_circle.txt', delimiter=',', dtype=dtype)
matplotlib.pyplot.scatter(circleData['raJ2000'],circleData['decJ2000'])
matplotlib.pyplot.show()

squareData = numpy.loadtxt('agn_square.txt', delimiter=',', dtype=dtype)
matplotlib.pyplot.scatter(squareData['raJ2000'], squareData['decJ2000'])
matplotlib.pyplot.show()

rectangleData = numpy.loadtxt('agn_rectangle.txt', delimiter=',', dtype=dtype)
matplotlib.pyplot.scatter(rectangleData['raJ2000'], rectangleData['decJ2000'])
matplotlib.pyplot.show()

class myCartoonCatalog(InstanceCatalog):
    column_outputs = ['raJ2000', 'decJ2000', 'lsst_u', 'shifted_u_magnitude']

    def get_shifted_u_magnitude(self):
        u = self.column_by_name('lsst_u') #this gets the lsst_u value for every object in the
                                          #catalog as a numpy array
        return u + 4.0

cartoonAgn = myCartoonCatalog(galaxyDB, obs_metadata=circleObsMetadata)
cartoonAgn.write_catalog('cartoon_agn.txt')

#re-import these packages in case we need to restart the kernel
import numpy
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catUtils.baseCatalogModels import GalaxyTileObj
from lsst.sims.catalogs.measures.instance import InstanceCatalog

from lsst.sims.catUtils.mixins import PhotometryGalaxies, VariabilityGalaxies

class variableAgnCatalog(InstanceCatalog, PhotometryGalaxies, VariabilityGalaxies):
    
    cannot_be_null = ['uAgn'] #again, we only want AGN
    
    #note that we are using [u,g,r,i,z,y]Agn as the baseline magnitudes
    #rather than lsst_[u,g,r,i,z,y].  The VariabilityGalaxies mixin operates
    #by calculating the baseline magnitude from the object's SED, rather
    #than reading in the value from the database.  These should give the
    #same answer, but they do not have to (if, for example, we changed 
    #reddening models between now and when the database was created).
    #[u,g,r,i,z,y]Agn are provided by the PhotometryGalaxies mixin defined
    #in sims_photUtils/../Photometry.py, which is why we have to inherit from
    #that class as well
    column_outputs = ['galid', 'raJ2000', 'decJ2000',
                      'uAgn', 'rAgn', 'zAgn',
                      'delta_uAgn', 'delta_rAgn', 'delta_zAgn']
    
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

variableObsMetadata = ObservationMetaData(pointingRA=25.0, pointingDec=30.0,
                                          boundType='circle', boundLength=0.05,
                                          mjd=57086)

galaxyDB = GalaxyTileObj()
variableAgn = variableAgnCatalog(galaxyDB, obs_metadata=variableObsMetadata)
variableAgn.write_catalog('variable_agn.txt')

class baselineAgnCatalog(InstanceCatalog, PhotometryGalaxies):
    
    cannot_be_null = ['uAgn'] #again, we only want AGN
    
    #note that we are using [u,g,r,i,z,y]Agn as the baseline magnitudes
    #rather than lsst_[u,g,r,i,z,y].  The VariabilityGalaxies mixin operates
    #by calculating the baseline magnitude from the object's SED, rather
    #than reading in the value from the database.  These should give the
    #same answer, but they do not have to (if, for example, we changed 
    #reddening models between now and when the database was created).
    #[u,g,r,i,z,y]Agn are provided by the PhotometryGalaxies mixin defined
    #in sims_photUtils/../Photometry.py, which is why we have to inherit from
    #that class as well
    column_outputs = ['galid', 'raJ2000', 'decJ2000',
                      'uAgn', 'rAgn', 'zAgn']
    
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

baselineAgn = baselineAgnCatalog(galaxyDB, obs_metadata=variableObsMetadata)
baselineAgn.write_catalog('baseline_agn.txt')

variableDtype = numpy.dtype([('galid',(str,100)), ('ra', numpy.float), ('dec', numpy.float),
                             ('u', numpy.float), ('r', numpy.float), ('z', numpy.float),
                             ('delta_u', numpy.float), ('delta_r',numpy.float),
                             ('delta_z',numpy.float)])

variableData = numpy.genfromtxt('variable_agn.txt', dtype=variableDtype, delimiter=', ')

baselineDtype = numpy.dtype([('galid',(str,100)), ('ra', numpy.float), ('dec', numpy.float),
                             ('u', numpy.float), ('r', numpy.float), ('z', numpy.float)])

baselineData = numpy.genfromtxt('baseline_agn.txt', dtype=baselineDtype, delimiter=', ')

maxError = -1.0
for base in baselineData:
    var = variableData[numpy.where(variableData['galid']==base['galid'])[0][0]]
    error = numpy.abs(var['u']-base['u']-var['delta_u'])

    if error>maxError:
        maxError=error

    error = numpy.abs(var['r']-base['r']-var['delta_r'])
    if error>maxError:
        maxError=error
    
    error = numpy.abs(var['z']-base['z']-var['delta_z'])
    if error>maxError:
        maxError=error

print 'maxError: ',maxError

#re-import these packages in case we need to restart the kernel
import numpy
from lsst.sims.utils import ObservationMetaData
from lsst.sims.catUtils.baseCatalogModels import GalaxyTileObj
from lsst.sims.catalogs.measures.instance import InstanceCatalog

from lsst.sims.catUtils.mixins import PhotometryGalaxies, VariabilityGalaxies

class variableAgnCatalogCheat(InstanceCatalog, PhotometryGalaxies, VariabilityGalaxies):
    
    cannot_be_null = ['uAgn'] #again, we only want AGN
    
    #note that we are using [u,g,r,i,z,y]Agn as the baseline magnitudes
    #rather than lsst_[u,g,r,i,z,y].  The VariabilityGalaxies mixin operates
    #by calculating the baseline magnitude from the object's SED, rather
    #than reading in the value from the database.  These should give the
    #same answer, but they do not have to (if, for example, we changed 
    #reddening models between now and when the database was created).
    #[u,g,r,i,z,y]Agn are provided by the PhotometryGalaxies mixin defined
    #in sims_photUtils/../Photometry.py, which is why we have to inherit from
    #that class as well
    column_outputs = ['galid', 'raJ2000', 'decJ2000',
                      'uAgn', 'rAgn', 'zAgn',
                      'delta_uAgn', 'delta_rAgn', 'delta_zAgn']
    
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}
    
    def get_sedFilenameBulge(self):
        ra = self.column_by_name('raJ2000') #to figure out how many objects are in the catalog
        nameList = []
        for rr in ra:
            nameList.append('None')
        return numpy.array(nameList)
    
    def get_sedFilenameDisk(self):
        return self.column_by_name('sedFilenameBulge')

galaxyDB = GalaxyTileObj()

variableObsMetadata = ObservationMetaData(pointingRA=25.0, pointingDec=30.0,
                                          boundType='circle', boundLength=0.05,
                                          mjd=57086)

variableAgn = variableAgnCatalogCheat(galaxyDB, obs_metadata=variableObsMetadata)
variableAgn.write_catalog('variable_agn_cheat.txt')

from lsst.sims.catUtils.utils import ObservationMetaDataGenerator

help(ObservationMetaDataGenerator.__init__)

help(ObservationMetaDataGenerator.getObservationMetaData)

import eups
import os
opsimdb = os.path.join(eups.productDir('sims_data'),'OpSimData','opsimblitz1_1133_sqlite.db')

gen = ObservationMetaDataGenerator(database=opsimdb, driver='sqlite')

obsMDresults = gen.getObservationMetaData(boundType='circle', boundLength=0.05,
                                          fieldRA=(20.0,30.0), fieldDec=(-65.0, -55.0), airmass = (1.4, 2.1))

for o in obsMDresults:
    print o.pointingRA, o.pointingDec,     o.phoSimMetaData['airmass'][0], o.mjd.TAI

variableAgn = variableAgnCatalogCheat(galaxyDB, obs_metadata=obsMDresults[0])
variableAgn.write_catalog('variable_agn_real_obs_metadata.txt')



