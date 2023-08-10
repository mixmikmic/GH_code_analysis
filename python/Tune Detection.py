import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import lsst.daf.persistence        as dafPersist
import lsst.afw.display            as afwDisplay
import lsst.afw.table              as afwTable

from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.pipe.tasks.calibrate         import CalibrateTask
from lsst.meas.algorithms.detection    import SourceDetectionTask
from lsst.meas.deblender               import SourceDeblendTask
from lsst.meas.base                    import SingleFrameMeasurementTask

disp  = afwDisplay.Display(1)

schema = afwTable.SourceTable.makeMinimalSchema()
algMetadata = dafBase.PropertyList()

config = CharacterizeImageTask.ConfigClass()
config.psfIterations = 1
charImageTask =         CharacterizeImageTask(None, config=config)

config = SourceDetectionTask.ConfigClass()
if True:
    config.thresholdValue = 30       # detection threshold in units of thresholdType
    if True:
        print "SourceDetectionTask.thresholdType: %s" % (
            SourceDetectionTask.ConfigClass.thresholdType.__doc__)
    config.thresholdType = "stdev"   # units for thresholdValue
if False:                    
    config.doTempLocalBackground = True  # Use local-background during detection step
sourceDetectionTask =   SourceDetectionTask(schema=schema, config=config)

sourceDeblendTask =     SourceDeblendTask(schema=schema)

config = SingleFrameMeasurementTask.ConfigClass()
sourceMeasurementTask = SingleFrameMeasurementTask(schema=schema, config=config,
                                                   algMetadata=algMetadata)

butler = dafPersist.Butler("/Volumes/RHLData/hsc-v13_0")

if False:
    dataId = dict(tract=9348, patch='7,6', filter='HSC-I')
    exposure = butler.get('deepCoadd_calexp',dataId)
else:
    dataId = dict(visit=29352, ccd=50)
    exposure = butler.get('calexp', dataId)
    
tab = afwTable.SourceTable.make(schema)

result = charImageTask.characterize(exposure)

result = sourceDetectionTask.run(tab, exposure)
sources = result.sources

sourceDeblendTask.run(exposure, sources)

sourceMeasurementTask.run(exposure, sources)

if False:
    sources.writeFits("outputTable.fits")
    exposure.writeFits("example1-out.fits")

sources = sources.copy(True)

good = np.logical_and.reduce([sources.get('base_PixelFlags_flag_saturatedCenter') == 0,
                              sources.get("deblend_nChild") == 0,
                              ])

if True:
    disp.mtv(exposure)
else:
    disp.erase()
    
disp.pan(1163, 533); disp.zoom(1)

with disp.Buffering():
    for s in sources[good]:
        disp.dot('+', *s.getCentroid(), ctype=afwDisplay.RED)

