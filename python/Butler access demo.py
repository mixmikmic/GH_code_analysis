get_ipython().magic('matplotlib inline')
import os
from matplotlib import pylab as plt
plt.rcParams['figure.figsize'] = (13,8)
import numpy

os.chdir('/global/cscratch1/sd/descdm/DC1/full_focalplane_undithered')

import lsst.afw.display as afw_display
import lsst.daf.persistence as daf_persistence
import lsst.afw.geom as afw_geom

calexpId = {'visit':1919421, 'filter':'r', 'raft':'2,2', 'sensor':'1,1'}
coaddId = {'patch':'5,5', 'tract':0, 'filter':'r'}

butler = daf_persistence.Butler('.')

calexp = butler.get('calexp', dataId=calexpId)
calexp_src = butler.get('src', dataId=calexpId)
coadd = butler.get('deepCoadd', dataId=coaddId)
coadd_src = butler.get('deepCoadd_meas', dataId=coaddId)

calexp_display = afw_display.getDisplay(frame=1)
coadd_display = afw_display.getDisplay(frame=2)
calexp_display.mtv(calexp)
coadd_display.mtv(coadd)

for field in calexp_src.schema:
    name = field.getField().getName()
    if 'flag' not in name:
        print(name+':', field.getField().getDoc())

for field in calexp_src.schema:
    name = field.getField().getName()
    if 'flag' in name:
        print(name+':', field.getField().getDoc())

with calexp_display.Buffering():
    for source in calexp_src:
        calexp_display.dot('o', source.getX(), source.getY())
with coadd_display.Buffering():
    for source in coadd_src:
        coadd_display.dot('x', source.getX(), source.getY())

modelFlux = coadd_src.getModelFlux()
modelFlux_flag = coadd_src.get('modelfit_CModel_flag')
psfFlux = coadd_src.getPsfFlux()
psfFlux_flag = coadd_src.get('base_PsfFlux_flag')
coadd_calib = coadd.getCalib()
modelMags = []
psfMags = []
for mflux, pflux, mflag, pflag in zip(modelFlux, psfFlux, modelFlux_flag, psfFlux_flag):
    if mflux > 0 and pflux > 0 and not (mflag or pflag):
        modelMags.append(coadd_calib.getMagnitude(mflux))
        psfMags.append(coadd_calib.getMagnitude(pflux))
modelMags = numpy.array(modelMags)
psfMags = numpy.array(psfMags)

plt.scatter(psfMags, psfMags-modelMags, alpha=.1)
plt.xlim(15., 30)
plt.ylim(-0.1, 3)
plt.show()



