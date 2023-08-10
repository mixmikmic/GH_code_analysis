import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import numpy as np
import cmocean as cmo
from matplotlib import cm

from scripts import analyseWaveReef as wrAnalyse

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

dataTIN = wrAnalyse.analyseWaveReef(folder='output',timestep=22)

dataTIN.regridTINdataSet()

dataTIN.plotdataSet(title='Elevation', data=dataTIN.z, color=cmo.cm.delta,crange=[-2000,2000])

dataTIN.plotdataSet(title='Total erosion/deposition [m]', data=dataTIN.dz, color=cmo.cm.balance,crange=[-20.,20.])

dataTIN.plotdataSet(title='Wave erosion/deposition [m]', data=dataTIN.wavesed, color=cmo.cm.balance,crange=[-5.,5.])

dataTIN.plotdataSet(title='Hillslope erosion/deposition [m]', data=dataTIN.cumhill, color=cmo.cm.balance,
                      crange=[-2.5,2.5])

dataTIN.plotdataSet(title='Wave induced stress', data=dataTIN.wavestress, color=cmo.cm.tempo,crange=[0.,2.],ctr='k')

dataTIN.plotdataSet(title='Reef absent/present', data=dataTIN.depreef, color=cm.Blues,
                      crange=[0,2])

dataTIN.plotdataSet(title='Erosion/Deposition [m]', data=dataTIN.dz, color=cmo.cm.balance,  
                      crange=[-25,25], erange=[100000,250000,540000,640000],
                      depctr=(-2,10),ctr='k',size=(10,10))

dataTIN.plotdataSet(title='Reef position', data=dataTIN.depreef, color=cmo.cm.balance,  
                      crange=[-2,2], erange=[100000,250000,540000,640000],
                      depctr=(1),ctr='k',size=(10,10))

dataTIN.getDepositedVolume(data=dataTIN.dz,time=11000.,erange=[100000,250000,540000,640000])
print '----'
dataTIN.getErodedVolume(data=dataTIN.dz,time=11000.,erange=[100000,250000,540000,640000])

dataTIN.getDepositedVolume(data=dataTIN.cumhill,time=11000.,erange=[100000,250000,540000,640000])
print '----'
dataTIN.getErodedVolume(data=dataTIN.cumhill,time=11000.,erange=[100000,250000,540000,640000])

dataTIN.getDepositedVolume(data=dataTIN.wavesed,time=11000.,erange=[100000,250000,540000,640000])
print '----'
dataTIN.getErodedVolume(data=dataTIN.wavesed,time=11000.,erange=[100000,250000,540000,640000])

r,c = np.where(dataTIN.depreef>0)
erodepreef = np.zeros(dataTIN.dz.shape)
erodepreef[r,c] = dataTIN.dz[r,c]
dataTIN.getDepositedVolume(data=erodepreef,time=11000.,erange=[100000,250000,540000,640000])
print '----'
dataTIN.getErodedVolume(data=erodepreef,time=11000.,erange=[100000,250000,540000,640000])

time = np.linspace(-10000.0, 0.0, num=21)
time,ero,depo = wrAnalyse.temporalGrowth(folder='output',step=22,vtime=time,
                                         smooth=50,vrange=[100000,250000,540000,640000])
wrAnalyse.temporalPlot(time,ero,depo,size=(8,5))



