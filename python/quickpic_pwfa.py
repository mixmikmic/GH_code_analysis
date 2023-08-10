import quickpic
dirname = 'data'
quickpic.runqpic(rundir=dirname)

from numpy import *
import h5py
import matplotlib.pyplot as pp

f=h5py.File('QEP01-XZ_00000001.h5','r')

dataname='/QEP01-XZ'

xaxis=f['/AXIS/AXIS1'][...]

yaxis=f['/AXIS/AXIS2'][...]

dataset=f[dataname]

data=dataset[...]

x=linspace(xaxis[0],xaxis[1],data.shape[0])

y=linspace(yaxis[0],yaxis[1],data.shape[1])

pp.axis([x.min(), x.max(), y.min(), y.max()])

pp.title('Plasma Density')

pp.pcolor(x,y,data,vmin=-10,vmax=0)

pp.colorbar()

pp.show()



