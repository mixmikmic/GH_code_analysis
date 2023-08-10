get_ipython().magic('load_ext memory_profiler')
get_ipython().magic('load_ext snakeviz')
get_ipython().magic('load_ext cython')
import holoviews as hv
hv.extension('bokeh','matplotlib')
from IPython.core import debugger
ist = debugger.set_trace

import numpy as np
pos = np.loadtxt('data/positions.dat')
box = np.loadtxt('data/box.dat')

#In order to make this test fair and 'pure python', 
#we need to convert the numpy position and box size arrays to python lists.
x,y,z = map(list,pos.T)
lx,ly,lz = box

print('Read {:d} positions.'.format(pos.shape[0]))
print('x min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[0],pos.max(0)[0]))
print('y min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[1],pos.max(0)[1]))
print('z min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[2],pos.max(0)[2]))

from math import cos,pi

def potentialEnergyFunk(r,width=1.0,height=10.0):
    '''
    Calculates the (soft) potential energy between two atoms
    
    Parameters
    ----------
    r: float
        separation distance between two atoms
    height: float
        breadth of the potential i.e. where the potential goes to zero
    width: float
        strength/height of the potential
    '''
    if r<width:
        return 0.5 * height * (1 + cos(pi*r/width))
    else:
        return 0

get_ipython().run_cell_magic('opts', 'Curve [width=600,show_grid=True,height=350]', "\ndr = 0.05          # spacing of r points\nrmax = 10.0        # maximum r value\npts = int(rmax/dr) # number of r points\nr = [dr*i for i in range(pts)]\n\ndef plotFunk(width,height,label='dynamic'):\n    \n    # Need to wrap potentialEnergyFunk for map call below to behave\n    funk = lambda r: potentialEnergyFunk(r,width,height)\n    \n    U = list(map(funk,r))\n    \n    return hv.Curve((r,U),kdims=['Separation Distance'],vdims=['Potential Energy'],label=label)\n    \ndmap = hv.DynamicMap(plotFunk,kdims=['width','height'])\ndmap = dmap.redim.range(width=((1.0,10.0)),height=((1.0,5.0)))\ndmap*plotFunk(10.0,5.0,label='width: 10., height: 5.')*plotFunk(1.0,1.0,label='width: 1., height: 1.')")

from math import sqrt

def calcTotalEnergy(x,y,z,lx,ly,lz):
    '''
    Parameters
    ----------
    x,y,z: lists of floats
        1-D lists of cartesian coordinate positions
    
    lx,ly,lz: floats
        simulation box dimensions
    '''
    
    #sanity check
    assert len(x) == len(y) == len(z)
    
    # store half box lengths for minimum image convention
    lx2 = lx/2.0
    ly2 = ly/2.0
    lz2 = lz/2.0
    
    U = 0
    
    #The next two loops simply loop over every element in the x, y, and z arrays
    for i,(x1,y1,z1) in enumerate(zip(x,y,z)):
        for j,(x2,y2,z2) in enumerate(zip(x,y,z)):
            
            # We only need to consider each pair once
            if i>=j:
                continue
           
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            dz = abs(z1 - z2)
                
            # The next few lines take care of applying the minimum image convention
            # This is neccesary because many/most molecular simulations use periodic boundary conditions
            if dx > lx2:
                dx -= lx
            if dy > ly2:
                dy -= ly
            if dz > lz2:
                dz -= lz
                
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            
            U += potentialEnergyFunk(dist)
    
    return U
                

get_ipython().run_cell_magic('prun', '-D prof/python.prof', 'energy = calcTotalEnergy(x,y,z,lx,ly,lz)')

memprof = get_ipython().magic('memit -o calcTotalEnergy(x,y,z,lx,ly,lz)')

usage = memprof.mem_usage[0]
incr = memprof.mem_usage[0] - memprof.baseline
with open('prof/python.memprof','w') as f:
    f.write('{}\n{}\n'.format(usage,incr))

with open('energy/python.dat','w') as f:
    f.write('{}\n'.format(energy))





