get_ipython().magic('pylab inline')

import numpy as np
dx = 50.0 # from grid size
xshots = [] 
for i in xrange(100):
    sx = 5000.0 + 2*dx*i
    st0 = sx - 3550 # first station (Stack Array)
    stations = [ st0 + 2*dx*i for i in xrange(70)]
    xshots.append([sx, stations]) 

nsamples = 917
dummy = np.zeros(70)

figure(figsize=(5,5))

for i in xrange(100):
    plot(xshots[i][1], dummy+i, 'g+')  # stations
    plot(xshots[i][0], i, 'r*')  # source

ylabel("shot number")
xlabel("line position (m)")
pylab.show()

def zmutelinear(data, t0i, dkdi):
    ntrc = data.shape[1]    
    zmax = data.shape[0]
    for i in xrange(ntrc):
        imute = t0i + int(round(dkdi*i))
        if imute < 0 or imute > zmax:
            break
        data[0:imute, i] = 0.0
    return data

nshot = 8

get_ipython().magic('cd Fatiando')
import numpy as np
shots = np.load('arrayshots_70s.npy')
print shots.shape
figure(figsize=(5,5))
dt = 0.0043
shots[nshot][:225][:] = 0.0
imshow(shots[nshot], aspect=22, origin='upper', cmap=mpl.cm.gray_r, vmin=-.1, vmax=.1, extent=[0, 60, dt*917, 0.])

figure()
get_ipython().magic('cd Fatiando')
get_ipython().magic('cd ..')
from PIL import Image
img = Image.open('wedge.tif')
img.load()
img = img.convert('I') # gray scale, convert format
data  = np.asarray(img, dtype=np.float32)
data[:][data[:] == 255.] = 2500.0 # overburden
data[:][data[:] == 0. ] = 3500 # the triangle
data[:][data[:] == 146.] = 2300.0 # the first slab bellow the triangle
data[:][data[:] == 36.] = 2700.0 # the last slab
cshape = data.shape
print cshape
imshow(data)

# Get shot coordinate
sx, sz = ((xshots[nshot][0]-xshots[nshot][1][0])/dx, 0.0) # subtract source position from first station
# the source will be re-positioned in a new grid
# get the entire v. model in the spread
sx = sx*dx/2 
i_start = xshots[nshot][1][0]/dx # fst station
vmig = data[:, i_start:i_start+140:2] # 70 stations 35 each side
vshape = vmig.shape
print sx, sz, i_start, vshape

imshow(vmig, aspect=.5)

get_ipython().magic('cd Fatiando')
from fatiando.seismic import migration

dz = 25. # still the same from simulation
area = [0, vshape[1]*dx, 0, vshape[0]*dz]
# avoiding spatial alias same is as last simulation parameters
fc = 11.27

shotmigrated = migration.pre_rtmshot(shots[50], dt, vmig, area, fc, (sx, sz))

imshow(shotmigrated, aspect=0.22, origin='upper', cmap=mpl.cm.gray_r, vmin=-.001, vmax=.001)

shotmigrated[:,:35] = zmutelinear(shotmigrated[:,:35], 140, -14./3.)
shotmigrated[:,35:] = zmutelinear(shotmigrated[:,35:], 0, 14./3.)
figure(figsize=(5,5))
imshow(shotmigrated, aspect=0.32, origin='upper', cmap=mpl.cm.gray_r, vmin=-.001, vmax=.001)

import numpy as np
dx = 50.0 # from grid size
xshots = [] 
for i in xrange(100):
    sx = 5000.0 + 2*dx*i
    st0 = sx - 3550 # first station (Stack Array)
    stations = [ st0 + 2*dx*i for i in xrange(70)]
    xshots.append([sx, stations]) 

nsamples = 917

get_ipython().magic('cd Fatiando')
import numpy as np
import numpy
shots = np.load('arrayshots_70s.npy')
print shots.shape
dt = 0.0043

get_ipython().magic('cd Fatiando')
get_ipython().magic('cd ..')
from PIL import Image
img = Image.open('wedge.tif')
img.load()
img = img.convert('I') # gray scale, convert format
data  = np.asarray(img, dtype=np.float32)
data[:][data[:] == 255.] = 2500.0 # overburden
data[:][data[:] == 0. ] = 3500 # the triangle
data[:][data[:] == 146.] = 2300.0 # the first slab bellow the triangle
data[:][data[:] == 36.] = 2700.0 # the last slab
cshape = data.shape
print cshape

def zmutelinear(data, t0i, dkdi):
    ntrc = data.shape[1]    
    zmax = data.shape[0]
    for i in xrange(ntrc):
        imute = t0i + int(round(dkdi*i))
        if imute < 0 or imute > zmax:
            break
        data[0:imute, i] = 0.0
    return data

get_ipython().magic('cd Fatiando')
from fatiando.seismic import migration

migrated_shots = numpy.zeros((160, 169))  # stacked traces of migrated shots, 169 stations from 100 split-spread shots

dt = 0.0043
for nshot in xrange(100):
    shots[nshot][:225][:] = 0.0
    # Get shot coordinate
    sx, sz = ((xshots[nshot][0]-xshots[nshot][1][0])/dx, 0.0) # subtract source position from first station
    # the source will be re-positioned in a new grid
    # get the entire v. model in the spread
    sx = sx*dx/2 
    # get velocities on trace positions
    i_start = xshots[nshot][1][0]/dx # fst station index
    vmig = data[:, i_start:i_start+140:2] # 70 stations 35 each side
    vshape = vmig.shape
    dz = 25. # still the same from simulation
    area = [0, vshape[1]*dx, 0, vshape[0]*dz]
    # avoiding spatial alias same is as last simulation parameters
    fc = 11.27
    shotmigrated = migration.pre_rtmshot(shots[nshot], dt, vmig, area, fc, (sx, sz))
    shotmigrated[:,:35] = zmutelinear(shotmigrated[:,:35], 140, -14./3.)
    shotmigrated[:,35:] = zmutelinear(shotmigrated[:,35:], 0, 14./3.)
    
    # stack it
    
    migrated_shots[:, nshot:nshot+70] += shotmigrated[:, :]
    
    #figure(figsize=(5,5))
    #imshow(shotmigrated, aspect=0.32, origin='upper', cmap=mpl.cm.gray_r, vmin=-.001, vmax=.001)
    print "\nshot ", nshot, " migrated"    

