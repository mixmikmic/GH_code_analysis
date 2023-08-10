get_ipython().magic('pylab inline')

get_ipython().magic('cd Fatiando')
from fatiando.vis import mpl

import numpy as np
cmps_stacked = np.load('cmp_stacked_01_12_2015.npy')
print cmps_stacked.shape
figure(figsize=(12,6))
sim_dt=0.00433034793813
mpl.seismic_image(cmps_stacked, dt=1, vmin=-0.1, vmax=0.1, aspect='auto')

# 3225 to 16575 by 50
# 50 dx becomes aproximatelly 
icmps = [ 64+i for i in xrange(268)]

get_ipython().magic('cd Fatiando')
get_ipython().magic('cd ..')
from PIL import Image
import numpy as np
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
imshow(data, extent=[0, 400*50, 160*25, 0], aspect='auto', origin='upper')

vmig = data[:, icmps]
vshape = vmig.shape
print vshape
figure(figsize=(5,5))
imshow(vmig, origin='upper')

dx = 50. # spacing between cmps
dz = 25. # still the same from simulation
area = [0, vshape[1]*dx, 0, vshape[0]*dz]
vmig = vmig/2. # half velocity due zero-offset section aproximation
dt = sim_dt
maxit = 846
snapshots = 1 # every 3 iterations plots one

get_ipython().magic('cd Fatiando')
from fatiando.seismic import migration
rtmsimulation = migration.rt_scalar(vmig, area, dt, maxit, cmps_stacked, snapshot=1, padding=50)
import numpy as np
movie = np.zeros(((maxit/snapshots)+2, vshape[0], vshape[1]))
i = 0
for t, u  in rtmsimulation:
    movie[i] = u
    sys.stdout.write("\rprogressing .. %.1f%% time %.3f"%(100.0*float(t)/maxit, (dt*t)))
    sys.stdout.flush()
    i += 1
sys.stdout.write("\niteration %d"%(i))

# there is no need for that much of depth
get_ipython().magic('cd Fatiando')
from fatiando.vis import mpl

def pstep(iter):
    figure(figsize=(8,3))
    tmv = movie[iter]
    mpl.seismic_image(tmv, dt=25, vmin=-.5, vmax=.6, aspect='auto') 
    #imshow(tmv, cmap=mpl.cm.gray_r, vmin=-.1, vmax=.1, extent=area, origin='lower')

pstep(300)
pstep(500)
pstep(600)
pstep(700)
pstep(800)
pstep(845)





