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

get_ipython().magic('pylab inline')
figure(figsize(10,10))
imshow(data)

get_ipython().magic('cd Fatiando')
from fatiando.seismic import wavefd
# Set the parameters of the finite difference grid
dx = 50. # spacing
dz = 25.
velocity = data
# avoiding spatial alias and less numerical dispersion based on plane waves v=l*f and Alford et al.
# we are using 5
eps = 0.98*1./(5*max(dx, dz)*min(1./(2*dx),1./(2*dz)))
fc = eps*np.min(velocity)/(max(2*dx, 2*dz))  
# fc for a Ricker wavelet to garante the maximum fc above 
fricker = (2./3.)*fc/np.sqrt(np.pi)
sim = wavefd.Scalar(velocity, (dx, dz))
sim.add_point_source((100, 3), wavefd.Ricker(1., fricker, delay=1./fricker))
print fc, np.min(velocity)/(fc*max(dx,dx)), eps, sim.dt
print velocity.shape

wf = wavefd.Ricker(1., fricker, delay=1./fricker)
t = np.linspace(0, 300*sim.dt, 300)
figure(figsize(5,2))
plot(t, wf(t))
print "time shift (s)", (t[wf(t).argmax()]), " samples to shitft ", wf(t).argmax()

sim.run(900)

sim.dt*sim.simsize

sim.explore()

#sim.animate(every=5, cutoff=1, blit=True, fps=20, dpi=50, embed=True, writer='ffmpeg')

# that's default image size for this interactive session
def pstep(iter):     
    figure(figsize=(7, 7))     
    background = (velocity-3500)*10**(-6.0)         
    imshow((-background+sim[iter])[::-1], cmap=mpl.cm.gray_r, vmin=-.02, vmax=.02, origin='lower')  
pstep(104)
pstep(504)
pstep(650)
pstep(750)
pstep(800)

#sim[:].shape

get_ipython().magic('cd Fatiando')

# if I want 400 I just could have those but I just want to look at 100 traces
from fatiando.vis import mpl

sx = 50
shot = np.zeros((900, 100))
shot[:, :] = sim[:, 3, sx:sx+100] # TODO: fix boundary/padding to avoid loosing 2 indexes due 4order space
# needed indexes for calculation

figure(figsize=(3,5))
#mpl.seismic_wiggle(shot, dt=sim.dt, scale=350.)
mpl.seismic_image(shot, dt=sim.dt, aspect='auto', vmin=-0.01, vmax=0.01)

get_ipython().magic('pylab inline')

shots_x = [] # index x shot coordinates per shot
stations_x = [] # index x station coordinates per shot

for i in xrange(100):
    shot_x = 100 + 2*i
    station_0 = shot_x -1 -2*35 # first station (stack array)
    array_stations = [ station_0 + 2*i for i in xrange(70) ] # dx_shot = dx_station
    shots_x.append(shot_x) 
    stations_x.append(array_stations)

dummy = np.zeros(70)
figure(figsize=(10,10))

for i in xrange(100):
    plot(stations_x[i], dummy+i, 'g+')  # stations
    plot(shots_x[i], i, 'r*')  # source

ylabel("shot number")
xlabel("line position * dx (m)")
pylab.show()

from fatiando.seismic import wavefd
# Set the parameters of the finite difference grid dx = 50. spacing 
dx = 50
dz = 25. 
velocity = data 
# avoiding spatial alias and less numerical dispersion based on plane waves v=l*f and Alford et al. 
eps = 0.98*1./(5*max(dx, dz)*min(1./(2*dx),1./(2*dz))) 
fc = eps*np.min(velocity)/(max(2*dx, 2*dz))  
# for Gauss source the spectrum is really controled by fc (the highest frequency)
# But for Ricker no!! fc is the central frequency! So fc/2 gives 
# approximately fc as the highest frequency for a Ricker wavelet (take a look at ffts to clarify)
fc = fc/2. # for use with Ricker wavelet

def rum_sim_shot(source_array, iter=900):
    source_x, array_stations = source_array
    sim = wavefd.Scalar(velocity, (dx, dz))     
    sim.add_point_source((source_x, 3), wavefd.Ricker(1., fc, 1./fc))   
    sim.run(iter)
    # getting seismograms at zindex=3 or 3*25= 75 meters :-( TODO:
    return sim[:, 3, array_stations] # TODO: same as above padding/boundary etc..

print 'shot position: ', shots_x[0]
print 'stations: ', stations_x[0]

get_ipython().magic('timeit')
shot0 = None
shot0 = rum_sim_shot((shots_x[0], stations_x[0]))

from fatiando.vis.mpl import seismic_image

figure(figsize=(8,10))
seismic_image(shot0, dt=sim.dt, vmin=-0.01, vmax=0.01)

get_ipython().magic('timeit')
from multiprocessing import Pool
print('Simulating...')
pool = Pool()
shots = pool.map(rum_sim_shot, zip(shots_x, stations_x))
pool.close()
print('Finished')

figure(figsize=(8,10))
seismic_image(shots[75], dt=sim.dt, vmin=-0.01, vmax=0.01)

print shots[0].shape

zeroffset = np.array([shots[i] for i in range(100)])

print zeroffset.shape

figure(figsize=(8,10))
seismic_image(zeroffset[:, :, 35].transpose(), dt=sim.dt, vmin=-0.01, vmax=0.01)

arrayshots = np.zeros((100, 900, 70))
for i in range(100):
    arrayshots[i] = shots[i]

np.save('arrayshots_70s_900_07_12_2015', arrayshots)



