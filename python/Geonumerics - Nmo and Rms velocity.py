def vrms(vi, ds):
    """
    Calculate RMS velocity from:

    * vi : ndarray
        interval velocity
    * ds : ndarray
        layer size
    
    return Rms velocity array
    """
    twt = 2*(ds/vi) # two way time
    vi2_twt = (vi**2)*twt
    return np.sqrt(vi2_twt.cumsum()/twt.cumsum())    

get_ipython().magic('pylab inline')

from matplotlib import pylab as pyplot
import numpy as np
# vrms calculation from intervalar velocity and
# layer size (Parana basin case 1300 meters)
vi = np.array([ 1200.0, 4500.0, 2900.0, 3200, 3300 ])
ds = np.array([ 300.0, 700.0, 600.0, 1200, 1000 ])    
v_rms = vrms(vi, ds)
twt = 2*(ds/vi) # two way time
fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(twt.cumsum(), vi, 'g-^', label=r'$\mathbf{V_{interval}}$')
ax.plot(twt.cumsum(), v_rms, 'b-^', label=r'$\mathbf{V_{rms}}$')
ax.set_xlabel("Twt (s)")
ax.set_ylabel("Velocity (m/s)")
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(twt.cumsum())
ax2.set_xticklabels(ds.cumsum())
ax2.xaxis.grid()
ax2.set_xlabel("Depth (m)")
ax.legend()

get_ipython().magic('pylab inline')

dx = 50.0 # from grid size
startx = 100*dx # 5000
htraces = [] # trace header shot x coordinate and station x coordinate
for i in xrange(100): # each shot
    sx = startx + 100*i  # 2*dx = 100 meters 
    for j in xrange(70): # each station
        htraces.append((sx, sx-3550+100*j)) # source and receiver coordinate for this trace

figure(figsize=(7,5))
for i in xrange(100):
    xstations = zip(*htraces[70*i:(70*(i+1))])[1]
    xsource = zip(*htraces[70*i:(70*(i+1))])[0][0]
    plot(xstations, [ i for j in range(70)], 'g+')  # stations
    plot(xsource, i, 'r*')  # source
ylabel("shot number")
xlabel("line position (m)")
pylab.show()

htraces[0] # first of 7000 traces

htraces = [ (htraces[i][0], htraces[i][1], (htraces[i][0]+htraces[i][1])/2.,  
               abs(htraces[i][0]-htraces[i][1])) for i in range(7000) ]

htraces[0] # first of 7000 traces

# (xsource, xstation, cmp, offset)
htraces[0:5] # first 25 of 7000 traces

get_ipython().magic('cd Fatiando')
get_ipython().magic('cd ..')

import numpy as np
shots = np.load('arrayshots_70s_900_07_12_2015.npy')
print shots.shape

time_shift = shots[0, :, 35].argmax()
dtraces = np.zeros((7000, shots.shape[1]-time_shift))
p=0
for i in xrange(100):
    for j in xrange(70):
        dtraces[p, :] = shots[i, :shots.shape[1]-time_shift, j]
        p += 1

import pandas

tuple_traces = [((dtraces[i, :]), htraces[i][0], htraces[i][1], htraces[i][2], htraces[i][3]) for i in range(7000)]
df_traces = pandas.DataFrame(tuple_traces, columns=['traces', 'x_source', 'x_station', 'cmp', 'abs_offset'], 
                      dtype=np.float64)

figure(figsize(10,2))
df_traces.cmp.hist(bins=df_traces.cmp.nunique())

figure(figsize(10,2))
df_traces.abs_offset.hist(bins=df_traces.abs_offset.nunique())

print df_traces.cmp.unique()#[190:200]

CMPi = 8625
cmpn = df_traces[ df_traces['cmp'] == CMPi  ].traces.values
cmpnffsets =  df_traces[ df_traces['cmp'] == CMPi  ].abs_offset.values
cmpn = np.array([ cmpn[i] for i in range(size(cmpn)) ])
cmpn = cmpn.transpose()
print cmpn.shape

get_ipython().magic('cd Fatiando')

from scipy import signal
from fatiando.vis import mpl

sim_dt = 0.00433034793813

#Now create a lowpass Butterworth to lowpass the source frequency
fir = signal.firwin(527, 0.15)
for i in xrange(35):
    cmpn[:, i] = signal.convolve(cmpn[:, i], fir, 'same') # first must be the larger array
    
cmpn[:200,:] = 0.0 # gambiarra instead of fk for direct arrivals
figure(figsize=(4, 6), dpi=200)
dt = 0.00433034793813
mpl.seismic_image(cmpn, dt=sim_dt, vmin=-.01, vmax=0.01, aspect='auto')
mpl.seismic_wiggle(cmpn, dt=sim_dt, scale=10)

get_ipython().magic('cd Fatiando')
get_ipython().magic('cd ..')

CMPi = 172

from PIL import Image
import numpy as np
from matplotlib import pylab

img = Image.open('wedge.tif')
img.load()
img = img.convert('I') # gray scale, convert format
vdata  = np.asarray(img, dtype=np.float32)
vdata[:][vdata[:] == 255.] = 2500.0 # overburden
vdata[:][vdata[:] == 0. ] = 3500 # the triangle
vdata[:][vdata[:] == 146.] = 2300.0 # the first slab bellow the triangle
vdata[:][vdata[:] == 36.] = 2700.0 # the last slab
vdata = vdata[3:,:] # remove upper part not used in simulation (source and geophones were at z=3)
vshape = vdata.shape

print vshape
figure(figsize=(6,10))
ax = pylab.subplot(211)
ax.plot([CMPi, CMPi], [0, vshape[0]], 'w--')
ax.imshow(vdata, aspect='auto')
#pylab.colorbar(ax, orientation='horizontal')

vi = vdata[:, CMPi]
ds = np.ones(vshape[0])*25
vrmsi = vrms(vi, ds)
twt = 2*(ds/vi) # two way time
twt = twt.cumsum()
ax = pylab.subplot(212)
ax.plot(twt, vi, 'g-', label=r'$\mathbf{V_{interval}}$')
ax.plot(twt, vrmsi, 'b-', label=r'$\mathbf{V_{rms}}$')
ax.set_xlabel("Twt (s)")
ax.set_ylabel("Velocity (m/s)")
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(twt[::30])
ax2.set_xticklabels(ds.cumsum()[::30])
ax2.xaxis.grid()
ax2.set_xlabel("Depth (m)")
ax.legend()

dt = sim_dt
ns = cmpn.shape[0]

from scipy.interpolate import UnivariateSpline as ispline
# linear with extrapolation possible, cubic is bad at end and start
fs = ispline(twt, vrmsi, bbox=[0.0, ns*dt], k=1) 
ntwt = np.linspace(0.0, ns*dt, ns)
svrmsi = fs(ntwt)

get_ipython().magic('cd Fatiando')

from fatiando.seismic import utils

figure(figsize(8,6))
pylab.subplot(121)
svrmsi *= 0.92 # small tuning for better stacks,... RMS not exactly NMO velocity for non horizontal layer!!
# apply here just to look pretty, 
cmpnmo = utils.nmo(cmpn, cmpnffsets, svrmsi, dt=sim_dt)
mpl.seismic_image(cmpnmo, aspect='auto', vmin=-0.05, vmax=0.05)
#ylim(ns*sim_dt, 0)
pylab.subplot(122)
utils.plot_vnmo(cmpn, cmpnffsets, svrmsi, dt=sim_dt, vmin=-0.05, vmax=0.05)
ylim(ns*sim_dt, 0)
#mpl.seismic_wiggle(cmpnmo, scale=200)

from scipy.interpolate import UnivariateSpline as ispline
from fatiando.seismic import utils

ds = np.ones(vshape[0])*25 # from dz = 25 simulation grid
dt = 0.00433034793813


ncmps = df_traces.cmp.nunique()
cmps = df_traces.cmp.unique()
cmps_stacked = np.zeros((ns, ncmps))

for i in xrange(ncmps):

    print(i)
    CMPi = cmps[i]
    df_cmp = df_traces[ df_traces['cmp'] == CMPi  ]
    cmpn = df_cmp.traces.values
    cmpnoffsets =  df_cmp.abs_offset.values
    cmpn = np.array([ cmpn[p] for p in range(size(cmpn)) ])
    cmpn = cmpn.transpose()
    
    vi = vdata[:, CMPi//50]  # interval velocity  
    vrmsi = utils.vrms(vi, ds) # rms velocity
    twt = 2*(ds/vi) # exact two way time from interval velocity
    twt = twt.cumsum() 
    # linear with extrapolation possible, cubic is bad at end and start
    ntwt = np.linspace(0.0, ns*dt, ns)
    svrms = fs(ntwt)           
    cmpn[:180, :] = 0.0 # mute all instead of fk for direct arrivals
    svrms *= 0.92 # small tuning for better stacks
    cmp_nmo = utils.nmo(cmpn, cmpnoffsets, svrms, dt)   
    cmps_stacked[:, i] = cmp_nmo.sum(1) # stack all traces

get_ipython().magic('cd Fatiando')
from fatiando.vis import mpl

figure(figsize=(15,7))
mpl.seismic_image(cmps_stacked, dt=dt, vmin=-.1, vmax=0.1, aspect='auto')
#mpl.seismic_wiggle(cmps_stacked, dt=sim_dt, scale=50)

np.save('cmp_stacked_01_12_2015', cmps_stacked)

cmps_stacked.shape

cmp8625fft = np.fft.fft2(cmp8625)
cmp8625fk = np.abs(cmp8625fft)
figure(figsize=(10, 10))
imshow(cmp8625fk[:int(917/2)][:], aspect=.8, extent=[0, 30, 0, 1./(2*dt)], origin='lower')
ylabel('frequency (Hz)')
xlabel('wave number (k)')
ylim([0., 50])

get_ipython().magic('cd Fatiando')
dt = 0.00433034793813

def apply_nmo(nmo_args, dt=dt):
    cmpn, cmpnffsets, svrmsi = nmo_args
    sim = wavefd.Scalar(velocity, (dx, dz))     
    sim.add_point_source((source_x, 0), wavefd.Gauss(1., fc))   
    sim.run(iter)
    return sim[:, 3, array_stations]

from multiprocessing import Pool
print('Simulating...')
pool = Pool()
shots = pool.map(rum_sim_shot, zip(shots_x, stations_x))
pool.close()
print('Finished')

