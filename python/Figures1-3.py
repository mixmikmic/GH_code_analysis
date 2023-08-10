import rebound
print(rebound.__build__)
import numpy as np
import warnings
from ctypes import cdll, byref
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    get_ipython().magic('matplotlib inline')
    import matplotlib.pyplot as plt

def setup(sim):
    try:
        cgr = cdll.LoadLibrary("gr_force.so")
        gr_force = cgr.gr_force
    except:
        print("Using fall-back python implementation of GR force. This might be slower than using the C version.")
        def gr_force(sim):
            ps = sim.contents.particles
            source = ps[0]
            prefac1 = 6.*source.m*source.m/1.0130251e+08
            for i in range(1,sim.contents.N):
                p = ps[i]
                dp = p - source
                r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z
                prefac = prefac1/(r2*r2)
                ps[i].ax -= prefac*dp.x
                ps[i].ay -= prefac*dp.y
                ps[i].az -= prefac*dp.z
                ps[0].ax += p.m/source.m*prefac*dp.x
                ps[0].ay += p.m/source.m*prefac*dp.y
                ps[0].az += p.m/source.m*prefac*dp.z
    sim.additional_forces=gr_force            

def gr_potential(sim):
    source = sim.particles[0]
    mu = sim.G*source.m
    prefac = 3.*mu*mu/1.0130251e+08
    grpot = 0.
    for i in range(1,sim.N):
        pi = sim.particles[i] - source
        r2 = pi.x*pi.x + pi.y*pi.y + pi.z*pi.z
        grpot -= prefac*sim.particles[i].m/r2
    return grpot

sa = rebound.SimulationArchive("restart_0013.bin",setup=setup)

get_ipython().run_cell_magic('timeit', '-n1 -r1 ', 'fig = plt.figure(figsize=(7, 3)) \nax = plt.subplot(111)\nax.set_xscale("log")\nax.set_yscale("log")\nax.set_ylim([1e-13,1e-9])\nsim0 = sa.getSimulation(0.)\ne0 = sim0.calculate_energy()+gr_potential(sim0)\n\nif False:\n    times = np.logspace(3., 8, 2500)\n    data = np.zeros((2,len(times)))\n    for j, sim in enumerate(sa.getSimulations(times, mode=\'close\')):\n        data[0][j] = sim.t\n        data[1][j] = np.abs((e0-sim.calculate_energy()-gr_potential(sim))/e0)    \n    ax.scatter(data[0]/(2.*np.pi), data[1],marker=".");\n    ax.set_xlim([data[0][0]/(2.*np.pi),sa.tmax/(2.*np.pi)])    \n\ntimes = np.logspace(7., np.log10(sa.tmax), 2000)\ndata = np.zeros((2,len(times)))\nfor j, sim in enumerate(sa.getSimulations(times)):\n    data[0][j] = sim.t\n    data[1][j] = np.abs((e0-sim.calculate_energy()-gr_potential(sim))/e0)\n    \nax.set_xlim([data[0][0]/(2.*np.pi),sa.tmax/(2.*np.pi)])        \nax.set_xlabel("time [yrs]")\nax.set_ylabel("relative energy error")\nax.scatter(data[0]/(2.*np.pi), data[1],marker=".",color="black",alpha=0.24);\nax.plot([1e5,1e10],2e-10*np.sqrt([1e5,1e10])/1e5);\nplt.savefig("f_energy.pdf", format=\'pdf\', bbox_inches=\'tight\', pad_inches=0)')

def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

get_ipython().run_cell_magic('timeit', '-n1 -r1 ', 'fig = plt.figure(figsize=(7, 2)) \nax = plt.subplot(111)\ntimes = np.linspace(sa.tmin, sa.tmax, 20000)\ndata = np.zeros((2,len(times)))\nle =0\nfor i, sim in enumerate(sa.getSimulations(times)):\n    data[0][i] = sim.t/2/np.pi/1e9\n    data[1][i] = sim.particles[1].e\nax.set_xlim([0,data[0][-1]])            \nn=30\nax.set_xlabel("time [Gyrs]")\nax.set_ylabel("eccentricity of Mercury")\nax.plot(data[0], data[1], color="black",alpha=0.32);\nax.plot(data[0][n//2:-n//2+1], moving_average(data[1],n=n),color="black");\nplt.savefig("f_eccentricity.pdf", format=\'pdf\', bbox_inches=\'tight\', pad_inches=0)')

from multiprocessing import Pool
def thread_init(*rest):
    import rebound
    global sat2        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sat2 = rebound.SimulationArchive("restart_0013.bin",setup=setup)
def analyze(t):
    sim = sat2.getSimulation(t,mode="close")
    return [sim.t/2/np.pi,sim.particles[3].e]        
pool = Pool(initializer=thread_init) 

# First panel
times = np.linspace(0.1, 801e3*2.*np.pi, 1000)
res = np.array(pool.map(analyze,times))
fig = plt.figure(figsize=(7, 2)) 
ax = plt.subplot(111)
ax.set_xlim([0,res[-1][0]])            
ax.set_xlabel("time [yrs]")
ax.set_ylabel("eccentricity of Earth")
ax.plot(res[:,0], res[:,1], color="black");
plt.savefig("f_eccentricity_earth.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

# Second panel
times = np.linspace(5e9*2*np.pi+0., 5e9*2*np.pi+801e3*2.*np.pi, 1000)
res = np.array(pool.map(analyze,times))
fig = plt.figure(figsize=(7, 2)) 
ax = plt.subplot(111)
ax.set_xlim([res[1][0],res[-1][0]])            
ax.set_xlabel("time [yrs]")
ax.set_ylabel("eccentricity of Earth")
ax.plot(res[:,0], res[:,1], color="black");
plt.savefig("f_eccentricity_earth5.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

pool.close()



