# Plotting w(k)
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

N = 5
k = np.linspace(-N,N,N*20)
w_p = 1
c = 1
w = np.sqrt(w_p**2 + c**2 * k**2)
cline = k
plt.plot(k,w, label='$\omega$(k)')
plt.plot(k,cline, label='slope = c')
plt.xlabel('k [$c/\omega_p$]')
plt.ylabel('$\omega$ (in units of $\omega_p$)')
plt.xlim(-N,N)
plt.ylim(0,N+1)
plt.grid(b=True, which='major', axis='both')
plt.legend(loc=0)
plt.show()

import osiris
get_ipython().run_line_magic('matplotlib', 'inline')

# vth/c = 0.02
dirname = 'v02'
osiris.runosiris(rundir=dirname,inputfile='v02.txt')

dirname = 'v02'
osiris.field(rundir=dirname, dataset='e3', time=100)

dirname = 'v02'
osiris.field(rundir=dirname, dataset='e3', space=100)

dirname = 'v02'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.02, vmin=-7, vmax=2, plot_or=3) 
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.02, vmin=-7, vmax=2, plot_or=3, show_theory=True) 

dirname = 'v02'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,0.4], vth = 0.02, vmin=-7, vmax=2, show_theory=True, 
               debye=True, plot_or=3) 

# vth/c = 0.05
dirname = 'v05'
osiris.runosiris(rundir=dirname,inputfile='v05.txt')

dirname = 'v05'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.05, vmin=-7, vmax=2, plot_or=3) 
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.05, vmin=-7, vmax=2, plot_or=3, show_theory=True) 

dirname = 'v05'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,0.4], vth = 0.05, vmin=-7, vmax=2, show_theory=True, 
               plot_or=3, debye=True) 

# vth/c = 0.2
dirname = 'v20'
osiris.runosiris(rundir=dirname,inputfile='v20.txt')

dirname = 'v20'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.20, vmin=-5, vmax=4, plot_or=3) 
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,10], vth = 0.20, vmin=-5, vmax=4, plot_or=3, show_theory=True) 

dirname = 'v20'
osiris.plot_wk(rundir=dirname, wlim=[0,10], klim=[0,0.4], vth = 0.20, vmin=-5, vmax=4, show_theory=True, plot_or=3, debye=True)

