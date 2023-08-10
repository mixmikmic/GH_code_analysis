

import osiris
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpmath import *
plt.rc('font',size=20,family="serif")
get_ipython().run_line_magic('matplotlib', 'inline')

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 32 

karray=np.arange(0.0,0.8,0.01)
[damping, frequency]=osiris.landau(karray)
damping_estimate=np.sqrt(np.pi/8)*1.0/(karray*karray*karray+1e-5)*np.exp(-1.5)*np.exp(-1/(2*karray*karray+1e-5))
damping_estimate[0]=0.0
frequency[0]=1.0
 
   
plt.figure(figsize=(8,6))
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
plt.plot(karray,np.abs(damping),'r',label='Exact')
plt.plot(karray,damping_estimate,'b',label='Jackson/dfdv estimate')
plt.xlabel('$k \lambda_D$')
plt.ylabel('$\gamma/\omega_{pe}$')
plt.title('Landau Damping Rate vs k')
# plt.xlim(0,0.3)
plt.legend()

  
plt.show()


plt.figure(figsize=(8,6))
plt.plot(karray[2:],100*(1-np.abs(damping_estimate[2:]/np.abs(damping[2:]))),'r',label='Relative Error in Damping')
plt.legend()
plt.xlabel('$k \lambda_D$')

plt.title('Percentage Error in the Jackson Estimate')
plt.show()
landau_exact=scipy.interpolate.interp1d(karray,np.abs(damping),kind='cubic')
landau_estimate=scipy.interpolate.interp1d(karray,damping_estimate,kind='cubic')



plt.figure(figsize=(8,6))
bohm_gross=(1+3*karray*karray)
plt.plot(karray,bohm_gross,'b',label='Bohm Gross')
plt.plot(karray, frequency,'r', label='Exact')
plt.title('Exact Frequency vs Bohm Gross Dispersion')
plt.xlabel('$k \lambda_D$')


plt.legend()
plt.show()

dirname = 'landau-kd035'
osiris.run_upic_es(rundir=dirname,inputfile='landau-kd035.txt')

dirname = 'landau-kd035'

osiris.plot_tk_arb(dirname,'Ex',klim=0.5,tlim=50)
#rundir, field, title='potential', klim=5,tlim=100

dirname = 'landau-kd035'
landau_exact=scipy.interpolate.interp1d(karray,np.abs(damping),kind='cubic')
landau_estimate=scipy.interpolate.interp1d(karray,damping_estimate,kind='cubic')
theory_exact=landau_exact(26.0*np.pi/256.0)
theory_estimate=landau_estimate(26.0*np.pi/256.0)
osiris.plot_tk_landau_theory(dirname,'Ex',modeno=13,theory1=theory_exact,theory2=theory_estimate,init_amplitude=1.18,tlim=50)
#rundir, field, title='potential', klim=5,tlim=100

dirname = 'landau-kd054'
osiris.run_upic_es(rundir=dirname,inputfile='landau-kd054.txt')

dirname = 'landau-kd054'


osiris.plot_tk_arb(dirname,'Ex',klim=0.6,tlim=25)
#rundir, field, title='potential', klim=5,tlim=100

dirname = 'landau-kd054'

landau_exact=scipy.interpolate.interp1d(karray,np.abs(damping),kind='cubic')
landau_estimate=scipy.interpolate.interp1d(karray,damping_estimate,kind='cubic')
theory_exact=landau_exact(44.0*np.pi/256.0)
theory_estimate=landau_estimate(44.0*np.pi/256.0)
osiris.plot_tk_landau_theory(dirname,'Ex',modeno=22,theory1=0.18,theory2=0.14,init_amplitude=1.3,tlim=25)
#rundir, field, title='potential', klim=5,tlim=100



