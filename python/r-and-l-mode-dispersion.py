#############
# here we specify the plasma conditions 
wp=1
wc=0.5
#############

from scipy.optimize import fsolve 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#first we define the range of k's of interest, "k" here means "ck"
karray=np.arange(0,10,0.05)
nk=karray.shape[0]

def rwave_disp(w,omegap,omegac,ck):
    ratio=omegac/omegap
    y=(ck*ck)/(omegap*omegap) - (w*w)/(omegap*omegap) + 1/(1-wc/w)
    return y

def lwave_disp(w,omegap,omegac,ck):
    ratio=omegac/omegap
    y=(ck*ck)/(omegap*omegap) - (w*w)/(omegap*omegap) + 1/(1+wc/w)
    return y

wR=0.5*(wc+np.sqrt(4*wp*wp+wc*wc))
wL=0.5*(np.sqrt(4*wp*wp+wc*wc)-wc)

warrayL=np.zeros(karray.shape[0]); warrayR1=np.zeros(karray.shape[0]); warrayR2=np.zeros(karray.shape[0]);
wLarray=np.zeros(karray.shape[0]); wR1array=np.zeros(karray.shape[0]); wR2array=np.zeros(karray.shape[0]); 
#wHarray=np.zeros(karray.shape[0])

wLarray[:]=wL
wR1array[:]=0.01
wR2array[:]=wR
#wHarray[:]=np.sqrt(wp*wp+wc*wc)

warrayL[0]=wL
warrayR1[0]=0.01
warrayR2[0]=wR
for ik in range(1,nk):
    warrayR2[ik]=fsolve(rwave_disp,warrayR2[ik-1],args=(wp,wc,karray[ik]))
    warrayR1[ik]=fsolve(rwave_disp,warrayR1[ik-1],args=(wp,wc,karray[ik]))
    warrayL[ik]=fsolve(lwave_disp,warrayL[ik-1],args=(wp,wc,karray[ik]))

plt.plot(karray,warrayR1,'r',label='R-wave dispersion')
plt.plot(karray,warrayR2,'r')
plt.plot(karray,warrayL,'b',label='L-wave dispersion')
plt.plot(karray,wR2array,'r--',label='$\omega_R$')
plt.plot(karray,wLarray,'b--',label='$\omega_L$')
plt.plot(karray, karray,'g--',label='$\omega/k=c$')
plt.xlabel('wave number $[ck/\omega_{pe}]$')
plt.ylabel('frequency $[\omega_{pe}]$')
plt.title('R wave dispersion relation')
plt.legend()
plt.xlim([karray[0],karray[nk-1]])
plt.ylim([0,5])#karray[nk-1]+1.0])
plt.legend(loc=0, fontsize=8)
plt.show()

import osiris
get_ipython().run_line_magic('matplotlib', 'inline')

dirname = 'therm-b05-rl'
osiris.runosiris(rundir=dirname,inputfile='therm-b05-rl.txt')

dirname = 'therm-b20-rl'
osiris.runosiris(rundir=dirname,inputfile='therm-b20-rl.txt')

dirname = 'therm-b05-rl'
osiris.field(rundir=dirname, dataset='e2', time=100)

dirname = 'therm-b05-rl'
osiris.field(rundir=dirname, dataset='e2', space=100)

dirname = 'therm-b05-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vth = 0.1, b0_mag=0.5, vmin=-5, vmax=2, plot_or=3) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vth = 0.1, b0_mag=0.5, vmin=-5, vmax=2, plot_or=3, 
               show_theory=True) 

dirname = 'therm-b05-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth = 0.02, b0_mag=0.5, plot_or=2) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth = 0.02, b0_mag=0.5, plot_or=2, 
               show_theory=True) 

dirname = 'therm-b05-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth=0.02, b0_mag=0.5, plot_or=1) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth=0.02, b0_mag=0.5, plot_or=1, 
               show_theory=True) 

dirname = 'therm-b20-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vmin=-5, vmax=2, vth = 0.1, b0_mag=2.0, plot_or=3) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vmin=-5, vmax=2, vth = 0.1, b0_mag=2.0, plot_or=3, 
               show_theory=True) 

dirname = 'therm-b20-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vmin=-5, vmax=2, vth=0.02, b0_mag=2.0, plot_or=2) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,15], vmin=-5, vmax=2, vth=0.02, b0_mag=2.0, plot_or=2, 
               show_theory=True) 

dirname = 'therm-b20-rl'
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth = 0.005, b0_mag=2.0, plot_or=1) 
osiris.plot_wk_rl(rundir=dirname, wlim=[0,5], klim=[0,25], vmin=-5, vmax=2, vth = 0.005, b0_mag=2.0, plot_or=1, 
               show_theory=True) 

