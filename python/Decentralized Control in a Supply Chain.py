get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import control.matlab as control

Kd = .5
[num,den] = control.pade(1,5)
D = Kd*control.tf(num,den)
I = control.tf([1],[1,0])

Hyd = I*(D-1)

def myBode(H):
    w = np.logspace(-3,2,401)
    mag,phase,omega = control.bode(H,w,deg=True,Plot=False);
    
    wc = np.interp(-180.0,np.flipud(phase),np.flipud(omega))
    gc = np.interp(wc,omega,mag)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.loglog(omega,mag)
    plt.ylim(.01,10)
    plt.loglog([wc,wc],plt.ylim(),'r--')
    plt.loglog(plt.xlim(),[1,1],'g--')
    plt.grid()
    plt.title('Open Loop Transfer Function')
    
    plt.subplot(2,2,3)
    plt.semilogx(omega,phase)
    plt.ylim(-360,360)
    plt.semilogx(plt.xlim(),[-180,-180],'g--')
    plt.semilogx([wc,wc],plt.ylim(),'r--')
    plt.grid()

    
myBode(Hyd)

from ipywidgets import interact

w = np.logspace(-1,2,401)

def sim(torder=0, kp=.1, alpha = 0):

    num,den = control.pade(torder,5)
    p = control.tf(num,den)
    g = control.tf([1],[1,0])
    k = control.tf([kp],[1])
    mag,phase,omega = control.bode(g*p*k,w,deg=True,Hz=True, Plot=False);
    magc, phasec, omegac = control.bode(g*k/(1+g*p*k),w,deg=True,Hz=True, Plot=False);
    maga, phasea, omegaa = control.bode(alpha + (1-alpha)*g*k/(1+g*p*k),w,deg=True,Hz=True, Plot=False);
    
    wc = np.interp(-180.0,np.flipud(phase),np.flipud(omega))
    gc = np.interp(wc,omega,mag)

    
    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.loglog(omega,mag)
    plt.ylim(.01,10)
    plt.loglog([wc,wc],plt.ylim(),'r--')
    plt.loglog(plt.xlim(),[1,1],'g--')
    plt.grid()
    plt.title('Open Loop Transfer Function')
    
    plt.subplot(2,2,3)
    plt.semilogx(omega,phase)
    plt.ylim(-360,360)
    plt.semilogx(plt.xlim(),[-180,-180],'g--')
    plt.semilogx([wc,wc],plt.ylim(),'r--')
    plt.grid()
    
    plt.subplot(2,2,2)
    plt.loglog(omegac,magc)
    plt.loglog(omegaa,maga)
    plt.ylim(.01,10)
    plt.grid()
    plt.title('Closed Loop Transfer Function')
    
    plt.subplot(2,2,4)
    plt.semilogx(omegac,phasec)
    plt.semilogx(omegaa,phasea)
    plt.ylim(-360,360)
    plt.grid()

interact(sim, torder=(0,8,.02), kp=(.1,5,.05), alpha = (-1,1,.01))

cont

