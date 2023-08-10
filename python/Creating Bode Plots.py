get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import control

import warnings
warnings.filterwarnings('ignore')

# requires coefficients of the numerator and denominator polynomials
# the coefficients are given starting with the highest power of s

G = 0.2*control.tf([0.5,1],[1.5,0.5,1])
print(G)

(num,den) = control.pade(0.25,3)
Gp = control.tf(num,den)*G
print(Gp)

mag,phase,omega = control.bode(Gp)

w = np.logspace(-1.5,1,200)
mag,phase,omega = control.bode(Gp,w)

mag,phase,omega = control.bode(Gp,w,Hz=True,dB=True,deg=False)

w = np.logspace(-1,1)
mag,phase,omega = control.bode(Gp,w);
plt.tight_layout()

# find the cross-over frequency and gain at cross-over
wc = np.interp(-180.0,np.flipud(phase),np.flipud(omega))
Kcu = np.interp(wc,omega,mag)

print('Crossover freq = ', wc, ' rad/sec')
print('Gain at crossover = ', Kcu)

mag,phase,omega = control.bode(Gp,w);
plt.tight_layout()

ax1,ax2 = plt.gcf().axes     # get subplot axes

plt.sca(ax1)                 # magnitude plot
plt.plot(plt.xlim(),[Kcu,Kcu],'r--')
plt.plot([wc,wc],plt.ylim(),'r--')
plt.title("Gain at Crossover = {0:.3g}".format(Kcu))

plt.sca(ax2)                 # phase plot
plt.plot(plt.xlim(),[-180,-180],'r--')
plt.plot([wc,wc],plt.ylim(),'r--')
plt.title("Crossover Frequency = {0:.3g} rad/sec".format(wc))



