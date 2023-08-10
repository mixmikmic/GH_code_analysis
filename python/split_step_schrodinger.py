#NAME: Schrödinger Equation
#DESCRIPTION: Solving the 1D time dependent Schrödinger Equation using the split step method.

import pycav.display as display

display.display_animation('barrier.mp4')

display.display_animation('linwell.mp4')

import numpy as np

import pycav.pde as pde
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def oneD_gaussian(x,mean,std,k0):
    return np.exp(-((x-mean)**2)/(4*std**2)+ 1j*x*k0)/(2*np.pi*std**2)**0.25

def V(x):
    V_x = np.zeros_like(x)
    V_x[np.where(abs(x) < 0.5)] = 1.5
    return V_x

N_x = 2**11
dx = 0.05
x = dx * (np.arange(N_x) - 0.5 * N_x)

dt = 0.01
N_t = 2000

p0 = 2.0
d = np.sqrt(N_t*dt/2.)

psi_0 = oneD_gaussian(x,x.max()-10.*d,d,-p0)

psi_x,psi_k,k = pde.split_step_schrodinger(psi_0, dx, dt, V, N_t, x_0 = x[0])

real_psi = np.real(psi_x)
imag_psi = np.imag(psi_x)
absl_psi = np.absolute(psi_x)
abs_psik = np.absolute(psi_k)

fig = plt.figure(figsize = (10,10))
ax1 = plt.subplot(211)
line1_R = ax1.plot(x,real_psi[:,0],'b')[0]
line1_I = ax1.plot(x,imag_psi[:,0],'r')[0]
line1_A = ax1.plot(x,absl_psi[:,0],'k')[0]
line_V = ax1.plot(x,0.5*V(x),'k',alpha=0.5)[0]
ax1.set_ylim((real_psi.min(),real_psi.max()))
ax1.set_xlim((x.min(),x.max()))

ax2 = plt.subplot(212)
line2 = ax2.plot(k,abs_psik[:,1],'k')[0]
ax2.set_ylim((abs_psik.min(),abs_psik.max()))
ax2.set_xlim((-10,10))

def nextframe(arg):
    line1_R.set_data(x,real_psi[:,10*arg])
    line1_I.set_data(x,imag_psi[:,10*arg])
    line1_A.set_data(x,absl_psi[:,10*arg])
    line2.set_data(k,abs_psik[:,10*arg])
    
animate = anim.FuncAnimation(fig, nextframe, frames = int(N_t/10), interval = 50, repeat = False)
animate = display.create_animation(animate,temp = True)
display.display_animation(animate)





