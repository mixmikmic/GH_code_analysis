from IPython.display import YouTubeVideo
# A driven double pendulum
YouTubeVideo('7DK1Eayyj_c')

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

def non_linear_θ(ℓ,θ0,t):
    '''The solution for θ for the non-linear pendulum.'''
    # use special functions
    from scipy import special
    k = np.sin(θ0/2.0)
    K = special.ellipk(k*k)
    (sn,cn,dn,ph) = special.ellipj(K-np.sqrt(g/l)*t,k*k)
    return 2.0*np.arcsin(k*sn)

from scipy.constants import pi as π
from scipy.constants import g

# constants and intitial conditions
ℓ = 0.25 # m
Δt = 0.001 # s

t = np.arange(0.0,4.0,Δt)
θ,ω = np.zeros_like(t),np.zeros_like(t)
θ[0] = π/4.0 # rad

for n in range(t.size-1):
    θ[n+1] = θ[n] + ω[n]*Δt
    ω[n+1] = ω[n] -(g/ℓ)*np.sin(θ[n+1])*Δt

# the exact solution
plt.plot(t,non_linear_θ(ℓ,θ[0],t), label='Exact')

# the Euler-Cromer method
plt.plot(t[::20],θ[::20], 'o', mfc='None', markersize = 6, label='Euler Cromer method')
plt.legend(loc='lower left',frameon=True)

plt.xlabel('Time [s]')
plt.ylabel('θ(t) [rad]')

from scipy.constants import g
from scipy.constants import pi as π
def euler(t,FD,ℓ,θ0,ω0,γ,ΩD):
    ''' Semi-implicit Euler Method for the non-linear, dissipative, driven pendulum.'''
    
    Δt = t[1]-t[0]
    ω,θ = np.zeros_like(t),np.zeros_like(t)
    θ[0],ω[0] = θ0,ω0
    
    # perform the numerical integration
    for n in range(t.size-1):
        ω[n+1] = ω[n] + (-(g/ℓ)*np.sin(θ[n]) - γ*ω[n] + FD*np.sin(ΩD*t[n]))*Δt
        θ[n+1] = θ[n] + ω[n+1]*Δt
        
        # keep theta in [-pi,pi)
        if θ[n+1] < -π: θ[n+1] += 2.0*π
        if θ[n+1] >= π: θ[n+1] -= 2.0*π 

    return θ,ω

params = ℓ,θ0,ω0,γ,ΩD = g, 0.2, 0.0, 0.5, 2.0/3.0
FD = [0.0,0.5,0.75,1.0,1.25,1.5]
Δt = 0.04
t = np.arange(0.0,60,Δt)

θ,ω = euler(t,0.0,*params)

plt.plot(t,θ)
plt.title(r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % 0.0)
plt.xlabel('Time [s]')
plt.ylabel('θ(t) [rad]')

params = g, 0.2, 0.0, 0.5, 2.0/3.0
θ,ω = euler(t,0.2,*params)

plt.plot(t,θ)
plt.title(r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % 0.2)
plt.xlabel('Time [s]')
plt.ylabel('θ(t) [rad]')

F = [0.0,0.5,0.75,1.0,1.25,1.5]

# create a subplot array
fig, axes = plt.subplots(2,3, figsize=(12,10), sharex=True)
fig.subplots_adjust(wspace=0.3)
for i, ax in enumerate(axes.flat):
    θ,ω = euler(t,F[i],*params)
    ax.plot(t,θ, label=r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i])
    ax.legend(frameon=True, loc='lower right')

# set axis labels
[ax.set_ylabel('θ(t) [rad]') for ax in axes[:,0]]
[ax.set_xlabel('Time [s]') for ax in axes[-1,:]]

θ0 = [0.2,0.21,0.22] #  initial conditions
F = [0.5,1.2] # driving force

# compare the initial conditions
fig, axes = plt.subplots(2,1,sharex=True,sharey=False, figsize=(6,8))
for i, ax in enumerate(axes.flat):
    for j,cθ in enumerate(θ0):
        label = r'$\theta_0 = %4.2f\,\mathrm{rad}$' % cθ
        params = ℓ,cθ,ω0,γ,ΩD
        θ,ω = euler(t,F[i],*params)
        ax.plot(t,θ,label=label)
    ax.legend(frameon=True, loc='lower left')
    ax.text(70,0.0,r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i],fontsize=30)
    
    # set axis labels
    ax.set_ylabel('θ(t) [rad]')

axes[-1].set_xlabel('Time [s]')

F = [0.5,1.2] # driving force [1/s^2]
θ0 = 0.2 # initial angle [rad]
params = [(ℓ,θ0,ω0,γ,ΩD),(ℓ,θ0+1.0E-4,ω0,γ,ΩD)]

Δθ = []
fig, axes = plt.subplots(2,1,sharex=True, sharey=False, squeeze=False, figsize=(6,8))
for i, ax in enumerate(axes.flat):
    label = r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i]
    
    θ1,ω = euler(t,F[i],*params[0])
    θ2,ω = euler(t,F[i],*params[1])
    Δθ.append(np.abs(θ1-θ2))
    
    ax.semilogy(t, Δθ[i], label=label)
    ax.legend(loc='best', frameon=False)
    
    # set axis labels
    ax.set_ylabel('|Δθ(t)| [rad]')

axes[1,0].set_xlabel('Time [s]')

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
F = [0.5,1.2]

# Linear fitting function
def linear(x,a0,a1):
    return a0 + a1*x

# find the local maxima
popt = [0,0]
for i,cF in enumerate(F):
    ind = argrelextrema(np.log(Δθ[i]),np.greater)[0]
    extθ = np.log(Δθ[i][ind])
    popt[i], pcov = curve_fit(linear,t[ind],extθ)

# Now plot the results of the fit
fig, axes = plt.subplots(2,1,sharex=True, sharey=False, squeeze=False, figsize=(6,8))
for i, ax in enumerate(axes.flat):
    labellam = r'$\lambda = %4.2f\; \mathrm{s}^{-1}$' % popt[i][1]
    
    ax.semilogy(t, Δθ[i], ',', markeredgewidth=0.0)
    ax.semilogy(t, np.exp(linear(t,popt[i][0],popt[i][1])), linewidth=3.0, label=labellam)
      
    # set labels and legend
    ax.set_ylabel('|Δθ(t)| [rad]')
    ax.text(70,1.0E-6,r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i],fontsize=30)
    ax.legend()
axes[1,0].set_xlabel('Time [s]')

params = ℓ,θ0,ω0,γ,ΩD
F = [0.5,1.2]
blue = '#2078b5'

fig, axes = plt.subplots(2,1,sharex=True, sharey=False, squeeze=False, figsize=(6,8))
for i, ax in enumerate(axes.flat):
    labelF = r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i]
    theta,omega = euler(t,F[i],*params)
    
    ax.scatter(theta, omega, s=1.0, color=blue, label=labelF)

    # set axis labels and legends
    ax.set_ylabel('ω [rad/s]')
    ax.set_xlim(-π,π)
    ax.legend(loc='upper right')

axes[1,0].set_xlabel('θ(t) [rad]')

# Generate more phase space data
Δt = 2.0*π/(ΩD*100)
longt = np.arange(0,10000,Δt)

F = [0.5,1.2]
θ,ω = [],[]
for cF in F:
    th,w = euler(longt,cF,*params)
    θ.append(th)
    ω.append(w)

# Get the in-phase time slices
inPhase = []
n = 0
while True:
    m = int(2.0*π*n/(longt[1]*ΩD) + 0.5)
    if m > 1 and m < len(longt):
        inPhase.append(m)
    elif m >= len(longt):
        break
    n += 1

#Exploit the ability of numpy arrays to take a list of indices as their index
inPhaset = longt[inPhase]

orange = '#ff7f0f'
colors = orange,blue
plt.figure(figsize=(8,8))
for i in range(2):
    labelF = r'$F_\mathrm{D} = %3.1f\; \mathrm{s}^{-2}$' % F[i]
    plt.scatter(θ[i][inPhase], ω[i][inPhase], s=1.0, color=colors[i], label=labelF)
    
plt.title('Strange Attractors')
plt.legend(fontsize=16)
plt.xlabel('θ [rad]')
plt.ylabel('ω [rad/s]')
plt.xlim(-π,π);

F = [1.35, 1.44, 1.465] 
θ0 = 0.2
params = ℓ,θ0,ω0,γ,ΩD

fig, axes = plt.subplots(3,1,sharex=True, sharey=True, squeeze=False, figsize=(6,10))
for i, ax in enumerate(axes.flat):
        labelF = r'$F_\mathrm{D} = %5.3f\; \mathrm{s}^{-2}$' % F[i]
        θ,ω = euler(t,F[i],*params)
        ax.plot(t, θ, label=labelF)
        ax.legend(loc="lower left", frameon=True, prop={'size':16})
    
        # set axis labels
        ax.set_ylabel('θ(t) [rad]')
        
axes[-1,0].set_xlabel('Time [s]')

run = False
F = np.arange(0,1.5,0.0025)
if run:
    θ = np.zeros([len(inPhase[10:]),len(F)])
    for i,cF in enumerate(F):
        th,ω = euler(longt,cF,*params)
        θ[:,i] = th[inPhase[10:]]
    np.savetxt('data/theta.dat',theta)

θ = np.loadtxt('data/theta.dat')
plt.figure(figsize=(7,7))
for i,cF in enumerate(F):
    plt.scatter(cF*np.ones_like(θ[:,i]), θ[:,i], s=3.0, marker=',', c=blue, edgecolors='None') 

plt.ylabel('θ [rad]')
plt.xlabel(r'$F_\mathrm{D}\; [\mathrm{s}^{-2}]$')
plt.xlim(1.34,1.5);
plt.ylim(0,π);



