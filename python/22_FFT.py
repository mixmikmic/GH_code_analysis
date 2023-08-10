import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

from scipy.constants import pi as π
def fourier_transform(t,y):
    '''Return the discrete Fourier transform of y.'''
    N = y.size
    Δt = t[1]-t[0]
    ŷ = np.zeros([N],dtype=complex)
    
    for k in range(N):
        ŷ[k] = np.sum(y*np.exp(-complex(0,1)*2*π*np.arange(N)*k/N))
        
    ω = 2*π*np.arange(N)/(N*Δt)
    
    return ω,ŷ

def fast_fourier_transform(t,y):
    '''Return the fast Fourier transform of y.'''
    ŷ = np.fft.fft(y)
    ω = 2*π*np.fft.fftfreq(t.size,t[1]-t[0])
    return ω,ŷ

# the time series
Δt = 0.1
t = np.arange(0.0,50.0,Δt)
ω0 = 2.0*π*(0.2)
ϕ = 0.5*π
y = np.sin(ω0*t + ϕ) + np.sin(2*ω0*t)

# Plot the time series
plt.figure(1)
plt.plot(t,y)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [arb.]')

# Get the discrete FT
ω,ŷ = fast_fourier_transform(t,y)

# Plot the real and imaginary parts of the DFT
plt.figure(2)
plt.plot(ω,ŷ.real,label='real.')
plt.plot(ω,ŷ.imag,linestyle='--',label='imag.')
plt.xlim(0,np.max(ω))
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Transform [arb.]')
plt.legend(loc='lower right')

import timeit
N = np.array([2**n for n in range(6,15)])
dft,fft = [],[]
for cN in N:
    t = np.linspace(0,20,cN)
    y = np.sin(t)
    
    dft.append(timeit.timeit('fourier_transform(t,y)', number=1, globals=globals()))
    fft.append(timeit.timeit('fast_fourier_transform(t,y)', number=1, globals=globals()))
    
dft = np.array(dft)
fft = np.array(fft)

from scipy.optimize import curve_fit

def N2(x,*a):
    '''N^2'''
    return a[0] + a[1]*x**2

def NlogN(x,*a):
    '''power function.'''
    return a[0] + a[1]*x*np.log2(x)

# perform the fits
a1,a1_cov = curve_fit(N2,N,dft,p0=(1,1))
a2,a2_cov = curve_fit(NlogN,N,fft,p0=(1,1))

# plot the timing results
plt.loglog(N,dft,'o', markersize=6, mec='None', mfc=colors[0], label='DFT')
plt.loglog(N,fft,'o', markersize=6, mec='None', mfc=colors[1], label='FFT')

# Plot the fits
plt.loglog(N,N2(N,*a1),'-', color=colors[0], zorder=0, linewidth=2, label=r'$N^2$')
plt.loglog(N,NlogN(N,*a2),'-', color=colors[1], zorder=0, linewidth=2, label=r'$N\log_2N$')

plt.xlim(50,20000)
plt.legend(loc='upper left')
plt.xlabel('N')
plt.ylabel('Execution Time [s]')

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
FD = [0.5,5.2]
Δt = 0.01

# We determine the maximum time such that N = 2*n
N = int(2**(np.ceil(np.log2(int(500.0/Δt)))))
t = np.arange(0.0,N*Δt,Δt)

fig, axes = plt.subplots(2,2,sharex=False, sharey=False, figsize=(10,10))
fig.subplots_adjust(wspace=0.5,hspace=0.5)
for i in range(2):
    θ,cω = euler(t,FD[i],*params)
    ω,θ̂ = fast_fourier_transform(t,θ)
    axes[i,0].plot(t,θ,color=colors[i],lw=2)
    axes[i,1].plot(ω,np.abs(θ̂),color=colors[i],lw=1)
    axes[i,1].set_xlim(0,50)
    



