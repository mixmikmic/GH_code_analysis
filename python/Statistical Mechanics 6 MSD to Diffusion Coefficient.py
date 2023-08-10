import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

Ndim = 2

N = 10000
dp = 1e-6
nu = 8.9e-4
T = 293
kB = 1.38e-23
pi = np.pi
T = 10000.0
dt = T/N

def get_Dtheor(T, Ndim, dp, nu):
    Dtheor = (kB*T)/(3*Ndim*pi*dp*nu)
    return Dtheor
    
Dtheor = get_Dtheor(T,Ndim,dp,nu)
print(Dtheor)

# Variance of step size distribution
# (units of m)
var = 2*Dtheor*dt
stdev = np.sqrt(var)
print(stdev)

# Single random diffusion walk

# mean 0, std dev computed above
dx = stdev*np.random.randn(N,)
dy = stdev*np.random.randn(N,)

x = np.cumsum(dx)
y = np.cumsum(dy)

plt.plot(x, y, '-')
plt.xlabel('x'); plt.ylabel('y');
plt.title("Brownian Motion 2D Walk")
plt.show()

# Compute MSD versus lag time 
# 0 to sqrt(N) avoids bias of longer lag times
upper = int(round(np.sqrt(N)))
msd = np.zeros(upper,)
lag = np.zeros(upper,)

for i, p in enumerate(range(1,upper+1)):
    lagtime = dt*p
    delx = ( x[p:] - x[:-p] )
    dely = ( y[p:] - y[:-p] )
    msd[i] = np.mean(delx*delx + dely*dely)
    lag[i] = lagtime

m, b = np.polyfit(lag, msd, 1)

plt.loglog(lag, msd, 'o')
plt.loglog(lag, m*lag+b, '--k')

plt.xlabel('Lag time (s)')
plt.ylabel('MSD (m)')
plt.title('Linear Fit: MSD vs. Lag Time')

plt.show()

print("linear fit:")
print("Slope = %0.2g"%(m))
print("Intercept = %0.2g"%(b))

# Slope is:
# v = dx / dt
# v = 2 D / dt
# Rearrange:
# D = v * dt / 2
v = m
Dempir = (v*dt)/2

err = (np.abs(Dtheor-Dempir)/Dtheor)*100

print("Theoretical D:\t%0.4g"%(Dtheor))
print("Empirical D:\t%0.4g"%(Dempir))
print("Percent Error:\t%0.4g"%(err))
print("\nNote: this result is from a single realization. Taking an ensemble yields a more accurate predicted D.")

def msd_ensemble(T, Ndim, dp, nu, N, Nwalks):
    Dtheor = get_Dtheor(T, Ndim, dp, nu)
    
    ms    = []
    msds  = []
    msdxs = []
    msdys = []
    lags  = []
    
    for w in range(Nwalks):
    
        # Single random diffusion walk
        # mean 0, std dev computed above
        dx = stdev*np.random.randn(N,)
        dy = stdev*np.random.randn(N,)
        # accumulate
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        # Compute MSD versus lag time 
        # 0 to sqrt(N) avoids bias of longer lag times
        upper = int(round(np.sqrt(N)))
        msd  = np.zeros(upper,)
        msdx = np.zeros(upper,)
        msdy = np.zeros(upper,)
        lag  = np.zeros(upper,)

        for i, p in enumerate(range(1,upper+1)):
            lagtime = dt*p
            delx = ( x[p:] - x[:-p] )
            dely = ( y[p:] - y[:-p] )
            msd[i] = np.mean((delx*delx + dely*dely)/2)
            msdx[i] = np.mean(delx*delx)
            msdy[i] = np.mean(dely*dely)
            lag[i] = lagtime

        slope, _ = np.polyfit(lag, msd, 1)
    
        ms.append( slope )
        msds.append( msd )
        msdxs.append(msdx)
        msdys.append(msdy)
        lags.append( lag )
    
    
    return (ms, msds, msdxs, msdys, lags)

Ndim = 2

N = 10000
dp = 1e-6
nu = 8.9e-4
T = 293
kB = 1.38e-23
pi = np.pi
T = 10000.0
dt = T/N

Nwalks = 1000

slopes, msds, msdxs, msdys, lags = msd_ensemble(T, Ndim, dp, nu, N, Nwalks)

Dempir = np.mean((np.array(slopes)*dt)/2)

err = (np.abs(Dtheor-Dempir)/Dtheor)*100

print("Theoretical D:\t%0.4g"%(Dtheor))
print("Empirical D:\t%0.4g"%(Dempir))
print("Percent Error:\t%0.4g%%"%(err))

print("\nUsing an ensemble of %d particles greatly improves accuracy of predicted D."%(N))

for i, (msd, lag) in enumerate(zip(msdxs,lags)):
    if(i>200):
        break
    plt.loglog(lag,msd,'b',alpha=0.1)

for i, (msd, lag) in enumerate(zip(msdys,lags)):
    if(i>200):
        break
    plt.loglog(lag,msd,'r',alpha=0.1)
    
for i, (msd, lag) in enumerate(zip(msds,lags)):
    if(i>200):
        break
    plt.loglog(lag,msd,'k',alpha=0.1)
    
plt.xlabel('Lag Time (m)')
plt.ylabel('MSD (s)')
plt.title('MSD vs Lag Time: \nMSD X (blue), MSD Y (red), MSD MAG (black)')
plt.show()



