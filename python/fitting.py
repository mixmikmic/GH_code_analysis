import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('matplotlib inline')
# comment out the following if you're not on a Mac with HiDPI display
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

def solve_ns(A, b): return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
def clsq(A, b, C, d):
    """An 'exact' constrained least squared solution of Ax= b, s.t. Cx= d"""
    from scipy.linalg import qr
    p= C.shape[0]
    Q, R= qr(C.T)
    xr, AQ= np.linalg.solve(R[:p].T, d), np.dot(A, Q)
    xaq= solve_ns(AQ[:, p:], b- np.dot(AQ[:, :p], xr))
    return np.dot(Q[:, :p], xr)+ np.dot(Q[:, p:], xaq)
def cpf(x, y, x_c, y_c, n):
    """Constrained polynomial fit based on clsq solution."""
    from numpy.polynomial.polynomial import Polynomial as P, polyvander as V
    x_c = np.array([x_c])
    y_c = np.array([y_c])
    return P(clsq(V(x, n), y, V(x_c, n), y_c))

L=pd.read_csv('qsiwell2_frm.csv')

vp=L.VP.values
z=L.DEPTH.values
mask = np.isfinite(vp) # mask to select valid (i.e., not NaNs) velocity samples

# anchor points
z0=1000
vp0=1600
vs0=800
rho0=1.5

# final trends will be calculated over zfit that goes from z0=seabottom to 3000m
zfit = np.linspace(z0,3000,100)

# simple linear fit
fit0 = np.polyfit(z[mask],vp[mask],1)
trend0 = np.polyval(fit0,zfit)

# constrained polynomial fit
p = cpf(z[mask], vp[mask], z0 , vp0, 1)
trend1 = p(zfit)

plt.figure(figsize=(6,10))
plt.plot(vp,z,'k',lw=1,label='Vp')
plt.plot(trend0,zfit,'r',lw=4,label='np.polyfit')
plt.plot(trend1,zfit,'k',alpha=0.4,lw=8,label='cpf')
plt.plot(vp0,z0,'ko',ms=10)
plt.ylim(3000,0)
plt.axhline(y=z0, color='b')
plt.legend()
plt.grid()

# constrained linear fit
m = np.linalg.lstsq(z[mask][:,None]-z0,vp[mask][:,None]-vp0)[0]
trend2 = m.flatten()*(zfit-z0)+vp0

plt.figure(figsize=(6,10))
plt.plot(vp,z,'k',lw=1,label='Vp')
plt.plot(trend0,zfit,'r',lw=4,label='np.polyfit')
plt.plot(trend1,zfit,'k',alpha=0.4,lw=8,label='cpf')
plt.plot(trend2,zfit,'y',lw=2,label='np.linalg.lstsq constrained')
plt.plot(vp0,z0,'ko',ms=10)
plt.ylim(3000,0)
plt.axhline(y=z0, color='b')
plt.legend(fontsize='medium')
plt.grid()

from scipy.optimize import curve_fit

def fpow(x, a, b):
    return a * np.power(x,b)

def flog(x, a, b):
    return a * np.log(x) + b

def fexp(z, a, b, c):
    return a / (1+c*np.exp(-z/b))

# to constrain curve_fit first I add the anchor point to the top of the log
# then create a sigma vector with a absurdly low value (=high weight)
# corresponding to this first point 
z1=np.insert(z[mask],0,z0)
vp1=np.insert(vp[mask],0,vp0)
sigma =np.ones(z1.size)
sigma[0] = 0.001

# constrained polynomial fit 2nd degree
p = cpf(z[mask], vp[mask], z0 , vp0, 2)
trend3 = p(zfit)

# curve_fit of function a/(1+c*exp(-z/z0))
pE, _ = curve_fit(fexp, z[mask], vp[mask], p0=(5000,2000,1), maxfev=9999)
trendE = fexp(zfit, *pE)
pEc, _ = curve_fit(fexp, z1, vp1, p0=(5000,2000,1), sigma=sigma, maxfev=9999)
trendEc = fexp(zfit, *pEc)

# curve_fit of function a*x**b
pP, _ = curve_fit(fpow, z[mask], vp[mask])
trendP = fpow(zfit, *pP)
pPc, _ = curve_fit(fpow, z1, vp1,sigma=sigma)
trendPc = fpow(zfit, *pPc)

# curve_fit of function a*log(x)+b
pL, _ = curve_fit(flog, z[mask], vp[mask])
trendL = flog(zfit, *pL)
pLc, _ = curve_fit(flog, z1, vp1,sigma=sigma)
trendLc = flog(zfit, *pLc)

f, ax = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(12,10))
ax[0].plot(trendE,zfit, label='V = a/(1+c*e^(-z/b)')
ax[0].plot(trendP,zfit, label='V = a*Z^b')
ax[0].plot(trendL,zfit, label='V = a*ln(Z)+b')
ax[1].plot(trendEc,zfit, label='V = a/(1+c*e^(-z/b) constrained')
ax[1].plot(trendPc,zfit, label='V = a*Z^b constrained')
ax[1].plot(trendLc,zfit, label='V = a*ln(Z)+b constrained')
ax[0].set_title('unconstrained non-linear fits')
ax[1].set_title('constrained non-linear fits')
for aa in ax:
    aa.plot(vp,z,'k',lw=1,label='Vp')
    aa.plot(trend2,zfit,'k',lw=8, alpha=0.4,label='np.linalg.lstsq constrained')
    aa.plot(vp0,z0,'ko',ms=10)
    aa.set_ylim(3000,0)
    aa.axhline(y=z0, color='b')
    aa.legend(fontsize='medium')
    aa.grid()

vpb_detrend=np.nanmean(vpb-vp) # I know, "detrend" is not the most appropriate name for a variable
vpg_detrend=np.nanmean(vpg-vp)

print('shift to be applied to calculated trend for BRINE: {} m/s'.format(vpb_detrend))
print('shift to be applied to calculated trend for GAS: {} m/s'.format(vpg_detrend))

#trend=trend2.copy()  # using the linear trend as starting trend
trend=trendLc.copy() # using the non-linear logarithmic trend as starting trend
trend_brine=trend+vpb_detrend
trend_gas=trend+vpg_detrend

f, ax = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(12,10))
ax[0].plot(trend, zfit, 'k', lw=5, alpha=0.5, label='Initial trend')

ax[1].plot(trend_brine, zfit, 'b', lw=5, alpha=0.5, label='brine')
ax[1].plot(trend_gas, zfit, 'r', lw=5, alpha=0.5, label='gas')

ax[0].set_title('Trendline on insitu data')
ax[1].set_title('Trendline for various fluids')
for aa in ax:
    aa.plot(vp,z,'k',lw=1,label='Vp')
    aa.plot(vp0,z0,'ko',ms=10)
    aa.set_ylim(3000,0)
    aa.axhline(y=z0, color='b')
    aa.legend(fontsize='medium')
    aa.grid()



