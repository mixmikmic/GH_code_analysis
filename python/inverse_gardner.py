import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
sympy.init_printing(use_latex='mathjax')

get_ipython().magic('matplotlib inline')
# comment out the following if you're not on a Mac with HiDPI display
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

d, V, f, rho = sympy.symbols('d, V, f, rho')
gardner_expr=sympy.Eq(rho, d*V**f)

gardner_expr

sympy.solve(gardner_expr,V)

print(sympy.solve(gardner_expr,V))

vp_test=np.linspace(2000, 4000)
rho_test=0.31*vp_test**0.25

noise=np.random.uniform(0.1,.3,vp_test.shape)
rho_test += noise

plt.figure(figsize=(6,6))
plt.plot(rho_test,vp_test,'ok')
plt.xlabel('Density [g/cc]'), plt.xlim(1.8,3)
plt.ylabel('Vp [m/s]'), plt.ylim(1500,5000)
plt.grid()

from scipy.optimize import curve_fit
def inv_gardner(rho, d, F):
    return (rho/d)**(1/F)

popt_synt, pcov = curve_fit(inv_gardner,rho_test,vp_test)
print(popt_synt)

rho_synt=np.linspace(1, 3, 50)
vp_synt=inv_gardner(rho_synt, *popt_synt)

plt.figure(figsize=(6,6))
plt.plot(rho_test,vp_test,'ok')
plt.plot(rho_synt, vp_synt, '-r', lw=2, label=r'$ V_p = (%.2f / \rho)^{1/%.2f}$' % (popt_synt[0],popt_synt[1]))
plt.xlabel('Density [g/cc]'), plt.xlim(1.8,3)
plt.ylabel('Vp [m/s]'), plt.ylim(1500,5000)
plt.grid()
plt.legend()

ll=np.recfromcsv('qsiwell2_frm.csv')
z    =ll['depth']
gr   =ll['gr']
vpb  =ll['vp_frmb']
rhob =ll['rho_frmb']
nphi =ll['nphi']

ztop=2025
zbot=2250

f,ax = plt.subplots(1,3,figsize=(12,8))
ax[0].plot(gr,z,'-k')
ax[0].set_xlabel('GR')
ax[1].plot(nphi,z,'-g')
ax[1].set_xlabel('NPHI',color='g'), ax[1].set_xlim(0.45,-0.15)
ax1bis = ax[1].twiny()
ax1bis.plot(rhob, z, 'k')
ax1bis.set_xlabel('RHOB', color='k'),  ax1bis.set_xlim(1.95,2.95)
ax[2].plot(vpb,z,'-k')
ax[2].set_xlabel('Vp')
for aa in ax:
    aa.set_ylim(zbot,ztop)
    aa.grid()

vp1=vpb.copy()
hole=(z>2075) & (z<2125)
vp1[hole]=np.nan

fig,ax = plt.subplots(1,3,figsize=(12,8))
ax[0].plot(gr,z,'-k')
ax[0].set_xlabel('GR')
ax[1].plot(nphi,z,'-g')
ax[1].set_xlabel('NPHI',color='g'), ax[1].set_xlim(0.45,-0.15)
ax1bis = ax[1].twiny()
ax1bis.plot(rhob, z, 'k')
ax1bis.set_xlabel('RHOB', color='k'),  ax1bis.set_xlim(1.95,2.95)
ax[2].plot(vp1,z,'-k')
ax[2].set_xlabel('Vp')
for aa in ax:
    aa.set_ylim(zbot,ztop)
    aa.grid()

f1=(z>=ztop) & (z<=zbot) & (np.isfinite(vp1))
f2=(z>=ztop) & (z<=2150) & (np.isfinite(vp1))  & (gr>75)

fig,ax = plt.subplots(1,2,figsize=(10,6))
pl0=ax[0].scatter(rhob[f1],vpb[f1],20,c=gr[f1],cmap='rainbow',edgecolors='None',vmin=50,vmax=100)
pl1=ax[1].scatter(rhob[f2],vpb[f2],20,c=gr[f2],cmap='rainbow',edgecolors='None',vmin=50,vmax=100)
for aa in ax:
    aa.set_xlabel('Density [g/cc]'), aa.set_xlim(2.0,2.5)
    aa.set_ylabel('Vp [m/s]'), aa.set_ylim(1800,3500)
    aa.grid()
cax = fig.add_axes([0.2, 0.02, 0.6, 0.025])
fig.colorbar(pl0, cax=cax, orientation='horizontal')
plt.suptitle('Velocity-density (color=GR)', fontsize='x-large')
ax[0].set_title('Depth {}-{}'.format(ztop,zbot))
ax[1].set_title('Depth {}-{}, GR>{}'.format(ztop,2150,75))

popt, pcov = curve_fit(inv_gardner, rhob[f2], vpb[f2])
print(popt)

rho_gardner=np.linspace(1, 3, 50)
vp_gardner=inv_gardner(rho_gardner, *popt)

fig,ax = plt.subplots(1,2,figsize=(10,6))
pl0=ax[0].scatter(rhob[f1],vpb[f1],20,c=gr[f1],cmap='rainbow',edgecolors='None',vmin=50,vmax=100)
pl1=ax[1].scatter(rhob[f2],vpb[f2],20,c=gr[f2],cmap='rainbow',edgecolors='None',vmin=50,vmax=100)
ax[1].plot(rho_gardner, vp_gardner, '-k', lw=2, label=r'$ V_p = (%.2f / \rho)^{1/%.2f}$' % (popt[0],popt[1]))
for aa in ax:
    aa.set_xlabel('Density [g/cc]'), aa.set_xlim(2.0,2.5)
    aa.set_ylabel('Vp [m/s]'), aa.set_ylim(1800,3500)
    aa.grid()
cax = fig.add_axes([0.2, 0.02, 0.6, 0.025])
fig.colorbar(pl0, cax=cax, orientation='horizontal')
ax[1].legend()
plt.suptitle('Velocity-density (color=GR)', fontsize='x-large')
ax[0].set_title('Depth {}-{}'.format(ztop,zbot))
ax[1].set_title('Depth {}-{}, GR>{}'.format(ztop,2150,75))

vp_gardner=inv_gardner(rhob, *popt)  # apply inverse Gardner fit to density log to derive Vp
vp_gardner[~hole]=np.nan             # blanks out all the data outside of the 'hole'
vp_rebuilt=vp1.copy()                # creates a copy of the original Vp (with hole)
vp_rebuilt[hole]=vp_gardner[hole]    # fills the hole with vp_gardner

fig,ax = plt.subplots(1,3,figsize=(12,8))
ax[0].plot(gr,z,'-k')
ax[0].set_xlabel('GR')
ax[1].plot(vp1,z,'-k')
ax[1].plot(vp_gardner,z,'-r')
ax[1].set_xlabel('Vp')
ax[1].set_xlim(2000,3500)
ax[2].plot(vpb,z,'-c')
ax[2].plot(vp_rebuilt,z,'-k')
ax[2].set_xlabel('Vp')
ax[2].set_xlim(2000,3500)
for aa in ax:
    aa.set_ylim(2150,2050)
    aa.grid()
plt.suptitle('Reconstruction of velocity log through Inverse Gardner fit', fontsize='xx-large')

rho_redux=rho_test[::5]
vp_redux=vp_test[::5]

vp_synt_redux=inv_gardner(rho_redux, *popt_synt)

plt.figure(figsize=(6,8))
plt.plot(rho_redux,vp_redux,'ok', ms=10)
plt.plot(rho_synt, vp_synt, '-r', lw=2, label=r'$ V_p = (%.2f / \rho)^{1/%.2f}$' % (popt_synt[0],popt_synt[1]))
plt.plot(rho_redux, vp_synt_redux, '.r', ms=10)

for i in range(rho_redux.size):
    x1,y1=rho_redux[i],vp_redux[i]
    x2,y2=rho_redux[i],vp_synt_redux[i]
    plt.plot([x1,x2],[y1,y2],color='k',ls='--')

plt.xlabel('Density [g/cc]'), plt.xlim(2.2,2.7)
plt.ylabel('Vp [m/s]'), plt.ylim(1500,4500)
plt.legend()

