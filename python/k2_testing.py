get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np

t, f = np.loadtxt('k2_full.txt', unpack=True)
#m = np.absolute(t) < 1.5
#t=t[m]; f=f[m]; f_err=f_err[m]
f_err = 0.00026*np.ones_like(f)
plt.plot(t, f, '.');
#plt.xlim(2.45677e6 + 4.5, 2.45677e6+5.5)

from transitfit import LightCurve, Planet, TransitModel

# Here, I am initializing planet with priors on period, epoch 
#  a factor of 10 more generous than the reported values in the paper,
#  in order to be able to fit for it.
epoch = 2454833 + 1942.1659 # from paper
planet = Planet((14.5665,0.02), (epoch, 0.02), 4.73/24)

lc = LightCurve(t, f, f_err, texp=1626./86400, planets=[planet],
               detrend=False, rhostar=(3.92,1.43), 
               dilution=(0.01, 0.005)) #rhostar, dilution from paper
mod = TransitModel(lc, fix_zp=True, width=3)

pars = lc.default_params
mod.plot_planets(pars);

mod.fit_emcee(nburn=300, niter=100);

mod.triangle(['dilution', 'rho', 'b_1', 'rprs_1', 'ecc_1',
              'period_1', 'epoch_1']);

mod.samples[['period_1','epoch_1']].describe()

t0 = mod.samples['epoch_1'].mean()
P = mod.samples['period_1'].mean()
m1 = (t > t0-0.5) & (t < t0+0.5)
m2 = (t > t0+P-0.5) & (t < t0+P+0.5)
m3 = (t > t0+2*P-0.5) & (t < t0 + 2*P+0.5)

fig, axes = plt.subplots(3,1, figsize=(8,12))

for ax,m,n in zip(axes, [m1, m2, m3], [0,1,2]):
    ax.plot(t[m], f[m], '.');
    nsamples = 200
    inds = np.random.randint(len(mod.samples), size=nsamples)
    for _, s in mod.samples.iloc[inds].iterrows():
        fmod = mod.evaluate(tuple(s))
        ax.plot(t[m], fmod[m], 'k', alpha=0.03)
    ax.axvline(t0 + n*P, ls=':', color='k')
    ax.set_xlim(t0+n*P-0.5, t0+n*P+0.5)

lc2 = LightCurve(t, f, f_err, texp=1626./86400, planets=[planet],
                detrend=False, 
                 dilution=(0.01, 0.005)) #dilution from paper, no rhostar
mod2 = TransitModel(lc2, fix_zp=True, fix_circular=True, width=3)

mod2.fit_emcee(nburn=300, niter=100);
mod2.triangle(['dilution', 'rho', 'b_1', 'rprs_1', 'ecc_1',
              'period_1', 'epoch_1']);

mod2.samples[['period_1','epoch_1']].describe()

mod2.save_hdf('k2_circular_model.h5')

