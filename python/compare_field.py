get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from sklearn.neighbors.kde import KernelDensity

from astropy.time import Time
from astropy import units as u

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

# CDS X-Match between GCK and GALEX GR5

gck_gr5 = '1504810534436A.csv'
df = pd.read_csv(gck_gr5)
df.columns

plt.figure(figsize=(8,8))
plt.scatter(df[u'nuv_mag'], df[u'NUVmag'], s=4)
plt.xlim(20,11)
plt.ylim(20,11)
plt.xlabel('GALEX GR5 NUV (mag)')
plt.ylabel('GCK NUV (mag)')

plt.figure(figsize=(8,8))
plt.scatter(df[u'nuv_mag'], df[u'NUVmag'], s=4)
plt.xlim(17,16)
plt.ylim(17,16)
plt.xlabel('GALEX GR5 NUV (mag)')
plt.ylabel('GCK NUV (mag)')

plt.figure(figsize=(8,8))
plt.scatter(df[u'nuv_mag'], df[u'nuv_mag'] - df[u'NUVmag'], s=4)
plt.scatter(16.46, 16.46 - 16.499, s=80, c='r')
plt.xlim(17,16)
plt.ylim(-.2,.2)
plt.xlabel('GALEX GR5 NUV (mag)')
plt.ylabel('GR5 - GCK NUV (mag)')



GCK_near = 'GCK_seach.txt'
GR6_near = 'galex_1846338921.csv'

gck = pd.read_table(GCK_near, delimiter='|', names=('ra', 'dec', 'pl','gck','ra2000','de2000','nuvmag','e_nuvmag',
                                                    'nuvflux','e_nuvflux', 'nuvsn', 'drad','KIC'), comment='#')
gr6 = pd.read_csv(GR6_near)

plt.figure(figsize=(11,11))
plt.scatter(gck['ra'], gck['dec'], alpha=0.5)
plt.scatter(gr6['ra'], gr6['dec'], alpha=0.85,s=2)
plt.plot(301.5644, 44.45684, '+', markersize=50, c='k', alpha=0.5)

# match the datasets within 1arcsecond

mtch = np.zeros(len(gr6['ra'])) - 1

dlim = 1. / 3600.

for k in range(len(gr6['ra'])):
    dist = np.sqrt((gck['ra'].values - gr6['ra'].values[k])**2 + 
                   (gck['dec'].values - gr6['dec'].values[k])**2)
    x = np.where((dist <= dlim))[0]
    
    if len(x) > 0:
        mtch[k] = x[0]


ok = np.where((mtch > -1))[0]

plt.figure(figsize=(8,8))
plt.scatter(gr6['nuv_mag'].values[ok], gck['nuvmag'][mtch[ok]], alpha=0.75)
plt.plot([20,14],[20,14], c='r', alpha=0.5)
plt.xlim(20,14)
plt.ylim(20,14)
plt.xlabel('GALEX GR6 NUV (mag)')
plt.ylabel('GCK NUV (mag)')

plt.figure(figsize=(8,8))
plt.errorbar(gr6['nuv_mag'].values[ok], gck['nuvmag'][mtch[ok]], yerr=gck['e_nuvmag'][mtch[ok]], 
             linestyle='none', marker='o')
plt.plot([20,14],[20,14], c='r', alpha=0.5)
plt.xlim(20,14)
plt.ylim(20,14)
plt.xlabel('GALEX GR6 NUV (mag)')
plt.ylabel('GCK NUV (mag)')
plt.savefig('GCK_GR6.png', dpi=150, bbox_inches='tight', pad_inches=0.25)

print(gck['nuvmag'][mtch[ok]].values[0], gr6['nuv_mag'][ok][0])
print(gr6['nuv_mag'][ok][0] - gck['nuvmag'][mtch[ok]].values[0])

okc = np.where((mtch > -1) & (gr6['nuv_mag'] < 19))[0]

print(np.shape(okc))

_ = plt.hist(gr6['nuv_mag'].values[okc] - gck['nuvmag'][mtch[okc]])
plt.plot( (gr6['nuv_mag'][ok][0] - gck['nuvmag'][mtch[ok]].values[0]) * np.ones(2), [0,2])

plt.figure(figsize=(8,8))
plt.errorbar(gr6['nuv_mag'].values[ok], gr6['nuv_mag'].values[ok] - gck['nuvmag'][mtch[ok]], 
             yerr=gck['e_nuvmag'][mtch[ok]], linestyle='none', marker='o')
plt.xlim(19,14.)
plt.ylim(-.2,.2)
plt.xlabel('GALEX GR6 NUV (mag)')
plt.ylabel('GR6 $-$ GCK NUV (mag)')

X = np.array(gr6['nuv_mag'].values[okc] - gck['nuvmag'][mtch[okc]].values)[:,None]

kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(gck['e_nuvmag'][mtch[okc]]) * 2 ).fit(X)
kde

X_new = np.linspace(-.25,.25,1000)[:, None]
log_dens = np.exp(kde.score_samples(X_new))
plt.plot(X_new[:,0], log_dens)
plt.plot( (gr6['nuv_mag'][ok][0] - gck['nuvmag'][mtch[ok]].values[0]) * np.ones(2), [0,8])
_ = plt.hist(gr6['nuv_mag'].values[okc] - gck['nuvmag'][mtch[okc]], histtype='step')
plt.xlim(-0.14, 0.14)
plt.ylim(0,7.2)
plt.xlabel('GR6 $-$ GCK NUV (mag)')
plt.savefig('hist_diff.png', dpi=150, bbox_inches='tight', pad_inches=0.25)

kde.score_samples(X)



