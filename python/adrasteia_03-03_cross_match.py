#! cat /Users/gully/.ipython/profile_default/startup/start.ipy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord

d1 = pd.read_csv('../../ApJdataFrames/data/Luhman2012/tbl1_plusSimbad.csv') #local version

d1 = d1[~d1.RA.isnull()]

d1.columns

c1 = SkyCoord(d1.RA.values, d1.DEC.values, unit=(u.hourangle, u.deg), frame='icrs')

df_list = []

for i in range(16):
    df_list.append(pd.read_csv('../data/TgasSource_000-000-{:03d}.csv'.format(i)))

tt = pd.concat(df_list, ignore_index=True)

plt.figure(figsize=(10,4))
ax = sns.jointplot(tt.ra, tt.dec, kind='hex', size=8)
ax.ax_joint.plot(c1.ra.deg, c1.dec.deg, '.', alpha=0.5)

cg = SkyCoord(tt.ra.values, tt.dec.values, unit=(u.deg, u.deg), frame='icrs')

idx, d2d, blah = c1.match_to_catalog_sky(cg)

vec_units = d2d.to(u.arcsecond)
vec = vec_units.value

bins = np.arange(0, 4, 0.2)
sns.distplot(vec, bins=bins, kde=False),
plt.xlim(0,4)
plt.xlabel('match separation (arcsec)')

len(set(idx)), idx.shape[0]

tt_sub = tt.iloc[idx]
tt_sub = tt_sub.reset_index()
tt_sub = tt_sub.drop('index', axis=1)

d1 = d1.reset_index()
d1 = d1.drop('index', axis=1)

x1 = pd.concat([d1, tt_sub], axis=1)

x1.shape

col_order = d1.columns.values.tolist() + tt_sub.columns.values.tolist()
x1 = x1[col_order]
x0 = x1.copy()

x0['xmatch_sep_as'] = vec

x0['Gaia_match'] = vec < 2.0 #Fairly liberal, 1.0 might be better.

plt.figure(figsize=(8,4))
bins = np.arange(2, 14, 0.2)
sns.distplot(x0.parallax[x0.Gaia_match], bins=bins)
#sns.distplot(1.0/(x0.parallax[x0.Gaia_match]/1000.0))
plt.xlabel('Parallax (mas)')
plt.savefig('../results/luhman_mamajek2012.png', dpi=300)

x0.Gaia_match.sum(), len(d1)

plt.figure(figsize=(10,4))
ax = sns.jointplot(tt.ra, tt.dec, kind='hex', size=8, xlim=(230,255), ylim=(-40,-10))
ax.ax_joint.plot(c1.ra.deg, c1.dec.deg, '.', alpha=0.5)
ax.ax_joint.scatter(x0.ra[x0.Gaia_match], x0.dec[x0.Gaia_match], 
                    s=x0.parallax[x0.Gaia_match]**3*0.2, c='r',alpha=0.5)

