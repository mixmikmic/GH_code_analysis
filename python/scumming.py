get_ipython().magic('matplotlib inline')
from __future__ import division
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from vis_common import load_store, load_games

from crawl_data import CANON_SPECIES, CANON_BGS

f = load_games()
print "Loaded data frame with {} records and {} columns".format(
    len(f), len(f.columns))

ilost = ~f['won']
iwon = f['won']
iquit = f['howdied'] == 'quit'

FS = (10, 6) # Reasonable default figsize

species = f['species'].cat.categories
drac_species = [sp for sp in species if 'draconian' in sp]
idrac = f['species'].isin(drac_species)

f.columns

# Temple scumming!
f.groupby('temple_xl').size().plot.bar(figsize=FS);



from matplotlib import colors
fig, ax = plt.subplots(figsize=FS)
df = f[~f['temple_depth'].isnull()].head(100000)
x = df['temple_depth'].values
y = df['temple_xl'].values
max_plvl =  12
bins = [[3.5, 4.5,5.5,6.5, 7.5], np.arange(.5,max_plvl+1.5, 1)]
norm = colors.PowerNorm(.5)
plt.hist2d(x, y, bins, 
          norm=norm,
           #normed=True,
           #ax=ax,
         );
ax.set_xlabel('depth')
ax.set_ylabel('player level')
plt.colorbar(norm=norm);
ax.set_xticks([4, 5, 6, 7]);
ax.set_yticks(range(1,max_plvl+1));
# TODO: take an equal number per depth

zealots = ['berserker', 'chaos knight', 'abyssal knight']
izealot = f['bg'].isin(zealots)
f[~izealot].groupby('temple_depth')['won'].sum().plot.bar(figsize=FS);
# I am actually amazed by this

f[~izealot].groupby('temple_depth')['won'].sum()

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from numpy.random import randn

#normal distribution center at x=0 and y=5
x = randn(100000)
y = randn(100000)+5

H, xedges, yedges, img = plt.hist2d(x, y, norm=LogNorm())
extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(H, cmap=plt.cm.jet, extent=extent, norm=LogNorm())
fig.colorbar(im, ax=ax)
plt.show()



f['howdied'].cat.categories

death_lvls = f.loc[ilost & idrac].groupby('level').size()
death_lvls.plot.bar(title='Draconian deaths by level');
# No peak at lvl 7. Guess players really aren't scumming drac colours.
# But actually looking specifically at the quitting numbers below contradicts that. Why isn't
# this phenom visible in this graph? Hypothesis: the quitters do tend to increase the death rate
# at level 7, but those who stick with their drac get a new power at level 7 that increases their
# survivability, which acts as a countervaling force
# TODO: Some day I'll understand matplotlib primitives well enough to do stuff like putting plots
# next to each other.

f.loc[ilost].groupby('level').size().plot.bar(title='Deaths by player level');

# Load the 'raw' data frame (which doesn't exclude games that were quit at level 1)
fr = load_frame(raw=True)
idrac_raw = fr['species'].isin(drac_species)

fr.loc[iquit & idrac_raw].groupby('level').size()    .plot.bar(title='Draconian quitters by level');

drac_quits_per_lvl = (fr.loc[iquit & idrac_raw].groupby('level').size() /
    idrac_raw.sum())
nondrac_quits_per_lvl = (fr.loc[iquit & ~idrac_raw].groupby('level').size() /
    (~idrac_raw).sum())
# TODO: write a function for this pattern
w = .35
PLVLS = np.arange(1,28)
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(drac_quits_per_lvl.index, drac_quits_per_lvl.values, w, label='draconians', color='r')
ax.bar(nondrac_quits_per_lvl.index+w, nondrac_quits_per_lvl.values, w, label='non-draconians', color='b')
ax.legend()
ax.set_xlim(1,10);
ax.set_xticks(range(1,10));
ax.set_title('Quits per player level (normalized by # games)')
# AHA! There are draconian colour scummers out there!

fr['orig_species'] = fr['species'].map(get_original_species)
quits_per_sp = fr.loc[iquit].groupby('orig_species').size()[CANON_SPECIES].dropna()
games_per_sp = fr.groupby('orig_species').size()[CANON_SPECIES].dropna()
quit_rate_per_sp = 100 * quits_per_sp / games_per_sp
quit_rate_per_sp.sort_values(ascending=0).plot.bar(title='% of games quit per species');

# What about per background?
quits_per_bg = fr.loc[iquit].groupby('bg').size()[CANON_BGS].dropna()
games_per_bg = fr.groupby('bg').size()[CANON_BGS].dropna()
quit_rate_per_bg = 100 * quits_per_bg / games_per_bg
quit_rate_per_bg.sort_values(ascending=0).plot.bar(title='% of games quit per background');
# Wanderer scumming!

# What about DS scummers?
#fr.loc[iquit & (fr['species'] == 'demonspawn')].groupby('level').size().plot.bar();
# TODO: compare with all species/non-DS
# TODO: win rate of monstrous DS vs. non-monstrous

ids = fr['species'] == 'demonspawn'
ds_quits_per_lvl = (100 * fr.loc[iquit & ids].groupby('level').size() /
    ids.sum())
nonds_quits_per_lvl = (100 * fr.loc[iquit & ~ids].groupby('level').size() /
    (~ids).sum())
# TODO: write a function for this pattern
w = .35
PLVLS = np.arange(1,28)
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(ds_quits_per_lvl.index, ds_quits_per_lvl.values, w, label='demonspawn', color='r')
ax.bar(nonds_quits_per_lvl.index+w, nonds_quits_per_lvl.values, w, label='non-demonspawn', color='b')
ax.legend()
ax.set_xlim(1,10);
ax.set_xticks(range(1,10));
ax.set_title('Quits per player level (as % of all games)')
del ids

f[f['howdied']=='starved'].groupby('level').size().plot.bar();

print "{} games out of {} ended in starvation".format(
    (f['howdied']=='starved').sum(),
    len(f)
)

(f[f['howdied']=='starved'].groupby('version').size() /
 f.groupby('version').size())\
    .plot.bar();
    
# okay, .1 and .11 did have the most starvation deaths, proportionally, but not by a large margin

# Okay, which colours are people really sad to get?
f.loc[iquit & (f['level'] == 7) & idrac].groupby('species').size()    .where(lambda x: x > 0).dropna().sort_values().plot.bar(title='Draconian level 7 quitters');
# Not normalized, but that's fine since there's very little spread in the number of games per colour

