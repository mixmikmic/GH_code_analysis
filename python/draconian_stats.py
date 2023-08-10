get_ipython().magic('matplotlib inline')
from __future__ import division
from vis_common import load_frame, STORE
from crawl_data import CANON_SPECIES
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

f = load_frame(include=[])
ilost = STORE['ilost']
iwon = STORE['iwon']
iquit = STORE['iquit']

species = f['species'].cat.categories
drac_species = [sp for sp in species if 'draconian' in sp]
idrac = f['species'].isin(drac_species)
colored_dracs = [sp for sp in drac_species if sp != 'draconian']
cdrac_index = f['species'].isin(colored_dracs)

fig, ax = plt.subplots(figsize=(11,7))
colours = [(.2,.2,.2), 'pink', (1,.9,0), 'aliceblue', 'ivory', 'purple', 'green', 'crimson', 'grey']
color_winrates = (f[cdrac_index].groupby('species')['won'].mean().dropna() * 100).sort_values()
xrange = np.arange(len(colours))
labels = [name.split()[0] for name in color_winrates.index]
bars = ax.bar(xrange, color_winrates.values, 
               color=colours, tick_label=labels, 
              edgecolor='black', lw=1,
              )
bars[1].set_hatch('.')
ax.set_title('Win % per draconian colour');

games = f[cdrac_index].groupby('species').size()[colored_dracs].rename('games played')
wins = f.loc[cdrac_index & iwon].groupby('species').size()[colored_dracs].rename('games won')
pct = (100 * wins/games).rename('win %')
drac_table = pd.concat([games, wins, pct], axis=1).sort_values('win %', ascending=0)
dt2 = drac_table.copy()
dt2.index = dt2.index.astype('object')
dt2 = dt2.append(drac_table.sum().rename('Overall'))
ov = dt2.loc['Overall']
ov['win %'] = ov['games won'] * 100 / ov['games played']
cdrac_wr = ov['win %'] / 100
print dt2
# TODO: number formatting

from scipy.stats import binom_test

def binom_pvalue(row):
    played, won = row[['games played', 'games won']]
    return binom_test(won, played, cdrac_wr, alternative='two-sided')

drac_table.apply(lambda d: binom_pvalue(d), axis=1).rename('p-value')

from scipy.stats import chi2_contingency

chi2, p, dof, _ = chi2_contingency(
    drac_table[ ['games played', 'games won']].values
)
print "chi^2 = {}\tp-value = {}\tdegrees of freedom={}".format(chi2, p, dof)

import math
from scipy.stats import norm
def z_test(x1, n1, x2, n2):
    p1 = x1/n1
    p2 = x2/n2
    pbar = (x1+x2)/(n1+n2)
    qbar = (1 - pbar)
    numerator = p1 - p2
    denom = math.sqrt(pbar * qbar * (1/n1 + 1/n2))
    return numerator/denom

n1, x1 = drac_table[['games played', 'games won']].loc['grey draconian']
std_norm = norm()


def ztest_pvalue(row):
    n2, x2 = row[['games played', 'games won']]
    z = z_test(x1, n1, x2, n2)
    return std_norm.cdf(-abs(z)) * 2 # times 2 for 2-tailed test. Which I assume is appropriate.

drac_table.apply(lambda d: ztest_pvalue(d), axis=1).rename('p-value')

def ztest_pair_pvalue(row):
    n1, x1, n2, x2 = row[['games played_x', 'games won_x', 'games played_y', 'games won_y']]
    z = z_test(x1, n1, x2, n2)
    return std_norm.cdf(-abs(z)) * 2

def holm_bonferonni(row, m):
    alpha = .05
    rank, p = row[['rank', 'p']]
    thresh = alpha / (m-rank)
    return thresh

drac1 = drac_table[['games played', 'games won']].copy()
drac1['tmp'] = 1
drac1['species'] = drac_table.index
drac2 = drac1.copy()
pairs = pd.merge(drac1, drac2, on='tmp')
# One row for each distinct pair with non-identical species
iuniq = pairs['species_x'] < pairs['species_y']
pairs = pairs.loc[iuniq]
paired_p = pairs.apply(lambda d: ztest_pair_pvalue(d), axis=1).rename('p')
paired_p.sort_values(inplace=True)
paired_p = paired_p.to_frame()
paired_p['rank'] = range(len(paired_p))
paired_p['thresh'] = paired_p.apply(lambda d: holm_bonferonni(d, len(paired_p)), axis=1)
pairs.loc[paired_p.index][['species_x', 'species_y']]
under = paired_p['p'] <= paired_p['thresh']
print "Significant pairs...\n"
print pairs.loc[under[under].index][['species_x', 'species_y']]

# Output from `!lg * recentish dr s=race / won o=%`
sequell_data = [
    ('Grey', 146, 3407),
    ('Green', 89, 3386),
    ('Red', 90, 3457),
    ('Purple', 85, 3319),
    ('White', 77, 3486),
    ('Pale', 71, 3456),
    ('Black', 66, 3452),
    ('Yellow', 65, 3446),
    ('Mottled', 46, 3101),
]

ps = []
for d1, w1, n1 in sequell_data:
    for d2, w2, n2 in sequell_data:
        if d1 <= d2:
            continue
        z = z_test(w1, n1, w2, n2)
        p = std_norm.cdf(-abs(z)) *2
        ps.append([p, d1, d2])
        
ps.sort()
for i, (p, d1, d2) in enumerate(ps):
    alpha = .05
    thresh = alpha / (len(ps) - i)
    if p < thresh:
        print '{}\t{}\tp = {}'.format(d1, d2, p)

print 'p = {:.4f}'.format(
    paired_p['p'].loc[(pairs['species_x']=='red draconian') & (pairs['species_y']=='yellow draconian')].values[0]
)

from bin_conf import binconf

err_low = drac_table.apply(
    lambda row: binconf(row['games won'], (row['games played']-row['games won']), c=.95)[0],
    axis=1) * 100
err_high = drac_table.apply(
    lambda row: binconf(row['games won'], (row['games played']-row['games won']), c=.95)[1],
    axis=1) * 100
cwr = color_winrates
# matplotlib wants offsets, not positions
err_low = color_winrates - err_low[cwr.index]
err_high = err_high[cwr.index] - color_winrates

fig, ax = plt.subplots(figsize=(11,7))
colours = [(.2,.2,.2), 'pink', (1,.9,0), 'aliceblue', 'ivory', 'purple', 'green', 'crimson', 'grey']
cdrac_index = f['species'].isin(colored_dracs)
color_winrates = (f[cdrac_index].groupby('species')['won'].mean().dropna() * 100).sort_values()
xrange = np.arange(len(colours))
labels = [name.split()[0] for name in color_winrates.index]
bars = ax.bar(xrange, color_winrates.values, 
               color=colours, tick_label=labels, 
              edgecolor='black', lw=1,
              yerr=[err_low[color_winrates.index].values, err_high[color_winrates.index].values],
              )
bars[1].set_hatch('.')
ax.set_title('Win rate per draconian colour (with 95% confidence intervals)');
ax.grid(axis='y');

