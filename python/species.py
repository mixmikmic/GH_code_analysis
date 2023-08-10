get_ipython().magic('matplotlib inline')
from __future__ import division
from vis_common import load_frame, STORE
from crawl_data import CANON_SPECIES
from plotting_helpers import xlabel_pct, ylabel_pct, plot_percent
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import IPython.core.display as di

f = load_frame(include=['saw_temple', 'saw_lair', 'saw_vaults', 'saw_zot'])
print "Loaded data frame with {} records and {} columns".format(len(f), len(f.columns))

# Some reusable indices. These are just boolean arrays I can use to index into my data
# frame to get games that were lost, or won, or quit
ilost = STORE['ilost']
iwon = STORE['iwon']
iquit = STORE['iquit']

species = f['species'].cat.categories
drac_species = [sp for sp in species if 'draconian' in sp]
idrac = f['species'].isin(drac_species)
colored_dracs = [sp for sp in drac_species if sp != 'draconian']
cdrac_index = f['species'].isin(colored_dracs)

def get_original_species(sp):
    return 'draconian' if sp in drac_species else sp
    
# The draconian species presents a little problem. When draconians hit level 7, they 
# get a random colour, and their species label changes accordingly. We'd rather lump
# black/red/green/baby draconians all into one species, or we'll get some funny results.
# (For example, the win rate for coloured draconians is very high, because of the survivor
# bias. On the other hand, baby draconians win 0% of the time, because it's basically 
# impossible to win the game before level 7.)
f['orig_species'] = f['species'].map(get_original_species)

pr = (f.groupby('orig_species').size() / len(f)).sort_values()
ax = plot_percent(pr, True, title='Pick rate by species (% of games)', figsize=(10,8));
ax.set_xlabel('% of all games picked');

wr = f.groupby('orig_species')['won'].mean().sort_values()
ax = plot_percent(wr, True, title='Win rate by species', figsize=(10,8))
ax.grid(axis='x');

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,6))
idd = f['species'] == 'deep dwarf' # reusable index
dd_winr = f[idd].groupby('version')['won'].mean()
dd_plays = f[idd].groupby('version').size()
plot_percent(dd_winr, ax=ax1, title="Win rate")
dd_plays.plot.bar(ax=ax2, title="Games played");

s = 25
alpha = 0.3
n = 250
lw=.5
def plot_turn_time(figsize=(8,6)):
    ax = f.loc[iwon & ~idd].head(n)        .plot.scatter(x='time', y='turns', color='red', marker="s", figsize=figsize, loglog=1,
                      label='non-DD winners', alpha=alpha, s=s, lw=lw);
    f.loc[iwon & idd].head(n)        .plot.scatter(x='time', y='turns', color='blue', label='deep dwarf winners',
                      marker="o", alpha=alpha, s=s, lw=lw, ax=ax)

    l, r = 5*10**3, 5 * 10**5
    b, t = 10**4, 10**6
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ax.legend(loc=4);
    ax.set_title("Time taken vs. turns")
    return ax
plot_turn_time();

bots = STORE['bots']
ax = plot_turn_time((10,7))
bots.loc[iwon].head(n)    .plot.scatter(x='time', y='turns', color='green', label='bot winners',
                  marker="^", alpha=alpha, s=s, lw=lw, ax=ax)

l, r = 10**3, 5 * 10**5
b, t = 10**4, 10**6
ax.set_xlim(l, r)
ax.set_ylim(b, t)
for turns_per_second in [1, 2, 4, 8, 16]:
    ax.plot(
        [l, t/turns_per_second],
        [l*turns_per_second, t],
        color='black', lw=lw/2,
    )
ax.legend(loc=4);

from adjustText import adjust_text
sp = pr.index
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(pr.values, wr[sp].values)
texts = [plt.text(pr.loc[species], wr.loc[species], species) for species in sp]
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=.5), force_points=.7, force_text=.3)
ax.set_title("Pick rate vs. Win rate")
ax.set_xlabel('Pick rate')
xlabel_pct(ax)
ax.set_ylabel('Win rate');
ax.set_ylim(bottom=0, top=0.025)
ylabel_pct(ax)

# Kinda slow. Probably a more efficient way to do this, without looping. 
# I wish the pandas group by documentation was better.
dr_given_level = []
ioct = f['species'] == 'octopode'
for lvl in range(1, 28):
    dr_given_level.append( (~ioct & ilost & (f['level'] == lvl)).sum() / 
                          (0.0+(~ioct & (f['level'] >= lvl)).sum()) )
# TODO: download more RAM
import gc; gc.collect()
oc_dr_given_level = []
for lvl in range(1, 28):
    oc_dr_given_level.append( (ilost & (f['level'] == lvl) & ioct).sum() / 
                          (0.0+((f['level'] >= lvl) & ioct).sum()) )
gc.collect();

fig, ax = plt.subplots(figsize=(11,6))
common_kwargs = dict(marker='.', markersize=7, linestyle=':', lw=.6)
ax.plot(np.arange(1,28), dr_given_level, color='brown', label='non-octopodes', **common_kwargs)
ax.plot(np.arange(1,28), oc_dr_given_level, color='b', label='octopodes', **common_kwargs)
ax.legend(loc=9)
ax.set_xlim(1, 27.25);
ax.set_xticks(range(1,28))
ylabel_pct(ax)
ax.set_title("Given that you just reached player level X, what's your chance of dying there?");
ax.grid(axis='both')

extended = f.loc[f['nrunes'] > 5].groupby('orig_species').size()
ax = (extended / f.loc[f['nrunes'] >= 3].groupby('orig_species').size()).sort_values() .plot.barh(title='% of games electing to go into extended', figsize=(8,6));
xlabel_pct(ax)
ax.grid(axis='x');

octo_death_places = f.loc[ilost & (f['species'] == 'octopode') & (f['level'] > 24)].groupby('wheredied').size()
octo_death_places.name = 'octopodes'
human_death_places = f.loc[ilost & (f['species'] == 'kobold') & (f['level'] > 24)].groupby('wheredied').size()
human_death_places.name = 'kobolds'
dp = pd.concat([octo_death_places, human_death_places], axis=1)
print "Where do octopodes/kobolds die after level 24?"
print dp.select(lambda i: dp.loc[i].sum() > 10).sort_values(by='octopodes', ascending=0)

geq_three_runes = f[f['nrunes']>=3].groupby('orig_species').size()

three_rune_wins = f.loc[iwon & (f['nrunes']==3)].groupby('orig_species').size()
three_rune_deathspots = ['dungeon', 'realm of zot', 'depths']
three_rune_win_attempts = (three_rune_wins + 
    f[(f['nrunes']==3) & f['wheredied'].isin(three_rune_deathspots)].groupby('orig_species').size())

zot_success = three_rune_wins / three_rune_win_attempts
ax = plot_percent(zot_success.sort_values(ascending=False), figsize=(11,4), 
             title="What % of 3-rune win attempts succeed, once 3 runes are earned?")
ax.grid(axis='y', lw=.3);

adjusted_wr = (geq_three_runes / f.groupby('orig_species').size()) * zot_success
wr2 = wr
fig, ax = plt.subplots(figsize=(11,11))
x = range(len(wr2.index))
ax.barh(x, wr2.values, label="Raw win rate", color="b", zorder=1)
ax.barh(x, adjusted_wr.loc[wr2.index].values, label="Adjusted win rate", color="crimson", zorder=0)
ax.legend()
ax.set_ylim(-.8, len(x)-.2)
ax.set_yticks(x)
ax.set_yticklabels(wr2.index)
ax.grid(axis='x')
xlabel_pct(ax);

w = .8
fig, ax = plt.subplots(figsize=(11,8))
temple_wr = f[f['saw_temple'] == True].groupby('orig_species')['won'] .mean().dropna()[wr.index]
for label, df, color in [('made it to temple', temple_wr, 'pink'), ('baseline', wr, 'brown')]:
    ax.barh(np.arange(len(df.index)), df.values, w, label=label, color=color)
ax.set_ylim(-w, len(wr))
ax.set_yticks(np.arange(len(wr)))
ax.set_yticklabels(wr.index)
xlabel_pct(ax)
ax.set_title("Base win rate vs. win rate at temple")
ax.grid(axis='x')
ax.legend(loc=4);

width = 1
fig, ax = plt.subplots(figsize=(11,11))
colours = ['brown', 'pink', 'green', 'grey', 'violet']
branches = ['D:1', 'temple', 'lair', 'vaults', 'zot']
# Order species by win rate given vaults
sp = f[f['saw_vaults']==True].groupby('orig_species')['won'].mean().sort_values(ascending=0).index
ranked = []
for branch, colour in reversed(zip(branches, colours)):
    if branch == 'D:1':
        df = f
    else:
        df = f[f['saw_'+branch] == True]
    branch_wr = df.groupby('orig_species')['won'].mean()[sp]
    ax.barh(range(len(branch_wr.index)), branch_wr.values, 
           width, label=branch, color=colour, linewidth=.5, edgecolor='black',
          )
    ranked.append(branch_wr.sort_values(ascending=0).index)
ranked.reverse()
ax.set_title("Win rate given branch reached")
ax.set_yticks(np.arange(len(branch_wr.index)))
ax.set_ylim(-.5, len(branch_wr.index)-.5)
ax.set_yticklabels(branch_wr.index)
xlabel_pct(ax)
ax.legend();

ranked = np.asarray(ranked).T
cm = plt.get_cmap('jet')
color_indices = np.linspace(0, 1, ranked.shape[0])
canon_sort = list(ranked.T[0])
alpha = .3
def color_species(sp):
    # I think maybe this can be accomplished with style.background_gradient, but I wrote this before seeing it.
    i = canon_sort.index(sp)
    ctup = cm(color_indices[i])[:-1]
    csstup = tuple(map(lambda x: int(x*255), ctup)) + (alpha,)
    return ';'.join(['background-color: rgba{}'.format(csstup),
                    'text-align: center',
                    'font-size: 16px',
                     'padding: .3em, .6em',
            ])
                              
df = pd.DataFrame(ranked, columns=branches)
s = df.style    .applymap(color_species)    .set_caption('Species ranked by win rate given milestone reached')

# Hack to avoid obscure rendering issues with the HTML generated by 
# pandas' style.render() (not XHTML compliant) and kramdown
from BeautifulSoup import BeautifulSoup as BS
def sanitize_style(s):
    soup = BS(s.render())
    return soup.prettify()
    
di.display(di.HTML(sanitize_style(s)))

fig, ax = plt.subplots(figsize=(11,7))
colours = [(.2,.2,.2), 'pink', (1,.9,0), 'aliceblue', 'ivory', 'purple', 'green', 'crimson', 'grey']
color_winrates = f[cdrac_index].groupby('species')['won'].mean().dropna().sort_values()
xrange = np.arange(len(colours))
labels = [name.split()[0] for name in color_winrates.index]
bars = ax.bar(xrange, color_winrates.values, 
               color=colours, tick_label=labels, 
              edgecolor='black', lw=1,
              )
bars[1].set_hatch('.')
ylabel_pct(ax)
ax.set_title('Win % per draconian colour');

