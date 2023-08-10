from __future__ import print_function, division

import string
import random
import cPickle as pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import thinkstats2
import thinkplot
import matplotlib.pyplot as plt

import ess

# colors by colorbrewer2.org
BLUE1 = '#a6cee3'
BLUE2 = '#1f78b4'
GREEN1 = '#b2df8a'
GREEN2 = '#33a02c'
PINK = '#fb9a99'
RED = '#e31a1c'
ORANGE1 = '#fdbf6f'
ORANGE2 = '#ff7f00'
PURPLE1 = '#cab2d6'
PURPLE2 = '#6a3d9a'
YELLOW = '#ffff99'
BROWN = '#b15928'

get_ipython().magic('matplotlib inline')

store = pd.HDFStore('ess.resamples.h5')

country_map = ess.make_countries(store)

FORMULA1 = ('netuse_f ~ inwyr07_f + yrbrn60_f + yrbrn60_f2 + '
            'edurank_f + hincrank_f +'
            'tvtot_f + rdtot_f + nwsptot_f + hasrelig_f')

FORMULA2 = ('netuse_f ~ inwyr07_f + yrbrn60_f + yrbrn60_f2 + '
            'edurank_f + hincrank_f +'
            'tvtot_f + rdtot_f + nwsptot_f + rlgdgr_f')

num = 201
ess.process_all_frames(store, country_map, num, 
                       smf.ols, FORMULA1, 1)

ess.process_all_frames(store, country_map, num,
                       smf.ols, FORMULA2, 2)

store.close()

with open('ess5.pkl', 'wb') as fp:
    pickle.dump(country_map, fp)

with open('ess5.pkl', 'rb') as fp:
    country_map = pickle.load(fp)

plot_counter = 1

def save_plot(flag=True):
    """Saves plots in png format.
    
    flag: boolean, whether to save or not
    """
    global plot_counter
    if flag:
        root = 'ess5.%2.2d' % plot_counter
        thinkplot.Save(root=root, formats=['png'])
        plot_counter += 1

xlabel1 = 'Difference in level of Internet use (0-7 scale)'
xlabel2 = 'Difference in level of Internet use (0-7 scale)'

xlim = [-1.0, 3.5]

reload(ess)
t = ess.extract_ranges(country_map, 'yrbrn60_f', 'hasrelig_f')
ess.plot_cis(t, GREEN2)
thinkplot.Config(title='Year born',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'inwyr07_f', 'hasrelig_f')
ess.plot_cis(t, GREEN1)
thinkplot.Config(title='Interview year',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'edurank_f', 'hasrelig_f')
ess.plot_cis(t, ORANGE2)
thinkplot.Config(title='Education (relative rank)',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'hincrank_f', 'hasrelig_f')
ess.plot_cis(t, ORANGE1)
thinkplot.Config(title='Income (relative rank)',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'tvtot_f', 'hasrelig_f')
ess.plot_cis(t, RED)
thinkplot.Config(title='Television watching',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'rdtot_f', 'hasrelig_f')
ess.plot_cis(t, BLUE1)
thinkplot.Config(title='Radio listening',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

t = ess.extract_ranges(country_map, 'nwsptot_f', 'hasrelig_f')
ess.plot_cis(t, BLUE2)
thinkplot.Config(title='Newspaper reading',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

reload(ess)
t = ess.extract_ranges(country_map, 'hasrelig_f', 'hasrelig_f')
ess.plot_cis(t, BROWN)
thinkplot.Config(title='Religious affiliation',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

reload(ess)
cdfnames = ['tvtot_f', 'hasrelig_f', 'rdtot_f', 
            'nwsptot_f', 'hincrank_f', 'edurank_f',
            'inwyr07_f', 'yrbrn60_f']
ess.plot_cdfs(country_map, ess.extract_ranges, cdfnames=cdfnames)
thinkplot.Config(xlabel=xlabel1,
                 xlim=xlim,
                 legend=True,
                 loc='lower right')
save_plot()

xlim = [-1.5, 3.5]

t = ess.extract_ranges2(country_map, 'yrbrn60_f', 'hasrelig_f')
ess.plot_cis(t, GREEN2)
thinkplot.Config(title='Year born',
                 xlabel=xlabel2, xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'inwyr07_f', 'hasrelig_f')
ess.plot_cis(t, GREEN1)
thinkplot.Config(title='Interview year',
                 xlabel=xlabel2, xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'edurank_f', 'rlgdgr_f')
ess.plot_cis(t, ORANGE2)
thinkplot.Config(title='Education rank',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'hincrank_f', 'hasrelig_f')
ess.plot_cis(t, ORANGE1)
thinkplot.Config(title='Income rank',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'tvtot_f', 'hasrelig_f')
ess.plot_cis(t, RED)
thinkplot.Config(title='Television watching',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'rdtot_f', 'hasrelig_f')
ess.plot_cis(t, BLUE1)
thinkplot.Config(title='Radio listening',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'nwsptot_f', 'hasrelig_f')
ess.plot_cis(t, BLUE2)
thinkplot.Config(title='Newspaper reading',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

t = ess.extract_ranges2(country_map, 'rlgdgr_f', 'hasrelig_f')
ess.plot_cis(t, BROWN)
thinkplot.Config(title='Degree of religiosity',
                 xlabel=xlabel2,
                 xlim=xlim)
save_plot()

reload(ess)
cdfnames = ['tvtot_f', 'rlgdgr_f', 'rdtot_f', 
            'nwsptot_f', 'hincrank_f', 'edurank_f',
            'inwyr07_f', 'yrbrn60_f']
ess.plot_cdfs(country_map, ess.extract_ranges2, cdfnames=cdfnames)
thinkplot.Config(xlabel=xlabel2,
                 xlim=xlim,
                 ylabel='CDF',
                 loc='lower right')
save_plot()

reload(ess)
varnames = ['inwyr07_f', 'yrbrn60_f', 'hasrelig_f', 'edurank_f', 
            'tvtot_f', 'hincrank_f', 'rdtot_f', 'nwsptot_f']

ts = ess.make_table(country_map, varnames, ess.extract_ranges)
ess.print_table(ts)

varnames = ['inwyr07_f', 'yrbrn60_f', 'rlgdgr_f', 'edurank_f', 
            'tvtot_f', 'hincrank_f', 'rdtot_f', 'nwsptot_f']

ts = ess.make_table(country_map, varnames, ess.extract_ranges2)
ess.print_table(ts)





