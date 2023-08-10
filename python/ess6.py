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

FORMULA1 = ('treatment ~ inwyr07_f + yrbrn60_f + yrbrn60_f2 + '
            'edurank_f + hincrank_f +'
            'tvtot_f + rdtot_f + nwsptot_f')

def compute_delta(group, country):
    group['yrbrn60_f2'] = group.yrbrn60_f ** 2
    group['propensity'] = np.nan
    group['treatment'] = np.nan
        
    # quantize netuse to get treatment variable
    # choose threshold close to the median
    netuse = group.netuse_f
    thresh = netuse.median()
    if thresh < 1:
        thresh = 1
    group.treatment = (netuse >= thresh).astype(int)

    # compute propensities
    model = smf.logit(FORMULA1, data=group)    
    results = model.fit(disp=False)
    group.propensity = results.predict(group)
    
    # divide into treatment and control groups
    treatment = group[group.treatment == 1]
    control = group[group.treatment == 0]
    
    # sort the propensities of the controls (for fast lookup)
    series = control.propensity.sort_values()

    # look up the propensities of the treatment group
    # to find (approx) closest matches in the control group
    indices = series.searchsorted(treatment.propensity)
    indices[indices < 0] = 0
    indices[indices >= len(control)] = len(control)-1
    
    # use the indices to select the matches
    control_indices = series.index[indices]
    matches = control.loc[control_indices]

    # find distances and differences
    distances = (treatment.propensity.values - 
                 matches.propensity.values)
    differences = (treatment.rlgdgr_f.values - 
                   matches.rlgdgr_f.values)
    
    # select differences with small distances
    caliper = differences[abs(distances) < 0.001]

    # return the mean difference
    delta = np.mean(caliper)
    return delta

def process_frame(df, country_map):
    grouped = df.groupby('cntry')
    for code, group in grouped:
        country = country_map[code]

        # compute mean difference between matched pairs
        delta = compute_delta(group, country)
        d = dict(delta=delta)
        country.add_params(d)

def process_all_frames(store, country_map, num=201):
    """Loops through the store and processes frames.
    
    store: store
    country_map: map from code to Country
    num: how many resamplings to process
    reg_func: function used to compute regression
    formula: string Patsy formula
    model_num: which model we're running
    """
    for i, key in enumerate(store.keys()):
        if i >= num:
            break
        print(i, key)
        df = store.get(key)
        process_frame(df, country_map)

process_all_frames(store, country_map, num=101)

store.close()

with open('ess6.pkl', 'wb') as fp:
    pickle.dump(country_map, fp)

with open('ess6.pkl', 'rb') as fp:
    country_map = pickle.load(fp)

len(country_map['DE'].param_seq)

plot_counter = 1

def save_plot(flag=False):
    """Saves plots in png format.
    
    flag: boolean, whether to save or not
    """
    global plot_counter
    if flag:
        root = 'ess6.%2.2d' % plot_counter
        thinkplot.Save(root=root, formats=['png'])
        plot_counter += 1

xlabel1 = 'Difference in religiosity (10 point scale)'

xlim = [-2.5, 1.0]

reload(ess)
t = ess.extract_vars(country_map, 'delta', None)
ess.plot_cis(t, PURPLE2)
thinkplot.Config(title='Internet use',
                 xlabel=xlabel1, xlim=xlim)
save_plot()

reload(ess)
cdfnames = ['delta']
ess.plot_cdfs(country_map, ess.extract_vars, cdfnames=cdfnames)
thinkplot.Config(xlabel=xlabel1,
                 xlim=xlim,
                 ylabel='CDF',
                 loc='upper left')
save_plot()

reload(ess)
varnames = ['delta']

ts = ess.make_table(country_map, varnames, ess.extract_vars)
ess.print_table(ts)





