get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

from ipywidgets import interact
from IPython import display

from flight_safety.queries import (get_events_accidents, 
        get_aircrafts_accidents, get_occurrences_accidents,
        get_seq_of_events_accidents, get_flight_time_accidents,get_events_all, get_aircrafts_all,
        get_flight_crew_accidents)

mpl.rcParams['figure.figsize'] = 10, 6

mpl.rcParams['font.size'] = 20

con = sqlite3.connect('data/avall.db')

events = get_events_accidents(con)

aircraft = get_aircrafts_accidents(con)

occurrences = get_occurrences_accidents(con)

seq_of_events = get_seq_of_events_accidents(con)

flight_time = get_flight_time_accidents(con)

events_all = get_events_all(con)

aircraft_all = get_aircrafts_all(con)

aircraft_events_all = pd.merge(aircraft_all, events_all, right_index = True, left_on ='ev_id')

cond_incid = aircraft_events_all['ev_type'] == 'INC'
cond_accd  = aircraft_events_all['ev_type'] == 'ACC'
cond_far  = aircraft_events_all.far_part.isin(['121 ', '125 '])

try:
    occurrences.phase_flt_spec_gross.cat.add_categories('TOTAL', inplace=True)
except ValueError:
    pass

phases_per_occurence = pd.crosstab(occurrences.Occurrence_Code, occurrences.phase_flt_spec_gross)
phases_per_occurence['TOTAL'] = phases_per_occurence.sum(axis=1)
phases_per_occurence.sort_values('TOTAL', inplace=True, ascending=False)
phases_per_occurence.iloc[:10].iloc[:, 1:-1]

phases_per_occurence.iloc[:10].loc[:, 'TOTAL'].plot.barh();

aux = occurrences.Occurrence_Code.value_counts().iloc[0:10]
aux
occurrences_red = occurrences[occurrences['Occurrence_Code'].isin(aux.index)]

occurrences_per_phase = pd.crosstab(occurrences_red.phase_flt_spec_gross, occurrences_red.Occurrence_Code)
occurrences_per_phase = occurrences_per_phase.loc[['Standing', 'Taxi', 'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach',
       'Landing', 'Maneuvering']]

occurrences_per_phase

plt.figure(figsize=(20, 20))
occurrences_per_phase_ = occurrences_per_phase.iloc
plt.matshow(occurrences_per_phase.values, cmap=plt.cm.Blues)
plt.xticks(np.arange(occurrences_per_phase.shape[1]),
           occurrences_per_phase.columns.tolist(),
           rotation='vertical')
plt.yticks(np.arange(occurrences_per_phase.shape[0]),
           occurrences_per_phase.index.tolist());

occurrences_ = pd.merge(occurrences, events, left_on='ev_id', right_index=True)

inj_cols = ['inj_tot_f', 'inj_tot_s', 'inj_tot_m', 'inj_tot_n', 'inj_tot_t']

inj_per_occ = occurrences_.groupby('Occurrence_Code')[inj_cols].sum()
inj_per_occ.sort_values('inj_tot_t', inplace=True, ascending=False)

inj_per_occ.loc[inj_per_occ.index[:10], ['inj_tot_t', 'inj_tot_n']].plot.barh(stacked=True);

inj_per_occ['inj_tot_s+m'] = inj_per_occ['inj_tot_s'] + inj_per_occ['inj_tot_m']

inj_per_occ.loc[inj_per_occ.index[:10], ['inj_tot_f', 'inj_tot_s+m']].plot.barh(stacked=True);

try:
    aircraft.damage.cat.add_categories(['TOTAL'], inplace=True)
except ValueError:
    pass

damage_per_Occurrence = pd.crosstab(occurrences.Occurrence_Code, aircraft.damage)
damage_per_Occurrence['TOTAL'] = damage_per_Occurrence.sum(axis=1)
damage_per_Occurrence.sort_values('TOTAL', inplace=True, ascending=False)
damage_per_Occurrence.head()

damage_per_Occurrence.loc[damage_per_Occurrence.index[1:10], ['DEST', 'SUBS', 'MINR', 'NONE']].plot.barh(stacked=True);

# Influence of fire in an accident

aircraft_with_phase = aircraft[aircraft.phase_flt_spec_gross != 0]

# f: fatal
# m: medium
# n: none
# s: serious
# t: f+s+m

# TODO: borrar injuries
injury_types = ['inj_tot_f', 'inj_tot_s', 'inj_tot_m', 'inj_tot_n', 'inj_tot_t']

aircraft_2 = aircraft_with_phase.join(events, on='ev_id', how='inner', rsuffix='e')

injuries_per_phase = aircraft_2[injury_types + ['phase_flt_spec_gross']].groupby('phase_flt_spec_gross').sum()
injuries_per_phase.sort_values('inj_tot_t', inplace=True, ascending=False)
injuries_per_phase = injuries_per_phase.iloc[:10]
injuries_per_phase['inj_tot_s+m'] = injuries_per_phase['inj_tot_s'] + injuries_per_phase['inj_tot_m']
injuries_per_phase

def tot_events_and_mean_injuries_by_factor(factor):
    ac_ = aircraft_2[[
    'ev_id',
    'inj_tot_f', 'inj_tot_m', 'inj_tot_n', 'inj_tot_s', 'inj_tot_t',
    'apt_dist', 'apt_dir', 'light_cond', 'sky_cond_nonceil',
    'gust_ind', 'gust_kts',
    'damage', 'acft_fire', 'acft_expl', 'acft_make', 'acft_model', 'acft_category',
    'afm_hrs', 'afm_hrs_last_insp', 'num_eng', 'far_part', 
    ]]
    
    gby = ac_.groupby([factor])
    r = gby.agg({'ev_id': 'count',
                 'inj_tot_f':'mean',
                 'inj_tot_m':'mean',
                 'inj_tot_n':'mean',
                 'inj_tot_s':'mean',
                 'inj_tot_t':'mean'}
           )
    
    if factor in ('acft_fire', 'acft_expl'):
        r = r.loc[['GRD ', 'IFLT', 'NONE', 'UNK ']]
    elif factor in ('gust_ind'):
        r = r.loc[['N', 'Y']]
    
    return r.loc[:, ['ev_id', 'inj_tot_f']]

# EN acft_fire u acft_expl drop blaco y BOTH
# EN gust_ind drop blanco
# EVALUAR light_cond y sky_con_nonceil

tot_events_and_mean_injuries_by_factor('acft_fire')



