#if needed 
#from __future__ import print_function

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df=pd.read_csv('dengue_features_train.csv')
labels = pd.read_csv('dengue_labels_train.csv')
test = pd.read_csv('dengue_features_test.csv')

df.fillna(method='ffill', inplace=True)

df['week_start_date']=pd.to_datetime(df['week_start_date'])

df['date_shifted_2wk'] = df['week_start_date'].apply(lambda x: x - pd.DateOffset(weeks=2))
df['date_shifted_1wk'] = df['week_start_date'].apply(lambda x: x - pd.DateOffset(weeks=1))

df=pd.merge(df, labels, on=['city', 'year', 'weekofyear'])

plt.figure()
ax = df[['week_start_date', 'total_cases']].plot()
ax.set_ylabel('total cases')

fig,ax = plt.subplots(figsize=(10,5))
for i in np.unique(df.week_start_date.dt.year.values):
    plt.plot(df[df.week_start_date.dt.year==i].week_start_date.T,df[df.week_start_date.dt.year==i].total_cases,label='%d'%(i));
    plt.legend();
    plt.title('Dengue Fever Cases per Year');
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
#plt.savefig('dengue.png')
    ax.legend(bbox_to_anchor=(1.25, 1.05))

### reset axis
df.index = df['week_start_date']
del df['week_start_date']

figs, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 4))

ax = df[df['city']=='sj'].total_cases.plot(ax=axes[0], label="San Juan", color='g')
ax.set_title('SAN JOSE')
ax.set_xticklabels([])
ax.set_xlabel("")

# plot iq
ax1 = df[df['city']=='iq'].total_cases.plot(ax=axes[1], label="Iquitos")
ax1.set_title('IQUITOS')
ax1.set_xlabel("YEAR")

plt.suptitle("DENGUE ACTUAL CASES")

pv = pd.pivot_table(df, index=df.index.year, columns=df.city,
                    values='total_cases', aggfunc='sum')

pv.plot()

pv2 = pd.pivot_table(df, index=df.index.month, columns=df.city,
                    values='total_cases', aggfunc='sum')
pv2

pv2.plot()

pv3 = pd.pivot_table(df, index=df.index.month, columns=df.index.year,
                    values='total_cases', aggfunc='sum')

fig = plt.figure()
ax = pv3.plot()

ax.legend(bbox_to_anchor=(1.25, 1.05))

df.groupby(['year']).total_cases.sum().plot(kind='bar')

#a look at the worst year - 1994

fig,ax = plt.subplots(figsize=(10,5))
for i in np.unique(df.year.values):
    if df[df.index.year==i].total_cases.mean() >= 100:
    #df[df.year==i].total_cases > 850:

        plt.plot(df[df.year==i].total_cases,label='%d'%(i));
        plt.legend();
        plt.title('Dengue Fever Cases per Year');
        plt.xlabel('Week of Year')
        plt.ylabel('Number of Cases')
    

weather_cols=[
 #'city',
 #'week_start_date',
 #'date_shifted_2wk',
 #'date_shifted_1wk',
 'total_cases',
 'station_avg_temp_c',
 'station_diur_temp_rng_c',
 'station_max_temp_c',
 'station_min_temp_c',
 'station_precip_mm']

precipitation_cols=[
  'city',
 'week_start_date',
 'date_shifted_2wk',
 'date_shifted_1wk',
 'total_cases',
 'reanalysis_air_temp_k',
 'reanalysis_avg_temp_k',
 'reanalysis_dew_point_temp_k',
 'reanalysis_max_air_temp_k',
 'reanalysis_min_air_temp_k',
 'reanalysis_precip_amt_kg_per_m2',
 'reanalysis_relative_humidity_percent',
 'reanalysis_sat_precip_amt_mm',
 'reanalysis_specific_humidity_g_per_kg',
 'reanalysis_tdtr_k',
]

vegetation_cols=[
  'city',
 'week_start_date',
 'date_shifted_2wk',
 'date_shifted_1wk',
 'total_cases',
 'ndvi_ne',
 'ndvi_nw',
 'ndvi_se',
 'ndvi_sw'
]

df['ndvi_ne'][0:10]

df['ndvi_ne'].plot()

# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
ax1.plot(df['ndvi_ne'])
ax1.set_title('NDVI NE')
ax2.plot(df['ndvi_nw'])
ax2.set_title('NDVI NW')
ax3.plot(df['ndvi_se'])
ax3.set_title('NDVI SE')
ax4.plot(df['ndvi_sw'])
ax4.set_title('NDVI SW')

df['ndvi_ne'].min()

df['ndvi_all_directions']=(df['ndvi_ne']+df['ndvi_nw']+df['ndvi_se']+df['ndvi_sw'])/4

df[['ndvi_all_directions','ndvi_se','ndvi_sw','ndvi_ne','ndvi_nw','ndvi_desc']].head()

def get_ndvi_category(x):
    x=float(x)
    if x < 0.1: return 'water'
    if x >= 0.1 and x <0.2: return 'rock/sand'
    if x >=0.2 and x<0.5 : return 'shrub/grassland'
    if x > 0.6: return 'forest'
    return 'other'

df['ndvi_desc'] = df.ndvi_all_directions.apply(get_ndvi_category)

df['ndvi_desc'].value_counts().plot(kind='bar')

df[df['city']=='sj'].ndvi_desc.value_counts().plot(kind='bar')

df[df['city']=='iq'].ndvi_desc.value_counts().plot(kind='bar')

#compare both precipitation columns
df[['reanalysis_sat_precip_amt_mm','station_precip_mm']].plot()

