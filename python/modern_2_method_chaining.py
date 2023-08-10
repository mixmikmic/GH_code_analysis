get_ipython().magic('matplotlib inline')

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='ticks', context='talk')

import prep

def read(fp):
    df = (pd.read_csv(fp)
            .rename(columns=str.lower)
            .drop('unnamed: 36', axis=1)
            .pipe(extract_city_name)
            .pipe(time_to_datetime, ['dep_time', 'arr_time', 'crs_arr_time', 'crs_dep_time'])
            .assign(fl_date=lambda x: pd.to_datetime(x['fl_date']),
                    dest=lambda x: pd.Categorical(x['dest']),
                    origin=lambda x: pd.Categorical(x['origin']),
                    tail_num=lambda x: pd.Categorical(x['tail_num']),
                    unique_carrier=lambda x: pd.Categorical(x['unique_carrier']),
                    cancellation_code=lambda x: pd.Categorical(x['cancellation_code'])))
    return df

def extract_city_name(df):
    '''
    Chicago, IL -> Chicago for origin_city_name and dest_city_name
    '''
    cols = ['origin_city_name', 'dest_city_name']
    city = df[cols].apply(lambda x: x.str.extract("(.*), \w{2}", expand=False))
    df = df.copy()
    df[['origin_city_name', 'dest_city_name']] = city
    return df

def time_to_datetime(df, columns):
    '''
    Combine all time items into datetimes.

    2014-01-01,0914 -> 2014-01-01 09:14:00
    '''
    df = df.copy()
    def converter(col):
        timepart = (col.astype(str)
                       .str.replace('\.0$', '')  # NaNs force float dtype
                       .str.pad(4, fillchar='0'))
        return pd.to_datetime(df['fl_date'] + ' ' +
                               timepart.str.slice(0, 2) + ':' +
                               timepart.str.slice(2, 4),
                               errors='coerce')
    df[columns] = df[columns].apply(converter)
    return df

output = 'data/flights.h5'

if not os.path.exists(output):
    df = read("data/627361791_T_ONTIME.csv")
    df.to_hdf(output, 'flights', format='table')
else:
    df = pd.read_hdf(output, 'flights', format='table')
df.info()

(df.dropna(subset=['dep_time', 'unique_carrier'])
   .loc[df['unique_carrier']
       .isin(df['unique_carrier'].value_counts().index[:5])]
   .set_index('dep_time')
   # TimeGrouper to resample & groupby at once
   .groupby(['unique_carrier', pd.TimeGrouper("H")])
   .fl_num.count()
   .unstack(0)
   .fillna(0)
   .rolling(24)
   .sum()
   .rename_axis("Flights per Day", axis=1)
   .plot()
)
sns.despine()

import statsmodels.api as sm

get_ipython().magic("config InlineBackend.figure_format = 'png'")
flights = (df[['fl_date', 'tail_num', 'dep_time', 'dep_delay', 'distance']]
           .dropna()
           .sort_values('dep_time')
           .assign(turn = lambda x:
                x.groupby(['fl_date', 'tail_num'])
                 .dep_time
                 .transform('rank').astype(int)))

fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(x='turn', y='dep_delay', data=flights, ax=ax)
sns.despine()

plt.figure(figsize=(15, 5))
(df[['fl_date', 'tail_num', 'dep_time', 'dep_delay', 'distance']]
    .dropna()
    .assign(hour=lambda x: x.dep_time.dt.hour)
    .query('5 < dep_delay < 600')
    .pipe((sns.boxplot, 'data'), 'hour', 'dep_delay'))
sns.despine()

planes = df.assign(year=df.fl_date.dt.year).groupby("tail_num")
delay = (planes.agg({"year": "count",
                     "distance": "mean",
                     "arr_delay": "mean"})
               .rename(columns={"distance": "dist",
                                "arr_delay": "delay",
                                "year": "count"})
               .query("count > 20 & dist < 2000"))
delay.head()

X = delay['dist'].values
y = delay['delay']

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

@prep.cached('flights-gp')
def fit():
    kernel = (1.0 * RBF(length_scale=10.0, length_scale_bounds=(1e2, 1e4))
        + WhiteKernel(noise_level=.5, noise_level_bounds=(1e-1, 1e+5)))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X.reshape(-1, 1), y)
    return gp

gp = fit()
X_ = np.linspace(X.min(), X.max(), 1000)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)

ax = delay.plot(kind='scatter', x='dist', y = 'delay', figsize=(12, 6),
                color='k', alpha=.25, s=delay['count'] / 10)

ax.plot(X_, y_mean, lw=2, zorder=9)
ax.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                y_mean + np.sqrt(np.diag(y_cov)),
                alpha=0.25)

sizes = (delay['count'] / 10).round(0)

for area in np.linspace(sizes.min(), sizes.max(), 3).astype(int):
    plt.scatter([], [], c='k', alpha=0.7, s=area,
                label=str(area * 10) + ' flights')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

ax.set_xlim(0, 2100)
ax.set_ylim(-20, 65)
sns.despine()
plt.tight_layout()

