get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

trips = pd.read_csv('2015_trip_data.csv',
                    parse_dates=['starttime', 'stoptime'],
                    infer_datetime_format=True)

ind = pd.DatetimeIndex(trips.starttime)
trips['date'] = ind.date.astype('datetime64')
trips['hour'] = ind.hour

hourly = trips.pivot_table('trip_id', aggfunc='count',
                           index=['usertype', 'date'], columns='hour').fillna(0)
hourly.head()

from sklearn.decomposition import PCA
data = hourly[np.arange(24)].values
data_pca = PCA(2).fit_transform(data)
hourly['projection1'], hourly['projection2'] = data_pca.T

hourly['total rides'] = hourly.sum(axis=1)

hourly.plot('projection1', 'projection2', kind='scatter', c='total rides', cmap='Blues_r');

plt.savefig('figs/pca_raw.png', bbox_inches='tight')

from sklearn.mixture import GMM
gmm = GMM(3, covariance_type='full', random_state=2)
data = hourly[['projection1', 'projection2']]
gmm.fit(data)

# require high-probability cluster membership
hourly['cluster'] = (gmm.predict_proba(data)[:, 0] > 0.6).astype(int)

from datetime import time
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.1)
times = pd.date_range('0:00', '23:59', freq='H').time
times = np.hstack([times, time(23, 59, 59)])

hourly.plot('projection1', 'projection2', c='cluster', kind='scatter', 
            cmap='rainbow', colorbar=False, ax=ax[0]);

for i in range(2):
    vals = hourly.query("cluster == " + str(i))[np.arange(24)]
    vals[24] = vals[0]
    ax[1].plot(times, vals.T, color=plt.cm.rainbow(255 * i), alpha=0.05, lw=0.5)
    ax[1].plot(times, vals.mean(0), color=plt.cm.rainbow(255 * i), lw=3)
    ax[1].set_xticks(4 * 60 * 60 * np.arange(6))
    
ax[1].set_ylim(0, 60);
ax[1].set_ylabel('Rides per hour');

fig.savefig('figs/pca_clustering.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.05)

for i, col in enumerate(['Annual Member', 'Short-Term Pass Holder']):
    hourly.loc[col].plot('projection1', 'projection2',  c='cluster', kind='scatter', 
                         cmap='rainbow', colorbar=False, ax=ax[i]);
    ax[i].set_title(col + 's')
    
fig.savefig('figs/pca_annual_vs_shortterm.png', bbox_inches='tight')

usertype = hourly.index.get_level_values('usertype')
weekday = hourly.index.get_level_values('date').dayofweek < 5
hourly['commute'] = (weekday & (usertype == "Annual Member"))

fig, ax = plt.subplots()

hourly.plot('projection1', 'projection2', c='commute', kind='scatter', 
            cmap='binary', colorbar=False, ax=ax);

ax.set_title("Annual Member Weekdays vs Other")

fig.savefig('figs/pca_true_weekends.png', bbox_inches='tight')

mismatch = hourly.query('cluster == 0 & commute')
mismatch = mismatch.reset_index('usertype')[['usertype', 'projection1', 'projection2']]
mismatch

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2014-08', '2015-10', return_name=True)
holidays_all = pd.concat([holidays,
                          "2 Days Before " + holidays.shift(-2, 'D'),
                          "Day Before " + holidays.shift(-1, 'D'),
                          "Day After " + holidays.shift(1, 'D')])
holidays_all = holidays_all.sort_index()
holidays_all.head()

holidays_all.name = 'holiday name'  # required for join
joined = mismatch.join(holidays_all)
joined['holiday name']

set(holidays) - set(joined['holiday name'])

fig, ax = plt.subplots()

hourly.plot('projection1', 'projection2', c='cluster', kind='scatter', 
            cmap='binary', colorbar=False, ax=ax);

ax.set_title("Holidays in Projected Results")

for i, ind in enumerate(joined.sort_values('projection1').index):
    x, y = hourly.loc['Annual Member', ind][['projection1', 'projection2']]
    if i % 2:
        ytext = 20 + 3 * i
    else:
        ytext = -8 - 4 * i
    ax.annotate(joined.loc[ind, 'holiday name'], [x, y], [x , ytext], color='black',
                ha='center', arrowprops=dict(arrowstyle='-', color='black'))
    ax.scatter([x], [y], c='red')
    
for holiday in (set(holidays) - set(joined['holiday name'])):
    ind = holidays[holidays == holiday].index[0]
    #ind = ind.strftime('%Y-%m-%d')
    x, y = hourly.loc['Annual Member', ind][['projection1', 'projection2']]
    ax.annotate(holidays.loc[ind], [x, y], [x + 20, y + 30], color='black',
                ha='center', arrowprops=dict(arrowstyle='-', color='black'))
    ax.scatter([x], [y], c='#00FF00')

ax.set_xlim([-60, 60])
ax.set_ylim([-60, 60])

fig.savefig('figs/pca_holiday_labels.png', bbox_inches='tight')

