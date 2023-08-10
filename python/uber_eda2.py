import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
get_ipython().magic('matplotlib inline')

df = pd.read_csv('../data/organized_uber.csv', parse_dates=['record_time'])
df.set_index('record_time', inplace=True)
df.index = df.index - pd.Timedelta(hours=7)
df['hour'] = df.index.hour
df['dayofmonth'] = df.index.day
df['date'] = df.index.date
df['dayofweek'] = df.index.dayofweek
df['weekofyear'] = df.index.weekofyear

cities = df['city'].unique().tolist()
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    data = df.query("display_name == 'uberX' and city == @cities[@i]")['avg_price_est'].resample('H').mean()
    mean_price = data.mean()
    axs.plot(data, label='uberX')
    axs.axhline(mean_price, color='r', ls='--', label='mean price, {}'.format(mean_price))
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

cities = df['city'].unique().tolist()
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    data = df.query("display_name == 'uberX' and city == @cities[@i]")['avg_price_est'].resample('H')
    mean_price = data.mean()
    axs.plot(data.diff(periods=1), label='uberX')
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

cities = df['city'].unique().tolist()
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    data = df.query("display_name == 'uberX' and city == @cities[@i]")['avg_price_est'].resample('H')
    mean_price = data.mean()
    axs.plot(data.diff(periods=24), label='uberX')
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

cities = df['city'].unique().tolist()
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    daily = df.query("display_name == 'uberX' and city == @cities[@i]")['avg_price_est'].resample('D').mean()
    hourly = df.query("display_name == 'uberX' and city == @cities[@i]")[['avg_price_est','dayofweek']].resample('H').mean()
    mean_price = hourly['avg_price_est'].mean()
    axs.plot(hourly['avg_price_est'], label='uberX hourly')
    axs.fill_between(hourly.index, 0, hourly['avg_price_est'].max()*1.1, where=hourly['dayofweek'] >= 5, alpha=0.2, label='weekends')
    axs.plot(daily, label='uberX daily')
    axs.axhline(mean_price, color='r', ls='--', label='mean price, {}'.format(mean_price))
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.set_ylim([0,hourly['avg_price_est'].max()*1.1])
    axs.legend(loc='upper right')
plt.tight_layout()

import statsmodels.api as sm

d_hr = d.resample('H').mean()
d_dates = np.unique(d_hr.index.date)
for city in cities:
    hourly = df.query("display_name == 'uberX' and city == @city").resample('H').mean()
    decomposition = sm.tsa.seasonal_decompose(hourly['avg_price_est'].values, model='additive', freq=24)  
    fig = decomposition.plot()
    fig.set_figheight(15)
    fig.set_figwidth(20)
    fig.tight_layout()
    axs_f = fig.get_axes()[0]
    axs_l = fig.get_axes()[-1]
    start, end = axs_l.get_xlim()
    stepsize=end/d_dates.shape[0]
    axs_l.xaxis.set_ticks(np.arange(start, end, stepsize))
    axs_l.set_xticklabels(d_dates, rotation='45', fontsize=15)
    axs_l.set_xlabel("", fontsize=20)
#     axs_f.set_title(city.capitalize() + " UberX Data Decomposition From 2-15-16 to 4-14-16", fontsize=30);
    axs_a = fig.get_axes()
    labels = ['Observed Prices','Trend','Seasonal','Residual']
    for i, ax in enumerate(axs_a):
        ax.tick_params(labelsize=15)
        ax.set_ylabel('', fontsize=20)

df.columns

features = ['avg_price_est','trip_duration','trip_distance',
            'surge_multiplier','surge_minimum_price','pickup_estimate',
            'capacity','base_price','base_minimum_price','cost_per_minute',
            'cost_per_distance','cancellation_fee','service_fees','hour','day',
           'date','dayofweek','weekofyear']
fs = df[features].corr()
mask = np.zeros_like(fs)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(14,12))
with sns.axes_style("white"):
    sns.heatmap(fs, mask=mask, square=True, annot=True, linewidths=1.5)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.tight_layout();

c = df.query("display_name == 'uberX' and city == 'denver'")[['surge_minimum_price','avg_price_est']].resample('H')
sns.jointplot(x="surge_minimum_price", y="avg_price_est", data=c);

sub_features = ['avg_price_est','surge_minimum_price','base_price',
                'base_minimum_price','cost_per_minute','cost_per_distance',
               'cancellation_fee']
scmat = df.query("display_name == 'uberX' and city == 'denver'")[sub_features]
sns.pairplot(scmat);

cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]")['avg_price_est'].resample('H')
        axs.plot(data, label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

agg_wkly_dates.values()

df.columns

from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

city = 'denver'
cartype = 'uberX'
dayofweek = 0
dateofint = '03/23/16'
sdates = np.unique(df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].index.date)
sday = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(sdates[1])].resample('H')
print sdates
# print sday
fig, ax = plt.subplots(figsize=(20,6))
for i, date in enumerate(sdates[1:]):
    data = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(date)].resample('H')
    ax.plot_date(sday.index.to_pydatetime(), data.values, 'o-', label=date);
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
    ax.xaxis.set_minor_formatter(DateFormatter('%H'))
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
    ax.set_title('{} price spike on {} in {}'.format(cartype, dateofint, city))
    ax.set_ylabel('average price estimate')
    ax.set_xlabel('hour')
    if i == 5:
        ax.axvline(pd.to_datetime('2016-02-22 16:00:00'), color='r', ls='--', alpha=0.5, label='denver price spike')
    ax.legend(loc='best');

from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

sdates = np.unique(df.query("city == 'seattle' and display_name == 'uberX' and dayofweek == 2")['avg_price_est'].index.date)
sday = df.query("city == 'seattle' and display_name == 'uberX' and dayofweek == 2")['avg_price_est'].ix[str(sdates[0])].resample('H').mean()
print sdates
# print sday
fig, ax = plt.subplots(figsize=(20,8))
for i, date in enumerate(sdates[:-3]):
    data = df.query("city == 'seattle' and display_name == 'uberX' and dayofweek == 2")['avg_price_est'].ix[str(date)].resample('H').mean()
    ax.plot_date(sday.index.to_pydatetime(), data.values, 'o-', label=date);
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
    ax.xaxis.set_minor_formatter(DateFormatter('%H'))
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
#     ax.set_title('uberX price spike on 03/09/16 in seattle')
#     ax.set_ylabel('average price estimate')
#     ax.set_xlabel('hour')
    ax.tick_params(labelsize=15, which="both")
#     if i == 5:
#         ax.axvline(pd.to_datetime('2016-02-17 18:00:00'), color='r', ls='--', alpha=0.5, label='bieber concert at 7:30pm')
    ax.legend(loc='best', fontsize=15);

city = 'denver'
cartype = 'uberX'
dayofweek = 2
dateofint = '03/23/16'
sdates = np.unique(df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].index.date)
sday = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(sdates[0])].resample('H')
print sdates
# print sday
fig, ax = plt.subplots(figsize=(20,6))
for i, date in enumerate(sdates[:-1]):
    data = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(date)].resample('H')
    ax.plot_date(sday.index.to_pydatetime(), data.values, 'o-', label=date);
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
    ax.xaxis.set_minor_formatter(DateFormatter('%H'))
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
    ax.set_title('{} price spike on {} in {}'.format(cartype, dateofint, city))
    ax.set_ylabel('average price estimate')
    ax.set_xlabel('hour')
    if i == 5:
        ax.axvline(pd.to_datetime('2016-02-17 12:00:00'), color='r', ls='--', alpha=0.5, label='heavy snow in denver')
    ax.legend(loc='best');

city = 'denver'
cartype = 'uberX'
dayofweek = 4
dateofint = '03/18/16'
sdates = np.unique(df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].index.date)
sday = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(sdates[0])].resample('H')
print sdates
# print sday
fig, ax = plt.subplots(figsize=(20,6))
for i, date in enumerate(sdates):
    data = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(date)].resample('H')
    ax.plot_date(sday.index.to_pydatetime(), data.values, 'o-', label=date);
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
    ax.xaxis.set_minor_formatter(DateFormatter('%H'))
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
    ax.set_title('{} price spike on {} in {}'.format(cartype, dateofint, city))
    ax.set_ylabel('average price estimate')
    ax.set_xlabel('hour')
    if i == 5:
        ax.axvline(pd.to_datetime('2016-02-19 06:00:00'), color='r', ls='--', alpha=0.5, label='heavy snow in denver')
    ax.legend(loc='best');

city = 'ny'
cartype = 'uberX'
dayofweek = 1
dateofint = '02/23/16'
sdates = np.unique(df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].index.date)
sday = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(sdates[0])].resample('H')
print sdates
# print sday
fig, ax = plt.subplots(figsize=(20,6))
for i, date in enumerate(sdates):
    data = df.query("city == @city and display_name == @cartype and dayofweek == @dayofweek")['avg_price_est'].ix[str(date)].resample('H')
    ax.plot_date(sday.index.to_pydatetime(), data.values, 'o-', label=date);
    ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
    ax.xaxis.set_minor_formatter(DateFormatter('%H'))
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
    ax.set_title('{} price spike on {} in {}'.format(cartype, dateofint, city))
    ax.set_ylabel('average price estimate')
    ax.set_xlabel('hour')
    if i == 6:
        ax.axvline(pd.to_datetime('2016-02-16 16:00:00'), color='r', ls='--', alpha=0.5, label='ny price spike')
    ax.legend(loc='best');

from collections import defaultdict

cities = df['city'].unique().tolist()
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
agg_wkly_dates = defaultdict(str)
cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, axs = plt.subplots(figsize=(20,8))
for start, end in wkly_dates:
    data = df.query("city == @cities[3] and display_name == @cartypes[0]").ix[start:end].resample('H').mean()
    axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
    dates = np.unique(data.index.date)
#     for j,date in enumerate(dates):
#         if str(date) not in agg_wkly_dates[j]:
#             agg_wkly_dates[j] += str(date) + '\n'
#     axs.set_ylabel('Average Price Estimate', fontsize=20)
#     axs.set_xlabel('Time', fontsize=20)
    week_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    axs.tick_params(labelsize=15)
    start, end = axs.get_xlim()
    stepsize = end / dates.shape[0]
    axs.set_xticks(np.arange(start, end, stepsize))
    axs.set_xticklabels(week_names)
    axs.legend(loc='upper right', fontsize=20)
#     axs.set_title("New York UberX Weekly Prices From 2-22-16 to 3-27-16", fontsize=30)
plt.tight_layout()

from collections import defaultdict

cities = df['city'].unique().tolist()
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
agg_wkly_dates = defaultdict(str)
cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[å@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        for j,date in enumerate(dates):
            if str(date) not in agg_wkly_dates[j]:
                agg_wkly_dates[j] += str(date) + '\n'
        axs.set_ylabel('avg_price_est')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(agg_wkly_dates.values())
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))
plt.tight_layout()

fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('avg_price_est')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))
#     if cities[i] == 'denver':
#         axs.axvline(pd.to_datetime('2016-03-24').date(), label='heavy snow', color='r', linestyle='--')

pd.to_datetime('2016-03-24').date()

fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[1]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('avg_price_est')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[1]))

fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[2]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('avg_price_est')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[2]))

fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[3]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['avg_price_est'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('avg_price_est')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[3]))

from scipy.stats import pearsonr

d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est'].resample('H')
for lag in [1,12,24]:
    g = sns.JointGrid(d, d.shift(lag), ratio=100).set_axis_labels("lag {}".format(lag),"avg_price_estimate")
    g.plot_joint(sns.regplot, fit_reg=False)
    g.annotate(pearsonr)
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off();

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_pacf(data, lags, label):
    """
    Input: Amount of lag
    Output: Plot of ACF/PACF
    """
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data, lags=lags, ax=ax1)
    ax1.set_ylabel('correlation for each lag')
    ax1.set_xlabel('lag')
    ax1.set_title('Autocorrelation - {}'.format(label))
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data, lags=lags, ax=ax2)
    ax2.set_xlabel('lag')
    ax2.set_ylabel('correlation for each lag')
    plt.tight_layout()
for city in cities:
    hourly = df.query("display_name == 'uberX' and city == @city")['avg_price_est'].resample('H')
    plot_acf_pacf(hourly, lags=48, label=city)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_pacf(data, lags, label):
    """
    Input: Amount of lag
    Output: Plot of ACF/PACF
    """
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data, lags=lags, ax=ax1)
    ax1.set_ylabel('correlation for each lag')
    ax1.set_xlabel('lag')
    ax1.set_title('Autocorrelation - {}'.format(label))
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data, lags=lags, ax=ax2)
    ax2.set_xlabel('lag')
    ax2.set_ylabel('correlation for each lag')
    plt.tight_layout()
cities = df['city'].unique().tolist()
for city in cities:
    minutely = df.query("display_name == 'uberX' and city == @city")['avg_price_est']
    plot_acf_pacf(minutely, lags=60, label=city)

for city in cities:
    daily = df.query("display_name == 'uberX' and city == @city")['avg_price_est'].resample('D')
    plot_acf_pacf(daily, lags=21, label=city)

from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

s = df.query("display_name == 'uberX' and city == 'denver'").resample('H').ix['2016-02-16':'2016-02-17']
fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(s.index.to_pydatetime(), s['avg_price_est'].values, '-');

ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');

sub_s = df.query("display_name == 'uberX' and city == 'denver'").resample('H').ix['2016-02-16']
fig, ax = plt.subplots(figsize=(8,6))
ax.plot_date(sub_s.index.to_pydatetime(), sub_s['avg_price_est'].values, '-');

ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');

sub_s['time'] = np.arange(1, sub_s.shape[0]+1)

y = sub_s['avg_price_est'].values
X = sub_s['time']
model = sm.OLS(y, sm.add_constant(X)).fit()
fig, ax = plt.subplots(figsize=(8,6))
ax.plot_date(sub_s.index.to_pydatetime(), sub_s['avg_price_est'].values, '-');

ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
pd.Series(model.fittedvalues, index=sub_s.index).plot();
plt.setp(ax.get_xticklabels(), rotation=0);
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');
model.summary()

s['time'] = np.arange(1, s.shape[0]+1)

new_s = s[['avg_price_est','time']].ix['2016-02-17']
new_X = new_s['time']
res = model.predict(sm.add_constant(new_X))

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(s.index.to_pydatetime(), s['avg_price_est'].values, '-');

ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');
pd.Series(model.fittedvalues, index=sub_s.index).plot(label='fitted');
plt.plot(new_s.index,res, label='predicted')
plt.setp(ax.get_xticklabels(), rotation=0);
ax.legend(loc='best');

sub_s.columns

features = ['trip_duration','trip_distance','pickup_estimate',
            'capacity','base_price','base_minimum_price','cost_per_minute',
            'cost_per_distance','cancellation_fee','service_fees','hour','day',
           'dayofweek','weekofyear']
y = sub_s['avg_price_est'].values
X = sub_s[features]
model = sm.OLS(y, sm.add_constant(X)).fit()
fig, ax = plt.subplots(figsize=(8,6))
ax.plot_date(sub_s.index.to_pydatetime(), sub_s['avg_price_est'].values, '-');

ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
pd.Series(model.fittedvalues, index=sub_s.index).plot();
plt.setp(ax.get_xticklabels(), rotation=0);
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');
model.summary()

from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

s = df.query("display_name == 'uberX' and city == 'denver' and weekofyear >= 8 and weekofyear <= 10").resample('H')
fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(s.index.to_pydatetime(), s['avg_price_est'].values, '-');

# ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
# ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=7))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('hour')
ax.set_ylabel('average price estimate');
ax.set_title('denver');

from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, HourLocator

s = df.query("display_name == 'uberX' and city == 'denver' and weekofyear >= 8 and weekofyear <= 9").resample('H')
fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(s.index.to_pydatetime(), s['avg_price_est'].values, '-');

# ax.xaxis.set_minor_locator(HourLocator(byhour=range(24), interval=1))
# ax.xaxis.set_minor_formatter(DateFormatter('%H'))
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('daily')
ax.set_ylabel('average price estimate');
ax.set_title('denver');

import statsmodels.api as sm
import scipy.stats as scs

# df['lag_1'] = df['avg_price_est'].diff(periods=1)
features = ['avg_price_est','city','display_name','trip_duration','trip_distance','hour','dayofweek','weekofyear','pickup_estimate','surge_multiplier']
sample = pd.get_dummies(df[features], columns=['city','display_name','hour','dayofweek','weekofyear']).drop(['city_chicago','display_name_uberASSIST','hour_0','dayofweek_0','weekofyear_7'], axis=1).dropna()

train_set = sample.query("weekofyear_8 == 1 or weekofyear_9 == 1")
test_set = sample.query("weekofyear_10 == 1")

y = train_set['avg_price_est'].values
X = train_set[train_set.columns[1:]]
model = sm.OLS(y, sm.add_constant(X)).fit()
y_pred = model.fittedvalues
trainplot = pd.concat([train_set, pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)
trainplot['resid'] = trainplot['y_pred'] - trainplot['avg_price_est']
trainplot = trainplot.query("display_name_uberX == 1 and city_denver == 1").resample('H')

print model.summary()

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(trainplot.index.to_pydatetime(), trainplot['avg_price_est'].values, '-', label='true');
ax.plot_date(trainplot.index.to_pydatetime(), trainplot['y_pred'].values, '-', label='prediction');
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('daily')
ax.set_ylabel('average price estimate');
ax.set_title('denver uberX prediction');
ax.legend(loc='upper right')

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(trainplot.index.to_pydatetime(), trainplot['resid'].values, 'o', label='resid', alpha=0.4);
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.axhline(0, color='r', ls='--')
ax.set_xlabel('daily')
ax.set_ylabel('residuals');
ax.set_title('denver uberX residuals');
ax.legend(loc='upper right')

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(model.resid.resample('H').index.to_pydatetime(), model.resid.resample('H').values, 'o', label='resid', alpha=0.4);
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.axhline(0, color='r', ls='--')
ax.set_xlabel('daily')
ax.set_ylabel('model residuals');
ax.set_title('denver uberX model residuals');
ax.legend(loc='upper right')

plot_acf_pacf(trainplot['resid'], lags=48, label='denver uberX')

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(trainplot['y_pred'], trainplot['resid'])
ax.set_title("Fitted Values vs Residual - Check for Constant Variance (Heteroscedasticity)")
ax.set_xlabel("fitted values")
ax.set_ylabel("residual")

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(trainplot['resid'])
ax.set_title("Histogram of Residuals - Check for Normal Distribution of Residuals")
ax.set_xlabel("residuals")
ax.set_ylabel("frequency")

fig, ax = plt.subplots(figsize=(10,6))
sm.graphics.qqplot(model.resid, scs.norm, line='q', ax=ax)
ax.set_title('Residual qqplot - Check for Normal Distribution of Residuals');
line1 = ax.get_lines()[1]
line1.set_linestyle('--')
line1.set_alpha(0.8)
line2 = ax.get_lines()[0]
line2.set_color('royalblue')
line2.set_alpha(0.5)

# df['lag_1'] = df['avg_price_est'].diff(periods=1)
features = ['avg_price_est','city','display_name','trip_duration', 'trip_distance','hour','dayofweek','weekofyear','pickup_estimate']
sample = pd.get_dummies(df[features], columns=['city','display_name','hour','dayofweek','weekofyear']).drop(['city_chicago','display_name_uberASSIST','hour_0','dayofweek_0','weekofyear_7'], axis=1).dropna()

fs = sample.corr()
mask = np.zeros_like(fs)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(14,12))
with sns.axes_style("white"):
    sns.heatmap(fs, mask=mask, square=True, annot=False, linewidths=1.5)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.tight_layout();

test_pred = model.predict(test_set[test_set.columns[1:]])
testplot = pd.concat([test_set.reset_index(), pd.DataFrame(test_pred, columns=['y_pred'])], axis=1).set_index('record_time')
testplot['resid'] = testplot['y_pred'] - testplot['avg_price_est'] 
testplot = testplot.query("display_name_uberX == 1 and city_denver == 1").resample('H')

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(testplot.index.to_pydatetime(), testplot['avg_price_est'].values, '-', label='true');
ax.plot_date(testplot.index.to_pydatetime(), testplot['y_pred'].values, '-', label='prediction');
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('daily')
ax.set_ylabel('average price estimate');
ax.set_title('denver uberX prediction');
ax.legend(loc='upper right')

fig, ax = plt.subplots(figsize=(20,6))
ax.plot_date(testplot.index.to_pydatetime(), testplot['resid'].values, 'o', label='resid');
ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7), interval=1))
ax.xaxis.set_major_formatter(DateFormatter('\n\n%a\n%D'))
ax.xaxis.grid(True, which="minor")
ax.set_xlabel('daily')
ax.set_ylabel('average price estimate');
ax.set_title('denver uberX prediction');
ax.legend(loc='upper right');

from matplotlib.dates import strpdate2num

cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['trip_distance']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]").resample('H')
        axs.plot(data['trip_distance'], label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('trip_distance')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    for date in ['2016-02-22','2016-02-29','2016-03-07','2016-03-14','2016-03-21','2016-03-28']:
        axs.axvline(pd.to_datetime(date), color='r', linestyle='--')
    axs.legend(loc='upper right')
plt.tight_layout()

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['trip_distance'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('trip_distance')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)

cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['trip_duration']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]")['trip_duration'].resample('H')
        axs.plot(data, label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('trip_duration')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    for date in ['2016-02-22','2016-02-29','2016-03-07','2016-03-14','2016-03-21','2016-03-28']:
        axs.axvline(pd.to_datetime(date), color='r', linestyle='--')
    axs.legend(loc='upper right')
plt.tight_layout()

np.unique(df.query("weekofyear == 7").index.date)

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['surge_multiplier'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('surge_multiplier')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['surge_minimum_price'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('surge_minimum_price')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['capacity'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('capacity')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['base_price'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('base_price')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['base_minimum_price'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('base_minimum_price')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['cost_per_minute'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('cost_per_minute')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['cost_per_distance'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('cost_per_distance')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['cancellation_fee'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('cancellation_fee')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

# can't use this
wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name in @cartypes").ix[start:end].resample('H')
        axs.plot(data.reset_index()['service_fees'])
        dates = np.unique(data.index.date)
        axs.set_ylabel('service_fees')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.set_title(cities[i])

features = ['avg_price_est','trip_duration','trip_distance',
            'surge_multiplier','surge_minimum_price','pickup_estimate',
            'capacity','base_price','base_minimum_price','cost_per_minute',
            'cost_per_distance','cancellation_fee','service_fees','hour','day',
           'date','dayofweek','minute']
cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['surge_multiplier']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]")['surge_multiplier'].resample('H')
        axs.plot(data, label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('surge_multiplier')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

features = ['avg_price_est','trip_duration','trip_distance',
            'surge_multiplier','surge_minimum_price','pickup_estimate',
            'capacity','base_price','base_minimum_price','cost_per_minute',
            'cost_per_distance','cancellation_fee','service_fees','hour','day',
           'date','dayofweek','minute']
cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['surge_minimum_price']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]")['surge_minimum_price'].resample('H')
        axs.plot(data, label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('surge_minimum_price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    axs.legend(loc='upper right')
plt.tight_layout()

features = ['avg_price_est','trip_duration','trip_distance',
            'surge_multiplier','surge_minimum_price','pickup_estimate',
            'capacity','base_price','base_minimum_price','cost_per_minute',
            'cost_per_distance','cancellation_fee','service_fees','hour','day',
           'date','dayofweek','minute']
cartypes = ['uberX','uberXL','uberBLACK','uberSUV']
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['pickup_estimate']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    for cartype in cartypes:
        data = df.query("display_name == @cartype and city == @cities[@i]")['pickup_estimate'].resample('3H')
        axs.plot(data, label=cartype)
    axs.set_title(cities[i])
    axs.set_ylabel('pickup_estimate')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
    for date in ['2016-02-22','2016-02-29','2016-03-07','2016-03-14','2016-03-21','2016-03-28']:
        axs.axvline(pd.to_datetime(date), color='r', linestyle='--')
    axs.legend(loc='upper right')
plt.tight_layout()

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['pickup_estimate'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('pickup_estimate')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[1]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['pickup_estimate'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('pickup_estimate')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[1]))

df.columns

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[2]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['pickup_estimate'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('pickup_estimate')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[2]))

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[3]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['pickup_estimate'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('pickup_estimate')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[3]))

df.columns

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['hour'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('hour')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['dayofweek'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('dayofweek')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['weekofyear'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('weekofyear')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))

tues = np.unique(df.query("city == 'denver' and display_name == 'uberX' and dayofweek == 1")[['avg_price_est']].index.date)
print tues
plt.figure(figsize=(20,10))
for date in tues:
    data = df.query("city == 'denver' and display_name == 'uberX' and dayofweek == 1")[['avg_price_est']].ix[str(date)].resample('H')
    plt.plot(data.values, label=date, marker='o')
plt.legend(loc='best');

df['price_range'] = df['high_estimate'] - df['low_estimate']

df.query("city == 'denver' and display_name == 'uberX'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'ny' and display_name == 'uberX'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'sf' and display_name == 'uberX'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'seattle' and display_name == 'uberX'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'chicago' and display_name == 'uberX'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'denver' and display_name == 'uberXL'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'seattle' and display_name == 'uberXL'").ix['2016-03-22'][['avg_price_est','low_estimate','high_estimate','price_range']].head()

df.query("city == 'denver' and display_name == 'uberXL'").ix['2016-03-22']['price_range'].hist();

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['price_range'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('price_range')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))

df['price_per_range'] = df['avg_price_est'] / df['price_range'].astype(float)

wkly_dates = [('2016-02-22','2016-02-28'), ('2016-02-29','2016-03-06'), 
              ('2016-03-07','2016-03-13'), ('2016-03-14','2016-03-20'),
             ('2016-03-21','2016-03-27')]
fig, ax = plt.subplots(5,1, figsize=(20,30))
for i, axs in enumerate(ax.reshape(5,)):
    for start, end in wkly_dates:
        data = df.query("city == @cities[@i] and display_name == @cartypes[0]").ix[start:end].resample('H')
        axs.plot(data.reset_index()['price_per_range'], label='{} --> {}'.format(start,end))
        dates = np.unique(data.index.date)
        axs.set_ylabel('price_per_range')
        axs.set_xlabel('time')
        start, end = axs.get_xlim()
        stepsize = end / dates.shape[0]
        axs.set_xticks(np.arange(start, end, stepsize))
        axs.set_xticklabels(dates)
        axs.legend(loc='upper right')
        axs.set_title('{} - {}'.format(cities[i], cartypes[0]))



