import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
get_ipython().magic('matplotlib inline')

df = pd.read_csv('data/organized_uber.csv', parse_dates=['record_time'])
df['hour'] = df.record_time.dt.hour
df['day'] = df.record_time.dt.day
df['date'] = df.record_time.dt.date
df['dayofweek'] = df.record_time.dt.dayofweek
df['minute'] = df.record_time.dt.minute
df.set_index('record_time', inplace=True)

df.city.unique()

df[df.columns[:10]].head()

df[df.columns[10:19]].head()

df[df.columns[20:]].head()

df.describe()

df.info()

df.corr()['avg_price_est'].sort_values(ascending=False)

df[(df['display_name'] == 'uberX') & (df['city'] == 'denver')]. ix['2016-02-16'].groupby('hour').mean()['avg_price_est'].plot(marker='o', figsize=(10,6), label='2016-02-16')
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Denver');

dates = ['2016-02-16','2016-02-17','2016-02-18','2016-02-19','2016-02-20','2016-02-21']
dofwk = ['Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cmap = cm.get_cmap('seismic')
for i,date in enumerate(dates):
    df[(df['display_name'] == 'uberX') & (df['city'] == 'denver')].     ix[date].groupby('hour').mean()['avg_price_est'].     plot(marker='o', figsize=(12,8), color=cmap((2*i)/10.), label='{} - {}'.format(date,dofwk[i]))
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Denver For a Week');

dates = ['2016-02-22','2016-02-23','2016-02-24','2016-02-25','2016-02-26','2016-02-27','2016-02-28']
dofwk = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cmap = cm.get_cmap('seismic')
for i,date in enumerate(dates):
    df[(df['display_name'] == 'uberX') & (df['city'] == 'denver')].     ix[date].groupby('hour').mean()['avg_price_est'].     plot(marker='o', figsize=(12,8), color=cmap((1.5*i)/10.), label='{} - {}'.format(date,dofwk[i]))
plt.xlabel('Hour')
plt.ylabel('Average Price')
plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Denver For a Week');

dates = ['2016-02-22','2016-02-23','2016-02-24','2016-02-25','2016-02-26','2016-02-27','2016-02-28']
dofwk = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cmap = cm.get_cmap('seismic')
cities = df.city.unique()
fig, ax = plt.subplots(5,1, figsize=(12,40))
for i, axs in enumerate(ax.reshape(5,)):
    for j,date in enumerate(dates):
        bydate = df[(df['display_name'] == 'uberX') & (df['city'] == cities[i])].         ix[date].groupby('hour').mean()['avg_price_est']
        axs.plot(bydate, marker='o', color=cmap((1.5*j)/10.), label='{} - {}'.format(date,dofwk[j]))
    axs.set_xlabel('Hour')
    axs.set_ylabel('Average Price')
    # plt.axvline(8, ls='--', color='k', alpha=0.3)
    axs.legend(loc='upper right')
    axs.set_xticks(np.arange(0,24,1))
    axs.set_title('Fluctuation in UberX Pricing Within 24 Hours in {} For a Week'.format(cities[i].capitalize()));

for city in ['ny']:
    mean = df[(df['display_name'] == 'uberX') & (df['city'] == city)].     ix['2016-02-20'].groupby('hour').mean()['avg_price_est']
    mean.plot(marker='o', figsize=(12,8), label='{}'.format(city))
    std = df[(df['display_name'] == 'uberX') & (df['city'] == city)].     ix['2016-02-20'].groupby('hour').std()['avg_price_est']
    (mean+std).plot(linestyle='--', figsize=(12,8))
    (mean-std).plot(linestyle='--', figsize=(12,8))
    break
plt.xlabel('Hour')
plt.ylabel('Average Price')
# plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Different Cities on Weekdays');

for city in df['city'].unique():
    df[(df['display_name'] == 'uberX') & (df['city'] == city)].     ix['2016-02-22':'2016-02-26'].groupby('hour').mean()['avg_price_est'].plot(marker='o', figsize=(12,8), label='{}'.format(city))
plt.xlabel('Hour')
plt.ylabel('Average Price')
# plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Different Cities on Weekdays');

df['display_name'].value_counts()

for city in df['city'].unique():
    print city, df.query("city == @city")['display_name'].unique()

df['display_name'].replace(['UberBLACK','UberSUV','UberSELECT','uberT','Yellow WAV','ASSIST','PEDAL','For Hire','#UberTAHOE','uberCAB','WarmUpChi'], 
                           ['uberBLACK','uberSUV','uberSELECT','uberTAXI','uberWAV','uberASSIST','uberPEDAL','uberTAXI','uberTAHOE','uberTAXI','uberWARMUP'], inplace=True)

df['display_name'].value_counts()

for city in df['city'].unique():
    print city, df.query("city == @city")['display_name'].unique()

cartypes = ['uberX','uberXL','uberSELECT','uberBLACK','uberSUV']
for cartype in cartypes:
    df[(df['display_name'] == cartype) & (df['city'] == 'denver')].     ix['2016-02-16'].groupby('hour').mean()['avg_price_est'].plot(marker='o', figsize=(12,8), label='{}'.format(cartype))
plt.xlabel('Hour')
plt.ylabel('Average Price')
# plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in Car Services Pricing Within 24 Hours in Denver on Feb 16th');

cartypes = ['uberX','uberXL','uberSELECT','uberBLACK','uberSUV']
dates = ['2016-02-16','2016-02-17','2016-02-18','2016-02-19','2016-02-20','2016-02-21']
dofwk = ['Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(6,1, figsize=(12,25))
for i, axs in enumerate(ax.reshape(6,)):
    for cartype in cartypes:
        bycar = df[(df['display_name'] == cartype) & (df['city'] == 'denver')].ix[dates[i]].groupby('hour').mean()['avg_price_est']
        axs.plot(bycar, marker='o', label='{}'.format(cartype))
    axs.set_xlabel('Hour')
    axs.set_ylabel('Average Price')
    # plt.axvline(8, ls='--', color='k', alpha=0.3)
    axs.legend(loc='upper right')
    axs.set_xticks(np.arange(0,24,1))
    axs.set_yticks(np.arange(0,251,50))
    axs.set_title('Fluctuation in Car Services Pricing Within 24 Hours in Denver on {} - {}'.format(dates[i],dofwk[i]))
plt.tight_layout();

df[(df['city'] == 'denver') & (df['display_name'] == 'uberX')]['avg_price_est'].plot(figsize=(12,8));
# avg price estimate per minute from feb 17th - feb 29th 2016 for denver/uberX

# df[df['city'] == 'denver'].reset_index().groupby(['record_time','hour']).mean()['avg_price_est'].plot(figsize=(12,8));
df[(df['city'] == 'denver') & (df['display_name'] == 'uberX')].resample('T')['avg_price_est'].plot(figsize=(12,8));
# resample by minute

hourly = df[(df['city'] == 'denver') & (df['display_name'] == 'uberX')].resample('H')
hourly['avg_price_est'].plot(figsize=(12,8));
plt.fill_between(hourly.index, 0, hourly['avg_price_est'].max()*1.1, where=hourly['dayofweek'] >= 5, alpha=0.2, label='weekends');
plt.legend(loc='best');
plt.ylim([0,hourly['avg_price_est'].max()*1.1])
# resample by hour

trip_avg = df.groupby('city').mean()[['trip_duration','trip_distance','avg_price_est']].sort_values(by=['trip_duration'],ascending=False)
trip_avg

trip_avg['trip_minutes'] = trip_avg['trip_duration'] / 60.
trip_avg

trip_avg['avg_price_per_min'] = trip_avg['avg_price_est'] / trip_avg['trip_minutes']
trip_avg

df['trip_minutes'] = df['trip_duration'] / 60.
df['avg_price_per_min'] = df['avg_price_est'] / df['trip_minutes']

for city in df['city'].unique():
    df[(df['display_name'] == 'uberX') & (df['city'] == city)].     ix['2016-02-16':'2016-02-19'].groupby('hour').mean()['avg_price_per_min'].plot(marker='o', figsize=(12,8), label='{}'.format(city))
plt.xlabel('Hour')
plt.ylabel('Average Price Per Trip Minute')
# plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Different Cities on Weekdays');

for city in df['city'].unique():
    df[(df['display_name'] == 'uberX') & (df['city'] == city)].     ix['2016-02-27':'2016-02-28'].groupby('hour').mean()['avg_price_per_min'].plot(marker='o', figsize=(12,8), label='{}'.format(city))
plt.xlabel('Hour')
plt.ylabel('Average Price Per Trip Minute')
# plt.axvline(8, ls='--', color='k', alpha=0.3)
plt.legend(loc='best')
plt.xticks(np.arange(0,24,1))
plt.title('Fluctuation in UberX Pricing Within 24 Hours in Different Cities on Weekends');

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_pacf(data, lags):
    """
    Input: Amount of lag
    Output: Plot of ACF/PACF
    """
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data, lags=lags, ax=ax2)
data = df.ix[:'2016-02-20'].query("display_name == 'uberX' and city == 'denver'")
plot_acf_pacf(data['avg_price_est'], lags=60)

df.ix[:'2016-02-20'].query("display_name == 'uberX' and city == 'denver'")['avg_price_est'].plot(figsize=(20,8));

cities = df['city'].unique().tolist()
fig, ax = plt.subplots(5,1, figsize=(20,30))
d = df.query("display_name == 'uberX' and city == 'denver'")['avg_price_est']
dates = np.unique(d.index.date)
for i, axs in enumerate(ax.reshape(5,)):
    data = df.query("display_name == 'uberX' and city == @cities[@i]")['avg_price_est']
    axs.plot(data)
    axs.set_title(cities[i])
    axs.set_ylabel('average ride price')
    axs.set_xlabel('day')
    axs.set_xticks(dates)
    axs.set_xticklabels(dates, rotation='45')
plt.tight_layout()



