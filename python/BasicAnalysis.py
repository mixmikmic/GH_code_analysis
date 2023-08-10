get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

import os
if not os.path.exists('open_data_year_one.zip'):
    get_ipython().system('curl -O https://s3.amazonaws.com/pronto-data/open_data_year_one.zip')
    get_ipython().system('unzip open_data_year_one.zip')

trips = pd.read_csv('2015_trip_data.csv',
                    parse_dates=['starttime', 'stoptime'],
                    infer_datetime_format=True)
trips.columns

# Find the start date
ind = pd.DatetimeIndex(trips.starttime)
trips['date'] = ind.date.astype('datetime64')
trips['hour'] = ind.hour

# Count trips by date
by_date = trips.pivot_table('trip_id', aggfunc='count',
                            index='date',
                            columns='usertype', )

# Count trips by weekday
weekly = by_date.pivot_table(['Annual Member', 'Short-Term Pass Holder'],
                             index=by_date.index.weekofyear,
                             columns=by_date.index.dayofweek)

by_date.sort_values('Short-Term Pass Holder', ascending=False)

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.1)

def add_inset(ax, rect, *args, **kwargs):
    box = ax.get_position()
    inax_position = ax.transAxes.transform(rect[0:2])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    new_rect = list(infig_position) + [box.width * rect[2], box.height * rect[3]]
    return fig.add_axes(new_rect, *args, **kwargs)

color_cycle = plt.rcParams['axes.color_cycle']
for i, col in enumerate(['Annual Member', 'Short-Term Pass Holder']):
    by_date[col].plot(ax=ax[i], title=col, color=color_cycle[i])
    ax[i].set_title(col + 's')

with sns.axes_style('whitegrid'):
    inset = [add_inset(ax[0], [0.07, 0.6, 0.2, 0.32]),
             add_inset(ax[1], [0.07, 0.6, 0.2, 0.32])]

for i, col in enumerate(['Annual Member', 'Short-Term Pass Holder']):
    inset[i].plot(range(7), weekly[col].values.T, color=color_cycle[i], lw=2, alpha=0.05);
    inset[i].plot(range(7), weekly[col].mean(0), color=color_cycle[i], lw=3)
    inset[i].set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
    inset[i].yaxis.set_major_locator(plt.MaxNLocator(5))
    inset[i].set_ylim(0, 500)
    inset[i].set_title('Average by Day of Week')
    
fig.savefig('figs/daily_trend.png', bbox_inches='tight')

ind = pd.DatetimeIndex(trips.date)
ind.dayofweek

trips['weekend'] = (ind.dayofweek > 4)
hourly = trips.pivot_table('trip_id', aggfunc='count',
                           index=['date'], columns=['usertype', 'weekend', 'hour'])
fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.1)
fmt = plt.FuncFormatter(lambda x, *args: '{0}:00'.format(int(x)))

for weekend in (False, True):
    axi = ax[int(weekend)]
    for i, col in enumerate(['Annual Member', 'Short-Term Pass Holder']):
        vals = hourly[col][weekend].values
        vals = np.hstack([vals, vals[:, :1]])
        axi.plot(range(25), vals.T,
                 color=color_cycle[i], lw=1, alpha=0.05)
        axi.plot(range(25), np.nanmean(vals, 0),
                 color=color_cycle[i], lw=3, label=col)
    axi.xaxis.set_major_locator(plt.MultipleLocator(4))
    axi.xaxis.set_major_formatter(fmt)
    axi.set_ylim(0, 60)
    axi.set_title('Saturday - Sunday' if weekend else 'Monday - Friday')
    axi.legend(loc='upper left')
    axi.set_xlabel('Time of Day')
ax[0].set_ylabel('Number of Trips')
fig.suptitle('Hourly Trends: Weekdays and Weekends', size=14);

fig.savefig('figs/hourly_trend.png', bbox_inches='tight')

stations = pd.read_csv('2015_station_data.csv')
pronto_shop = dict(id=54, name="Pronto shop",
                   terminal="Pronto shop",
                   lat=47.6173156, long=-122.3414776,
                   dockcount=100, online='10/13/2014')
stations = stations.append(pronto_shop, ignore_index=True)

# Here we query the Google Maps API for distances between trips

from time import sleep

def query_distances(stations=stations):
    """Query the Google API for bicycling distances"""
    latlon_list = ['{0},{1}'.format(lat, long)
                   for (lat, long) in zip(stations.lat, stations.long)]

    def create_url(i):
        URL = ('https://maps.googleapis.com/maps/api/distancematrix/json?'
               'origins={origins}&destinations={destinations}&mode=bicycling')
        return URL.format(origins=latlon_list[i],
                          destinations='|'.join(latlon_list[i + 1:]))

    for i in range(len(latlon_list) - 1):
        url = create_url(i)
        filename = "distances_{0}.json".format(stations.terminal.iloc[i])
        print(i, filename)
        get_ipython().system('curl "{url}" -o {filename}')
        sleep(11) # only one query per 10 seconds!


def build_distance_matrix(stations=stations):
    """Build a matrix from the Google API results"""
    dist = np.zeros((len(stations), len(stations)), dtype=float)
    for i, term in enumerate(stations.terminal[:-1]):
        filename = 'queried_distances/distances_{0}.json'.format(term)
        row = json.load(open(filename))
        dist[i, i + 1:] = [el['distance']['value'] for el in row['rows'][0]['elements']]
    dist += dist.T
    distances = pd.DataFrame(dist, index=stations.terminal,
                             columns=stations.terminal)
    distances.to_csv('station_distances.csv')
    return distances

# only call this the first time
import os
if not os.path.exists('station_distances.csv'):
    # Note: you can call this function at most ~twice per day!
    query_distances()

    # Move all the queried files into a directory
    # so we don't accidentally overwrite them
    if not os.path.exists('queried_distances'):
        os.makedirs('queried_distances')
    get_ipython().system('mv distances_*.json queried_distances')

    # Build distance matrix and save to CSV
    distances = build_distance_matrix()

distances = pd.read_csv('station_distances.csv', index_col='terminal')
distances.iloc[:5, :5]

stacked = distances.stack() / 1609.34  # convert meters to miles
stacked.name = 'distance'
tmp = trips.join(stacked, on=['from_station_id', 'to_station_id'])
trips['distance'] = tmp['distance']

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.1)

trips['minutes'] = trips.tripduration / 60
trips.groupby('usertype')['minutes'].hist(bins=np.arange(61), alpha=0.5, normed=True, ax=ax[0]);
ax[0].set_xlabel('Duration (minutes)')
ax[0].set_ylabel('relative frequency')
ax[0].set_title('Trip Durations')
ax[0].text(34, 0.09, "Free Trips\n\nAdditional Fee", ha='right',
           size=18, rotation=90, alpha=0.5, color='red')
ax[0].legend(['Annual Members', 'Short-term Pass'])
ax[0].axvline(30, linestyle='--', color='red', alpha=0.3);

trips['minutes'] = trips['tripduration'] / 60
trips['speed'] = trips.distance * 60 / trips.minutes
trips.groupby('usertype')['speed'].hist(bins=np.linspace(0, 20, 50), normed=True,
                                        alpha=0.5, ax=ax[1])
ax[1].set_xlabel('Minimum Speed (mph)')
ax[1].legend(['Annual Members', 'Short-term Pass Holders'])
ax[1].set_title('Rider Speed');

fig.savefig('figs/duration_and_speed.png', bbox_inches='tight')

g = sns.FacetGrid(trips, col="usertype", hue='usertype', size=6)
g.map(plt.scatter, "distance", "speed", s=4, alpha=0.2)
g.axes[0, 0].axis([0, 10, 0, 25]);

for axi, col in zip(g.axes.flat, ['Annual Member', 'Short-Term Pass Holder']):
    axi.text(9.5, 16.6, "Free Trips\n\nAdditional Fee", ha='right',
             size=17, rotation=45, alpha=0.5, color='red')
    axi.plot([0, 20], [0, 40], '--r', alpha=0.3)
    axi.axis([0, 10, 0, 20])
    axi.set_title(col + 's')
    axi.set_xlabel('Minimum Distance (miles)')
g.axes[0, 0].set_ylabel('Minimum Speed (MPH)');

g.fig.savefig('figs/distance_vs_speed.png', bbox_inches='tight')

# Get elevations from the Google Maps API

def get_station_elevations(stations):
    """Get station elevations via Google Maps API"""
    URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="
    locs = '|'.join(['{0},{1}'.format(lat, long)
                     for (lat, long) in zip(stations.lat, stations.long)])
    URL += locs
    get_ipython().system('curl "{URL}" -o elevations.json')


def process_station_elevations():
    """Convert Elevations JSON output to CSV"""
    import json
    D = json.load(open('elevations.json'))
    def unnest(D):
        loc = D.pop('location')
        loc.update(D)
        return loc
    elevs = pd.DataFrame([unnest(item) for item in D['results']])
    elevs.to_csv('station_elevations.csv')
    return elevs

# only run this the first time:
import os
if not os.path.exists('station_elevations.csv'):
    get_station_elevations(stations)
    process_station_elevations()

elevs = pd.read_csv('station_elevations.csv', index_col=0)
elevs.head()

# double check that locations match
print(np.allclose(stations.long, elevs.lng))
print(np.allclose(stations.lat, elevs.lat))

stations['elevation'] = elevs['elevation']
elevs.index = stations['terminal']

trips['elevation_start'] = trips.join(elevs, on='from_station_id')['elevation']
trips['elevation_end'] = trips.join(elevs, on='to_station_id')['elevation']
trips['elevation_gain'] = trips['elevation_end'] - trips['elevation_start']

len(trips)

g = sns.FacetGrid(trips, col="usertype", hue='usertype')
g.map(plt.hist, "elevation_gain", bins=np.arange(-145, 150, 10))
g.fig.set_figheight(6)
g.fig.set_figwidth(16);

# plot some lines to guide the eye
for lim in range(60, 150, 20):
    x = np.linspace(-lim, lim, 3)
    for ax in g.axes.flat:
        ax.fill(x, 100 * (lim - abs(x)),
                color='gray', alpha=0.1, zorder=0)
        
g.axes[0, 0].set_title('Annual Members')
g.axes[0, 0].set_xlabel('Elevation Gain (meters)')
g.axes[0, 0].set_ylabel('Number of Rides')
g.axes[0, 1].set_title('Short-Term Pass Holders')
g.axes[0, 1].set_xlabel('Elevation Gain (meters)');

counts = trips.groupby(['usertype', np.sign(trips.elevation_gain)])['trip_id'].count().unstack()
percents = (100 * (counts.T / counts.sum(axis=1)).T).astype(int)


for i, col in enumerate(['Annual Member', 'Short-Term Pass Holder']):
    g.axes[0, i].text(0.98, 0.98,
                      ("Total downhill trips: {c[0]} ({p[0]}%)\n"
                       "Total flat trips: {c[1]} ({p[1]}%)\n"
                       "Total uphill trips: {c[2]} ({p[2]}%)".format(c=counts.loc[col].values,
                                                                     p=percents.loc[col].values)),
                      transform=g.axes[0, i].transAxes, ha='right', va='top', fontsize=14)

g.fig.savefig('figs/elevation.png', bbox_inches='tight')

print("total downhill trips:", (trips.elevation_gain < 0).sum())
print("total uphill trips:  ", (trips.elevation_gain > 0).sum())

weather = pd.read_csv('2015_weather_data.csv', index_col='Date', parse_dates=True)
weather.columns

by_date = trips.groupby(['date', 'usertype'])['trip_id'].count()
by_date.name = 'count'
by_date = by_date.reset_index('usertype').join(weather)

# add a flag indicating weekend
by_date['weekend'] = (by_date.index.dayofweek >= 5)

#----------------------------------------------------------------
# Plot Temperature Trend
g = sns.FacetGrid(by_date, col="weekend", hue='usertype', size=6)
g.map(sns.regplot, "Mean_Temperature_F", "count")
g.add_legend();

# do some formatting
g.axes[0, 0].set_title('')
g.axes[0, 1].set_title('')
g.axes[0, 0].text(0.05, 0.95, 'Monday - Friday', va='top', size=14,
                  transform=g.axes[0, 0].transAxes)
g.axes[0, 1].text(0.05, 0.95, 'Saturday - Sunday', va='top', size=14,
                  transform=g.axes[0, 1].transAxes)
g.fig.text(0.45, 1, "Trend With Temperature", ha='center', va='top', size=16);
for ax in g.axes.flat:
    ax.set_xlabel('Mean Temperature (F)')
g.axes.flat[0].set_ylabel('Rider Count')
g.fig.savefig('figs/temperature.png', bbox_inches='tight')

#----------------------------------------------------------------
# Plot Precipitation
g = sns.FacetGrid(by_date, col="weekend", hue='usertype', size=6)
g.map(sns.regplot, "Precipitation_In ", "count")
g.add_legend();

# do some formatting
g.axes[0, 0].set_ylim(-50, 600);
g.axes[0, 0].set_title('')
g.axes[0, 1].set_title('')
g.axes[0, 0].text(0.95, 0.95, 'Monday - Friday', ha='right', va='top', size=14,
                  transform=g.axes[0, 0].transAxes)
g.axes[0, 1].text(0.95, 0.95, 'Saturday - Sunday', ha='right', va='top', size=14,
                  transform=g.axes[0, 1].transAxes)
g.fig.text(0.45, 1, "Trend With Precipitation", ha='center', va='top', size=16);
for ax in g.axes.flat:
    ax.set_xlabel('Precipitation (inches)')
g.axes.flat[0].set_ylabel('Rider Count');
g.fig.savefig('figs/precipitation.png', bbox_inches='tight')

