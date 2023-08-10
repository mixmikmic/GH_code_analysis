get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

trips = pd.read_csv('2015_trip_data.csv',
                    parse_dates=['starttime', 'stoptime'],
                    infer_datetime_format=True)
t_start = pd.DatetimeIndex(trips['starttime'])
t_stop = pd.DatetimeIndex(trips['stoptime'])
trips['date'] = t_start.date.astype('datetime64')
trips['starttime'] = t_start.time
trips['stoptime'] = t_stop.time
trips['minuteofday'] = t_start.hour * 60 + t_start.minute
trips['minutes'] = trips.tripduration / 60.

stations = pd.read_csv('2015_station_data.csv')
pronto_shop = dict(id=54, name="Pronto shop",
                   terminal="Pronto shop",
                   lat=47.6173156, long=-122.3414776,
                   dockcount=100, online='10/13/2014')
stations = stations.append(pronto_shop, ignore_index=True)
distances = pd.read_csv('station_distances.csv', index_col='terminal')
distances /= 1609.34  # convert meters to miles

trips['distance'] = [distances.loc[ind] for ind in
                     zip(trips.from_station_id, trips.to_station_id)]
trips['speed'] = trips.distance * 60 / trips.minutes

trips.head()

groups = trips.groupby(['from_station_id', 'to_station_id'])
paired = groups.aggregate({'distance':'mean', 'trip_id':'count',
                           'from_station_name':'first',
                           'to_station_name':'first'})
paired.rename(columns={'trip_id': 'count'}, inplace=True)

countmat = paired['count'].unstack()
total = countmat + countmat.T
total.values.flat[::total.shape[0] + 1] /= 2
paired['total'] = total.stack()

fig = plt.figure()
ax = plt.axes(yscale='log')
ax.plot(paired['distance'], paired['total'], '.k')
ax.set_xlabel('distance between stations')
ax.set_ylabel('number of trips');
ax.plot(6.83, 95, 's', ms=30, mec='red', mfc='none', mew=1)

fig.savefig('figs/trips_by_distance.png', bbox_inches='tight')

station_id_map = trips.groupby('from_station_id')['from_station_name'].first()

def get_group(id1, id2, include_reverse=False):
    query = '(usertype == "Annual Member")'
    
    if include_reverse:
        query += (' & ((from_station_id == "{0}" & to_station_id == "{1}") |'
                 '(from_station_id == "{1}" & to_station_id == "{0}"))')
    else:
        query += ' & (from_station_id == "{0}" & to_station_id == "{1}")'
    return trips.query(query.format(id1, id2))

fig, ax = plt.subplots()

id1, id2 = paired.query('distance > 6 & total > 50').reset_index()['from_station_id']
for route in [(id1, id2), (id2, id1)]:
    group = get_group(*route)
    names = station_id_map[route[0]], station_id_map[route[1]]
    lines = ax.plot(group['starttime'], group['minutes'], 'o', ms=5,
                    label="{0} $\\to$ {1}".format(*(' '.join(n.split()[:2]) for n in names)))
    color = lines[0].get_color()
    ax.plot(group['stoptime'], group['minutes'], 'o', ms=5, color=color)
    for i in range(group.shape[0]):
        ax.plot([group['starttime'].values[i], group['stoptime'].values[i]],
                2 * [group['minutes'].values[i]], '-', color=color, alpha=0.3)

ax.text(0.98, 0.02, "{0} born in {1}".format(group['gender'].iloc[0], int(group['birthyear'].iloc[0])),
        ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

ax.xaxis.set_major_locator(plt.MultipleLocator(2 * 60 * 60))
ax.legend(loc='best', fontsize=12)
ax.set_title('{0} \nto\n {1}'.format(*names))
ax.set_xlabel('ride start time')
ax.set_ylabel('ride duration');

fig.savefig('figs/power_user.png', bbox_inches='tight')

from sklearn.cluster import MeanShift

def compute_compactness(group, min_samples=25, bandwidth=5):
    """Return a measure of the compactness of the group"""
    # arrange data to cluster: divide minuteofday by 10
    # to increase effective bandwidth by a factor of 10
    X = np.vstack([group.minuteofday / 10,
                   group.minutes]).T
    if X.shape[0] < min_samples:
        return 0

    # compute the meanshift clusters, and count number of points in each
    c = MeanShift(bandwidth=bandwidth).fit_predict(X)
    counts = pd.Series(c).groupby(c).count()
    
    # Select only the points from the dominant cluster
    c = pd.Series(c).map(counts == counts.max())
    
    if c.sum() < min_samples or c.sum() < 0.9 * len(c):
        return 0
    else:
        return bandwidth / X[c.values].std(1).max()
    
compute_compactness(get_group('BT-03', 'UD-01'))

subset = trips.query('usertype == "Annual Member"')
groups = subset.groupby(['from_station_id', 'to_station_id'])
compactness = groups.apply(compute_compactness).fillna(0).unstack()

ranked = compactness.unstack().sort_values(ascending=False)
ranked.iloc[:10]

def analyze_pair(id1, id2):
    subset = get_group(id1, id2, True).copy()
    unique = pd.value_counts(subset['birthyear'])
    subset = subset[subset['birthyear'] == unique.index[0]]
    
    def mean_time(col):
        #s = pd.DatetimeIndex(s.astype('datetime64'))
        return np.mean([c.hour * 3600 + c.minute * 60 + c.second
                        for c in col])

    AMPM = subset.groupby('from_station_id')['starttime'].aggregate(mean_time)
    AMPM.sort_values(inplace=True)
    subset['AMPM'] = np.where(subset.from_station_id == np.argmin(AMPM),
                              'morning', 'afternoon')
    
    print('{0} -> {1}'.format(id1, id2))
    print(pd.value_counts(subset['birthyear']))
    print(pd.value_counts(subset['gender']))
    print("distance:", distances.loc[id1, id2])
    print("Date Range:", subset.date.min(), "to", subset.date.max())
    print("-----------------------------")
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
    fig.suptitle('{0} $\longrightarrow$ {1} '
                 ''.format(station_id_map[id1],
                           station_id_map[id2]),
                 size=14)
    
    ax[0].text(0.02, 0.98, "{1:d} total trips\ndistance = {0:.2f} mi".format(distances.loc[id1, id2],
                                                                             len(subset)),
            ha='left', va='top', transform=ax[0].transAxes, fontsize=14)
    
    names = station_id_map[AMPM.index[0]], station_id_map[AMPM.index[1]]
    
    colors = plt.rcParams['axes.color_cycle']
    for from_station, color in zip(AMPM.index, colors):
        half = subset[subset.from_station_id == from_station]
        if AMPM.index[0] == from_station:
            order = names
        else:
            order = names[::-1]
        ax[0].scatter(half['starttime'].values, half['minutes'].values,
                      c=color,
                      label="{0} $\\to$ {1}".format(*(' '.join(n.split()[:2]) for n in order)))
        ax[0].scatter(half['stoptime'].values, half['minutes'].values,
                      c=color)
        for i in range(half.shape[0]):
            ax[0].plot([half['starttime'].values[i], half['stoptime'].values[i]],
                       2 * [half['minutes'].values[i]], '-', color=color, alpha=0.3)
        
    ax[0].legend(loc='lower right', fontsize=14)
    ax[0].xaxis.set_major_locator(plt.MultipleLocator(4 * 60 * 60))
    ax[0].set_ylabel('trip duration (minutes)')
    
    ns_in_day = 24 * 60 * 60 * 1E9
    subset['daynumber'] = (subset.date - subset.date.iloc[0]).astype(int) / ns_in_day
    
    def dateformat(x, *args):
        return str(subset.date.iloc[0] + pd.datetools.timedelta(days=int(x))).split()[0]
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(dateformat))

    for AMPM in ['morning', 'afternoon']:
        sns.regplot('daynumber', 'minutes', data=subset.query('AMPM == "{0}"'.format(AMPM)),
                    ax=ax[1])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('Date')
    ax[1].text(0.98, 0.98, "{0} born in {1}".format(subset['gender'].iloc[0], int(subset['birthyear'].iloc[0])),
               ha='right', va='top', transform=ax[1].transAxes, fontsize=14)

    return fig

for i, ind in enumerate([0, 1, 3, 4, 7]):
    fig = analyze_pair(*ranked.index[ind])
    fig.savefig('figs/extreme-user-{0}.png'.format(i + 1))

