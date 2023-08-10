import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 5) # set default size of plots

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_context("talk")
sns.set_palette('Set2', 10)

from gtfs_utils import *
LOCAL_ZIP_PATH = 'data/sample/gtfs_181218.zip' 

conn = ftp_connect()
ftp_dir = get_ftp_dir(conn)
UPTODATE = 90 #days
our_uptodateness = get_uptodateness(ftp_dir, local_zip_path=LOCAL_ZIP_PATH)

if our_uptodateness > UPTODATE:
    get_ftp_file(conn)
    get_ftp_file(conn, file_name = 'Tariff.zip', local_zip_path = 'data/sample/tariff.zip' )

conn.quit()

tariff_df = extract_tariff_df(local_zip_path = 'data/sample/tariff.zip')
tariff_df.head()

import partridge as ptg

service_ids_by_date = ptg.read_service_ids_by_date(LOCAL_ZIP_PATH)
service_ids = service_ids_by_date[datetime.date(2017, 12, 21)]

feed = ptg.feed(LOCAL_ZIP_PATH, view={
    'trips.txt': {
        'service_id': service_ids,
    },
})

s = feed.stops
s.head()

r = feed.routes
r.head()

t = (feed.trips
     .assign(route_id=lambda x: pd.Categorical(x['route_id']))
    )
t.head()

def to_timedelta(df):
    '''
    Turn time columns into timedelta dtype
    '''
    cols = ['arrival_time', 'departure_time']
    numeric = df[cols].apply(pd.to_timedelta, unit='s')
    df = df.copy()
    df[cols] = numeric
    return df

f = (feed.stop_times[['trip_id', 'departure_time', 'arrival_time', 'stop_id']]
     .assign(date = datetime.date(2017, 12, 21))
     .merge(s[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id']], on='stop_id')
     # Much faster joins and slices with Categorical dtypes
     .merge(tariff_df.groupby(['zone_id', 'zone_name']).size().reset_index()[['zone_id', 'zone_name']], on='zone_id')
     .assign(zone_id=lambda x: pd.Categorical(x['zone_id']))
     .assign(zone_name=lambda x: pd.Categorical(x['zone_name']))
     .merge(t[['trip_id', 'route_id']], on='trip_id')
     .merge(r[['route_id', 'route_short_name', 'route_long_name']], on='route_id')
     .assign(route_id=lambda x: pd.Categorical(x['route_id']))
     .pipe(to_timedelta)
    )
f.head()

f.dtypes

f_no_nans = (f
             .assign(zone_name = lambda x: x.zone_name.cat.add_categories('').fillna(''))
             .assign(zone_id = lambda x: x.zone_id.cat.add_categories('').fillna(''))
             .assign(route_id = lambda x: x.route_id.cat.add_categories('').fillna(''))
            )
f_no_nans.fillna('')[f_no_nans.route_short_name.isnull()].head()

route_counts = (f_no_nans.fillna('')
 .groupby(['zone_name', 'route_id', 'route_short_name', 'route_long_name'])
 .size().sort_values(ascending=False).head(20).to_frame().rename(columns={0:'count'}))
route_counts

top_stops = (s.set_index('stop_name')
             .loc[:,'stop_id']
             .map(f.stop_id.value_counts())
             .sort_values(ascending=False)
             .head(20).to_frame())

def to_timedelta(df):
    '''
    Turn time columns into timedelta dtype
    '''
    cols = ['arrival_time', 'departure_time']
    numeric = df[cols].apply(pd.to_timedelta, unit='s')
    df = df.copy()
    df[cols] = numeric
    return df


se = (feed.stop_times.groupby(['trip_id'])
     .agg({'departure_time': 'min',
          'arrival_time': 'max'})
     .pipe(to_timedelta)
     .sort_values(['arrival_time', 'departure_time']))
se.head()

departures = pd.Series(1, se.departure_time).resample('1Min').sum()
departures.head()

arrivals =  pd.Series(1, se.arrival_time).resample('T').sum()
arrivals.head()

onroad = pd.concat([pd.Series(1, se.departure_time),  # departed add 1
                           pd.Series(-1, se.arrival_time)  # arrived substract 1
                           ]).resample('1Min').sum().cumsum().ffill()

df = (pd.concat([departures, arrivals, onroad], axis=1).reset_index()
        .rename(columns={'index': 'time', 0:'departures', 1:'arrivals', 2:'onroad'}))
df.head()

idx = pd.DatetimeIndex(df.time+datetime.datetime(2017, 12, 21))
idx

(df.assign(time=idx)
 .set_index('time')
 .resample('1min').sum()
 .ewm(span=60)
 .mean()
 .plot())

def to_numeric(df):
    '''
    Turn timedelta columns into numeric dtype
    '''
    cols = ['time']
    numeric = df[cols].apply(pd.to_numeric)
    df = df.copy()
    df[cols] = numeric
    return df

# we have to convert to numeric in order to use time in lmplot
melted_df = (df.pipe(to_numeric)
             .melt(id_vars=['time']))

g = sns.lmplot(x="time", y="value", hue="variable", data=melted_df, 
           size=5, aspect=3, line_kws={'linewidth': 0}, ci=None)
g.set(xticks=np.arange(25)*60*60*1e9, xticklabels=np.arange(25))

g = sns.FacetGrid(melted_df, row="variable", hue="variable", size=4, aspect=3, sharey=False)
g = g.map(plt.scatter, "time", "value", edgecolor="w", alpha=0.6)
g.set(xticks=np.arange(25)*60*60*1e9, xticklabels=np.arange(25))

def makebarplot(df):
    time = df.iloc[:, 0]   # extract the x-axis data
    fig = plt.figure()            # get the matplotlib plot figure
    fig.set_size_inches(15, 3)     # set the size of the plot
    ax = fig.add_subplot(1, 1, 1) # add a plot to the figure; Subplot
    # is confusing, though.  The magical "(1, 1, 1)" here means there
    # will be one row, one column, and we are working with plot number
    # 1, all of which is the same as just one plot.  There is a little
    # more documentation on this at:
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
    # turn off the borders (called spines)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # set all of the ticks to 0 length
    ax.tick_params(axis=u'both', which=u'both',length=0)
    # hide everything about the x-axis
    ax.axes.get_xaxis().set_visible(False)
    # remove grid
    ax.grid(linewidth=0)
    
    barwidth = 1                  # remove gaps between bars
    color = ["red", "blue", "green"] # set the colors for
    for row in range(1, len(color)+1): # make as many rows as colors
        # extract the correct column
        ongoing = df.iloc[:, row]
        # scale the data to the maximum
        ongoing = (ongoing / ongoing.max())

        # draw a black line at the left end
        left = 10
        border_width = 20
        d = border_width
        ax.barh(row, d, barwidth, color="black",
                left=left, edgecolor="none",
                linewidth=0)
        left += d
        # fill in the horizontal bar with the right color density
        # (alpha)
        for d, c in pd.concat((time, ongoing), axis=1).itertuples(index=False):
            ax.barh(row, d, barwidth,
                    alpha=0.9*c+.01,
                    color=color[row-1],
                    left=left,
                    edgecolor="none",
                    linewidth=0)
            left += d

        # draw a black line at the right end
        d = border_width
        ax.barh(row, d, barwidth,
                color="black",
                left=left, edgecolor="none",
                linewidth=0)
    # label the rows
    plt.yticks([1, 2, 3], ['departures', 'arrivals', 'onroad'], size=12)

makebarplot(df.fillna(0)
            .set_index('time').resample('5T').max().reset_index()
            .pipe(to_numeric))

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(3, rot=-.4, light=.7)
g = sns.FacetGrid(melted_df, row="variable", hue="variable", aspect=3, size=4, palette=pal, sharex=False, sharey=False)

# Draw the densities in a few steps
g.map(sns.kdeplot, "value", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, size=16, fontweight="bold", color=color, 
            ha="right", va="center", transform=ax.transAxes)

g.map(label, "value")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.1)

# Remove axes details that don't play will with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

route_filter = route_counts.reset_index().route_id
fr = (f[f.route_id.isin(route_filter)].groupby(['trip_id', 'route_id'], as_index=False)
     .agg({'departure_time': 'min',
          'arrival_time': 'max'})
     .pipe(to_timedelta)
     .sort_values(['arrival_time', 'departure_time']))
# converting the categorical to numerical
fr['route_id'] = fr['route_id'].cat.remove_unused_categories()
fr.head()

fr['route_id'].cat.categories

fr.shape

departures = (fr[['departure_time', 'route_id']]
              .assign(count=1).rename(columns={'departure_time': 'time'}))
arrivals = (fr[['arrival_time', 'route_id']]
            .assign(count=-1).rename(columns={'arrival_time': 'time'}))

con = pd.concat([departures, arrivals]).set_index('time')

onroad_route = (con.groupby([pd.Grouper(freq='1Min'), 'route_id'])
                .sum()
                .unstack()
                .cumsum()
                .stack()
                .reset_index()
                .pipe(to_numeric))

g = sns.FacetGrid(onroad_route, hue='route_id', size=10, aspect=1)
g.map(plt.step, 'time', 'count', linewidth=1,).add_legend()
g.set(xticks=np.arange(25)*60*60*1e9, xticklabels=np.arange(25))

g = sns.FacetGrid(onroad_route, row='route_id', hue='route_id', size=1, aspect=10, 
                  row_order=route_filter,
                  margin_titles=True)
g.map(plt.step, 'time', 'count', linewidth=1)

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=0)
g.set(yticks=[5, 10, 15])
g.set(xticks=np.arange(25)*60*60*1e9, xticklabels=np.arange(25))

[plt.setp(ax.texts, text="") for ax in g.axes.flat] # remove the original texts
                                                    # important to add this before setting titles
g.set_titles(row_template = '{row_name}')
g.despine()

from bokeh.io import output_notebook, show
output_notebook()

from bokeh.plotting import figure
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.palettes import inferno, magma

# web mercator coordinates (got them here https://epsg.io/map)
Center_Israel = x_range, y_range = ((3852120,3852120+10e4), (3729820,3729820+10e4/1.3))

plot_width  = int(540)
plot_height = int(plot_width//1.3)

def base_plot(tools='pan,wheel_zoom,box_zoom,reset', active_drag='pan', 
              active_scroll='wheel_zoom', toolbar_location='left',
              plot_width=plot_width, plot_height=plot_height, **plot_args):
    p = figure(tools=tools, active_drag=active_drag, active_scroll=active_scroll,
               toolbar_location=toolbar_location,
                plot_width=plot_width, plot_height=plot_height,
                x_range=x_range, y_range=y_range, outline_line_color=None,
                min_border=0, min_border_left=0, min_border_right=0,
                min_border_top=0, min_border_bottom=0, **plot_args)
    
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def wgs84_to_web_mercator(df, lon="stop_lon", lat="stop_lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

stops = (s[['stop_lon', 'stop_lat', 'stop_id', 'zone_id']].set_index('stop_id')
         .pipe(wgs84_to_web_mercator)
         .assign(counts = f.stop_id.value_counts())
         .sort_values(by='counts', ascending=False))

pal = inferno(256)
c256 = 255 - pd.cut(stops.counts.fillna(0), 256).cat.codes
colors = [pal[c] for _, c in c256.iteritems()]

options = dict(line_color=None, fill_color=colors, size=5)

p = base_plot()
p.add_tile(CARTODBPOSITRON)

p.circle(x=stops['x'], y=stops['y'], **options)
show(p)

from bokeh.plotting import figure
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.palettes import inferno, magma

# web mercator coordinates (got them here https://epsg.io/map)
Center_Israel = x_range, y_range = ((3852120,3852120+10e4), (3729820,3729820+10e4/1.3))

plot_width  = int(540)
plot_height = int(plot_width//1.3)

def base_plot(tools='pan,wheel_zoom,box_zoom,reset', active_drag='pan', 
              active_scroll='wheel_zoom', toolbar_location='left',
              plot_width=plot_width, plot_height=plot_height, **plot_args):
    p = figure(tools=tools, active_drag=active_drag, active_scroll=active_scroll,
               toolbar_location=toolbar_location,
                plot_width=plot_width, plot_height=plot_height,
                x_range=x_range, y_range=y_range, outline_line_color=None,
                min_border=0, min_border_left=0, min_border_right=0,
                min_border_top=0, min_border_bottom=0, **plot_args)
    
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def wgs84_to_web_mercator(df, lon="stop_lon", lat="stop_lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

stops = (s[['stop_lon', 'stop_lat', 'stop_id']].set_index('stop_id')
         .pipe(wgs84_to_web_mercator)
         .assign(counts = f.stop_id.value_counts())
         .sort_values(by='counts', ascending=False))

pal = inferno(256)
c256 = 255 - pd.cut(stops.counts.fillna(0), 256).cat.codes
colors = [pal[c] for _, c in c256.iteritems()]

options = dict(line_color=None, fill_color=colors, size=5)

p = base_plot()
p.add_tile(CARTODBPOSITRON)

p.circle(x=stops['x'], y=stops['y'], **options)
show(p)

f[f.route_id.isin(['11525', '11526'])] # Line 6 Modi'in Illit, the same holds for line 5









































