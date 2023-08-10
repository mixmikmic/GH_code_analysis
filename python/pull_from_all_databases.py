from campaigns import campaigns
my_campaigns = {}
for db in campaigns:
    print db
    try:
        c = Campaign.objects.using(db).get(id=1)
        ##if 'simz' in db:
        if 'CANON' in c.name:
            print('{:25s} {}'.format(db, c.description))
            my_campaigns[db] = c
    except Exception as e:
        ##print('{:25s} *** {} ***'.format(db, e))
        pass

get_ipython().magic('time')
import geopandas as gpd
df = gpd.GeoDataFrame()
for db, c in my_campaigns.iteritems():
    c_sum = 0
    if c.startdate.year != 2013:
        continue
    for platform in Platform.objects.using(db).all():
        sdtp = SimpleDepthTime.objects.using(db).filter(activity__platform=platform)
        sdtp = sdtp.order_by('instantpoint__timevalue').values('activity__name',
                                 'activity__platform__name', 'activity__platform__color',
                                 'activity__maptrack', 'instantpoint__timevalue', 'depth')
        try:
            c_sum += sdtp.count()
            p_df = gpd.GeoDataFrame.from_records(sdtp, index='instantpoint__timevalue')
        except KeyError:
            ##print "No time series of {} from ()".format(platform, db)
            pass
        df = df.append(p_df)
    print "{} records from {}".format(c_sum, db)

# Monkey patch PostGIS/GEOS LineString with properties that GeoPandas's GeoSeries expects
from django.contrib.gis.geos.linestring import LineString
LineString.type = LineString.geom_type
LineString.bounds = LineString.extent

# GeoPandas default geometry column is named 'geometry'; and rename the color column
df = df.rename(columns={'activity__maptrack': 'geometry'})
df = df.rename(columns={'activity__platform__color': 'color'})

# Make our color column a value that's understood
df['color'] = '#' + df['color']

df['geometry'].values[0].bounds

df.dropna().total_bounds

coast_df = gpd.GeoDataFrame.from_file('./ne_10m_coastline.shp')

get_ipython().magic('matplotlib inline')
from pylab import plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18.0, 18.0)
ax = df.dropna().plot()
ax.set_autoscale_on(False)
coast_df.plot()

get_ipython().magic('matplotlib inline')
from pylab import plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18.0, 6.0)
fig, ax = plt.subplots(1,1)
ax.set_title('All MBARI CANON Campaign Data')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()

for p, c in df.set_index('activity__platform__name'
                        )['color'].to_dict().iteritems():
    pdf = df.loc[df['activity__platform__name'] == p]
    for a in pdf['activity__name'].unique():
        # Plot each activity individually so as not to connect them
        pdf.loc[pdf['activity__name'] == a].depth.plot(label=p, c=c)
        
# See http://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
_ = ax.legend(by_label.values(), by_label.keys(), loc='best', ncol=4)

from bokeh.plotting import figure, show
from bokeh.io import output_notebook

fig = figure(width = 900, height = 400,
             title = 'All MBARI CANON Campaign Data',
             x_axis_type="datetime",
             x_axis_label='Time (GMT)',
             y_axis_label = 'Depth (m)')

output_notebook(hide_banner=True)

# Negate the depth values until we figure out inverting a bokeh axis
df2 = df.copy()
df2['depth'] *= -1
for p, c in df2.set_index('activity__platform__name'
                         )['color'].to_dict().iteritems():
    pdf = df2.loc[df['activity__platform__name'] == p]
    for a in pdf['activity__name'].unique():
        adf = pdf.loc[pdf['activity__name'] == a]['depth']
        fig.line(x=adf.index, y=adf.values, line_color=c)
    
_ = show(fig)

mp_total = 0
for db, c in my_campaigns.iteritems():
    try:
        mpc = CampaignResource.objects.using(db).get(
            campaign=c, resource__name='MeasuredParameter_count')
        print('{:25s} {:-12,}'.format(db, int(mpc.resource.value)))
        mp_total += int(mpc.resource.value)
    except Exception:
        pass
    
print('{:25s} {:12s}'.format('', 12*'-'))
print('{:25s} {:-12,}'.format('total', mp_total))

def df_stats(platform):
    df = pd.DataFrame()
    for db, c in my_campaigns.iteritems():
        aps = ActivityParameter.objects.using(db).filter(activity__platform__name=platform)
        aps = aps.values('activity__startdate', 'parameter__name', 'mean', 'p025', 'p975')
        df = df.append(pd.DataFrame.from_records(aps))
        
    return df

get_ipython().magic('matplotlib inline')
from pylab import plt
plt.rcParams['figure.figsize'] = (14.0, 4.0)
plt.style.use('ggplot')
def ts_plot(df, parm):
    d = df[df['parameter__name'] == parm]
    d.plot(x='activity__startdate', marker='*')
    plt.ylabel(parm)

dorado_df = df_stats('dorado')
dorado_df.head()

for p in dorado_df.parameter__name.unique():
    ts_plot(dorado_df, p)



