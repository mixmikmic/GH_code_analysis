import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import fiona
import shapefile
from matplotlib.collections import PatchCollection
from sodapy import Socrata
from descartes import PolygonPatch
import requests
import json
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

df_price = pd.read_csv('/Users/fangjie/Downloads/Neighborhood/Neighborhood_Zhvi_AllHomes.csv')

df_price.head(5)

df_price[df_price.City == 'San Francisco'].head()

d[lambda df: df.columns[7:]].transpose().head()

fig, ax = plt.subplots(1,1,figsize = (10,6))
d = df_price[df_price.City == 'San Francisco']
d.set_index(d.RegionName, inplace = True)
t = d[lambda df: df.columns[7:]].transpose()
t.plot(ax = ax)

t.columns

# the mission district which has been on newsreport for gentrification
list_of_gentrification_areas = ['Mission']

# list_of_gentrification_areas = ['Mission','South of Market', 'Van Ness - Civic Center', 'China Town']
list_of_rich_area = ['']

fig, ax = plt.subplots(1,1,figsize = (10,6))
j = 0
t['rescaled_price_accumulate'] = 0
for i, n in enumerate(t.columns):
    t['rescaled_price'] = t[n]/t[n].min()
    
    # directly plot out gentrification areas:
    if n in list_of_gentrification_areas:
        t.rescaled_price.plot(ax = ax, label = n)
    
    # remember non-gentrification areas:
    elif not t.rescaled_price.isnull().values.any():
        t['rescaled_price_accumulate'] += t['rescaled_price']
        j += 1

# at end, plot areas except for gentrification areas on overage..
(t['rescaled_price_accumulate']/j).plot(ax = ax, label = 'Other Neighborhoods On Average')

ax.legend()
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.title("Gentrification in San Francisco: Mission vs. All Rest", fontsize = 15)
ax.set_ylabel("Median Housing Price Increase since 1996", fontsize = 12)

# let's try San Francisco

df_price = df_price[df_price.City == 'San Francisco']
# df_price = df_price[df_price.State == 'CA']
df_price['avg_price_2005'] = df_price[[col for col in list(df_price) if col.startswith('2005')]].mean(axis = 1, skipna=None)
df_price['avg_price_2017'] = df_price[[col for col in list(df_price) if col.startswith('2017')]].mean(axis = 1, skipna=None)
df_price['avg_price_increase_2005_to_2017'] = 100.0*(df_price['avg_price_2017'] / df_price['avg_price_2005'] - 1)
df_price.set_index(df_price.RegionID, inplace = True)

# load shapefile
# download boundary files of states here: https://www.zillow.com/howto/api/neighborhood-boundaries.htm
shapefilename = '/Users/fangjie/Downloads/ZillowNeighborhoods-CA/ZillowNeighborhoods-CA'
# shapefilename = '/Users/fangjie/Downloads/Zoning Districts/geo_export_ff16c4b8-5bf2-4e47-a498-6513bb52d237'
shp = fiona.open(shapefilename+'.shp')
coords = shp.bounds

coords = (-122.51494757834968,
 37.70808923327349,
 -122.35696687665978,
 37.81157429336938)

shp.close()
w, h = coords[2] - coords[0], coords[3] - coords[1]
extra = 0.01
coords

# create basemap object
m = Basemap(
    projection='tmerc', ellps='WGS84',
    lon_0=np.mean([coords[0], coords[2]]),
    lat_0=np.mean([coords[1], coords[3]]),
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - (extra * h), 
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + (extra * h),
    resolution='i',  suppress_ticks=True)

_out = m.readshapefile(shapefilename, name='SF', drawbounds=False, color='none', zorder=2)

# prepare data
# set up a map dataframe
df_map = pd.DataFrame({
    'poly': [Polygon(hood_points) for hood_points in m.SF],
    'name': [hood['Name'] for hood in m.SF_info],
    'city': [hood['City'] for hood in m.SF_info],
    'RegionID': [hood['RegionID'] for hood in m.SF_info],
})

# drop some duplicates
df_map = df_map.drop_duplicates(subset = ['RegionID'])
df_map = df_map[df_map.city == 'San Francisco']
df_map['RegionID'] = df_map.RegionID.astype('int')
df_map.set_index(df_map.RegionID, inplace=True)

# join data together
df_map = df_map.join(df_price, how='inner', on = 'RegionID', lsuffix='_map', rsuffix='_price')

df_map.avg_price_increase_2005_to_2017.hist()

# visualize
# We'll only use a handful of distinct colors for our choropleth. So pick where
# you want your cutoffs to occur. Leave zero and ~infinity alone.

breaks = [40.] + [50, 60, 70., 80.] + [100]

def self_categorize(entry, breaks):
    for i in range(len(breaks)-1):
        if entry > breaks[i] and entry <= breaks[i+1]:
            return i
    return -1
df_map['jenks_bins'] = df_map.avg_price_increase_2005_to_2017.apply(self_categorize, args=(breaks,))

jenks_labels = ['0-40%% Increase']+[">%d%% Increase"%(perc) for perc in breaks[:-1]]

# #Or, you could always use Natural_Breaks to calculate your breaks for you:
# from pysal.esda.mapclassify import Natural_Breaks
# breaks = Natural_Breaks(df_map.avg_price_increase_2005_to_2017, initial=0, k=5)
# df_map['jenks_bins'] = -1 #default value if no data exists for this bin
# df_map['jenks_bins'][df_map.avg_price_increase_2005_to_2017 > 0] = breaks.yb
# jenks_labels = ['', "baseline"]+["> %d increase"%(perc) for perc in breaks.bins[:-1]]

def custom_colorbar(cmap, ncolors, labels, **kwargs):    
    """Create a custom, discretized colorbar with correctly formatted/aligned labels.
    
    cmap: the matplotlib colormap object you plan on using for your graph
    ncolors: (int) the number of discrete colors available
    labels: the list of labels for the colorbar. Should be the same length as ncolors.
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable
        
    norm = BoundaryNorm(range(0, ncolors), cmap.N)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1)+0.5)
    colorbar.set_ticklabels(range(0, ncolors))
    colorbar.set_ticklabels(labels)
    return colorbar

figwidth = 10
fig = plt.figure(figsize=(figwidth, figwidth*h/w))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)

cmap = plt.get_cmap('Blues')
# draw neighborhoods with grey outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)
# apply our custom color values onto the patch collection
cmap_list = [cmap(val) for val in (df_map.jenks_bins.values - df_map.jenks_bins.values.min())/(
                  df_map.jenks_bins.values.max()-float(df_map.jenks_bins.values.min()))]
pc.set_facecolor(cmap_list)
ax.add_collection(pc)

#Draw a map scale

m.drawmapscale(coords[0], coords[1],
    coords[0], coords[1], 10.,
    fontsize=16, barstyle='fancy', labelstyle='simple',
    fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
    zorder=5, ax=ax,)

# ncolors+1 because we're using a "zero-th" color
cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
cbar.ax.tick_params(labelsize=16)

fig.suptitle("% Housing Price Increase in San Francisco", fontsize = 5, fontdict={'size':20, 'fontweight':'bold'}, y=0.92)
#ax.set_title("Using location data collected from my Android phone via Google Takeout", fontsize=14, y=0.98)
# qax.text(1.35, 0.04, "Collected from 2012-2014 on Android 4.2-4.4\nGeographic data provided by data.seattle.gov", 
#     ha='right', color='#555555', style='italic', transform=ax.transAxes)
# ax.text(1.35, 0.01, "BeneathData.com", color='#555555', fontsize=16, ha='right', transform=ax.transAxes)

plt.savefig('San Francisco housing price changes.png', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#F2F2F2')



