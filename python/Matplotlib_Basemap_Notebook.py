import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib notebook')

x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

get_ipython().magic('matplotlib inline')

x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

plt.figure()
x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlim(x_values.min()*1.1, x_values.max()*1.1)
plt.ylim(y_values.min()*1.1, y_values.max()*1.1)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

from matplotlib import style
style.available

style.use('fivethirtyeight')
plt.figure()
x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlim(x_values.min()*1.1, x_values.max()*1.1)
plt.ylim(y_values.min()*1.1, y_values.max()*1.1)
plt.xlabel('x values')
plt.ylabel('y values')
plt.savefig('example_plot.svg')
plt.show()

get_ipython().magic("config InlineBackend.figure_format='retina'")

plt.figure()
x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlim(x_values.min()*1.1, x_values.max()*1.1)
plt.ylim(y_values.min()*1.1, y_values.max()*1.1)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

get_ipython().magic("config InlineBackend.figure_formats = {'svg',}")

plt.figure()
x_values = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.xlim(x_values.min()*1.1, x_values.max()*1.1)
plt.ylim(y_values.min()*1.1, y_values.max()*1.1)
plt.xlabel('x values')
plt.ylabel('y values')
plt.savefig('plot.svg')
plt.show()

import pandas as pd

PA_Frack_Wells_Dec = pd.read_csv('gas_data/Oil_Gas_Well_Historical_Production_Report.csv')
PA_Frack_Wells_Dec.head()

PA_Frack_Wells_Dec.describe()

style.use('seaborn-notebook')
plt.figure()
plt.scatter(x='Gas_Production_Days',y='Gas_Quantity',data=PA_Frack_Wells_Dec)
plt.xlabel('number of days of gas production (in Dec 2015)')
plt.ylabel('gas production (thousand cubic feet)')
plt.ylim(-20000,PA_Frack_Wells_Dec.Gas_Quantity.max()*1.1)
plt.xlim(0,32)
plt.show()

from mpl_toolkits.basemap import Basemap
style.use('seaborn-notebook')

m = Basemap(projection='mill',
            llcrnrlat = 39,
            llcrnrlon = -83,
            urcrnrlat = 43,
            urcrnrlon = -72,
            resolution='l')

core_x,core_y = m(PA_Frack_Wells_Dec.Well_Longitude.tolist(),PA_Frack_Wells_Dec.Well_Latitude.tolist())

m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.fillcontinents(color='tan',lake_color='lightblue',zorder=0)
m.scatter(x=core_x,y=core_y,c=PA_Frack_Wells_Dec.Gas_Quantity.tolist(),cmap='viridis',vmin=0)
m.colorbar(label='gas quantity (Mcf)')
plt.title('Unconventional gas wells in PA (as of Dec 2015)',)
plt.show()

m = Basemap(projection='mill',
            llcrnrlat = 39,
            llcrnrlon = -83,
            urcrnrlat = 43,
            urcrnrlon = -72,
            resolution='l')

core_x,core_y = m(PA_Frack_Wells_Dec.Well_Longitude.tolist(),PA_Frack_Wells_Dec.Well_Latitude.tolist())

m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.fillcontinents(color='tan',lake_color='lightblue',zorder=0)
m.scatter(x=core_x,y=core_y,c=PA_Frack_Wells_Dec.Gas_Quantity.tolist(),cmap='viridis',vmin=0)
m.readshapefile('./gas_data/marcellus_extent', 'marcellus_extent',color='red',linewidth=1)
m.colorbar(label='gas quantity (Mcf)')
plt.title('Unconventional gas wells in PA (with Marcellus Shale extent)')
plt.savefig('Marcellus_Shale_Map.pdf')
plt.show()

high_producing_wells = PA_Frack_Wells_Dec[PA_Frack_Wells_Dec.Gas_Quantity>300000]

m = Basemap(projection='mill',
            llcrnrlat = 39,
            llcrnrlon = -83,
            urcrnrlat = 43,
            urcrnrlon = -72,
            resolution='l')

core_x,core_y = m(high_producing_wells.Well_Longitude.tolist(),high_producing_wells.Well_Latitude.tolist())

m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.fillcontinents(color='tan',lake_color='lightblue',zorder=0)
m.scatter(x=core_x,y=core_y,c=high_producing_wells.Gas_Quantity.tolist(),cmap='viridis',vmin=0)
m.readshapefile('./gas_data/marcellus_extent', 'marcellus_extent',color='red',linewidth=1)
m.colorbar(label='gas quantity (Mcf)')
plt.title('high producing unconventional gas wells in PA (> 300,000 Mcf)')
plt.show()

earthquakes_mag7 = pd.read_csv('./earthquake_data/results.tsv',sep='\t')
earthquakes_mag7.drop([358,391,411],inplace=True) #drop lines with empty latitude which are there as empty strings '    '
earthquakes_mag7.columns

# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawcoastlines()
m.drawmapboundary(fill_color='white')

earthquake_lon = earthquakes_mag7.LONGITUDE.astype(float).tolist()
earthquake_lat = earthquakes_mag7.LATITUDE.astype(float).tolist()
earthquake_x,earthquake_y = m(earthquake_lon,earthquake_lat)

m.scatter(earthquake_x,earthquake_y,
          c=earthquakes_mag7.EQ_MAG_MW.astype(float).tolist(),
          cmap='seismic')
m.colorbar(label='Earthquake magnitude')
plt.title("Earthquake locations (M>7) since 1900")
plt.show()

m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='white')
m.readshapefile('./earthquake_data/tectonicplates/PB2002_boundaries',
                'PB2002_boundaries',color='black',linewidth=1)

earthquake_lon = earthquakes_mag7.LONGITUDE.astype(float).tolist()
earthquake_lat = earthquakes_mag7.LATITUDE.astype(float).tolist()
earthquake_x,earthquake_y = m(earthquake_lon,earthquake_lat)

m.scatter(earthquake_x,earthquake_y,
          c=earthquakes_mag7.EQ_MAG_MW.astype(float).tolist(),
          cmap='seismic')
m.colorbar(label='Earthquake magnitude')
plt.title("Earthquake locations (M>7) since 1900")
plt.show()

# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='white')
m.readshapefile('./earthquake_data/tectonicplates/PB2002_boundaries',
                'PB2002_boundaries',color='black',linewidth=1)
m.readshapefile('./earthquake_data/tectonicplates/PB2002_orogens',
                'PB2002_orogens',color='green',linewidth=1)

earthquake_lon = earthquakes_mag7.LONGITUDE.astype(float).tolist()
earthquake_lat = earthquakes_mag7.LATITUDE.astype(float).tolist()
earthquake_x,earthquake_y = m(earthquake_lon,earthquake_lat)

m.scatter(earthquake_x,earthquake_y,
          c=earthquakes_mag7.EQ_MAG_MW.astype(float).tolist(),
          cmap='seismic')
m.colorbar(label='Earthquake magnitude')
plt.title("Earthquake locations (M>7) since 1900")
plt.show()

plt.figure()
plt.scatter(x='EQ_MAG_MW',y='TOTAL_DEATHS',data=earthquakes_mag7)
plt.ylabel('total deaths from Earthquake')
plt.xlabel('magnitude of Earthquake')
plt.show()

earthquakes_mag7_highd = earthquakes_mag7[earthquakes_mag7.TOTAL_DEATHS>500]
earthquakes_mag7_lowd = earthquakes_mag7[earthquakes_mag7.TOTAL_DEATHS<500]

m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='white',zorder=0)
m.drawcoastlines(zorder=1)

badquake_lon = earthquakes_mag7_highd.LONGITUDE.astype(float).tolist()
badquake_lat = earthquakes_mag7_highd.LATITUDE.astype(float).tolist()
badquake_year = earthquakes_mag7_highd.YEAR.tolist()
badquake_casualties = earthquakes_mag7_highd.DEATHS.tolist()
badquake_x,badquake_y = m(badquake_lon,badquake_lat)
m.scatter(badquake_x,badquake_y,c='red',s=30,label='deaths > 5000',zorder=10)

notasbadquake_lon = earthquakes_mag7_lowd.LONGITUDE.astype(float).tolist()
notasbadquake_lat = earthquakes_mag7_lowd.LATITUDE.astype(float).tolist()
notasbadquake_x,notasbadquake_y = m(notasbadquake_lon,notasbadquake_lat)
m.scatter(notasbadquake_x,notasbadquake_y,c='green',s=10,label='deaths < 5000')
plt.legend(loc=4)
plt.title("Earthquake locations (M>7) since 1900")
plt.show()

import mpld3

fig, ax = plt.subplots()

m = Basemap(projection='moll',lon_0=0,resolution='l')
m.drawmapboundary(fill_color='white',zorder=0)
m.drawcoastlines(zorder=1)

scatter = m.scatter(badquake_x,badquake_y,s=30,zorder=2)
ax.set_title("Earthquake locations (M>7) since 1900 with >500 casualties")

labels = earthquakes_mag7_highd.YEAR.astype(float).tolist()
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.display()

bad_quake_w_focal_depth = earthquakes_mag7_highd.dropna(subset = ['FOCAL_DEPTH', 'TOTAL_INJURIES', 'EQ_MAG_MW'])
notasbad_quake_w_focal_depth = earthquakes_mag7_lowd.dropna(subset = ['FOCAL_DEPTH', 'TOTAL_INJURIES', 'EQ_MAG_MW'])

bad_quake_w_focal_depth.head()

fig = plt.figure()

ax = fig.add_subplot(111, axisbg='#EEEEEE')
ax.grid(color='white', linestyle='solid')

x = np.random.normal(size=1000)
ax.hist(notasbad_quake_w_focal_depth.FOCAL_DEPTH.tolist(),bins=30,
        histtype='stepfilled', fc='lightblue', alpha=0.5, label='earthquakes with deaths < 500')
ax.hist(bad_quake_w_focal_depth.FOCAL_DEPTH.tolist(),bins=30, 
        histtype='stepfilled', fc='red', alpha=0.5, label='earthquakes with deaths > 500')
plt.legend()
plt.xlabel('depth of earthquake (km)')
plt.ylabel('number of earthquakes')
mpld3.display()

import folium
map_1 = folium.Map(location=[20, 100], zoom_start=4,
                   tiles='Mapbox Bright')
for n in range(0,len(badquake_lon)):
    map_1.simple_marker([badquake_lat[n],badquake_lon[n]],
                        popup='Year: '+ str(badquake_year[n]) + ' casualties: ' + str(badquake_casualties[n]))
map_1.save(outfile='code_output/earthquakes.html')
map_1

from bokeh.io import output_file, show
from bokeh.models import (GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool)

map_options = GMapOptions(lat=20, lng=100, map_type="roadmap", zoom=4)

plot = GMapPlot(x_range=DataRange1d(), 
                y_range=DataRange1d(), 
                map_options=map_options, title="Earthquakes")

source = ColumnDataSource(
    data=dict(
        lat=badquake_lat,
        lon=badquake_lon,
    )
)

circle = Circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
output_file("code_output/gmap_plot.html")
show(plot)



