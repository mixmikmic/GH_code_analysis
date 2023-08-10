#import pandas and numpy libraries
import pandas as pd
import numpy as np
import sys #sys needed only for python version
import seaborn as sns

#display versions of python and packages
print('\npython version ' + sys.version)
print('pandas version ' + pd.__version__)
print('numpy version ' + np.__version__)
print('seaborn version ' + sns.__version__)

# read in text file, which is tab delimited, and set global unique indentifier as the index column
df = pd.read_csv('ebird_rubythroated_Jan2013_Aug2015.txt', sep='\t', index_col='GLOBAL UNIQUE IDENTIFIER', 
                 error_bad_lines=False, warn_bad_lines=False, low_memory=False)
# received error on one row of data...
# pandas.parser.CParserError: Error tokenizing data. C error: Expected 44 fields in line 36575, saw 45
# set error_bad_lines to false so it would just skip that line and keep going
# received dtype error on some columns having mixed-type data. added low_memory=False option to resolve.

# set pandas to output all of the columns in output (was truncating)
pd.options.display.max_columns = 50

# display the first 10 rows to view example data (this time all columns)
print(df.head(n=2))

#display the number of rows
print(df.shape[0])

# easier to read example of values in 1 row. picked 1000 because it's not far south like head rows are
print(df.iloc[2000,:])

# narrow dataset down to just columns we need
df_loc_date = df.loc[:,('COUNTRY','STATE_PROVINCE','LOCALITY','COUNTY','LATITUDE','LONGITUDE','OBSERVATION DATE')]

# put the month, year, and date into their own columns
df_loc_date['OBS MONTH'] = df_loc_date['OBSERVATION DATE'].str[5:7].astype(int)
df_loc_date['OBS DAY'] = df_loc_date['OBSERVATION DATE'].str[8:10].astype(int)
df_loc_date['OBS YEAR'] = df_loc_date['OBSERVATION DATE'].str[0:4].astype(int)
# pretend they're all in the same year so we can get earliest month and day and treat like a date
df_loc_date['OBS MONTHDAY'] = '2014-' + df_loc_date['OBSERVATION DATE'].str[5:10].astype(str) 
df_loc_date['OBS MONTHDAY'] = pd.to_datetime(df_loc_date['OBS MONTHDAY'])

# add columns for rounded latitude and longitude to make it less granular
df_loc_date['SHORT LAT'] = df_loc_date['LATITUDE'].round(0)
df_loc_date['SHORT LONG'] = df_loc_date['LONGITUDE'].round(0)

print(df_loc_date.iloc[2000,:])

#show the plot here instead of in pop-up
get_ipython().magic('matplotlib inline')
# seaborn joint plot with latitude and longitude
sns.jointplot("LONGITUDE", "LATITUDE", data=df_loc_date.where((df_loc_date['LATITUDE'] > 36) & (df_loc_date['LATITUDE'] < 39 )))
sns.plt.show()
# seaborn joint plot with less-granular latitude and longitude
sns.jointplot("SHORT LONG", "SHORT LAT", data=df_loc_date.where((df_loc_date['LATITUDE'] > 36) & (df_loc_date['LATITUDE'] < 39 )))
sns.plt.show()

# get the earliest sighting in any year by short long and short lat group
df_first_seen = df_loc_date.loc[:,('SHORT LAT','SHORT LONG','OBS MONTHDAY')].groupby(['SHORT LAT','SHORT LONG']).min()

#here's how the pivot table looks
print(df_first_seen.head(n=20))

# and here's how to access a value by index
print(df_first_seen.dtypes)
print(df_first_seen.ix[9,-84])

# however, I don't want those values to be indexes, I want it to act like a normal dataframe
df_first_seen = df_first_seen.reset_index()
print(df_first_seen.dtypes)
lat96 = df_first_seen[df_first_seen['SHORT LAT'] == 9]
print(lat96[lat96['SHORT LONG'] == -84])
print(df_first_seen.shape)

# seaborn joint plot with less-granular latitude and longitude
sns.jointplot("SHORT LONG", "SHORT LAT", data=df_first_seen) #where((df_first_seen['SHORT LAT'] >= 36) & (df_first_seen['SHORT'] <= 39 )))
sns.plt.show()

sns.distplot(df_first_seen['OBS MONTHDAY'].apply(lambda x: int((x.timetuple().tm_yday/7))))
sns.plt.show()

#import packages needed for bokeh plots on date-related data
from bokeh.plotting import *
from bokeh.palettes import Spectral6
import datetime

#set plots to show up inline
output_notebook()

#create labels for "map" plot
p = figure(title = "Ruby-Throated Hummingbird Migration Map (colors tied to week of 1st observation)")
p.xaxis.axis_label = 'Longitude'
p.yaxis.axis_label = 'Latitude'
p.title_text_font_size = '8'

#determine the range or values
plotdata = df_first_seen #.where((df_first_seen['SHORT LAT'] >= 36) & (df_first_seen['SHORT LAT'] <= 39 ))
high = plotdata['OBS MONTHDAY'].max().timetuple().tm_yday/7 #highest year day in resulting set
low = plotdata['OBS MONTHDAY'].min().timetuple().tm_yday/7 #lowest year day in resulting set
#print(int(low),int(high)) #weeks 1 to 52
matrix = [ ([i*85] * 13) for i in range(4) ]
plotdata['WOY'] = plotdata['OBS MONTHDAY'].apply(lambda x: int((x.timetuple().tm_yday/7/((high-low)/52))))
r = [x for row in reversed(matrix) for x in row]# [x*4.4 for x in range(0,53)]
g = [x*4.4 for x in range(0,53)]
b = [x for row in matrix for x in row] #need one more value
b.append(255)
r.append(0)
#convert to hex values
colors = [
    "#%02x%02x%02x" % (int(r), 50, int(b)) for r,g,b in zip(r,g,b)
 ]
#print(plotdata['WOY'].max())
color_dict = {}
#assign a hex color to each value for first sighting 0-52 weeks
color_dict = dict(zip(range(0,53),colors))
#print(color_dict)
#assign colors to values in table to be plotted
colors = [color_dict[x*2] if x < 26 else color_dict[52] for x in plotdata['WOY']] #added *2 to skip though these colors faster, until > 26, then stay at max
#print(colors)

# NOTE FOR LATER: show example location name on select?

#create and display the "map"
p.circle(plotdata['SHORT LONG'], plotdata['SHORT LAT'],
        fill_color=colors, line_color = None, fill_alpha = 0.8 )
show(p)

# get the earliest sighting in any year by short long and short lat group
df_first_city_state = df_loc_date.loc[:,('SHORT LAT','SHORT LONG','COUNTRY','STATE_PROVINCE','COUNTY','LOCALITY','OBS MONTHDAY')].groupby(['COUNTRY','STATE_PROVINCE','COUNTY'])
df_first_city_state = df_first_city_state.agg({'OBS MONTHDAY' : np.min, 'SHORT LAT': np.mean, 'SHORT LONG': np.mean})
#pd.set_option('display.max_rows', len(df_first_city_state))
count_orig=df_loc_date.where((df_loc_date['COUNTRY'] == 'United States') & (df_loc_date['STATE_PROVINCE'] == 'Virginia') & (df_loc_date['COUNTY'] == 'Harrisonburg')).count()
print('There are',count_orig[1], 'observations in Harrisonburg, VA')
df_city_state_co_summary = df_first_city_state.ix['United States','Virginia','Harrisonburg']

#***NOTE***: the indexes appear to be changing each time I run this! need a better way to reference?
#solution: have to use field names instead of index numbers (they still work normally)
#print(df_city_state_co_summary)
print('AVG LATITUDE:',df_city_state_co_summary['SHORT LAT'])
print('AVG LONGITUDE:',df_city_state_co_summary['SHORT LONG'])
obs_mon = df_city_state_co_summary['OBS MONTHDAY'].timetuple().tm_mon
obs_day = df_city_state_co_summary['OBS MONTHDAY'].timetuple().tm_mday
print('EARLIEST SIGHTING:',str(obs_mon)+'/'+str(obs_day))
#pd.reset_option('display.max_rows')

#print(df_loc_date['COUNTRY'].unique().tolist())
#now allow the user to enter input from available countries, states, and cities
from ipywidgets import *
from IPython.display import display

countryselect = widgets.Dropdown(
    options=df_loc_date['COUNTRY'].unique().tolist(),
    value = 'United States',
    description='Country:',
)
#display(countryselect)

stateselect = widgets.Dropdown(
    options=df_loc_date['STATE_PROVINCE'][df_loc_date['COUNTRY']==countryselect.value].unique().tolist(),
    value = 'Virginia',
    description='State/Province:',
)
#display(stateselect)

coselect = widgets.Dropdown(
    options=df_loc_date['COUNTY'][df_loc_date['STATE_PROVINCE']==stateselect.value].unique().tolist(),
    value = 'Harrisonburg',
    description='County:',
)
#display(coselect)
#df_loc_date['COUNTY'].notnull()
#print(countryselect.value, stateselect.value, coselect.value)

def mapMigrationByCity(country, state, co):
    #this triggers on-change, so also reset dropdown selection values
    stateselect.options = df_loc_date['STATE_PROVINCE'][df_loc_date['COUNTRY']==countryselect.value].unique().tolist()
    coselect.options = df_loc_date['COUNTY'][df_loc_date['STATE_PROVINCE']==stateselect.value].unique().tolist()
    
    #print(country)
    #using alternate to ix because of "too many indexers" error when using variable
    idx = pd.IndexSlice
    try:
        df_city_state_co_summary = df_first_city_state.loc[idx[country,state,co],:]
    except:
        df_city_state_co_summary = []
        print("Combination of country=",country,", state=", state, " and county=", co, "not found.")      
        
    if len(df_city_state_co_summary) > 0:
        #***NOTE***: the indexes appear to be changing each time I run this! need a better way to reference?
        #solution: have to use field names instead of index numbers (they still work normally)
        #print(df_city_state_co_summary)
        print('AVG LATITUDE:',df_city_state_co_summary['SHORT LAT'])
        print('AVG LONGITUDE:',df_city_state_co_summary['SHORT LONG'])
        obs_mon = df_city_state_co_summary['OBS MONTHDAY'].timetuple().tm_mon
        obs_day = df_city_state_co_summary['OBS MONTHDAY'].timetuple().tm_mday
        print('EARLIEST SIGHTING:',str(obs_mon)+'/'+str(obs_day))
        #pd.reset_option('display.max_rows')

        #show map with distribution by this date
        max_date = pd.to_datetime('2014-' + str(obs_mon).zfill(2) + '-' + str(obs_day).zfill(2))
        plotdata_date = plotdata[plotdata['OBS MONTHDAY'] <= max_date]
        #print(plotdata_date)

        #create new map plot
        #create labels for "map" plot
        p2 = figure(title = "Ruby-Throated Hummingbird Migration Through " + str(obs_mon)+'/'+str(obs_day))
        p2.xaxis.axis_label = 'Longitude'
        p2.yaxis.axis_label = 'Latitude'
        p2.title_text_font_size = '8'
        colors = [color_dict[x*2] if x < 26 else color_dict[52] for x in plotdata_date['WOY']] #added *2 to skip though these colors faster, until > 26, then stay at max
        #map points
        p2.circle(plotdata_date['SHORT LONG'], plotdata_date['SHORT LAT'],fill_color=colors, line_color = None, fill_alpha = 0.8 )
        #get the actual color of the dot
        plotdata_loc = plotdata_date.loc[(plotdata_date['SHORT LONG'] == df_city_state_co_summary['SHORT LONG']) & (plotdata_date['SHORT LAT'] == df_city_state_co_summary['SHORT LAT'] )]
        color2 = [color_dict[x*2] if x < 26 else color_dict[52] for x in plotdata_loc['WOY']]
        #larger point to indicate county
        p2.circle_x(df_city_state_co_summary['SHORT LONG'], df_city_state_co_summary['SHORT LAT'],fill_color=color2, line_color = "black", fill_alpha = 0.8, size=15 )
        show(p2)

interact(mapMigrationByCity,country=countryselect, state=stateselect, co=coselect)



#show map with distribution by this date
#mon_day = '0501'
from bokeh.models import Range1d
myslider = IntSlider(min=1,max=365,description="Day of Year")
myslider.value = 60
#display(myslider)


def mapMigrationByDate(md):
    max_date = datetime.datetime(2014, 1, 1) + datetime.timedelta(md - 1)
    plotdata_date = plotdata[plotdata['OBS MONTHDAY'] <= max_date]
    #print(plotdata_date)
    obs_mon2 = max_date.timetuple().tm_mon
    obs_day2 = max_date.timetuple().tm_mday
    
    #create new map plot
    #create labels for "map" plot
    p3 = figure(title = "Ruby-Throated Hummingbird Migration Through " + str(obs_mon2)+'/'+str(obs_day2))
    p3.xaxis.axis_label = 'Longitude'
    p3.yaxis.axis_label = 'Latitude'
    # fixed axes on this one for visual effect
    p3.xaxis.bounds = [-130,-50]
    p3.yaxis.bounds = [5,60]
    p3.x_range = Range1d(-130,-50)
    p3.y_range = Range1d(5,60)
    p3.title_text_font_size = '8'
    colors = [color_dict[x*2] if x < 26 else color_dict[52] for x in plotdata_date['WOY']] #added *2 to skip though these colors faster, until > 26, then stay at max
    #map points
    p3.circle(plotdata_date['SHORT LONG'], plotdata_date['SHORT LAT'],fill_color=colors, line_color = None, fill_alpha = 0.8 )
    show(p3)

interact(mapMigrationByDate,md=myslider)
mapMigrationByDate(60)

# http://bokeh.pydata.org/en/latest/docs/user_guide/geo.html

#NOTE - use the Google Map for the city/state part above?

from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool, BoxZoomTool
)

map_options = GMapOptions(lat=35, lng=-90, map_type="roadmap", zoom=3)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, title="Google Map"
)

source = ColumnDataSource(
    data=dict(
        lat= plotdata['SHORT LAT'],
        lon= plotdata['SHORT LONG']
    )
)

circle = Circle(x="lon", y="lat", size=8, fill_color="blue", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

#NOTE - zoom not allowed?
plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), BoxZoomTool())
#output_file("gmap_plot.html")
show(plot)



