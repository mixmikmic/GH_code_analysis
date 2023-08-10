#import the libaries we need to use

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import us

##  Import display 
from IPython.display import display


### ipywidget libraries
from ipywidgets import HBox, VBox, IntSlider, Play, jslink
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

### bqplot libraries
import bqplot
from bqplot import (
    Axis, ColorAxis, LinearScale, DateScale, DateColorScale, OrdinalScale,
    OrdinalColorScale, ColorScale, Scatter, Lines, Figure, Tooltip
)
from bqplot import (
    Figure, Map, Mercator, Orthographic, ColorScale, ColorAxis,
    AlbersUSA, topo_load, Tooltip
)

#!python
names = ["date", "city", "state", "country", "shape", "duration_seconds",
         "duration_reported", "description", "report_date", "latitude",
         "longitude"]

fn = "data/ufo-scrubbed-geocoded-time-standardized.csv"
ufo = pd.read_csv(fn, names = names, parse_dates = ["date", "report_date"])

abbr_to_fits = us.states.mapping('abbr', 'fips')

##  If the state column has a nan values. 
##  replace it with -1. 

ufo["fips"] = ufo["state"].apply(lambda a: int(abbr_to_fits.get(str(a).upper(), -1))) #return value of key "a", if not avail, return -1

ufo["fips"].unique()

ufo["year"]=[y.year for y in ufo["date"].astype(datetime.datetime)]
only_date=[d.date() for d in ufo["date"].astype(datetime.datetime)]
only_time=[d.time() for d in ufo["date"].astype(datetime.datetime)]

ufo["time"] = only_time
ufo["time"].head()

#total sightings per state
fips_count = np.log10(ufo.groupby("fips")["duration_seconds"].count())

#total time in sightings per state
tot_time = np.log10(ufo.groupby("fips")["duration_seconds"].sum())

def sel_one_state(self, target):
    #print(target['data']['id'])
    if (len(states_map.selected)>0):
        states_map.selected=[]
        states_map.selected=[target['data']['id']]    

map_tooltip=Tooltip(fields=["name","id","color"],labels=["state", 'id',"Sight Count"])


map_styles = {'scales': {'projection': bqplot.AlbersUSA(),'color': bqplot.ColorScale(colors=["white","blue"])},
             'color':fips_count.to_dict()           
            }
states_map = Map(map_data=bqplot.topo_load('map_data/USStatesMap.json'), 
                        **map_styles,
                         tooltip=map_tooltip,
                         interactions = {'click': 'select', 'hover': 'tooltip'},
                         selected_style={'opacity': 1.5, 'fill': 'blue', 'stroke': 'white'},
                          unselected_style={'opacity': 1.0}
                       )

map_fig = bqplot.Figure(marks=[states_map], title='UFO sightings in USA states', legend=True)

states_map.display_legend=True
states_map.enable_hover
states_map.on_element_click(sel_one_state)

import math
# Function for getting the min and max year for the selected state
def get_min_max_year(id):
    min_year=ufo[ufo["fips"]==id]["date"].min().year
    max_year=ufo[ufo["fips"]==id]["date"].max().year
    return [min_year,max_year]


# function to get total sighting and total time of sighting for each state for a year range for normal or normalized values.
def get_state_data(state_id,from_year,to_year,data_type):
        
    state_sighting=ufo["fips"]==state_id
    state_ufo=ufo[state_sighting]
    
    total_sighting= [state_ufo["duration_seconds"][state_ufo["year"]==year].count() for year in np.arange(from_year,to_year+1)]  
    total_duration = [state_ufo["duration_seconds"][state_ufo["year"]==year].sum() for year in np.arange(from_year,to_year+1)]
    total_duration = np.nan_to_num(total_duration)
    
    if(data_type == 1 ):
        area=int(area_over_time[area_over_time.index.values == state_id][2014])
        total_sighting =  [total_sighting[i] / area for i in range(0,len(total_sighting) )]
        total_duration =  [total_duration[i] / area for i in range(0,len(total_duration) )]

    return [range(from_year, to_year +1 ),total_sighting,total_duration]


# function to get color based on sighting or total time of sighting.
def get_color(from_year,to_year):
    
    ndf = ufo[from_year:to_year]
    ndf.set_index('fips')
    if(ddl.value == 1): 
        ### Create new series 
        n_fips = fips_count
        n_col = ndf.groupby('fips')["duration_seconds"].count()
    else:
        n_fips = tot_time
        n_col = ndf.groupby('fips')["duration_seconds"].sum()

    n_fips.update(n_col)
  
    return(n_fips)


# function to update the scatter plot based on sighting or total duration.
def upd_scat_plot(state_id,from_year,to_year,aggregate): 
    
    scat_data = get_state_data(get_state_id(),from_year,to_year,ddl2.value)
    scat_plot.x = scat_data[0]
    scat_plot.y = scat_data[aggregate]
    if(aggregate == 1):
        tt_labels=["State","Id","Total Sightings"]
    else:
        tt_labels=["State","Id","Total Duration"]
    
    map_tooltip.labels =tt_labels

area = pd.read_csv("data/usa-area.csv")
#replace the columns of state abbreviation with fips number
area["fips"]=area["state"].apply(lambda a: int(abbr_to_fits.get(str(a).upper(), -1)))

area_over_time=pd.concat([area["area"]] * 109, axis=1)
area_over_time.columns=np.arange(1906,2015)
area_over_time.index=area["fips"]

area_over_time[area_over_time.index.values==48]

style = {'description_width': 'initial'}


## Range slider 
slider =  widgets.IntRangeSlider(
    value=[1920, 1960],
    min=ufo['year'].min(),
    max=ufo['year'].max(),
    step=4,
    description='Select Year Range',
    style=style,
    disabled=False,
    continuous_update=False,
    orientation='Horizontal',
    readout=True,
)


display(slider)


### Drop downlist for selecting the sights or counts 
ddl = widgets.Dropdown(
    options={'Sight Counts': 1, 'Total Duration': 2 },
    value=1,
    description='Aggregate By:',
)

### Drop downlist for selecting the nromalize or normal data  
ddl2 = widgets.Dropdown(
    options={'Normalized Data': 1, 'Normal Data': 2 },
    value=2,
    description='Select Data:',
)


togg_norm = widgets.ToggleButtons(options=['Normalized Data','Data'],
                                  description='Select Data format:',disabled=False,
                                  button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                  tooltips=['Normalized data ', 'Normal Data']
                                 )
#togg_norm.observe(upd_on_data_sel)
#ddl2
#map_fig

## First we define a Figure

default_state = 48 ## 
aggregate = ddl.value

x = LinearScale()
y = LinearScale()

col_sc = ColorScale()

axis_x = Axis( scale=x, label='Year Range')
axis_y = Axis( scale=y,  label='Total Sightings', orientation='vertical')

from_year, to_year, data_type  = slider.value[0],slider.value[1], ddl2.value
scat_data = get_state_data(default_state,from_year,to_year,data_type)


scat_plot = Lines(x=scat_data[0] , y=scat_data[aggregate], 
                scales={'x': x, 'y': y},                
                stroke='white',
                colors = ['orange'],
                    labels=['YEAR', 'Range' ]
                    
                   )   
plot_scat = Figure(axes=[axis_x,axis_y], marks=[scat_plot])

## Call back function for recreating the line plot
def upd_plot(self, target): 
    id = target['data']['id']
    #print(id)
    upd_scat_plot(id, slider.value[0], slider.value[1],ddl2.value)

##  function for getting the stateid of the selected map
def get_state_id():  
    if (len(states_map.selected) == 0 ): 
        state_id = default_state
    elif(len(states_map.selected) > 1):
        state_id = states_map.selected.pop()
    else:
        state_id = states_map.selected
    return state_id 

##call back for changing the color of the map. 
def upd_map(change): 
    if(change['new'] == 1 ):
        states_map.color=fips_count.to_dict()
    else:
        states_map.color=tot_time.to_dict()
                
def upd_axes(change):
    if(change['new'] == 1 ):
        scat_plot.labels='asdfad'
    else:
        states_map.color=tot_time.to_dict()
        
def upd_scat_ddl(change):
    if(change['new'] == 1):
        axis_y.label = "Total Sightings"
    else: 
        axis_y.label = "Total Duration"
    
    upd_scat_plot(get_state_id(), slider.value[0], slider.value[1], change['new']) 

## Function for changing the datatype     
def upd_on_data_sel(change):
    #print(change['new'])
    #print(ddl2.value)
    upd_scat_plot(states_map.selected, slider.value[0], slider.value[1],ddl2.value)

## Function for changing the datatype     
def upd_map_col(change):
    states_map.color = get_color(change['new'][0],change['new'][1]).to_dict()

def upd_map_col_norm(change):
    states_map.color = get_color(slider.value[0],slider.value[1]).to_dict()

## Update the plot on selection of the state. 

states_map.on_element_click(upd_plot)
states_map.selected=[]

## Update the color of map on selection of the year. 
#slider.observe(upd_map_col,names='value')

## Change color of map on selecting the total count or duration 
ddl.observe(upd_map , names='value')


## Change scatter plot on selecting the total count or duration
ddl.observe(upd_scat_ddl , names='value')


## Recreate the map and plots based on the data selection 
ddl2.observe(upd_on_data_sel , names='value')

## Recreate the map and plots based on the data selection 
slider.observe(upd_map_col, names='value')

## Recreate the map and plots based on the data selection 
ddl2.observe(upd_map_col_norm , names='value')

tt_labels=["State","Id","Total Sightings"]
map_tooltip.labels = tt_labels
display(slider)

states_map.color = fips_count.to_dict()
states_map.selected=[]
H2 = HBox(children = [ddl,ddl2])
H1 = HBox(children = [plot_scat,map_fig])
V1 = VBox(children=[H2, H1])
V1



