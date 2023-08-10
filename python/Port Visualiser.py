import json

import plotly
from plotly.graph_objs import Scatter, Layout

plotly.offline.init_notebook_mode(connected=True)

with open('PUB151_distances.json') as f:
    ports = json.load(f)

with open('junction_points.json') as f:
    junct = json.load(f)

def to_decimal(raw):
    """Convert d/m/s representation of long/lat to decimal degrees"""
    d = raw[:raw.index('°')]
    m = raw[raw.index('°')+1:raw.index('\'')]
    s = raw[raw.index('\'')+1:raw.index('"')]
    l = raw[raw.index('"')+1:raw.index('.')]
    decimal = float(d) + float(m)/60. + float(s)/3600.
    if l is 'W' or l is 'S':
        result = -decimal
    else:
        result = decimal
    return result

def parse_location(raw):
    lat, long = raw.split(' ')
    lat = to_decimal(lat)
    long = to_decimal(long)
    return lat, long

ports_temp = {}
for name, rest in ports.items():
    ports_temp[name] = parse_location(rest['location'])

locs = [loc for name, loc in ports_temp.items()] 

lats = [loc[0] for loc in locs]
longs = [loc[1] for loc in locs]

flight_path = [ dict(
    type = 'scattergeo',
    lat = lats,
    lon = longs,
    text = [name for name, loc in ports_temp.items()],
    name = 'My Trace',
    mode = 'markers',
    hoverinfo = "text",
    marker = dict(
        symbol = 'hexagon',
        color = 'black',
        size = 3,
    )
) ]
    
layout = dict(
        title = 'All Ports',
        showlegend = False,         
        geo = dict(
            resolution = 50,
            showland = True,
            showlakes = True,
            landcolor = 'rgb(220, 230, 220)',
            countrycolor = 'rgb(250, 250, 250)',
            lakecolor = 'rgb(255, 255, 255)',
            projection = dict( type="natural earth" ),
            coastlinewidth = 0,
            lataxis = dict(
                range = [ -90, 90 ],
#                 range = sorted([source.latitude, dest.latitude]),
                showgrid = False,
                tickmode = "linear",
                dtick = 10
            ),
            lonaxis = dict(
                range = [-180, 180],
#                 range = sorted([source.longitude, dest.longitude]),
                showgrid = False,
                tickmode = "linear",
                dtick = 20
            ),
        )
    )

fig = dict( data=flight_path, layout=layout )
plotly.offline.iplot( fig, validate=False )

junct_temp = {}
for name, loc in junct.items():
    junct_temp[name] = parse_location(loc)

locs = [loc for name, loc in junct_temp.items()] 
lats = [loc[0] for loc in locs]
longs = [loc[1] for loc in locs]

flight_path = [ dict(
    type = 'scattergeo',
    lat = lats,
    lon = longs,
    text = list(junct_temp.keys()),
    name = 'My Trace',
    mode = 'markers',
    hoverinfo = "text",
    marker = dict(
        symbol = 'hexagon',
        color = 'red',
        size = 4,
    )
) ]
    
layout = dict(
        title = 'Junction Points',
        showlegend = False,         
        geo = dict(
            resolution = 50,
            showland = True,
            showlakes = True,
            landcolor = 'rgb(220, 230, 220)',
            countrycolor = 'rgb(250, 250, 250)',
            lakecolor = 'rgb(255, 255, 255)',
            projection = dict( type="natural earth" ),
            coastlinewidth = 0,
            lataxis = dict(
                range = [ -90, 90 ],
#                 range = sorted([source.latitude, dest.latitude]),
                showgrid = False,
                tickmode = "linear",
                dtick = 10
            ),
            lonaxis = dict(
                range = [-180, 180],
#                 range = sorted([source.longitude, dest.longitude]),
                showgrid = False,
                tickmode = "linear",
                dtick = 20
            ),
        )
    )

fig = dict( data=flight_path, layout=layout )
plotly.offline.iplot( fig, validate=False )

