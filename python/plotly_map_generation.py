import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/katiebrennan/Documents'+
                 '/UW/CSE583/uwseds-group-nw-climate-crew'+
                 '/futurefish/data/tiny_site_test_dataset.csv')
df.head()

mapbox_access_token = 'pk.eyJ1IjoibWticmVubmFuIiwiYSI6ImNqYW12OGxjYjM1MXUzM28yMXhpdWE3NW0ifQ.EljNVtky3qEFfvJL80RgMQ'

scl = [[0, 'rgb(0, 102, 0)'],[0.2, 'rgb(0, 102, 0)'],       [0.2, 'rgb(128, 255, 0)'], [0.4, 'rgb(128, 255, 0)'],       [0.4, 'rgb(255, 255, 51)'], [0.6, 'rgb(255, 255, 51)'],       [0.6, 'rgb(255, 153, 51)'], [0.8, 'rgb(255, 153, 51)'],       [0.8, 'rgb(255, 6, 6)'], [1.0, 'rgb(255, 6, 6)']]

all_data = Scattermapbox(
        lon = df['Longitude'],
        lat = df['Latitude'],
        mode='markers',
        marker=Marker(
            size=8,
            symbol='circle',
            colorscale = scl,
            cmin = 1,
            color = df['Viability'],
            cmax = df['Viability'].max(),
            colorbar=dict(
                title="Viability of Salmon Life",
                titleside = 'top',
                tickmode = 'array',
                tickvals = [1.3,2.2,3.0,3.8,4.5],
                ticktext = ['Great','Good','Mmm?','Nope','Yikes!'],
                ticks = 'outside'    
            ),
        ),
    )

data = Data([all_data])

layout = dict(
    height = 500,
    width = 700,
    margin = dict( t=0, b=0, l=0, r=0 ),
    font = dict( color='#FFFFFF', size=11 ),
    paper_bgcolor = '#000000',
    #paper_bgcolor = '#50667f',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=46,
            lon=-119
        ),
        pitch=0,
        zoom=4.5,
        style='light'
    ),
)

figure = dict(data=data, layout=layout) 

py.iplot(figure, filename='basin')



