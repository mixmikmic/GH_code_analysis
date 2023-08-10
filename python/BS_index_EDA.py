from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import Counter

import numpy as np
import pandas as pd
import os
import json
import urllib.request 

brd_count = {'bin': 0,
             'bowl': 0, 
             'bucket': 0, 
             'cup': 0,
             'tire': 0,
             'pottedplant': 0, 
             'jar': 0, 
             'vase': 0}

with open('brd_sites.geojson', 'r') as file:
    data = json.load(file)

print(json.dumps(data['features'][0], ensure_ascii=False, indent=4))

rows = []
for i, feature in enumerate(data['features']):
    
    row = feature['properties'].copy()
    
    lng, lat = feature['geometry']['coordinates']
    row['lng'], row['lat'] = lng, lat
    
    row['date'] = row['date']['year']+'-'+row['date']['month']
    
    for degree in row['brd_sites']:
        
        detected_brd = row['brd_sites'][degree]['count']

        for cls in brd_count:
            if cls not in row:
                row[cls] = 0
            
            if cls in detected_brd:
                row[cls] += detected_brd[cls]
                
    _= row.pop('brd_sites')
    _= row.pop('directory')
    _= row.pop('image_name')

    rows.append(row)

df = pd.DataFrame.from_dict(rows)
df = df.drop('province', axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df = df.set_index('date')
df = df.sort_index()
df = df['2016']
df.head()
df.tail()
print('Total:',len(df))

URL = 'https://raw.githubusercontent.com/pcrete/Mosquito_Breeding_Sites_Detector/master/geojson/province/%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A8%E0%B8%A3%E0%B8%B5%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A.geojson'
with urllib.request.urlopen(URL) as url:
    data_polygon = json.loads(url.read().decode())

df_sum = []
polygons = []
for i, feature in enumerate(data_polygon['features']):
    prop = feature['properties']
    district, subdist = prop['AP_TN'], prop['TB_TN']
    value = df[(df.district == district) & (df.subdist == subdist)].copy()
    
    if len(value) == 0:
        continue
    
    total = 0
    tmp_sum = []
    for cls in brd_count:
        tmp_sum.append(value[cls].sum())
        total += value[cls].sum()

    df_sum.append([Counter(list(value.index)).most_common(1)[0][0],
                   district,
                   subdist]+tmp_sum+[total])
    
df_sum = pd.DataFrame.from_records(df_sum)
df_sum.columns = ['date','district','subdist',
                 'bin','bowl', 'bucket','cup','jar',
                 'pottedplant','tire','vase','total']
df_sum = df_sum.set_index('date')
df_sum.head()
df_sum.tail()
print('Total:',len(df_sum))

df_sum.to_csv('BS_summation.csv')

count = dict(Counter(df.index.month))
key, val = [], []
for k in count:
    key.append(k)
    val.append(count[k])

trace_bar_actual = go.Bar( 
    x = key,
    y = val,
    text = val,
    textposition = 'auto',
    marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
    opacity=0.8
)
layout = go.Layout(
    title='Data Points for each Year',
    height=550,
#     width=750,
    yaxis= dict(title='Frequency'),
    xaxis= dict(title='Year')
)
fig = go.Figure(data=[trace_bar_actual], layout=layout)
iplot(fig)

arr = []
subdist_list = df['subdist'].unique()
for subdist in subdist_list:
    tmp = df.loc[df['subdist'] == subdist].copy()
    arr.append([subdist, len(tmp)])

arr = pd.DataFrame.from_records(arr)
arr.columns = ['subdist', 'freq']
# arr = arr.sort_values('freq', ascending=0)


trace = go.Bar( 
    x = arr['subdist'],
    y = arr['freq'],
    text = arr['freq'],
    textposition = 'auto',
    marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
    opacity=0.8
)
layout = go.Layout(
    title='Data Points for each Subdistrict',
    height=550,
#     width=1700,
    yaxis= dict(title='Frequency'),
    xaxis= dict(title='Year')
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

arr = []
subdist_list = df_sum['subdist'].unique()
for subdist in subdist_list:
    tmp = df_sum.loc[df_sum['subdist'] == subdist].copy()
    arr.append([subdist, tmp['total'].values[0]])

arr = pd.DataFrame.from_records(arr)
arr.columns = ['subdist', 'freq']
# arr = arr.sort_values('freq', ascending=0)

trace = go.Bar( 
    x = arr['subdist'],
    y = arr['freq'],
    text = arr['freq'],
    textposition = 'auto',
    marker=dict(
                color='#6BE59A',
                line=dict(
                    color='#05B083',
                    width=1.5),
            ),
    opacity=0.8
)
layout = go.Layout(
    title='Number of detected breeding sites for each Subdistrict',
    height=550,
#     width=1700,
    yaxis= dict(title='Frequency'),
    xaxis= dict(title='Year')
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

mapbox_access_token = 'pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'

mean, sd = df_sum['total'].mean(), df_sum['total'].std()
print(mean, sd)

norm = mpl.colors.Normalize(vmin=mean-sd, vmax=mean+sd)
cmap = cm.Blues

polygons = []
for feature in data_polygon['features']:
    prop = feature['properties']
    province = prop['PV_TN']
    district = prop['AP_TN']
    subdist = prop['TB_TN']
    
    value = df_sum[(df_sum.district == district) & (df_sum.subdist == subdist)]['total'].mean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    r,g,b,a = m.to_rgba(value)
    r,g,b,a = str(int(r*255)), str(int(g*255)), str(int(b*255)), str(1.0)
    rgba = 'rgba('+r+','+g+','+b+','+a+')'
    
    polygons.append(
        dict(
            sourcetype = 'geojson',
            source = feature,
            type = 'fill',
            color = rgba
        )
    )


data = Data([
    Scattermapbox(
        lat=1,
        lon=1,
        mode='markers',
        marker=Marker(
            size=0
        ),
        text=['Montreal'],
    )
])

layout = Layout(
    autosize=True,
    hovermode='closest',
    width=1500,
    height=800,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=df.lat[0],
            lon=df.lng[0]
        ),
        pitch=0,
        zoom=8,
        style='light', # dark,satellite,streets,light
        layers=polygons,
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Montreal Mapbox')

