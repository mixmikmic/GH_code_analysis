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

from tqdm import tqdm
from collections import Counter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

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

df_village = pd.read_csv('village.csv') 
df_village.head()
df_village.tail()
print('Total:', len(df_village), '\nUnique village:', len(df_village['village'].unique()))

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

df_detect = pd.DataFrame.from_dict(rows)
df_detect = df_detect.drop('province', axis=1)
df_detect['village'] = None
df_detect.head()
df_detect.tail()
print('Total:',len(df_detect))

knn = NearestNeighbors(n_neighbors=50)
knn.fit(df_village[['lat','lng']].values)

distances, indices = knn.kneighbors([[9.29446814, 99.7715087]])
indices[0]

pbar = tqdm(total=len(df_detect))
for i, row_detect in df_detect.iterrows():

    distances, indices = knn.kneighbors([[row_detect['lat'], row_detect['lng']]])  
    tmp_village = df_village.iloc[indices[0]].copy()
    village_name = tmp_village.loc[(tmp_village['district'] == row_detect['district']) & 
                                   (tmp_village['subdist'] == row_detect['subdist'])
                                  ].head(1)['village'].values
    
    if len(village_name) != 0:
        row_detect['village'] = village_name[0]
        df_detect.iloc[i] = row_detect

    pbar.update(1)
pbar.close()

df_detect = df_detect.dropna(axis=0, how='any')
df_detect = df_detect.reset_index(drop=True)
len(df_detect)

df_detect['date'] = pd.to_datetime(df_detect['date'], format='%Y-%m')
df_detect = df_detect.set_index('date')
df_detect = df_detect.sort_index()
df_detect = df_detect['2016']
df_detect.head()

URL = 'https://raw.githubusercontent.com/pcrete/Mosquito_Breeding_Sites_Detector/master/geojson/province/%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A8%E0%B8%A3%E0%B8%B5%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%A3%E0%B8%B2%E0%B8%8A.geojson'
with urllib.request.urlopen(URL) as url:
    data_polygon = json.loads(url.read().decode())

df_sum = []
polygons = []
for i, feature in enumerate(data_polygon['features']):
    prop = feature['properties']
    district, subdist = prop['AP_TN'], prop['TB_TN']
    subdist_level = df_detect[(df_detect.district == district) & (df_detect.subdist == subdist)].copy()
    
    if len(subdist_level) == 0: continue
    
    for village in subdist_level['village'].unique():
        
        village_level = subdist_level[subdist_level['village'] == village].copy()
        
        total = 0
        tmp_sum = []
        for cls in brd_count:
            tmp_sum.append(village_level[cls].sum())
            total += village_level[cls].sum()

        df_sum.append([Counter(list(village_level.index)).most_common(1)[0][0],
                       district, subdist, village]+tmp_sum+[total])
    
df_sum = pd.DataFrame.from_records(df_sum)
df_sum.columns = ['date','district','subdist','village',
                 'bin','bowl', 'bucket','cup','jar',
                 'pottedplant','tire','vase','total']
df_sum = df_sum.set_index('date')
df_sum.head()
df_sum.tail()
print('Total:',len(df_sum))

df_sum.to_csv('BS_village_summation.csv')



