#!pip install --upgrade pixiedust
#!pip install --upgrade bokeh

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pixiedust

get_ipython().run_cell_magic('javascript', '', 'navigator.geolocation.getCurrentPosition(function(position) {\n  console.log(position.coords.latitude, position.coords.longitude);\nsetTimeout(function() {\nIPython.notebook.kernel.execute(\'lat="\' + position.coords.latitude + \'";\')\nIPython.notebook.kernel.execute(\'lon="\' + position.coords.longitude + \'";\')\n},5000)});')

# Barcelona
lat = str(41.3850639)
lon = str(2.1734035)

print(lat, lon)

# @hidden_cell
# Weather company data API credentials
username='xxxx'
password='xxxx'

line='https://'+username+':'+password+    '@twcservice.mybluemix.net/api/weather/v1/geocode/'+    lat+'/'+lon+'/forecast/intraday/10day.json?&units=m'
r=requests.get(line)
weather = json.loads(r.text)    

#print json.dumps(weather, indent=4, sort_keys=True)

df = pd.DataFrame.from_dict(weather['forecasts'][0],orient='index').transpose()
for forecast in weather['forecasts'][1:]:
    df = pd.concat([df, pd.DataFrame.from_dict(forecast,orient='index').transpose()])

df['date'] = df['fcst_valid_local'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S+0200'))

df = df.drop(['expire_time_gmt','num','qualifier','qualifier_code'],1)   
df = df.drop(['fcst_valid','fcst_valid_local','icon_extd','wdir_cardinal'],1)   
df = df.drop(['subphrase_pt1','subphrase_pt2','subphrase_pt3','class'],1)   
df = df.drop(['daypart_name','phrase_12char','phrase_22char','phrase_32char'],1)   

df.dtypes

df[['pop','wspd','rh','clds','wdir','temp']] = df[['pop','wspd','rh','clds','wdir','temp']].apply(pd.to_numeric)
df.dtypes

df['rain'] = df['pop'].as_matrix()
df=df.drop('pop',1)

df.head()

df = df.set_index('date',drop=False)
df.head()

import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))

df['temp'].plot(ax=axes[1], color='#6EEDD8',lw=3.0,sharex=True)
axes[1].set_title('Temperature',loc='left',fontsize=20)

df['rain'].plot(ax=axes[0], kind='bar', color='#C93D79',lw=2.0,sharex=True)
axes[0].set_title('Chance of rain',loc='left',fontsize=20)

cities = [
    ('Exeter',50.7184,-3.5339),
    ('Truro',50.2632,-5.051),
    ('Carmarthen',51.8576,-4.3121),
    ('Norwich',52.6309,1.2974),
    ('Brighton And Hove',50.8225,-0.1372),
    ('Bristol',51.44999778,-2.583315472),
    ('Durham',54.7753,-1.5849),
    ('Llanidloes',52.4135,-3.5883),
    ('Penrith',54.6641,-2.7527),
    ('Jedburgh',55.4777,-2.5549),
    ('Coventry',52.42040367,-1.499996583),
    ('Edinburgh',55.94832786,-3.219090618),
    ('Cambridge',52.2053,0.1218),
    ('Glasgow',55.87440472,-4.250707236),
    ('Kingston upon Hull',53.7457,-0.3367),
    ('Leeds',53.83000755,-1.580017539),
    ('London',51.49999473,-0.116721844),
    ('Manchester',53.50041526,-2.247987103),
    ('Nottingham',52.97034426,-1.170016725),
    ('Aberdeen',57.1497,-2.0943),
    ('Fort Augustus',57.1448,-4.6805),
    ('Lairg',58.197,-4.6173),
    ('Oxford',51.7517,-1.2553),
    ('Inverey',56.9855,-3.5055),
    ('Shrewsbury',52.7069,-2.7527),
    ('Colwyn Bay',53.2932,-3.7276),
    ('Newton Stewart',54.9186,-4.5918),    
    ('Portsmouth',50.80034751,-1.080022218)]   

icons=[]
temps=[]
for city in cities:
    lat = city[1]
    lon = city[2]
    line='https://'+username+':'+password+'@twcservice.mybluemix.net/api/weather/v1/geocode/'+str(lat)+'/'+str(lon)+'/observations.json?&units=m'
    r=requests.get(line)
    weather = json.loads(r.text)    
    icons=np.append(icons,weather['observation']['wx_icon'])    
    temps=np.append(temps,weather['observation']['temp']) 

dfmap = pd.DataFrame(cities, columns=['city','lat','lon'])    
dfmap['temp']=temps
dfmap['icon']=icons
dfmap.head(25)

from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
from itertools import izip
import urllib

matplotlib.style.use('bmh')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 12))

# background maps
m1 = Basemap(projection='mill',resolution=None,llcrnrlon=-7.5,llcrnrlat=49.84,urcrnrlon=2.5,urcrnrlat=59,ax=axes[0])
m1.drawlsmask(land_color='dimgrey',ocean_color='dodgerBlue',lakes=True)

m2 = Basemap(projection='mill',resolution=None,llcrnrlon=-7.5,llcrnrlat=49.84,urcrnrlon=2.5,urcrnrlat=59,ax=axes[1])
m2.drawlsmask(land_color='dimgrey',ocean_color='dodgerBlue',lakes=True)

# weather icons map
for [icon,city] in izip(icons,cities):
    lat = city[1]
    lon = city[2]
    try:
        pngfile=urllib.urlopen('https://github.com/ibm-cds-labs/python-notebooks/blob/master/weathericons/icon'+str(int(icon))+'.png?raw=true')
        icon_hand = read_png(pngfile)
        imagebox = OffsetImage(icon_hand, zoom=.15)
        ab = AnnotationBbox(imagebox,m1(lon,lat),frameon=False) 
        axes[0].add_artist(ab)
    except:
        pass
        
# temperature map    
for [temp,city] in izip(temps,cities):
    lat = city[1]
    lon = city[2]
    if temp>16: 
        col='indigo'
    elif temp>14:
        col='darkmagenta'
    elif temp>12:
        col='red'
    elif temp>10:
        col='tomato'
    elif temp>0:
        col='turquoise'
            
    x1, y1 = m2(lon,lat)
    bbox_props = dict(boxstyle="round,pad=0.3", fc=col, ec=col, lw=2)
    axes[1].text(x1, y1, temp, ha="center", va="center",
                size=11,bbox=bbox_props)
    
plt.tight_layout()  

display(df)

#display(dfmap)

