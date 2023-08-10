import urllib 

import requests
from bs4 import BeautifulSoup
import lxml
import urllib, json
import pandas as pd, numpy as np
import pprint
import datetime as dt 
from urllib.parse   import quote

r3.json()[0]

output = [[] for k in range(0,8)]


for count, item in reversed(list(enumerate(r3.json()))):
    
    output[0].append(r3.json()[count]['name'])
    output[1].append(r3.json()[count]['address'])
    output[2].append(r3.json()[count]['city'])
    output[3].append(r3.json()[count]['city_id'])
    output[4].append(r3.json()[count]['hotel_id'])
    output[5].append(r3.json()[count]['location']['latitude'])
    output[6].append(r3.json()[count]['location']['longitude'])
    output[7].append(r3.json()[count]['zip'])
    
    

df_ = pd.DataFrame(output).T
df_.columns = [['name','address','city','city_id','hotel_id','latitude','longitude','zip']]
df_.head()

output_photo = [[] for k in range(0,3)]


for count, item in reversed(list(enumerate(r3.json()))):
    
    output_photo[0].append(r4.json()[count]['hotel_id'])
    output_photo[1].append(r4.json()[count]['url_max300'])
    output_photo[2].append(r4.json()[count]['url_original'])
      

df_photo = pd.DataFrame(output_photo).T
df_photo.columns = [['hotel_id','url_max300','url_original']]

df_photo_ = df_photo.groupby('hotel_id').first().reset_index()
df_photo_.head()

df_merge = pd.merge(df_,df_photo_,  how='left',on = 'hotel_id')

df_merge.sort('hotel_id')

