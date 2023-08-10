import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy as scipy
import time
plt.style.use('seaborn-whitegrid')
get_ipython().magic('matplotlib inline')

df = pd.read_csv('twitch_games.csv')

import requests as r
giantbomb_api_key = 'XXXX'
giantbomb_url = 'https://www.giantbomb.com/api/game/'
giantbomb_ids = list(df[df['rank']<15]['giantbomb_id'].unique())
urls = []
for gbid in giantbomb_ids:
    if gbid == 0:
        continue
    request_url = giantbomb_url + str(gbid) + '/?api_key=' +giantbomb_api_key
    urls.append(request_url)

print urls[:10]

headers = r.utils.default_headers()

headers.update(
    {
        'User-Agent': 'XXXXX-UserAgent-1.0'
    }
)

for url in urls[1:]:
    print url
    id1 = url[url.find('game/')+5:]
    fname = id1[:id1.find('/')]
    raw_data = r.get(url, headers=headers).text
    f= open('giantbomb/' + fname+'.txt', 'w')
    f.write(raw_data.encode('utf-8').strip())
    f.close()
    time.sleep(5)

