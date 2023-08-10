import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import requests

url = 'https://www.modelr.org/plot.jpeg'

# Parameters
params = {
           'script': 'wedge_spatial.py',
           'theta': 0,
           'f': 25,
           'colourmap': 'Greys',
           'opacity': 0.5,
           'tslice': 0.15,
           'scale': '1.0,99',
           'base1': 'variable-density',
           'Rock0': '3000.0,1600.0,2500.0,50.0,50.0,50.0',
           'Rock1': '2770.0,1780.0,2185.0,50.0,50.0,50.0',
           'Rock2': '3000.0,1600.0,2500.0,50.0,50.0,50.0',
           'type': "scenario"
         }

r = requests.get(url, params=params)

r.content[:200]

from PIL import Image
from base64 import decodestring
from io import BytesIO

i = Image.open(BytesIO(r.content))

plt.figure(figsize=(8,18))
plt.imshow(i)
plt.show()

url = 'https://www.modelr.org/plot.json'

# Parameters
params = {
           'script': 'slab_builder.py',
           'interface_depth': 80,
           'x_samples': 350,
           'margin': 50,
           'left': '0,40',
           'right': '30,130',
           'layers': 3,
           'type': "model_builder"
         }

r = requests.get(url, params=params)

r.status_code

s = r.json()['data'].encode('utf-8')

import base64
b = base64.decodestring(s)

i = Image.open(BytesIO(b))

plt.imshow(i)
plt.show()

time   = range(0,99)
traces = range(1,350)
freqs  = range(10,100)

data = r.json()['data']

url_fm = 'https://www.modelr.org/forward_model.json'

p = {
      'data': data,
      'metadata':{'time': time,
      'trace': traces,
      'f': freqs}
    }



