import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import requests
from PIL import Image
from io import BytesIO

url = "https://farm9.staticflickr.com/8076/8326425318_3721a23141_b_d.jpg"
r = requests.get(url)
im = Image.open(BytesIO(r.content))

plt.imshow(im)
plt.show()

i = np.asarray(im)

i.shape

red = i[...,0]

red

plt.imshow(red)

def index(a, colours=8):
    b = a.reshape(a.size)
    b = float(colours) * b / np.amax(b)
    bins = np.linspace(0, colours-1, colours)
    c = np.digitize(b, bins)
    return c.reshape(a.shape)

red_2bit = index(red, 4)

plt.imshow(red_2bit, cmap="Greys")



