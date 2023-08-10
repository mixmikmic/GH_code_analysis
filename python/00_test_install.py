import tensorflow as tf
print("You have version %s" % tf.__version__)

get_ipython().magic('matplotlib inline')
import pylab
import numpy as np

# create some data using numpy. y = x * 0.1 + 0.3 + noise
x = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x))
y = x * 0.1 + 0.3 + noise

# plot it
pylab.plot(x, y, '.')

import PIL.Image as Image
import numpy as np
from matplotlib.pyplot import imshow

image_array = np.random.rand(200,200,3) * 255
img = Image.fromarray(image_array.astype('uint8')).convert('RGBA')
imshow(np.asarray(img))

import pandas as pd
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

