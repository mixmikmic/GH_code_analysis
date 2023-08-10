import os 

import numpy as np
import pandas as pd
import scipy 

import matplotlib.pyplot as plt

import keras

get_ipython().run_line_magic('matplotlib', 'inline')

img_dir = '../images'

df = pd.read_csv('../data/subset.csv')

df = df[['Id', 'ViewCount', 'LikeCount', 'DislikeCount']]
df = df[df['Id'] != '#NAME?']
df.drop_duplicates(subset='Id', keep='last')

df.head()

sample = df.iloc[0]

img_path = os.path.join(img_dir, '{0}.jpg'.format(sample['Id']))
img = scipy.misc.imread(img_path)

plt.imshow(img)



