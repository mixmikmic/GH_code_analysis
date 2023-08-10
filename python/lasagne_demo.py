get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import numpy as np
from helpers import load_dataset

from bokeh.plotting import figure, show
from bokeh.io import output_notebook

X_train, y_train, _, _, _, _ = load_dataset()

# the feature set X train is a stack of images 
a = X_train[2, 0, :, :]
print('Type of training data:', type(X_train))
print('Shape of training data:', X_train.shape)

import math

nplots = 10
ncols = 5
nrows = math.ceil(nplots / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4))
for i, label in enumerate(y_train):
    if i >= nplots: 
        break
    im = X_train[i, 0, :, :]
    ix = i % ncols
    iy = math.floor(i / ncols)
    axes[iy][ix].imshow(im, cmap="gray_r")

fig.savefig('digits_10.png')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
found_fours = 0
for i, label in enumerate(y_train):
    if label == 4:
        im = X_train[i, 0, :, :]
        ax = axes[found_fours]
        ax.imshow(im, cmap="gray_r")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        found_fours +=1
    if found_fours >= 3:
        break
        
fig.savefig('some_training_fours.png')

img_names=["my_four.tif",
          "my_four_fancy.tif",
          "my_four_small.tif"
          ]
fig, axes = plt.subplots(3, 1, figsize=(12, 4))
for f, ax in zip(img_names, axes):
    im = mpimg.imread(f)
    ax.imshow(im, cmap="gray_r")
    ax.xaxis.set_visible(False)    
    ax.yaxis.set_visible(False)
    ax.bbox
    
# fig.savefig('some_test_fours.png')

from predict import make_predictions
q = make_predictions()

img_names=["my_four.tif",
          "my_four_fancy.tif",
          "my_four_small.tif"
          ]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for f, ax in zip(img_names, axes):
    im = mpimg.imread(f)
    ax.imshow(im, cmap="gray_r")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
fig.savefig('test_fours.png')   

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axes[0].set_ylabel('Probability')
for row, ax in zip(q, axes):
    ax.stem(row)
    ax.set_xlabel('Class')
fig.savefig('test_fours_probs.png')

output_notebook()

from bokeh.charts import Bar, show
import pandas as pd

df = pd.DataFrame(q, index=['big_four', 'fancy_four', 'small_four'])
df

df.T.plot(kind='bar')
plt.xlabel('Class')
_ = plt.ylabel('Probability')

