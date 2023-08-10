import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# read labels for aircraft images
labels = pd.read_csv('aircraft.csv')
print(labels.head())
print(labels.shape)

# create list of images that contain aircraft
aircraft = labels[labels['aircraft']==True]['imageName']
print(aircraft.shape)
print(type(aircraft))

# view images stored in '/images' subdirectory
from skimage import io

plt.rcParams['figure.figsize'] = (20.0, 200.0)
f, axarr = plt.subplots(24, 2)

rownow = 0
colnow = 0
i = 0

for craft in aircraft:
    if i < aircraft.shape[0]:
        if colnow == 0:
            toRead = 'images/' + craft
            axarr[rownow, colnow].imshow(io.imread(toRead))
            colnow = 1
            i = i + 1
            if i == aircraft.shape[0]:
                axarr[-1, -1].axis('off')
        elif colnow == 1:
            toRead = 'images/' + craft
            axarr[rownow, colnow].imshow(io.imread(toRead))
            colnow = 0
            rownow = rownow + 1
            i = i + 1
            



