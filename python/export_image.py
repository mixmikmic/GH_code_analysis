from scipy.misc import imsave
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def build_imgarr(data):
    res = np.zeros((len(data), 48, 48)).astype('float32')

    for i in range(len(data)):
        pixl = data.iloc[i,:].pixels.split(' ')
        pixels = np.array(pixl)
        pixels = pixels.reshape(48, 48)
        res[i] = pixels
    
    return res

df = pd.read_csv(' ') # write your path in ' ' to read csv file

train = df[df.Usage == 'Training']
test = df[df.Usage == 'PrivateTest']
val = df[df.Usage == 'PublicTest']

X_train = build_imgarr(train)
X_test = build_imgarr(test)
X_val = build_imgarr(val)

get_ipython().run_cell_magic('time', '', "\ntarget = val # choose one of train, test and val\npath = ' ' # write your path in ' ' to export png files\n\nfor i in range(len(target)):\n    label = ''\n    if target.emotion.values[i] == 0:\n        label = 'Angry'\n    elif target.emotion.values[i] == 1:\n        label = 'Disgust'\n    elif target.emotion.values[i] == 2:\n        label = 'Fear'\n    elif target.emotion.values[i] == 3:\n        label = 'Happy'\n    elif target.emotion.values[i] == 4:\n        label = 'Sad'\n    elif target.emotion.values[i] == 5:\n        label = 'Surprise'\n    elif target.emotion.values[i] == 6:\n        label = 'Neutral'\n    \n    imsave(path + '{}_{}.png'.format(i, label), X_val[i])")



