import numpy as np
import random
import sys
import os

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

#Add the src folder to the sys.path list
sys.path.append('../src/')
sys.path.append('../lib/')
import data_config as dc
import rnn
from analysis import Analyzer

wormData = dc.kato_matlab.data()
rawData = Analyzer(wormData[0]['deltaFOverF'][:,0:100].T)
seq = rawData.timeseries
print(seq.shape)

gen = rnn.generate(seq)



