import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
from welly import Well
import lasio
import glob
from sklearn import neighbors
import pickle
welly.__version__

get_ipython().run_cell_magic('timeit', '', 'import os\nenv = %env')

pd.set_option('display.max_rows', 2000)

X = [1,2,3,4,5,6,4,3,3,5,4,3,8,3,5,33,345,8,345,23,4,23,4,2344,43,234]
placeholder = []
counter = 0
for num in X:
    print(counter)
    step = [counter,num]
    placeholder.append(step)
    counter += 1
#new_array = []
    
#X = np.random.random((10, 3)) 
X = placeholder
print(X)
tree = neighbors.KDTree(X, leaf_size=2) 

dist, ind = tree.query([[1,1]], k=3)  

dist

ind

ind[0]

X[ind[0][1]]

picks_dic = pd.read_csv('../../SPE_006_originalData/OilSandsDB/PICKS_DIC.TXT',delimiter='\t')
picks = pd.read_csv('../../SPE_006_originalData/OilSandsDB/PICKS.TXT',delimiter='\t')
wells = pd.read_csv('../../SPE_006_originalData/OilSandsDB/WELLS.TXT',delimiter='\t')
gis = pd.read_csv('../../well_lat_lng.csv')
picks_new=picks[picks['HorID']==13000]
picks_paleoz=picks[picks['HorID']==14000]
df_new = pd.merge(wells, picks_new, on='SitID')
df_paleoz = pd.merge(wells, picks_paleoz, on='SitID')
df_new=pd.merge(df_paleoz, df_new, on='SitID')
df_new.head()

gis

type(gis)

gis[['lat','lng']]

position = gis[['lat','lng']]
[position][0]

[position][0][0:1]

type([position][0][0:1])

type([position][0:1][0:1])

[position][0:1]

tree = neighbors.KDTree(position, leaf_size=2) 

dist, ind = tree.query([position][0][0:1], k=3)  

ind

dist

ind[0]

second_pos = ind[0][1]+1
second_pos

[position][0][2:second_pos]

[position][0][2:second_pos]



