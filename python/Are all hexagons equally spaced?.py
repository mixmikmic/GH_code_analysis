import matplotlib.pyplot as plt
import pandas as pd
'''
The script plotts the hexagonal pixels of FACT with their value.
It needs the data-csv in the same folder.
'''

df = pd.read_csv('camera_bild.csv')

df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')
plt.show()

# Calculating along the x-axis.
# Sorting the values.
values = sorted(df[df['y']==0]['x'].values)
print('Every coordinate:\n', values)

values_dict = {}
for i in range(len(values)-1):
    
    #Difference to the next coordinate
    dif = round( values[i+1] - values[i] ,5)
    
    
    if dif in values_dict:
        values_dict[dif] += 1
    else:
        values_dict[dif] = 0
        
        
print('\nDif count:\n',values_dict)

# Calculating along the y-axis.
# Sorting the values.
values = sorted(df[df['x']==0]['y'].values)
print('Every coordinate:\n', values)


values_dict = {}
for i in range(len(values)-1):
    
    #Difference to the next coordinate
    dif = values[i+1] - values[i]
    
    
    if dif in values_dict:
        values_dict[dif] += 1
    else:
        values_dict[dif] = 0
        
        
print('\nDif count:\n',values_dict)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('camera_bild.csv')
df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')

df.y=round(df.y/8.2175,0)+22
df.x=df.x/4.75+39+df.y-16

df.plot.scatter(x='x', y='y', c='data', cmap=plt.cm.Reds, marker='h')
plt.show()

position_dict = {}
for id, values in enumerate(df.values):
    position_dict[id] = [values[0], values[1]]
    
position_dict

import numpy as np
positions = np.array(list(position_dict.values()))
plt.scatter(positions[:,0], positions[:,1], s=10, c='g')
plt.show()

import pickle

pickle.dump( position_dict, open( "position_dict.p", "wb" ))
positions = pickle.load( open( "position_dict.p", "rb" ))



