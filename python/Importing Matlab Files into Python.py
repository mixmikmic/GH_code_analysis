import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

df = hdf5storage.loadmat('converted_PL11.mat')
df2 = hdf5storage.loadmat('converted_PL03.mat')

df2.keys()

# Requires 3 inputs to get to the base level of the arrays

# THIS CELL HAS DATA FROM "1st Charge"

print(df['#subsystem#'][0][0][0][2][0][0])
print('\n')
print(df['#subsystem#'][0][0][0][2][0][5])
print('\n')
len((df['#subsystem#'][0][0][0][2][0][0]))

df['#subsystem#'][0][0][0][7][0][0][0][0] # Gets time-sec

df['#subsystem#'][0][0][0]

len(df['data2'])

df['data2'][1][0]

df['#subsystem#'][0][0][0][count][0][0].flatten()



host_df = pd.DataFrame()

host_df['time'] = df['#subsystem#'][0][0][0][count][0][0].flatten()
host_df['datetime'] = df['#subsystem#'][0][0][0][count][0][1].flatten()
host_df['step'] = df['#subsystem#'][0][0][0][count][0][2].flatten()
host_df['cycle'] = df['#subsystem#'][0][0][0][count][0][3].flatten()
host_df['current_amp'] = df['#subsystem#'][0][0][0][count][0][4].flatten()
host_df['voltage'] = df['#subsystem#'][0][0][0][count][0][5].flatten()
host_df['charge_ah'] = df['#subsystem#'][0][0][0][count][0][6].flatten()
host_df['discharge_ah'] = df['#subsystem#'][0][0][0][count][0][7].flatten()

len(discharge_ah)

len(df['#subsystem#'][0][0][0])

data_dict_full_cycle = {}
count = 2

for idx in range(1, len(df['data2'])):
    operation = str(df['data2'][idx][0][0][0])
    start_date = str(df['data2'][idx][1][0][0])
    
    host_df = pd.DataFrame()
    
    host_df['time'] = df['#subsystem#'][0][0][0][count][0][0].flatten()
    host_df['datetime'] = df['#subsystem#'][0][0][0][count][0][1].flatten()
    host_df['step'] = df['#subsystem#'][0][0][0][count][0][2].flatten()
    host_df['cycle'] = df['#subsystem#'][0][0][0][count][0][3].flatten()
    host_df['current_amp'] = df['#subsystem#'][0][0][0][count][0][4].flatten()
    host_df['voltage'] = df['#subsystem#'][0][0][0][count][0][5].flatten()
    host_df['charge_ah'] = df['#subsystem#'][0][0][0][count][0][6].flatten()
    host_df['discharge_ah'] = df['#subsystem#'][0][0][0][count][0][7].flatten()
    
    data_dict_full_cycle[operation + ' ' + start_date] = host_df
    
    count = count + 7
    
    if count > len(df['#subsystem#'][0][0][0]):
        break
    
    #op.append(operation)
    #start_d.append(start_date)

data_dict_full_cycle['101 Full Cycles April 23, 2015']

# Requires 3 inputs to get to the base level of the arrays
# Every 7, it repeats

# THIS CELL HAS DATA FROM "1st Charge"

print(df['#subsystem#'][0][0][0][2][0][0])
print('\n')
print(df['#subsystem#'][0][0][0][2][0][5])# This is Time, or first column
print('\n')
len((df['#subsystem#'][0][0][0][2][0][0]))



len(df['#subsystem#'][0][0][0][9][0][0]) # This is the second dataset

(df['#subsystem#'][0][0][0][16][0][0])

58+7

len((df['#subsystem#'][0][0][0][86][0][0]))





df['#subsystem#'][0][0][0]

len(df2['#subsystem#'][0][0][0])

count = 2+(7*21)
df2['#subsystem#'][0][0][0][count][0][5]
#df2['#subsystem#'][0][0][0][count][0][1].flatten().shape

7*17

len(df2['#subsystem#'][0][0][0])

df2['converted_PL03'][15][2].shape == (0,0)

count = 9
host_df = pd.DataFrame()

host_df['time'] = df2['#subsystem#'][0][0][0][count][0][0].flatten()
host_df['datetime'] = df2['#subsystem#'][0][0][0][count][0][1].flatten()
host_df['step'] = df2['#subsystem#'][0][0][0][count][0][2].flatten()
host_df['cycle'] = df2['#subsystem#'][0][0][0][count][0][3].flatten()
host_df['current_amp'] = df2['#subsystem#'][0][0][0][count][0][4].flatten()
host_df['voltage'] = df2['#subsystem#'][0][0][0][count][0][5].flatten()
host_df['charge_ah'] = df2['#subsystem#'][0][0][0][count][0][6].flatten()
host_df['discharge_ah'] = df2['#subsystem#'][0][0][0][count][0][7].flatten()

data_dict_partial_cycle = {}
count = int(2)

for idx in range(1, len(df2['converted_PL03'])):
    operation = str(df2['converted_PL03'][idx][0][0][0])
    start_date = str(df2['converted_PL03'][idx][1][0][0])
    
    if df2['converted_PL03'][idx][2].shape == (0,0):
        pass
    
    else:
    
        host_df = pd.DataFrame()

        host_df['time'] = df2['#subsystem#'][0][0][0][count][0][0].flatten()
        host_df['datetime'] = df2['#subsystem#'][0][0][0][count][0][1].flatten
        host_df['step'] = df2['#subsystem#'][0][0][0][count][0][2].flatten()
        host_df['cycle'] = df2['#subsystem#'][0][0][0][count][0][3].flatten()
        host_df['current_amp'] = df2['#subsystem#'][0][0][0][count][0][4].flatten()
        host_df['voltage'] = df2['#subsystem#'][0][0][0][count][0][5].flatten()
        host_df['charge_ah'] = df2['#subsystem#'][0][0][0][count][0][6].flatten()
        host_df['discharge_ah'] = df2['#subsystem#'][0][0][0][count][0][7].flatten()

        data_dict_partial_cycle[operation + ' ' + start_date] = host_df

        count = count + 7

        if count > len(df2['#subsystem#'][0][0][0]):
            break
    
    #op.append(operation)
    #start_d.append(start_date)

data_dict_partial_cycle['50 Partial Cycles April 14, 2015'].tail()



