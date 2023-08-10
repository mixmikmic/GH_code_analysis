get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import interpolate

os.listdir('data/')  

# Helper function to concat and interpolate High and low sampled trace data, noormalised to dmax = 100%

def process_HL(input_df, name_str):   
    High_low_concat = pd.concat([input_df['High'], input_df['Low']], axis=0).sort_index().dropna()  # join the two arrays, drop the NAN
    x = High_low_concat.index.values
    y = High_low_concat.values
    f = interpolate.interp1d(x, y)   # returns an interpolate function
    xnew = np.arange(0, 350)         # new mm scale required
    ynew = f(xnew)                   # use interpolation function returned by `interp1d`
    ynew = 100*ynew/ynew.max()      # normalise to Dmax = 100%
    return pd.Series(data=ynew, index=xnew, name=name_str)
    
#CC13_s = process_HL(CC13, 'CC13')
#CC13_s.plot().legend();

CC13_40 = pd.read_csv('data/CC13 100FSD PDDs - 40x40 only.csv', index_col = 0)
diamond_40= pd.read_csv('data/Micro diamond 6MV PDD 40x40 only.csv', index_col = 0)
Diode_40= pd.read_csv('data/Photon Diode 6MV PDDs 40x40 only.csv', index_col = 0)

CC13_100 = pd.read_csv('data/CC13 100FSD PDDs - 100x100 only.csv', index_col = 0)
diamond_100  = pd.read_csv('data/Micro diamond 6MV PDD 100x100 only.csv', index_col = 0)
Diode_100  = pd.read_csv('data/Photon Diode 6MV PDDs 100x100 only.csv', index_col = 0)

fig_width = 12
fig_height = 4
plt.figure(figsize=(fig_width,fig_height))

CC13_100_processed = process_HL(CC13_100, 'CC13')
diamond_100_processed = process_HL(diamond_100, 'micro Diamond')
Diode_100_processed = process_HL(Diode_100, 'Diode')

Diode_100_processed.plot().legend()
CC13_100_processed.plot().legend()
diamond_100_processed.plot().legend()

plt.xlim(300,350)
plt.ylim(17, 23)

#plt.title('6MV PDD for 100 x 100 field, 100FSD, three detectors')
plt.ylabel('PDD (%)')
plt.xlabel('Distance (mm)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

def stats_at_depth(depth_mm):
    depth_list = [Diode_100_processed.loc[depth_mm], 
                  CC13_100_processed.loc[100], 
                  diamond_100_processed.loc[100]]
    return depth_list

my_stats = np.array(stats_at_depth(100))
my_stats

my_stats.mean()

100*((my_stats.max() - my_stats.min())/my_stats.mean())

fig_width = 12
fig_height = 4
plt.figure(figsize=(fig_width,fig_height))

process_HL(CC13_40, 'CC13').plot().legend()
process_HL(diamond_40, 'micro Diamond').plot().legend()
process_HL(Diode_40, 'Diode').plot().legend()

plt.xlim(0,100)
plt.ylim(30, 105)

plt.title('6MV PDD for 40 x 40 field, 100FSD, three detectors')
plt.ylabel('PDD (%)')
plt.xlabel('Distance (mm)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));



