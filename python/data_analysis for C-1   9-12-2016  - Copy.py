get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MicroDiamond_df_raw  = pd.read_table('6MV 100FSD Profiles MicroDiamond - RAW.csv', sep= ",", header=[1,4,5],  skiprows=[6]) # index_col=0,  .fillna(0).astype(float);
MicroDiamond_df_raw = MicroDiamond_df_raw.ix[:,1:]  # skip the first column

tuples_list = MicroDiamond_df_raw.columns
cross_type, field_type, depth_mm = list(zip(*tuples_list))    # unzip the original columns index

cross_type_strings = [ "%s" % x for x in cross_type ]
field_type_strings = [ "%s" % x for x in field_type ]
depth_mm_strings = [ "%s" % x for x in depth_mm ]    # get lists of strings

tuples = list(zip(*[cross_type_strings, field_type_strings, depth_mm_strings]))  #zip back into a list of tuples containing strings

index_multi = pd.MultiIndex.from_tuples(tuples)  # get required multiindex
#index_multi

MicroDiamond_df  = pd.read_table('6MV 100FSD Profiles MicroDiamond - RAW.csv', sep= ",", header=[1,4,5], skiprows=[6], index_col=0) # 
MicroDiamond_df.columns = index_multi
MicroDiamond_df.head()

MicroDiamond_df['Crossline']['100 x 100 mm'].plot()

Diode_df_raw  = pd.read_table('6MV Profiles Photon Diode  - RAW.csv', sep= ",",  header=[1,4,5], skiprows=[6]) # index_col=0,.fillna(0).astype(float);
#Diode_df.head()  # loaded data
Diode_df_raw = Diode_df_raw.ix[:,1:]  # skip the first column

tuples_list = Diode_df_raw.columns
cross_type, field_type, depth_mm = list(zip(*tuples_list))    # unzip the original columns index

#cross_type_strings, field_type_strings, depth_mm_strings = [], [], []  # clear the lists from memory

cross_type_strings = [ "%s" % x for x in cross_type ]
field_type_strings = [ "%s" % x for x in field_type ]
depth_mm_strings = [ "%s" % x for x in depth_mm ]    # get lists of strings

tuples = list(zip(*[cross_type_strings, field_type_strings, depth_mm_strings]))  #zip back into a list of tuples containing strings
index_multi = pd.MultiIndex.from_tuples(tuples)  # get required multiindex

Diode_df  = pd.read_table('6MV Profiles Photon Diode  - RAW.csv', sep= ",", header=[1,4,5], skiprows=[6], index_col=0) # 
Diode_df.columns = index_multi
Diode_df.head()

Diode_df['Crossline']['100 x 100 mm'].plot()

CC13_df  = pd.read_table('6X Profiles CC13 RAW.csv', sep= ",", index_col=0, header=[1,4,5], skiprows=[6]).fillna(0).astype(float);
CC13_df.head()  # loaded data, comse in as multi-index, not sure why different

CC13_df['Crossline']['100 x 100 mm'].plot() # as multiindex index like this

def plot_profile(field_size, depth, plot_xlim):
    trace1 = Diode_df['Crossline'][field_size][depth]
    trace2 = MicroDiamond_df['Crossline'][field_size][depth]
    trace3 = CC13_df['Crossline'][field_size][depth]
    mean_trace = (trace1 + trace2 + trace3)/3.0

    plt.plot(trace1, color='red', label='Diode')
    plt.plot(trace2, color='green', label='MicroDiamond')
    plt.plot(trace3, color='black', label='CC13')
    plt.plot(mean_trace, color='blue', label='Mean')

    plt.title('Field %s and depth %s' % (field_size, depth))
    plt.xlim( [0,plot_xlim] ) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
plot_profile(field_size = '100 x 100 mm', depth = '300.0 mm', plot_xlim = 100)

plt.plot(trace1 - mean_trace, label='Diode - mean' )
plt.plot(trace2 - mean_trace, label='MicroDiamond - mean' )
plt.plot(trace3 - mean_trace, label='CC13 - mean' )
plt.axhline(color='k', linestyle='solid')
plt.xlim( [0,plot_xlim] ) 
plt.title('Difference trace')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

all_fields = set(Diode_df.columns.get_level_values(1))  |  set(MicroDiamond_df.columns.get_level_values(1)) | set(CC13_df.columns.get_level_values(1))
print(all_fields) # return the set of depths)

print('But in all')
intersect_fields = list(set(Diode_df.columns.get_level_values(1))  &  set(MicroDiamond_df.columns.get_level_values(1)) & set(CC13_df.columns.get_level_values(1)))
print(intersect_fields) 

depths = list(set(Diode_df.columns.get_level_values(2)))[1:]
depths

to_plot = [i+j for i,j in zip(intersect_fields, depths)]

to_plot



