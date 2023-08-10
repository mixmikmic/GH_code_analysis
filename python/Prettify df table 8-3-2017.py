from ipy_table import *
import numpy as np
import pandas as pd

ls

df_table = pd.read_csv('ALL_DATA_8_3_2017.csv')
df_table = df_table.fillna(value='-')   # swap the 
df_table

df_table = df_table[['VOI Material', 'Mass density', 'Stp PWR', 'CT number', 'AAA', 'AXB Dm', 'AXB Dw']]

headers = list(df_table.columns.values)
print(len(headers))
headers  # get headers as these are no included when recast to matrix

headers = np.asarray(headers).T
headers.shape

df_array = df_table.as_matrix()   # get table data as array

df_array.shape

df_array = np.insert(df_array, 0,headers, axis=0)   # re insert the headers
df_array.shape

#df_array

make_table(df_array)
apply_theme('basic')

#set_global_style(float_format='%0.3E') # experiment with 
set_global_style(float_format="%.2f")
set_global_style(align='center')

get_ipython().magic('pinfo set_cell_style')



