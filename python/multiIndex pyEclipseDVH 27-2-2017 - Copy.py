get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEclipseDVH import eclipse_DVH
from scipy import interpolate

ls

my_iterables = [['Case1', 'Case5'], ['AXB', 'AAA'], ['PTV CHEST', 'ITV', 'Oesophagus']]

my_index = pd.MultiIndex.from_product(my_iterables, names = ['Patient ID', 'Plan ID', 'Structure'])

my_index

df = pd.DataFrame(np.random.randn(12,3), index = my_index)

df = df.T     # for column based indexing

df

df.index = [0, 0.5, 1.0]   # set the index

df

df['Case1']['AAA']['ITV'].max()

df.xs('AAA', level='Plan ID', axis=1)

df.xs('AAA', level='Plan ID', axis=1).xs('PTV CHEST', level='Structure', axis=1)

df.xs('AAA', level='Plan ID', axis=1).xs('PTV CHEST', level='Structure', axis=1).max()

df.xs('PTV CHEST', level='Structure', axis=1)

df.xs('PTV CHEST', level='Structure', axis=1).plot().legend(loc='center left', bbox_to_anchor=(1, 0.5))

ls

def Load_patient(file):  # file = 'Case1_AAA.txt'  string
    with open(file, "r") as file_:
        my_file = [line for line in file_.readlines()]  # my_file is a list representation of the text file
    file_.close()        
    file_len = len(my_file)                                # number of lines in the file
    patID = my_file[1].split(':')[-1].strip('\n').strip()
    print(file + ' loaded \t patID = ' + patID)
        
    ## Get the structures
    structures_indexs_list = []
    structures_names_list  = []
    for i, line in enumerate(my_file):
        if line.startswith('Structure:'):
            structures_indexs_list.append(i)  
            structures_names_list.append(line.split(':')[-1].strip('\n').strip())
        
    # iterate through all structures and place Eclipse metrics into dataframe
    for i, index in enumerate(structures_indexs_list):
        start = structures_indexs_list[i]+18  # first line of DVH data
        if i < len(structures_indexs_list)-1:
            end = structures_indexs_list[i+1]-2  # find the last line of the DVH from the next index, BEWARE THE +1
        else:
            end = len(my_file)-2
        DVH_data = my_file[start:end]  # get list with data
            
        DVH_list = [line.split() for line in DVH_data]  # create list of lists split
        Rel_dose_pct, Dose_Gy, Ratio_pct = zip(*DVH_list) # unzip to 3 lists

        temp_DVH_df    = pd.DataFrame({'Dose_Gy' : Dose_Gy, structures_names_list[i]: Ratio_pct}).astype(float)
        if i == 0: 
            DVH_df = temp_DVH_df
        else:                          
            DVH_df = pd.merge(DVH_df, temp_DVH_df, on=['Dose_Gy'])
                
    DVH_df.set_index(keys='Dose_Gy', drop=True, inplace=True)
    return DVH_df

Case1_AAA_df = Load_patient('Case1_AAA.txt')

Case1_AAA_df.head()

dose_index = np.linspace(0,100, 2001)  # New dose index in range 0 - 100 Gy in 0.05 Gy steps

test_df = Case1_AAA_df['BODY']
test_df.head()

current_index = test_df.index.get_values() 
current_index.max()

x = test_df.index.get_values()    # to get list of values rather than index object. These dont go to 100 Gy so need to extend
y = test_df.values.flatten()      # to get array

f = interpolate.interp1d(x,y)   # returns an interpolate function
interp_test_df = pd.Series(data=f(dose_index), index=xnew)

get_ipython().magic('pinfo interpolate.interp1d')



