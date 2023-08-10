get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import os

ls

my_iterables = [['Case1', 'Case5'], ['AXB', 'AAA'], ['PTV CHEST', 'ITV', 'Oesophagus']]

my_index = pd.MultiIndex.from_product(my_iterables, names = ['Patient ID', 'Plan ID', 'Structure'])

my_index

df = pd.DataFrame(np.random.randn(12,5), index = my_index)

df = df.T     # for column based indexing

df.index = [0, 0.1, 0.3, 0.5, 5]

df

df['Case1']['AAA']['ITV'].max()

df.xs('AAA', level='Plan ID', axis=1)

df.xs('AAA', level='Plan ID', axis=1).xs('PTV CHEST', level='Structure', axis=1)

df.xs('AAA', level='Plan ID', axis=1).xs('PTV CHEST', level='Structure', axis=1).max()

df.xs('PTV CHEST', level='Structure', axis=1)

df.xs('PTV CHEST', level='Structure', axis=1).plot().legend(loc='center left', bbox_to_anchor=(1, 0.5))

def Load_patient(file):  # file = 'Case1_AAA.txt'  string
    with open(file, "r") as file_:
        my_file = [line for line in file_.readlines()]  # my_file is a list representation of the text file
    file_.close()        
    file_len = len(my_file)                                # number of lines in the file
    patID = my_file[1].split(':')[-1].strip('\n').strip()
    planID = my_file[10].split(':')[-1].strip('\n').strip()
    
        
    ## Get the structures
    structures_indexs_list = []
    structures_names_list  = []
    for i, line in enumerate(my_file):
        if line.startswith('Structure:'):
            structures_indexs_list.append(i)  
            structures_names_list.append(line.split(':')[-1].strip('\n').strip())
    
    
    ##structures_names_list = ['PTV CHEST', 'Foramen'] # hard code to limit number of structures and prevent memory issues
    
    print(file + ' loaded \t patID:' + patID + ' PlanID:' + planID + ' and number of structures is ' + str(len(structures_names_list)))
    dose_index = np.linspace(0,100, 2001)  # New dose index in range 0 - 100 Gy in 0.05 Gy steps
    
    data = np.zeros((dose_index.shape[0], len(structures_names_list)))
    
    # iterate through all structures and place interpolated DVH data in matrix
    for i, index in enumerate(structures_indexs_list):
        start = structures_indexs_list[i]+18  # first line of DVH data
        if i < len(structures_indexs_list)-1:
            end = structures_indexs_list[i+1]-2  # find the last line of the DVH from the next index, BEWARE THE +1
        else:
            end = len(my_file)-2
        DVH_data = my_file[start:end]  # get list with data
            
        DVH_list = [line.split() for line in DVH_data]  # create list of lists split
        Rel_dose_pct, Dose_Gy, Ratio_pct = zip(*DVH_list) # unzip to 3 lists, they are strings so conver to float
        
        Ratio_pct = np.asarray(Ratio_pct, dtype=np.float32)
        Dose_Gy = np.asarray(Dose_Gy, dtype=np.float32)
        ## Now working with dose data
      
        f = interpolate.interp1d(x=Dose_Gy,y=Ratio_pct, bounds_error=False, fill_value=0)   # returns a linear interpolate function, fill values outside range wiwth 0 

        data[:,i] = f(dose_index)
    
    my_iterables = [[patID], ['AAA'], structures_names_list]
    my_index = pd.MultiIndex.from_product(my_iterables, names = ['Patient ID', 'Plan ID', 'Structure'])

    df = pd.DataFrame(data.T, index = my_index)
    df = df.T
    df.index  = dose_index
    df.index.name = 'Dose (Gy)'
    return df

Case1_AAA_df = Load_patient('Case1_AAA.txt')

Case1_AAA_df.head()

Case1_AAA_df.plot()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

structure = 'PTV CHEST'
data = Case1_AAA_df.xs(structure, level='Structure', axis=1)
print('the D50% is ' + str(data.loc[50]))
plt.plot(data, label = structure)
plt.legend()

data.head()

data.ix[70.0][0]  # get the % at a given Gy

Case1_AAA_df.ix[65.0].max()   # get the % for all structures at a given Gy

#Case1_AAA_class_df = eclipse_DVH('Case1_AAA.txt')

# files = ['Case1_AAA.txt', 'Case1_AXB.txt', 'Case5_AAA.txt', 'Case5_AXB.txt']

files = os.listdir()   # return a list of files
txt_files = [file for file in files if file.endswith('.txt')]

for i, file in enumerate(txt_files):
    if i == 0:
        multi_df = Load_patient(file)        
    else:
        multi_df = pd.concat([multi_df, Load_patient(file)], axis=1)

multi_df.head()

multi_df.xs('PTV CHEST', level='Structure', axis=1).plot()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



