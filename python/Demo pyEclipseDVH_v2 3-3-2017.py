from pyEclipseDVH_v2 import List_txt, Load_patient, get_dmin, get_dmax, get_d_metric, Load_files_to_df
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

txt_files = List_txt()
txt_files

multi_df = Load_files_to_df(txt_files)

multi_df.head()

structure = 'PTV CHEST'
multi_df.xs(structure, level='Structure', axis=1).plot()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('The DVH for structure ' + structure)
plt.ylabel('Relative dose (%)')

multi_df.to_csv('All_data.csv')   # save all data to flat CSV

patient = 'Case1'
planID = 'AAA'
structure = 'PTV CHEST'
df = multi_df.xs(patient, level='Patient ID', axis=1).xs(planID, level='Plan ID', axis=1)[structure] # structure is final level so access like this
df.plot()
plt.legend()

get_dmin(df)      # very close to value in Eclipse text file - 49.72

get_dmax(df)

get_d_metric(df, 50.0)  



