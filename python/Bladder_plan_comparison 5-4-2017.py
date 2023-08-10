from pyEclipseDVH_v2 import List_txt, Load_patient, get_dmin, get_dmax, get_d_metric, Load_files_to_df
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Prescription = 36.0 # Gy

def diff_to_prescribed(dose, Prescribed_dose):
    return 100.0 + 100.0*(dose-Prescribed_dose)/Prescribed_dose

txt_files = List_txt()
txt_files

multi_df = Load_files_to_df(txt_files)

multi_df.to_csv('All_data.csv')

multi_df.columns = multi_df.columns.droplevel()
multi_df.head()

def plot_structure(structure, xlim):
    multi_df.xs(structure, level='Structure', axis=1).plot()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('The DVH for structure ' + structure)
    plt.ylabel('Relative dose (%)')
    plt.xlim(xlim)
    return

structure = 'PTV BLADDER'
plot_structure(structure, xlim=[30,40])

structure = 'Rectum'
plot_structure(structure, xlim=[0,40])

def d50(df):
    return diff_to_prescribed(get_d_metric(df, 50.0), Prescription)

d50_df = multi_df.apply(d50)    # function of form lambda function that takes a single argument

d50_df

for item in d50_df.index.values:
    print(item)

d50_df[('Planned', 'CTV36Gy')]

d50_df[('Replanned', 'CTV36Gy')]



