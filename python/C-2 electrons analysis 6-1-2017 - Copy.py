get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Profiles  = pd.read_table('RC_edit_electron_Profile_6_1_17.csv', sep= ",", index_col=0, header=[0,1,2], skiprows=[3]).fillna(0).astype(float);
Profiles.head()  # loaded data, comse in as multi-index, not sure why different

Profiles['6 MeV'].plot()
plt.title('Profiles, 6 MeV')
plt.ylabel('PDD (%)')
plt.xlabel('Distance (mm)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

PDD  = pd.read_table('RC_edit_electron_PDD_6_1_17.csv', sep= ",", index_col=0, header=[0,1,2], skiprows=[3]).fillna(0).astype(float);
#PDD.set_option('display.multi_sparse', True)
PDD.head()  # loaded data, comse in as multi-index, not sure why different

# ?PDD   # comes in as a dataframe, not multiindex

fig = plt.figure(figsize=(12, 8))
plt.plot(PDD['6 MeV']['100 x 100 mm'], label='6 MeV')
plt.plot(PDD['9 MeV']['100 x 100 mm'], label = '9 MeV')
plt.plot(PDD['12 MeV']['100 x 100 mm'], label = '12 MeV')
plt.plot(PDD['20 MeV']['100 x 100 mm'], label = '20 MeV')

plt.title('PDD, 100 x 100 mm')

plt.xlim(0,120)
plt.ylim(0,105)
plt.ylabel('PDD (%)')
plt.xlabel('Distance (mm)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

PDD.columns.levels[0]

PDD.columns

PDD.index.get_level_values(0)



