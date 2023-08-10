get_ipython().run_cell_magic('bash', '', "cat /proc/cpuinfo | grep 'processor\\|model name'")

get_ipython().run_cell_magic('bash', '', 'free -g')

from __future__ import print_function
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets.widgets import interact, Text
from IPython.display import display
import numpy as np

# use the notebook definition for interactive embedded graphics
# %matplotlib notebook

# use the inline definition for static embedded graphics
get_ipython().magic('matplotlib inline')

rcParam = {
    'figure.figsize': (12,6),
    'font.weight': 'bold',
    'axes.labelsize': 20.0,
    'axes.titlesize': 20.0,
    'axes.titleweight': 'bold',
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}

for key in rcParam:
    mpl.rcParams[key] = rcParam[key]

cbs_data = pd.read_csv('combined_data.csv',sep=',',na_values=['NA','.'],error_bad_lines=False);

cbs_data.head()

cbs_data_2015 = cbs_data.loc[cbs_data['YEAR'] == 2015];
#list(cbs_data_2015)

cbs_data_2015.describe()
#cbs_data_2015.YEAR.describe()

cbs_data_2015 = cbs_data_2015.dropna();
cbs_data_2015.describe()

cbs_data_2015.iloc[:,35:216].describe()

labels = cbs_data_2015["Vermogensmisdrijven_rel"].values
columns = list(cbs_data_2015.iloc[:,37:215])

features = cbs_data_2015[list(columns)];
features = features.apply(lambda columns : pd.to_numeric(columns, errors='ignore'))

print(labels[1:10])
features.head()

from sklearn.linear_model import RandomizedLasso

rlasso = RandomizedLasso(alpha='aic',verbose =True,normalize =True,n_resampling=3000,max_iter=100)
rlasso.fit(features, labels)

dfResults = pd.DataFrame.from_dict(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), list(features)), reverse=True))
dfResults.columns = ['Score', 'FeatureName']
dfResults.head(10)

dfResults.plot('FeatureName', 'Score', kind='bar', color='navy')
ax1 = plt.axes()
x_axis = ax1.axes.get_xaxis()
x_axis.set_visible(False)
plt.show()

plt.scatter(y=pd.to_numeric(cbs_data_2015['Vermogensmisdrijven_rel']),x=pd.to_numeric(cbs_data_2015['A_BED_GI']));
plt.ylabel('Vermogensmisdrijven_rel')
plt.xlabel('A_BED_GI ( Bedrijfsvestigingen: Handel en horeca )')
plt.show()

dfResults.tail(10)

plt.scatter(y=pd.to_numeric(cbs_data_2015['Vermogensmisdrijven_rel']),x=pd.to_numeric(cbs_data_2015['P_LAAGINKH']));
plt.ylabel('Vermogensmisdrijven_rel')
plt.xlabel('Perc. Laaginkomen Huish.')
plt.show()

plt.scatter(y=pd.to_numeric(cbs_data_2015['Gewelds_en_seksuele_misdrijven_rel']),x=pd.to_numeric(cbs_data_2015['P_GESCHEID']));
plt.ylabel('Gewelds_en_seksuele_misdrijven_rel')
plt.xlabel('Perc_Gescheiden')
plt.show()

