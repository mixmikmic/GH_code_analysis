get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

import notebook
#E = notebook.nbextensions.EnableNBExtensionApp()
#E.enable_nbextension('usability/python-markdown/main')
import pandas as pd
from scipy.optimize import curve_fit
import markdown
import sys
sys.path.append('/Users/vs/Dropbox/Python')
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import seaborn as sns
import os

bigfontsize=20
labelfontsize=16
tickfontsize=16
sns.set_context('talk')
plt.rcParams.update({'font.size': bigfontsize,
                     'axes.labelsize':labelfontsize,
                     'xtick.labelsize':tickfontsize,
                     'ytick.labelsize':tickfontsize,
                     'legend.fontsize':tickfontsize,
                     })

os.chdir('/Users/vs/Dropbox/Gaia/')


hip_df = pd.read_csv('/Users/vs/Dropbox/Gaia/Hipparcos_Cepheids.csv', skiprows=48, delimiter=';', header=None, names=('HIP','SpType','VarType','maxMag','l_minMag','minMag','Period','Notes','VarName'))

tycho_df = pd.read_csv('/Users/vs/Dropbox/Gaia/Tycho_Cepheids_noheader.csv', delimiter=';', header=None, names=('TYC','RAhms','DEdms','Vmag','RA','DE','HIP','BTmag','VTmag','BV'))

hip_df

tycho_df

matched_df = hip_df.merge(tycho_df, on='HIP')

matched_df

matched_df.columns

output_columns = ['HIP', 'TYC', 'VarName']
matched_df.to_csv('hip_tycho_cepheids.csv', sep='\t', header=True, index=False, columns=output_columns)

rachael_df = pd.read_csv('/Users/vs/Dropbox/Gaia/Rachaels_Cepheids', header=None)

rachael_df.columns = ['VarName']

rachael_df['Caps'] = rachael_df['VarName'].map(lambda x: str.upper(str.upper(x)))

rachael_df

matched_df['Caps'] = matched_df['VarName'].map(lambda x: str.upper(str.upper(x)))
matched_df

match_rachaels_df = matched_df.merge(rachael_df, on='Caps')
match_rachaels_df = match_rachaels_df.drop('VarName_y', 1)

match_rachaels_df = match_rachaels_df.rename(columns={'VarName_x':'VarName'})

found_df = match_rachaels_df['VarName']
found_df



