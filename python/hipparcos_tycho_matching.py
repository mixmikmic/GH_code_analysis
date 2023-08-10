get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

import notebook
E = notebook.nbextensions.EnableNBExtensionApp()
E.enable_nbextension('usability/python-markdown/main')
import pandas as pd
from scipy.optimize import curve_fit
import markdown
import sys
sys.path.append('/Users/vs/Dropbox/Python')
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import seaborn as sns

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

hipparcos = pd.read_csv("Hipparcos_RRL", sep='|')
hipparcos.rename(columns=lambda x: x.strip(), inplace=True) ## Stripping the whitespace out of the column names
hipparcos.rename(columns = {'RA(ICRS)':'RA', 'DE(ICRS)':'Dec'}, inplace=True) ## Getting rid of the (ICRS)
hipparcos = hipparcos.assign(RRName = lambda x: (x.VarName.str.replace('_', '')))
hipparcos = hipparcos.assign(Catname = 'Hipparcos')

tycho = pd.read_csv("Tycho_RRL", sep='|')
tycho.rename(columns=lambda x: x.strip(), inplace=True)
tycho.rename(columns = {'RAJ2000' : 'RA', 'DEJ2000' : 'Dec', 'VType' : 'VarType'}, inplace=True)
tycho = tycho.assign(RRName = lambda x: (x.Star.str.replace(' ', '')))
tycho = tycho.assign(Catname = 'Tycho')


hipparcos.info()

tycho.head()

hipparcos.head()

matched = pd.merge(left=hipparcos, right=tycho, how='left', left_on='RRName', right_on='RRName')

len(tycho) + len(hipparcos)

hiptyc = pd.concat([hipparcos, tycho], axis=0)

hiptyc.head()

hiptyc.to_csv('hiptyc.csv', columns=('RRName', 'RA', 'Dec', 'Catname'), header=('NAME', 'RA', 'DEC', 'CATNAME'), index=False)

tycho.to_csv('tycho.csv', columns=('RRName', 'RA', 'Dec'), header=('NAME', 'RA', 'DEC'), index=False)

spitzer_ipac = Table.read('hiptyc_obs_by_spitzer.ipac', format='ipac')

spitzer = spitzer_ipac.to_pandas()

spitzer.head()

spitzer['Tgt'], spitzer['Scrap'] = zip(*spitzer['Search_Tgt'].apply(lambda x: x.split(' ', 1)))
del spitzer['Scrap']
del spitzer['Search_Tgt']
s = spitzer.groupby(['Tgt'])
len(s)

wlf_obs = spitzer.loc[spitzer['PI']=='Freedman, Wendy']
kvj_obs = spitzer.loc[spitzer['PI']=='Johnston, Kathryn V']

wlf_targets = wlf_obs.groupby(['Tgt'])
n_wlf = len(wlf_targets.groups)
kvj_targets = kvj_obs.groupby(['Tgt'])
n_kvj = len(kvj_targets.groups)
n_wlf, n_kvj

wlf_obs['Search_Tgt_2'] = wlf_obs.Tgt
wlf_obs = wlf_obs.drop_duplicates(cols='Search_Tgt_2', take_last=True)
del wlf_obs['Search_Tgt_2']
wlf_obs.sort_index(inplace=True)

wlf_obs

other_obs = spitzer.loc[spitzer['PI']!='Freedman, Wendy'] ## Any observer except Wendy

other_targets = other_obs.groupby(['Tgt'])
n_other = len(other_targets.groups)
n_other

other_obs['Search_Tgt_2'] = other_obs.Tgt
other_obs = other_obs.drop_duplicates(cols='Search_Tgt_2', take_last=True)
del other_obs['Search_Tgt_2']
other_obs.sort_index(inplace=True)

other_obs

RA_Dec = SkyCoord(ra=(hiptyc.RA)*u.degree, dec=(hiptyc.Dec)*u.degree, frame='icrs')

hiptyc = hiptyc.assign(ra_hours = lambda x: (RA_Dec.ra.to_string('h')))

wlf_obs = wlf_obs.assign(Target_name = lambda x: (x.Target_name.str.replace('_', '')))

wlf_obs

spitzer_tycho_ipac = Table.read('tycho_obs_by_spitzer.ipac', format='ipac')

spitzer_tycho = spitzer_tycho_ipac.to_pandas()

spitzer_tycho.head()

spitzer_tycho['Tgt'], spitzer_tycho['Scrap'] = zip(*spitzer_tycho['Search_Tgt'].apply(lambda x: x.split(' ', 1)))
del spitzer_tycho['Scrap']
del spitzer_tycho['Search_Tgt']
s = spitzer_tycho.groupby(['Tgt'])
len(s)

wlf_obs_tycho = spitzer_tycho.loc[spitzer_tycho['PI']=='Freedman, Wendy']
kvj_obs_tycho = spitzer_tycho.loc[spitzer_tycho['PI']=='Johnston, Kathryn V']

wlf_obs_tycho['Search_Tgt_2'] = wlf_obs_tycho.Tgt
wlf_obs_tycho = wlf_obs_tycho.drop_duplicates(cols='Search_Tgt_2', take_last=True)
del wlf_obs_tycho['Search_Tgt_2']
wlf_obs_tycho.sort_index(inplace=True)

wlf_obs_tycho

spitzer_tycho



