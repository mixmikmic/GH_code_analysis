import pandas as pd
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

koi = pd.read_csv('http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative')
#koi = pd.read_excel('planets.xls')

koi.describe()

koi["kepid"]

ax = koi.plot(x='koi_period', y='koi_prad', c='koi_steff', s=koi['koi_srad']*30, colormap=plt.cm.hot, kind='scatter', 
              figsize=(10,6), xlim=(0.1,2000), loglog=True)

koi['koi_steff'].describe()

koi['koi_srad'].describe()

koi.index = koi['kepid']

koi.iloc[42]

koi.loc[11304958]

koi.kepler_name.isnull()

output = koi.ix[koi.kepler_name.isnull() == False]

output

output.to_excel('planets.xls')

