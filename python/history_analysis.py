get_ipython().magic('matplotlib inline')

import matplotlib
matplotlib.__version__

import numpy
numpy.__version__

import pandas
pandas.__version__

import astropy
astropy.__version__

import sys
sys.path

sys.path.pop(1)

sys.path

sys.path.append('/tigress/changgoo/pyathena-TIGRESS/src')

import ath_hst

ath_hst.__file__

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy

basedir='/tigress/changgoo/'
id='MHD_4pc_new'
hstfilename=basedir+id+'/id0/'+id+'.hst'

hst=ath_hst.read(hstfilename)

print hst.keys()

plt.plot(hst['time'],hst['sfr10'])

hstp=ath_hst.read_w_pandas(hstfilename)

hstp

plt.plot(hst['time'],hst['sfr10'])
plt.plot(hstp.time,hstp.sfr10)

hstp.plot(x='time',y=['sfr10','sfr40','sfr100'])

hstp.plot(x='time',y=['Mc','Mu','Mw','Mh1','Mh2'])

# to convert the mean density of each phase to fraction
phase=['c','u','w','h1','h2']
for p in phase:
    hstp['f'+p]=hstp['M'+p]/hstp['mass']
    hstp.plot(x='time',y='f'+p,ax=plt.gca())

hstp['H']=np.sqrt(hstp['H2']/hstp['mass'])
hstp.plot(x='time',y='H')

phase=['c','u','w','h1','h2']
for p in phase:
    hstp['H'+p]=np.sqrt(hstp['H2'+p]/hstp['M'+p])
    hstp.plot(x='time',y='H'+p,ax=plt.gca())
plt.yscale('log')

sn=ath_hst.read_w_pandas(hstfilename.replace('hst','sn'))

plt.plot(sn['time'],sn['x3'],'.')

runaway=sn['mass'] == 0.0
rsn=sn[runaway]
csn=sn[~runaway]

plt.plot(rsn['time'],rsn['x3'],'.')
plt.plot(csn['time'],csn['x3'],'.')



