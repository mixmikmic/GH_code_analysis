import numpy as np
import pandas as pd
from matplotlib import pylab as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic('cd pandas_data/')

get_ipython().system('head -n15 nvt_berendsen.gro')

get_ipython().system('tail -n5 nvt_berendsen.gro')

gro_header=['res_id', 'elem', 'atomno', 'x', 'y', 'z', 'vx', 'vy', 'vz']

df = pd.read_table('nvt_berendsen.gro', sep='\s+', skiprows=2, names=gro_header)

df.iloc[-1]=np.NaN      # last line contains the definition of the simulation box.
df.dropna(inplace=True)

df.tail()

df.head()

df['vel'] = (df.vx **2 + df.vy **2 + df.vz **2) **0.5

df.head()

df['vel'].hist(bins=50)

df.describe()

gb = df[ ['elem', 'vel' ] ].groupby('elem')
#.plot(kind='hist', bins=50, by='elem')

df.groupby(['elem']).mean()

df.groupby(['elem'])['vel'].describe()

df.groupby(['elem'])['vel'].count()

df.groupby(['elem'])['vel'].mean()

df.groupby(['elem'])['vel'].median()

df.groupby(['elem'])['vel'].std()

df.groupby(['elem'])['vel'].hist(bins=50, alpha=0.5)

df.groupby(['elem'])['vel'].plot(kind='hist', bins=50, alpha=0.7, legend=True)

fig1 = plt.figure(1, figsize=(15,5), )
plt.suptitle('Velocity Distribution with Berendsen Thermostat') # Figure super title

plt.subplot(1, 3, 1)
plot_h = df[ df['elem'] == 'H'  ]['vel'].plot(kind='hist', bins=50, xlim=(0.,5.))
plot_h.set_title('Hydrogen')

plt.subplot(1, 3, 2)
plot_c = df[ df['elem'] == 'C'  ]['vel'].plot(kind='hist', bins=20, xlim=(0.,5.))
plot_c.set_title('Carbon')

plt.subplot(1, 3, 3)
plot_cl= df[ df['elem'] == 'CL' ]['vel'].plot(kind='hist', bins=20, xlim=(0.,5.))
plot_cl.set_title('Chlorine')

df_bussi = pd.read_table('nvt_bussi.gro', sep='\s+', skiprows=2, names=gro_header)
df_bussi['vel'] = (df_bussi.vx **2 + df_bussi.vy **2 + df_bussi.vz **2) **0.5

fig1 = plt.figure(1, figsize=(15,5), )
plt.suptitle('Velocity Distribution with Bussi Thermostat')

plt.subplot(131)
plot_h = df_bussi[ df_bussi['elem']=='H' ]['vel'].plot(kind='hist', bins=50, xlim=(0.,5.))
plot_h.set_title('Hydrogen')

plt.subplot(132)
plot_c = df_bussi[ df_bussi['elem']=='C' ]['vel'].plot(kind='hist', bins=20, xlim=(0.,5.))
plot_c.set_title('Carbon')

plt.subplot(133)
plot_cl= df_bussi[ df_bussi['elem']=='CL']['vel'].plot(kind='hist', bins=20, xlim=(0.,5.))
plot_cl.set_title('Chlorine')

