get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context='notebook',style='ticks',font='serif',palette='muted',rc={"xtick.direction":"in","ytick.direction":"in"})

from astropy.io import ascii
from astropy.table import Table
tab = ascii.read('frebel10.tab',fill_values=[('--','0')])
df = Table.to_pandas(tab)

df

print df.columns

df['ke'][pd.isnull(df['ke'])] = 'NA'
keys = np.unique(df['ke'].dropna())
for key in keys:
    print key,np.sum(df['ke']==key)
# AL and NO have only a single star in the category, so let's put just them as uncategorized
df['ke'][df['ke']=='AL'] = 'NA'
df['ke'][df['ke']=='NO'] = 'NA'
keys = np.unique(df['ke'].dropna())
print keys

sns.distplot(df['[Fe/H]'].dropna(),hist=False,kde=True,rug=True)

my_elems = ['C','Mg','Ca','Sr','Ba','Eu']
for elem in my_elems:
    df['[{}/Fe]'.format(elem)] = df['[{}/H]'.format(elem)]-df['[Fe/H]']

sns.jointplot('[Fe/H]','[C/Fe]',data=df)

sns.jointplot('[Fe/H]','[C/Fe]',data=df,kind='kde')

df['CEMP'] = df['[C/Fe]'] > 0.7
print np.sum(df['CEMP']),np.sum(~df['CEMP'])
g = sns.FacetGrid(df,col="CEMP",margin_titles=True)
g.map(plt.hist,'[Fe/H]',normed=True)

key_colors = sns.color_palette('Set1',len(keys),desat=.5)
sns.palplot(key_colors)

my_columns = ['ke','[Fe/H]']+['[{}/Fe]'.format(elem) for elem in my_elems]
df2 = df[my_columns].dropna()

sns.pairplot(df2,hue='ke',palette=key_colors)



