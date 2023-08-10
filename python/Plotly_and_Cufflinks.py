import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
from plotly import __version__
print(__version__)

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

# Connect Javascript to Notebook
init_notebook_mode(connected=True)

# Cufflinks in Offline Mode
cf.go_offline()

df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split(' '))

df.head()

df2 = pd.DataFrame({'Category':['A','B','C'], 'Values':[32,43,50]})

df2.head()

# Interactive Plot
df.iplot()

# Scatter Plot
df.iplot(kind='scatter',x='A',y='B', mode='markers')

# Bar Plot
df2.iplot(kind='bar',x='Category',y='Values')

df.count().iplot(kind='bar')

df.sum().iplot(kind='bar')

df.head()

# Box Plot
df.iplot(kind='box')

# 3-D Surface Plot
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[500,400,300,200,100]})

df3.head()

df3.iplot(kind='surface')

# 3-D Surface Plot
df3 = pd.DataFrame({'x':[5,4,3,2,1],'y':[10,20,30,20,10],'z':[500,400,300,200,100]})

df3.iplot(kind='surface')

df['A'].iplot(kind='hist',bins=20)

df.iplot(kind='hist',bins=20)

df[['A','B']].iplot(kind='spread')

# Bubbel Plot
df.iplot(kind='bubble',x='A',y='B',size='C')

# Scatter Matrix Plot
df.scatter_matrix()

