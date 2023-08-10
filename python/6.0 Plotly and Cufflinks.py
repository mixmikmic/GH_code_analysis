import pandas as pd
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')

from  plotly import __version__

print(__version__)

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode  (connected=True)

cf.go_offline()

df=pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())

df.head()

df2=pd.DataFrame({'Category':['A','B','C'],'Values':[23,44,54]})
    

df.plot()

df.iplot()

df.iplot(kind ='scatter',x='A',y='B')

df.iplot(kind ='scatter',x='A',y='B', mode='markers')

df2.iplot(kind='bar',x='Category',y='Values')

df.iplot(kind='bar')

df.sum().iplot(kind='bar')



df.iplot(kind='box')



df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,22,12],'z':[9,6,5,3,1]})

df3

df3.iplot(kind='surface')

df3.iplot(kind='surface',colorscale='rdylbu')

df.iplot(kind='hist',colorscale='rdylbu')

df['A'].iplot(kind='hist',colorscale='rdylbu')

df[['A','B']].iplot(kind='spread',colorscale='rdylbu')

## Bubble Plot

df.iplot(kind ='bubble',x='A',y='B', size='C',colorscale='rdylbu')

df.scatter_matrix()



