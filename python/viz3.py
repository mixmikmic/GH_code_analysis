get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

df = pd.read_csv("output.csv")
#df.to_csv("output.csv", date_format='%Y%m%d')
#getting rid of undecided tweets
df = df[(df.prob <> .5)]
df.head()

#run these using your username and key
import plotly.tools as tls
tls.set_credentials_file(username='ayinmv', api_key='rq66z3hqx8')

import plotly.plotly as py
from plotly.graph_objs import *

#install cufflink package
import cufflinks as cf
print cf.__version__

airline_count = df.groupby(['airline']).count().prob.sort_values(axis=0, ascending=False, inplace=False)
airline_count.iplot(kind='bar', yTitle='Number of Tweets', title='Number of Tweets')

tls.embed('https://plot.ly/~ayinmv/33')

airline_count = df.groupby(['airline']).mean().prob.sort_values(axis=0, ascending=False, inplace=False)
airline_count.iplot(kind='bar', yTitle='Average Score', title='Average Score')

tls.embed('https://plot.ly/~ayinmv/63')

#df['created_at'].strftime("%d %b %Y")
#df['created_at'].apply(lambda x: x.strftime('%d%m%Y'))

pd.set_option('max_colwidth', 200)

df.sort_values(by=['followers_count'], ascending=[False])[['followers_count','text', 'prob', 'created_at']].head(20)

# gb = df.groupby('airline')    
# [gb.get_group(x) for x in gb.groups]

#gb.prob

#gb.iplot(kind='box')

#df.iplot([Box(y = np.random.randn(50), showlegend=False) for i in range(10)], show_link=False)

# for i, group in df.groupby('airline'):
#     plt.figure()
#     #group.plot(x='prob', y='followers_count')
#     group.hist('prob')

dfgroupby = pd.read_csv("outputgroupby.csv")
dfgroupby.head()

dfgroupby.iplot(kind='box', title='Score Quantiles')

tls.embed('https://plot.ly/~ayinmv/78')

