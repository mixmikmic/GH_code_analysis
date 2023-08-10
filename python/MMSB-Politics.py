import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
import mmsb

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

from IPython.core.debugger import Tracer
tracer = Tracer()

import warnings
warnings.filterwarnings('error')

data = pd.read_csv('../data/all_our_ideas/727/727_dat.csv', header=None)
names = pd.read_csv('../data/all_our_ideas/727/727_text_map.csv', header=None)[1]
data.head()

X = data[[0,1,2]].values
X.shape

V = max(X[:,1]) + 1
V

K = 8
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X, V, K, n_iter=300)
pd.Series(elbos).plot(figsize=[12,4])
max(elbos), elbos[-1]

pd.DataFrame(B).round(3)

I = pd.DataFrame(utils.get_interactions(X, V))
gamma_df = pd.DataFrame(gamma.T)
gamma_df = gamma_df[gamma_df[0] != gamma_df[1]] # Remove items with no answers
ptypes = gamma_df.idxmax(axis=1).sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')

plt.pcolor(I.ix[gamma_df.index][gamma_df.index], cmap='Blues')

gamma_df = pd.DataFrame(gamma.T, index=names[:V].apply(lambda x: x[:50]))
gamma_df = gamma_df[gamma_df[0] != gamma_df[1]] # Remove items with no answers
gamma_df.iloc[:12].T.plot(kind='bar', cmap='Accent', figsize=[12,6])

gamma_df[0].sort_values(ascending=False).iloc[:5]

gamma_df[1].sort_values(ascending=False).iloc[:5]

gamma_df[2].sort_values(ascending=False).iloc[:5]

gamma_df[3].sort_values(ascending=False).iloc[:5]

gamma_df[4].sort_values(ascending=False).iloc[:5]

gamma_df[5].sort_values(ascending=False).iloc[:5]

gamma_df[6].sort_values(ascending=False).iloc[:5]

gamma_df[7].sort_values(ascending=False).iloc[:5]



