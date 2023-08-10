import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mmsb
import utils

pd.options.display.max_columns = 30

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

from IPython.core.debugger import Tracer
tracer = Tracer()

import warnings
warnings.filterwarnings('error')

data = pd.read_csv('../data/all_our_ideas/4446/dat.csv', header=None)
text = pd.read_csv('../data/all_our_ideas/4446/text_map.csv', header=None)[1]
data.head()

X = data[[0,1,2]].values
X.shape

max(X[:,1]) # V

I = pd.DataFrame(utils.get_interactions(X))
plt.pcolor(I, cmap='Blues')

K = 3
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X, K, n_iter=300)
pd.Series(elbos).plot(figsize=[12,4])
max(elbos), elbos[-1]

pd.DataFrame(gamma).idxmax().value_counts()

ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')

pd.DataFrame(B).round(3)

gamma_df = pd.DataFrame(gamma.T, index=text.apply(lambda x: x[:30]))
gamma_df.iloc[:10].T.plot(kind='bar', cmap='Accent', figsize=[12,6])

gamma_df = pd.DataFrame(gamma.T, index=text.apply(lambda x: x[:50]))

def show_prototypes(ptype):
    return gamma_df[ptype][gamma_df[ptype] > .5].sort_values(ascending=False)

show_prototypes(0)

show_prototypes(1)

show_prototypes(2)



