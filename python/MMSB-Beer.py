import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mmsb
import utils

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

from IPython.core.debugger import Tracer
tracer = Tracer()

import warnings
warnings.filterwarnings('error')

data = pd.read_csv('../data/all_our_ideas/2565/2565_dat.csv', header=None)
text = pd.read_csv('../data/all_our_ideas/2565/2565_text_map.csv', header=None)[1]
data.head()

data = data[data[3] == '1bc8052fc357986cea6bf530ff4d5d3a'] # Most prolific user

X = data[[0,1,2]].values
X.shape

V = max(X[:,1]) + 1
V

I = pd.DataFrame(utils.get_interactions(X))

plt.pcolor(I, cmap='Blues')

K = 3
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X, K, n_iter=400)
ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')

pd.DataFrame(B).round(3)

gamma_df = pd.DataFrame(gamma.T, index=text.apply(lambda x: x[:50]))

gamma_df[0].sort_values(ascending=False).iloc[:10]

gamma_df[1].sort_values(ascending=False).iloc[:10]

gamma_df[2].sort_values(ascending=False).iloc[:10]



