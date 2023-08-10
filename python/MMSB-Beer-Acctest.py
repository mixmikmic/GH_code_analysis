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

np.random.shuffle(X)
X_train, X_test = X[:900], X[900:]
X_train.shape, X_test.shape

sum(X_test[:,2] == 0) / float(len(X_test))

I = pd.DataFrame(utils.get_interactions(X, V))

plt.pcolor(I, cmap='Blues')

K = 3
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X_train, V, K, n_iter=400)
ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')
probs = [gamma[:,p].dot(B).dot(gamma[:,q]) for p, q, v in X_test]
sum(X_test[:,2] == np.round(probs)) / float(len(X_test))

K = 3
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X_train, V, K, n_iter=400)
ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')
probs = [gamma[:,p].dot(B).dot(gamma[:,q]) for p, q, v in X_test]
sum(X_test[:,2] == np.round(probs)) / float(len(X_test))

pd.DataFrame(B).round(2)

K = 6
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X_train, V, K, n_iter=400)
ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')
probs = [gamma[:,p].dot(B).dot(gamma[:,q]) for p, q, v in X_test]
sum(X_test[:,2] == np.round(probs)) / float(len(X_test))

K = 4
gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X_train, V, K, n_iter=400)
ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')
probs = [gamma[:,p].dot(B).dot(gamma[:,q]) for p, q, v in X_test]
sum(X_test[:,2] == np.round(probs)) / float(len(X_test))

results = []
for K in xrange(1,15):
    gamma, phi_pq, phi_qp, B, elbos = mmsb.train_mmsb(X_train, V, K, n_iter=400)
    ptypes = pd.DataFrame(gamma).idxmax().sort_values().index
    plt.pcolor(I.ix[ptypes][ptypes], cmap='Blues')
    probs = [gamma[:,p].dot(B).dot(gamma[:,q]) for p, q, v in X_test]
    results.append((K, sum(X_test[:,2] == np.round(probs)) / float(len(X_test))))

x, y = zip(*results)
plt.scatter(x=x, y=y)

x, y = zip(*results)
plt.scatter(x=x, y=y)



