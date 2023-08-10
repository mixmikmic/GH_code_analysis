import sys
sys.path.append('/Users/IzmailovPavel/Documents/Education/GPproject/gplib/')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

from gplib.gpc import GPCLaplace, GPCSparse
from gplib.optim.methods import *
from gplib.covfun import SE

x_tr = np.load('../../../GPtf/data/mnist/features_tr.npy')
x_te = np.load('../../../GPtf/data/mnist/features_te.npy')

from keras.datasets import mnist

(_, y_tr), (_, y_te) = mnist.load_data()

data_name = 'mnist'

scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
x_te = scaler.transform(x_te)

y_tr = (y_tr %2 == 0).astype(float)
y_te = (y_te %2 == 0).astype(float)

x_tr = (x_tr + 1) / 2
x_te = (x_te + 1) / 2
y_tr = y_tr[:, None]
y_te = y_te[:, None]
y_tr[y_tr == 0] = -1
y_te[y_te == 0] = -1
num, dim = x_tr.shape
print('of objects:', num)
print('of features:', dim)
print(data_name)

ind_num = 500
print('Finding means...')
means = KMeans(n_clusters=ind_num, n_init=3, max_iter=100, random_state=241)
means.fit(x_tr)
inputs = means.cluster_centers_
print('...found')

gp = GPCSparse(SE(np.array([1., .5, .2])), inputs=inputs)

options = {'optimizer': LBFGS(disp=False, maxfun=5), 'maxiter': 5, 'disp':1}
res = gp.fit(x_tr, y_tr, method='JJ', options=options)

y_pred = gp.predict(x_te)

gp.get_quality(y_te, y_pred)

metric = lambda w: gp.get_prediction_quality(x_te, y_te, w)
x_lst, y_lst = res.plot_performance(metric, 't', freq=1)
plt.plot(x_lst, y_lst, '-bx', label='vi-JJ')
plt.ylim([.5, 1.])
plt.legend()
plt.xlabel('Seconds')
plt.ylabel('Accuracy')



