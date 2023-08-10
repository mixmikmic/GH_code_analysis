get_ipython().magic('reload_ext autoreload')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)
get_ipython().magic('aimport preprocess')
from preprocess.process import get_symbol
from preprocess.process import get_symbols_matrix
from preprocess.process import df_to_returns
from preprocess.process import get_windows_rets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from IPython.core.debugger import Tracer
import scipy as sp
from scipy.linalg import svd
from sklearn.decomposition import PCA
get_ipython().magic('matplotlib inline')

data_loc = "/home/dan/code/market_data/"
symbols = [line.rstrip('\n') for line in open(data_loc+'dow_jones_industrial.txt')]

train_start_date = "01/01/2013"
train_end_date = "12/31/2013"

test_start_date = "03/15/2015"
test_end_date = "03/21/2015"

#aapl = get_symbol("AAPL", data_loc, train_start_date, train_end_date)

test_symbols = symbols

aapl = get_symbol("AAPL", data_loc, train_start_date, train_end_date)

awins, arets= get_windows_rets(aapl, window_length=15, window_offset=1)

plt.plot(awins[0])
plt.show()
print(arets[0])

X = awins

pca = PCA()
pca.fit(X)

plt.plot(pca.explained_variance_, linewidth=3)
plt.title('Eigenvalue decay on PCA')
plt.savefig('eigenvals.png')
plt.show()

ci = 0
ri = 0
fig, axes = plt.subplots(nrows=5, ncols=3)
for i in range(15):
    axes[ri, ci].plot((1+pca.components_[i]).cumprod())
    axes[ri, ci].get_xaxis().set_visible(False)
    axes[ri, ci].get_yaxis().set_visible(False)
    ci += 1
    if ci == 3:
        ri += 1
        ci = 0
plt.savefig('pca_features.png')
plt.show()


plt.plot(aapl.Close)
plt.show()











