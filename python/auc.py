import numpy as np 
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

y_true = np.random.randint(0,2,1000000)
y_pred = np.random.rand(1000000)

fast_auc(y_true, y_pred)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_true, y_pred)

fast_auc(y_true, y_true)

get_ipython().run_line_magic('timeit', 'fast_auc(y_true, y_pred)')

get_ipython().run_line_magic('timeit', 'roc_auc_score(y_true, y_pred)')



