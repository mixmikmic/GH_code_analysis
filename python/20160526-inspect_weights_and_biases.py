import pickle as pkl
from pprint import pprint

def open_wb(path):
    with open(path, 'rb') as f:
        wb = pkl.load(f)
    
    return wb

wb = open_wb('../experiments/wbs/fp_linear-cf.score_sum-5000_iters-10_wb.pkl')

pprint(wb)

wb['layer1_LinearRegressionLayer']['linweights'].shape

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style('white')
sns.set_context('poster')

get_ipython().magic('matplotlib inline')

plt.bar(np.arange(1, 11) - 0.35, wb['layer1_LinearRegressionLayer']['linweights'])

wb = open_wb('../experiments/wbs/')

