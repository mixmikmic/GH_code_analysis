import pandas as pd
import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from load_save_csr import load_sparse_csr, save_sparse_csr
get_ipython().magic('matplotlib inline')

mi_df = pd.read_csv('feature_score.csv', index_col=0)
mut_brig = pd.read_csv('mutation_based_df.csv', index_col=0)

mi_nonz=mi_df[mi_df.score!=0]

mi_only_mu=pd.concat((mut_brig[mi_nonz.aamutation],mut_brig['medianBrightness']),axis=1)

bright_mi = []
for mut in xrange(len(mi_nonz.aamutation)):
    bright_mi.append(mi_only_mu[mi_only_mu[mi_nonz.aamutation.iloc[mut]]==1].medianBrightness)

mi_nonz = mi_nonz.reset_index(drop=True)
mi_nonz['mean_bright']=[np.mean(x) for x in bright_mi]
mi_nonz['std_bright']=[np.std(x) for x in bright_mi]

mi_nonz.sort_values('score', ascending=False).reset_index().mean_bright.plot(kind='line');

mi_nonz.sort_values(by='score', ascending=False)

mi_nonz[mi_nonz.aamutation == 'SN183Y']



