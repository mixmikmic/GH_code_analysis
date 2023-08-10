import numpy as np
import pandas as pd

# set up a toy dataset with 10 variables
r = 10
c = 10
np.random.seed([0])
toy_set = np.random.rand(r, c)
column_labels = ['v'+str(i) for i in range(1, c+1)]
toy_df = pd.DataFrame(toy_set, columns=column_labels)

toy_df.head()

toy_df.corr()

toy_df.corr()['v1']['v2']

import itertools

def corr_df(data):
    ''' 
    input: pandas DataFrame
    output: pandas DataFrame listing every possible pair of variables and their corresponding 
            correlation (rho-squared)
    '''
    # get column labels
    column_labels = data.columns
    
    # create the initial correlation table
    corr_df = data.corr()
    
    # create a generator that will iterate through all possible pairs of variables
    combs = itertools.combinations(column_labels, 2)
    
    # iterate through each pair, squaring the correlations
    corrs = [[comb, corr_df[comb[0]][comb[1]]**2] for comb in combs]
    
    # return a DataFrame of the correlations, sorted high-to-low
    return pd.DataFrame(corrs, columns=['Comb', 'R^2']).sort_values('R^2', ascending=False)

corr_df(toy_df).head()

# set up a larger dataset with 1000 variables
big_r = 10
big_c = 1000
big_set = np.random.rand(big_r, big_c)
big_column_labels = ['v'+str(i) for i in range(1, big_c+1)]
big_df = pd.DataFrame(big_set, columns=big_column_labels)

get_ipython().run_cell_magic('time', '', 'big_df.corr().head()')

get_ipython().run_cell_magic('time', '', 'big_corrs = corr_df(big_df)')

big_corrs[big_corrs['R^2'] >= .95]

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(big_df[['v471', 'v521']])
pca.explained_variance_ratio_

