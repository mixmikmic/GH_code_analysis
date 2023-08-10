import pandas as pd
import numpy as np 
from scipy import sparse, stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from load_save_csr import load_sparse_csr, save_sparse_csr
get_ipython().magic('matplotlib inline')

df = pd.read_csv('mutation_based_df.csv', index_col=0)

y = df['medianBrightness'].values
x = df.drop(df[['uniqueBarcodes', 'medianBrightness','std']], axis=1)

print len(y)==len(x)

#select a random subset if do'nt want to use all the data

idx = np.random.choice(len(y),1000, replace=False) 
y_ = y[idx]
x_ = x.ix[idx]

x_csr = sparse.csr_matrix(x_.values) # x to sparse

# freedom for RAM nation
del df
del x

mi = mutual_info_regression(x_csr,y_,n_neighbors=10) #compute the scores, it takes a while on all data

print len(mi)==len(x_.columns) #true

feature_score = pd.DataFrame(data={'aamutation': x_.columns, 'score':mi})
feature_score.sort_values(by=['score'], ascending= False).head()

mi_l=[]
for i in xrange(1,11):
    mi_l.append(mutual_info_regression(x_csr,y_,n_neighbors=i))

mi_df=pd.DataFrame(mi_l)

mi_df.T.describe()

