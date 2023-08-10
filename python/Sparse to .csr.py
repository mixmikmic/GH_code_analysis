import pandas as pd
import numpy as np 
from scipy import sparse 

df = pd.read_csv('mutation_based_df.csv', index_col=0)

df.head()

x = df.values.astype(np.int8)

np.shape(x)

x = sparse.csr_matrix(x)

x

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr('sparse_csr', x)

x1 = load_sparse_csr('sparse_csr.npz')

all(x1.data == x.data)

all(x1.indices == x.indices)

all(x1.indptr == x.indptr)



