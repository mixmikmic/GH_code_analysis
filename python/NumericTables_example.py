# Boilerplate
get_ipython().magic('matplotlib inline')

# Intel DAAL related imports
from daal.data_management import HomogenNumericTable

# Helpersfor getArrayFromNT and printNT. See utils.py
from utils import *

# Import numpy, matplotlib, seaborn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plotting configurations
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
plt.rcParams["figure.figsize"] = (12, 9)

import numpy as np
from daal.data_management import HomogenNumericTable

# The reshape is necessary because HomogenNumericTable constructor only takes array with fully defined dimensions. 
x = np.array([1., 2., 3., 4., 5., 6.]).reshape(1, 6)
x_nt = HomogenNumericTable(x)
print(x_nt.getNumberOfRows(), x_nt.getNumberOfColumns())

y_nt = HomogenNumericTable(x.reshape(6, 1))
print(y_nt.getNumberOfRows(), y_nt.getNumberOfColumns())

z_nt = HomogenNumericTable(x.reshape(2, 3))
print(z_nt.getNumberOfRows(), z_nt.getNumberOfColumns())

s = x.reshape(2, 3)
s_slice = s[:, :-1]
print(s_slice.flags['C'])

# DON'T DO THIS. s_slice is not C-contiguous!
# bad_nt = HomogenNumericTable(s_slice)

from daal.data_management import BlockDescriptor_Float64, readOnly

bd = BlockDescriptor_Float64()
z_nt.getBlockOfRows(0, z_nt.getNumberOfRows(), readOnly, bd)
z = bd.getArray()
z_nt.releaseBlockOfRows(bd)
print(z)

data = np.genfromtxt('./mldata/wine.data', dtype=np.double, delimiter=',', usecols=list(range(1, 14)), max_rows=5)
print(data.flags['C'])
data_nt = HomogenNumericTable(data)
print(data_nt.getNumberOfRows(), data_nt.getNumberOfColumns())

import pandas as pd
from utils import *

df = pd.DataFrame(np.random.randn(10, 5), columns = ['a', 'b', 'c', 'd', 'e'])
array = df.values
print(array.flags['C'])
print(array.shape)

array_nt = HomogenNumericTable(array)
print(array_nt.getNumberOfRows(), array_nt.getNumberOfColumns())

d = getArrayFromNT(array_nt)
df2 = pd.DataFrame(d, columns = ['a', 'b', 'c', 'd', 'e'])
print(df2)

from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.flags['C'])
# digits.data is NOT C-contiguous. We need to make it into the C-contiguous memory layout.
data = np.ascontiguousarray(digits.data, dtype = np.double)
data_nt = HomogenNumericTable(data[-100:])
print(data_nt.getNumberOfRows(), data_nt.getNumberOfColumns())

from scipy.sparse import csr_matrix
from daal.data_management import CSRNumericTable

# First, create a sparse matrix
values = np.array([2.0, 6.4, 1.7, 3.1, 2.2, 2.1, 3.8, 5.5])
col_ind = np.array([0, 2, 5, 3, 1, 4, 5, 6])
row_offset = np.array([0, 3, 4, 4, 6, 8])
sp = csr_matrix((values, col_ind, row_offset), dtype=np.double, shape=(5, 7))
print(sp.toarray())

# Then, create a CSRNumericTable based on the sparse matrix
sp_nt = CSRNumericTable(sp.data, sp.indices.astype(np.uint64), sp.indptr.astype(np.uint64), 7, 5)
print(sp_nt.getNumberOfRows(), sp_nt.getNumberOfColumns())
(values, col_ind, row_offset) = sp_nt.getArrays()
print("values = ", values)
print("col_ind = ", col_ind)
print("row_offset = ", row_offset)

