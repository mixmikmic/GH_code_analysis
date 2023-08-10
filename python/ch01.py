import pandas as pd
import numpy as np

# read data file to dataframe
df = pd.read_csv("../data/global_temp.txt", delim_whitespace=True, header=None, names=["Year", "TempChange"])

# get number of rows in data frame
n = df.shape[0]

# extract dependent data (temp change) and independent data (year) from dataframe into appropriate vector d and matrix G
d = df.as_matrix(columns=["TempChange"])
G = np.ones((n, 2))
G[:, 1] = df.as_matrix(columns=["Year"])[:,0]

print("d = ")
print(d[0:4])
print("")
print("G =")
print(G[0:4,:])

# initialize new matrix G
G = np.ones((n,3))

# extract independent data (year) from data frame
t = df.as_matrix(columns=["Year"])[:,0]

# insert independent data (and it's element-wise square) into the matrix G
G[:, 1] = t
G[:, 2] = np.multiply(t, t)

print(G[0:4, :])

G = np.zeros((8, 16))

# loop through rows
for i in range(0, 4):
    # loop through columns
    for j in range(0, 4):
        # row measurements
        k = i * 4 + j
        G[i, k] = 1
        
        # column measurements
        k = j * 4 + i
        G[i + 4, k] = 1
        
print(G)

from scipy.sparse import csr_matrix

# build a sparse matrix A, and vector v
A = csr_matrix([[1,2,0], [0,0,3], [4,0,5]])
v = np.array([1, 0, -1])

# multiply and output sparse matrix A and vector v
A.dot(v)

