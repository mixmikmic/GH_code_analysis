import numpy as np
data = np.genfromtxt("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", 
                     usecols=range(8))
data.shape

print data[32,:]

any(np.isnan(data[32,:]))

bad_rows = []

for i, line in enumerate(data):
    if any(np.isnan(line)):
        bad_rows.append(i)
    
count = 0
for i in bad_rows:
    data = np.delete(data,i-count,0)
    count = count + 1

np.isnan(data).any()

data.shape

A = data[:,1:7]
A.shape

y = data[:,0] # miles per gallon

get_ipython().magic('pinfo np.linalg.lstsq')

x, resid, rank, s = np.linalg.lstsq(A,y)

print x

