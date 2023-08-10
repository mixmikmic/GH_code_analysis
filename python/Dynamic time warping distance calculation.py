import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import import_data
import sort_data

charge_partial,discharge_partial = sort_data.charge_discharge('converted_PL03.mat')
charge_full,discharge_full = sort_data.charge_discharge('converted_PL11.mat')

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#Inputs are two nparray
x = np.array([1, 2, 3])
y = np.array([0.5, 1.5, 2.5])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)

x = np.array([1, 2, 3])
y = np.array([0.5, 1.5, 2.5,2.8])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)

#Distance between partial cycle #3 and full cycle #3
a = discharge_partial[3][['voltage']].values.flatten()
b = discharge_full[3][['voltage']].values.flatten()
distance, path = fastdtw(a, b, dist=euclidean)
print(distance)

#Distance between partial cycle #3 and full cycle #4
a = discharge_partial[3][['voltage']].values.flatten()
b = discharge_full[4][['voltage']].values.flatten()
distance, path = fastdtw(a, b, dist=euclidean)
print(distance)

a = discharge_partial[3][['voltage']].values.flatten()
b = discharge_full[5][['voltage']].values.flatten()
distance, path = fastdtw(a, b, dist=euclidean)
print(distance)

a = discharge_partial[3][['voltage']].values.flatten()
b = discharge_full[6][['voltage']].values.flatten()
distance, path = fastdtw(a, b, dist=euclidean)
print(distance)


num_samples = 61
group_size = 10

#
# create the main time series for each group
#

x = np.linspace(0, 5, num_samples)
scale = 4

a = scale * np.sin(x)
b = scale * (np.cos(1+x*3) + np.linspace(0, 1, num_samples))
c = scale * (np.sin(2+x*6) + np.linspace(0, -1, num_samples))
d = scale * (np.cos(3+x*9) + np.linspace(0, 4, num_samples))
e = scale * (np.sin(4+x*12) + np.linspace(0, -4, num_samples))
f = scale * np.cos(x)



timeSeries = pd.DataFrame()
ax = None
for arr in [a,b,c,d,e,f]:
    arr = arr + np.random.rand(group_size, num_samples) + np.random.randn(group_size, 1)
    df = pd.DataFrame(arr)
    timeSeries = timeSeries.append(df)

timeSeries = pd.DataFrame()
ax = None
for arr in [a,b,c,d,e,f]:
    arr = arr + np.random.rand(group_size, num_samples) + np.random.randn(group_size, 1)
    df = pd.DataFrame(arr)
    timeSeries = timeSeries.append(df)

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

Z = hac.linkage(X, 'ward')

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)



