get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn import manifold

featureNum = 32

isNum = np.ndarray(shape=(32), dtype=np.bool)
for i in range(len(isNum)):
    isNum[i] = True
    
la = True

with open('../../data/vandalism_preprocessed1.csv', newline='') as csvfile:
    data_reader = csv.reader(csvfile)
    firstLine = True
    for row in data_reader:
        if firstLine:
            firstLine = False
        else:
            for i, data in enumerate(row):
                if la:
                    la = False
                try:
                    f = float(data)
                except ValueError:
                    isNum[i] = False

isNum

la = [i for i in range(len(isNum)) if isNum[i]]

with open('../../data/vandalism_preprocessed1.csv', newline='') as csvfile:
    data_reader = csv.reader(csvfile)
    for row in data_reader:
        for i in la:
            print(row[i])
        break

X = np.ndarray(shape=(100000, len(la)), dtype=np.float32)

with open('../../data/vandalism_preprocessed1.csv', newline='') as csvfile:
    data_reader = csv.reader(csvfile)
    firstLine = True
    for i, row in enumerate(data_reader):
        if firstLine:
            firstLine = False
        else:
            X[i] = [float(row[j]) for j in la]

X[0]

X[1]

fig = plt.figure(figsize=(60,40))
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X[:4000])
ax = fig.add_subplot(2,5,10)
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()

fig = plt.figure(figsize=(60,40))
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X[4000:8000])
ax = fig.add_subplot(2,5,10)
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()

na = np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]])

na

np.delete(na, 0, 1)

X = np.delete(X, 0, 1)

Y = np.ndarray(shape=(100000), dtype=np.float32)
with open('../../data/vandalism_preprocessed1_ground_truth.csv', newline='') as csvfile:
    data_reader = csv.reader(csvfile)
    firstLine = True
    for i, row in enumerate(data_reader):
        if i == 100000:
            break
        if firstLine:
            firstLine = False
        else:
            Y[i] = 1.0 if row[2] == 'true' else 0.0

np.ones(Y)



