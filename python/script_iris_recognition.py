get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
iris.keys()

print(iris['DESCR'][:193] + '\n...')

iris['target_names'] #types of iris

iris['feature_names'] #description of each feature

print(iris['data'].shape)
iris['data'][:10] #contains the measurement of each flower

#an numpy array containing 0 to 2 values. Each one for a flower type. 0 for Setosa, 1 for Versicolor and 2 for Virginica
iris['target']

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)

print(X_train.shape)
print(X_test.shape)

fig, ax = plt.subplots(3, 3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X_train[:,j], X_train[:, i + 1], c=y_train, s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        
        if i == 2:
            ax[i,j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i,j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i,j].set_visible(False)

#Creating the model
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

#testing model
X_new = np.array([[5, 2.9, 1, 0.2]])
X_new.shape

prediction = knn.predict(X_new)
prediction

iris['target_names'][prediction]

#Accuracy of the model
knn.score(X_test, y_test)

