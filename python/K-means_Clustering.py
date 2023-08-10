import pandas as pd
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as sm

from sklearn import datasets
iris = datasets.load_iris()
#iris.data
#iris.feature_names
iris.target
#iris.target_names

x = pd.DataFrame(iris.data)
x.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth'] 

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

iris = x[['SepalLength', 'PetalLength']]

X= np.array ([[ 6,5],
 [ 6.2, 5.2],
 [ 5.8,4.8]])

model_1 = KMeans(n_clusters=3, random_state=42,max_iter=1,n_init=1, init = X ).fit(iris)
centroids_1 = model_1.cluster_centers_
labels_1=(model_1.labels_)
print(centroids_1)
print(labels_1)

model_10= KMeans(n_clusters=3, random_state=42,max_iter=10, n_init=1, init = X).fit(iris)
centroids_10 = model_10.cluster_centers_
labels_10=(model_10.labels_)
print(centroids_10)
print(labels_10)

model_11= KMeans(n_clusters=3, random_state=42,max_iter=11,n_init=1, init = X).fit(iris)
centroids_max = model_11.cluster_centers_
labels_max=(model_11.labels_)
print(centroids_max)
print(labels_max)

'''model_999= KMeans(n_clusters=3, random_state=42,max_iter=999).fit(iris)
centroids_max = model.cluster_centers_
labels_max=(model.labels_)
print(centroids_max)
print(labels_max)'''

# Set the size of the plot
plt.figure(figsize=(24,10))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
#colormap = {0: 'r', 1: 'g', 2: 'b'}

# Plot Original
plt.subplot(1, 4, 1)
plt.scatter(x.SepalLength, x.PetalLength, c="K", s=40)
plt.scatter(X[:,0],X[:,1],  c="b")
plt.title('Initial centroids')

# Plot the Models Classifications
plt.subplot(1, 4, 2)
plt.scatter(iris.SepalLength, iris.PetalLength, c=colormap[labels_1], s=40)
plt.scatter(centroids_1[:,0],centroids_1[:,1],  c="b")
plt.title('K Mean Clustering(iter=1)')


plt.subplot(1, 4, 3)
plt.scatter(iris.SepalLength, iris.PetalLength, c=colormap[labels_10], s=40)
plt.scatter(centroids_10[:,0],centroids_10[:,1],  c="b")
plt.title('K Mean Clustering (iter=10)')
                                                           
plt.subplot(1, 4, 4)
plt.scatter(iris.SepalLength, iris.PetalLength, c=colormap[labels_max], s=40)
plt.scatter(centroids_max[:,0],centroids_max[:,1],  c="b")
plt.title('K Mean Clustering (iter= MAX)')

plt.show()

def confusion(y,labels):
    cm = sm.confusion_matrix(y, labels)
    return cm

# Confusion Matrix (iter=1)
set_list = ["setosa","versicolor","virginica"]
cluster_list = ["c1", "c2", "c3"]
data = confusion(y, labels_1)
pd.DataFrame(data,cluster_list, set_list)

# Confusion Matrix (iter=10)
set_list = ["setosa","versicolor","virginica"]
cluster_list = ["c1", "c2", "c3"]
data = confusion(y, labels_10)
pd.DataFrame(data,cluster_list, set_list)

# Confusion Matrix (iter=max)
set_list = ["setosa","versicolor","virginica"]
cluster_list = ["c1", "c2", "c3"]
data = confusion(y, labels_max)
pd.DataFrame(data,cluster_list, set_list)

# Calculate purity of each confusion matrix
def Purity(cm):
    M=[]
    S=0
    for i in cm:
        k = max(i)
        M.append(k)
    for i in M:
        S+=i
    Purity=S/150 
    return Purity

metric_list = ["iter= 1", "iter= 10", "iter= MAX"]
set_list = ["Purity metric"]
data = np.array([Purity(confusion(y, labels_1)),Purity(confusion(y, labels_10)),Purity(confusion(y, labels_max))])
pd.DataFrame(data,metric_list, set_list)

#k=2 , random-state= 0
model = KMeans(n_clusters=2,).fit(x)
centroids = model.cluster_centers_
labels=(model.labels_)
print(centroids)
print(labels)

#Confusion matrix
set_list = ["setosa","versicolor","virginica"]
cluster_list = ["c1", "c2", "c3"]
data = confusion(y, labels)
pd.DataFrame(data,set_list, cluster_list)

print ("Purity(k=2)= %f " % Purity(confusion(y, labels)))

#k=3 , random-state= 0
model = KMeans(n_clusters=3,).fit(x)
centroids = model.cluster_centers_
labels=(model.labels_)
print(centroids)
print(labels)

#Confusion matrix
set_list = ["setosa","versicolor","virginica"]
cluster_list = ["c1", "c2", "c3"]
data = confusion(y, labels)
pd.DataFrame(data,set_list, cluster_list)

print ("Purity(k=3)= %f " % Purity(confusion(y, labels)))

#k=4 , random-state= 0
model = KMeans(n_clusters=4,).fit(x)
centroids = model.cluster_centers_
labels=(model.labels_)
print(centroids)
print(labels)


# Confusion Matrix 
set_list = ["setosa","versicolor","virginica","undefined"]
cluster_list = ["c1", "c2", "c3","c4"]
data = confusion(y, labels)
pd.DataFrame(data,set_list, cluster_list)

print ("Purity(k=4)= %f " % Purity(confusion(y, labels)))

#k=6 , random-state= 0
model = KMeans(n_clusters=6,).fit(x)
centroids = model.cluster_centers_
labels=(model.labels_)
print(centroids)
print(labels)

# Confusion Matrix 
set_list = ["setosa","versicolor","virginica","undefined_1","undefined_2","undefined_3"]
cluster_list = ["c1", "c2", "c3","c4","c5","c6"]
data = confusion(y, labels)
pd.DataFrame(data,set_list, cluster_list)

print ("Purity(k=6)= %f " % Purity(confusion(y, labels)))

