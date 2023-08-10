import pandas as pd
import numpy as np
from itertools import izip
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

#Computes the euclidean distance

def euclidean_distance(a,b):
    distance=np.sqrt(np.dot(a-b,a-b))
    return distance

#Computes cosine similarity between the two vectors

def cosine_distance(a,b):
       
    distance=1-np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))
    return distance

class KNearestNeighbors(object):
    
    def __init__(self,k,distance):
        self.k=k
        self.distance=distance
        self.X_train=np.asarray([])
        self.y_train=np.asarray([])
        
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        
    def predict(self,X):
        X=X.reshape((-1,self.X_train.shape[1]))
        
        #Creating matrix to store distance
        distances=np.zeros((X.shape[0],self.X_train.shape[0]))
        for i,x in enumerate(X):
            for j,x_train in enumerate(self.X_train):
                distances[i,j]=self.distance(x_train,x)
        #Storing the indices of top k elements where distance is in increasing order
        sorted_indices=distances.argsort()[:,:self.k]
        top_k = self.y_train[sorted_indices]  #sort and take top k
        result = np.zeros(X.shape[0])
        for i, values in enumerate(top_k):
            result[i] = Counter(values).most_common(1)[0][0]
        return result
        
        

#Creating a sample data and testing it on the kNN algorithm written

from sklearn.cross_validation import train_test_split

X, y = make_classification(n_samples=50, n_features=5, n_redundant=1, n_informative=2,
                               n_clusters_per_class=2, class_sep=5,
                               random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X, y)



knn = KNearestNeighbors(3, euclidean_distance)
knn.fit(X_train, y_train)

print len(X_test)
print len(y_test)

print "\tactual\tpredict\tcorrect?"
for i, (actual, predicted) in enumerate(izip(y_test, knn.predict(X_test))):
    print "%d\t%d\t%d\t%d" % (i, actual, predicted, int(actual == predicted))



