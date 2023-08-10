from sklearn.datasets import samples_generator

from sklearn.feature_selection import SelectKBest, f_regression
#f_regression - identifies which K features to select from given number of features

from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier

#generate Data
X,y = samples_generator.make_classification(n_samples=150, n_features=25, n_classes=3, random_state=7, n_informative=6)

X.shape

k_base_selector = SelectKBest(f_regression, k =9 )

classifier = ExtraTreesClassifier(n_estimators=50, max_depth=4)

##### Pipeline Creation

processor_pipeline = Pipeline([('selector',k_base_selector),('erf',classifier) ])

processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)

processor_pipeline.fit(X,y)

output = processor_pipeline.predict(X)

processor_pipeline.score(X,y)

status = processor_pipeline.named_steps['selector'].get_support()

#list comprehension
selected = [i for i,x in enumerate(status) if x]

selected

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.neighbors import NearestNeighbors

X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9],  
        [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], 
        [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]]) 

k = 5

test_datapoint = [4.3, 2.7]

# Plot input data  
plt.figure() 
plt.title('Input data') 
plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='black') 
plt.plot([4.3],[2.7],'ro')

knn_model = NearestNeighbors(n_neighbors=k).fit(X)

knn_model.kneighbors([test_datapoint])

X[12]

X[7]

import pandas as pd

house_data = pd.read_csv('house_rental_data.csv',index_col='Unnamed: 0')

knn_model = NearestNeighbors(n_neighbors=k).fit(house_data)

knn_model.kneighbors(house_data[1:2])

house_data.head()

house_data[1:2]

house_data[358:359]

house_data[442:443]

#### Using knn classifier

data = pd.read_csv('data.csv.txt',header=None)

X = data[[0,1]]

y = data[2]

num_neighbors = 12

from sklearn import neighbors

classifier = neighbors.KNeighborsClassifier(num_neighbors)

classifier.fit(X,y)

classifier.predict(X)

#Convert dataframe to numpy array using matrix
d = data.as_matrix()
plt.scatter(d[:,0], d[:,1], c= d[:,2])
plt.plot(1.77,2.67,'ro')
#d[:,0].shape

_, pts = classifier.kneighbors([[2.3,2]])

pts

pts[0]

data.loc[pts[0]]



