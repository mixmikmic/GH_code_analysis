# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import seaborn as sns
sns.set()

iris = sns.load_dataset('iris')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(15, 15))
sns.pairplot(iris,hue='species')

# Setosa is the most separable Flower



iris.head()

sns.kdeplot(data2=iris[iris['species'] == 'setosa'].sepal_length, data=iris[iris['species'] == 'setosa'].sepal_width)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.drop('species', axis=1), iris['species'], test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report

pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

from sklearn.grid_search import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid_model = GridSearchCV(SVC(), param_grid, verbose=3)
grid_model.fit(X_train, y_train)

pred = grid_model.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

grid_model.best_params_

grid_model.best_estimator_



