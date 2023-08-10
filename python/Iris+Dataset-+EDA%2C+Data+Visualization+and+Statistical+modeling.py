#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#reading the dataset 
iris=pd.read_csv('Iris.csv')

iris

#Information about Dataset
iris.describe()

iris.info()

#number of records per species
iris.Species.value_counts()

#relationship between Sepal length and width using Seaborn
sns.FacetGrid(iris, hue="Species", size=6).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.title("Relationship between Sepal Length and Width")

#relationship between Petal length and width 
sns.FacetGrid(iris, hue="Species", size=6).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
plt.title("Relationship between Petal Length and Width")

#using pairplot to check correlation between species

sns.pairplot(iris,hue='Species')

#correlation between parameters
corr_iris=iris.corr()

corr_iris

#vizualizing correlations using heatmap
sns.heatmap(corr_iris,annot=True)

key = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
species = iris['Species'].map(key)



iris

iris=pd.concat([iris,species],axis=1)

iris

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#linear modelling
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
print('Training Accuracy of Linear Model = {}'.format(lm.score(X_train, y_train)))
print('Testing  Accuracy of Linear Model = {}'.format(lm.score(X_test, y_test)))

predictions = lm.predict(X_test)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Logistic Regression

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train,y_train)
print('Training Accuracy  = {}'.format(logReg.score(X_train, y_train)))
print('Testing  Accuracy  = {}'.format(logReg.score(X_test, y_test)))

predictions2 = logReg.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions2))

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions2))
print('MSE:', metrics.mean_squared_error(y_test, predictions2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions2)))

#Decision Tree classifier

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini',max_depth=4,presort=True)
dt.fit(X_train,y_train)
print('Training Accuracy  = {}'.format(dt.score(X_train, y_train)))
print('Testing  Accuracy = {}'.format(dt.score(X_test, y_test)))

predictions3 = dt.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions3))

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions3))
print('MSE:', metrics.mean_squared_error(y_test, predictions3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions3)))

#KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
print('Training Accuracy = {}'.format(model.score(X_train, y_train)))
print('Testing  Accuracy = {}'.format(model.score(X_test, y_test)))

predictions4 = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions4))

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions4))
print('MSE:', metrics.mean_squared_error(y_test, predictions4))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions4)))

from sklearn.svm import LinearSVC

model2 = LinearSVC(C=3)
model2.fit(X_train,y_train)
print('Training Accuracy = {}'.format(model2.score(X_train, y_train)))
print('Testing  Accuracy = {}'.format(model2.score(X_test, y_test)))

predictions5 = model2.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions5))

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions5))
print('MSE:', metrics.mean_squared_error(y_test, predictions5))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions5)))

