import pandas as pd
import numpy as py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics

# load iris dataset and dump into a dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# Add target response into dataframe, i.e., species
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

# convert species into factors (integers) so that it can be compared later in graphviz
# as sklearn converts them into factors during processing (still, categorical data can be used in model)
df['species_factorize'], _ = pd.factorize(df['species'])

print df.head(n=2)
print df['species'].unique()
print df['species_factorize'].unique()
# so, we can see that 0 = setosa, 1 = versicolor, 2 = virginica

# time to sort the dataframes into two
# 1) the predictors 2) the response target
predictor = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
target = df['species']

print predictor.head(n=2)
print target.head(n=2)

# split the dataframe randomly using sklearn train_test_split function
# define the size of test dataframe 25% for test here
train_predictor, test_predictor, train_target, test_target = train_test_split(predictor, target, test_size=0.25)

# print shape
# test sample is about 25% (38) of total sample while train sample is 75% (112)
print test_predictor.shape
print train_predictor.shape

# set the classifier
clf = DecisionTreeClassifier()
# fit (train) the model. Arguments (training predictor, training response target)
model = clf.fit(train_predictor, train_target)
model

# put the test sample predictor in
predictions = model.predict(test_predictor)

# score the models, using a confusion matrix, and a percentage score
print sklearn.metrics.confusion_matrix(test_target,predictions)
print sklearn.metrics.accuracy_score(test_target, predictions)*100, '%'

# it is easier to use this package that does everything nicely for a perfect confusion matrix
from pandas_confusion import ConfusionMatrix
ConfusionMatrix(test_target, predictions)

# rank the importance of features
df2= pd.DataFrame(model.feature_importances_, index=df.columns[:-2])
df2.sort_values(by=0,ascending=False)

# petal width is most important (very important in fact) followed by petal length
# this can be better visualised in a graph form, see next code below

# create a .dot file of the tree using graphviz
# arguments include your model, output name, and feature names (predictors) for labelling)
from sklearn import tree
tree.export_graphviz(model, out_file = 'tree.dot', feature_names=iris.feature_names)

# convert .dot to .ps (postscript) so the file can be open
# otherwise, visualise it online by pasting the .dot code at this link (http://www.webgraphviz.com)

import subprocess
subprocess.call(['dot', '-Tps', 'tree.dot', '-o' 'tree.ps'])
# print of '0' means success, '1' means no success

