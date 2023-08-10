# Here we introduce Data science by starting with a common regression model(logistic regression). The example uses the Iris Dataset
# We also introduce Python as we develop the model. (The Iris dataset section is adatped from an example from Analyics Vidhya) 
# Python uses some libraries which we load first. 
# numpy is used for Array operations
# mathplotlib is used for visualization

import numpy as np
import matplotlib as mp
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

dataset = datasets.load_iris()

# Display the data
dataset

# first we need to understand the data

from IPython.display import Image
from IPython.core.display import HTML
Image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg")

Image("http://www.opengardensblog.futuretext.com/wp-content/uploads/2016/01/iris-dataset-sample.jpg")

# In statistics, linear regression is an approach for modeling the relationship between a scalar dependent variable y 
# and one or more explanatory variables (or independent variables) denoted X. There are differnt types of regressions that model the
# relationship between the independent and the dependent variables 

# In linear regression, the relationships are modeled using linear predictor functions whose unknown model 
# parameters are estimated from the data. Such models are called linear models.

# In mathematics, a linear combination is an expression constructed from a set of terms by multiplying 
# each term by a constant and adding the results (e.g. a linear combination of x and y would be any expression of the 
# form ax + by, where a and b are constants)

# Linear regression
Image("https://www.biomedware.com/files/documentation/spacestat/Statistics/Multivariate_Modeling/Regression/regression_line.png")

Image(url="http://31.media.tumblr.com/e00b481257fac723638b32271e611a2f/tumblr_inline_ntui2ohGy41sfzcxh_500.gif")

model = LogisticRegression()
model.fit(dataset.data, dataset.target)

expected = dataset.target
predicted = model.predict(dataset.data)

# classification metrics report builds a text report showing the main classification metrics
# In pattern recognition and information retrieval with binary classification, 
# precision (also called positive predictive value) is the fraction of retrieved instances that are relevant, 
# while recall (also known as sensitivity) is the fraction of relevant instances that are retrieved. 
# Both precision and recall are therefore based on an understanding and measure of relevance. 
# Suppose a computer program for recognizing dogs in scenes from a video identifies 7 dogs in a scene containing 9 dogs 
# and some cats. If 4 of the identifications are correct, but 3 are actually cats, the program's precision is 4/7 
# while its recall is 4/9.

# In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
# It considers both the precision p and the recall r of the test to compute the score: 
# p is the number of correct positive results divided by the number of all positive results, 
# and r is the number of correct positive results divided by the number of positive results that should have been returned. 
# The F1 score can be interpreted as a weighted average of the precision and recall

print(metrics.classification_report(expected, predicted))

# Confusion matrix 
# https://en.wikipedia.org/wiki/Confusion_matrix
# In the field of machine learning, a confusion matrix is a table layout that allows visualization of the performance 
# of an algorithm, typically a supervised learning one. 
# Each column of the matrix represents the instances in a predicted class 
# while each row represents the instances in an actual class (or vice-versa)

# If a classification system has been trained to distinguish between cats, dogs and rabbits, 
# a confusion matrix will summarize the results of testing the algorithm for further inspection. 
# Assuming a sample of 27 animals — 8 cats, 6 dogs, and 13 rabbits, the resulting confusion matrix 
# could look like the table below:

Image("http://www.opengardensblog.futuretext.com/wp-content/uploads/2016/01/confusion-matrix.jpg")

# In this confusion matrix, of the 8 actual cats, the system predicted that three were dogs, 
# and of the six dogs, it predicted that one was a rabbit and two were cats. 
# We can see from the matrix that the system in question has trouble distinguishing between cats and dogs, 
# but can make the distinction between rabbits and other types of animals pretty well. 
# All correct guesses are located in the diagonal of the table, so it's easy to visually 
# inspect the table for errors, as they will be represented by values outside the diagonal.

print (metrics.confusion_matrix(expected, predicted))

import pandas as pd

integers_list = [1,3,5,7,9] # lists are seperated by square brackets
print(integers_list)
tuple_integers = 1,3,5,7,9 #tuples are seperated by commas and are immutable
print(tuple_integers)
tuple_integers[0] = 11

#Python strings can be in single or double quotes
string_ds = "Data Science"

string_iot = "Internet of Things"

string_dsiot = string_ds + " for " + string_iot

print (string_dsiot)

len(string_dsiot)

# sets are unordered collections with no duplicate elements
prog_languages = set(['Python', 'Java', 'Scala'])
prog_languages

# Dictionaies are comma seperated key value pairs seperated by braces
dict_marks = {'John':95, 'Mark': 100, 'Anna': 99}

dict_marks['John']



