# Import pandas to read in data
import numpy as np
import pandas as pd

# Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import cross_validation

# Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import plotting
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

get_ipython().system('head -3 data/books.csv')

data = pd.read_csv("data/books.csv", quotechar="\"", escapechar="\\")

data.head()

X_text = data['review_text']
Y = data['positive']

# Create a vectorizer that will track text as binary features
binary_vectorizer = CountVectorizer(binary=True)

# Let the vectorizer learn what tokens exist in the text data
binary_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = binary_vectorizer.transform(X_text)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation AUCs
aucs = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="roc_auc", cv=5)

# Print out the average AUC rounded to three decimal points
print "Area under the ROC curve for our classifier is " + str(round(np.mean(aucs), 3))

# Create a vectorizer that will track text as binary features
count_vectorizer = CountVectorizer()

# Let the vectorizer learn what tokens exist in the text data
count_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = count_vectorizer.transform(X_text)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation AUCs
aucs = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="roc_auc", cv=5)

# Print out the average AUC rounded to three decimal points
print "Area under the ROC curve for our classifier is " + str(round(np.mean(aucs), 3))

# Create a vectorizer that will track text as binary features
tfidf_vectorizer = TfidfVectorizer()

# Let the vectorizer learn what tokens exist in the text data
tfidf_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = tfidf_vectorizer.transform(X_text)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation AUCs
aucs = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="roc_auc", cv=5)

# Print out the average AUC rounded to three decimal points
print "Area under the ROC curve for our classifier is " + str(round(np.mean(aucs), 3))

# Work with your teams here!
# Try different features, models, or both!
# What is the highest AUC you can get?

data = pd.read_csv("data/categorical.csv")

data

features_to_binarize = ["Gender", "Marital"]

# Go through each feature
for feature in features_to_binarize:
    # Go through each level in this feature (except the last one!)
    for level in data[feature].unique()[0:-1]:
        # Create new feature for this level
        data[feature + "_" + level] = pd.Series(data[feature] == level, dtype=int)
    # Drop original feature
    data = data.drop([feature], 1)

data

data['Satisfaction'] = data['Satisfaction'].replace(['Very Low', 'Low', 'Neutral', 'High', 'Very High'], 
                                                    [-2, -1, 0, 1, 2])

data



