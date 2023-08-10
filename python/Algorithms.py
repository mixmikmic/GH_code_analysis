# Loading necessary modules
import pandas as pd              # data processing; csv file i/o (pd.read_csv)
import numpy as np               # linear algebra
import matplotlib.pyplot as plt  # visulaization
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading the dataset
data = pd.read_csv('C:/Users/Dell-pc/Desktop/Minor Project/Dataset/Twitter_Suicide_Data_new.csv')

data.head()

len(data)

texts = []
labels = []
for i, label in enumerate(data['Sentiment']):
    texts.append(data['Content'][i])
    if label == 'Negative':
        labels.append(0)
    else:
        labels.append(1)

texts = np.asarray(texts)
labels = np.asarray(labels)

print("number of texts :" , len(texts))
print("number of labels: ", len(labels))

print(type(texts))

print(texts[0])

print(np.unique(labels))
print(np.bincount(labels))

174/129    #More negative

np.random.seed(42)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]

# we will use 80% of data as training, 20% as validation data
training_samples = int(303 * .8)
validation_samples = int(303 - training_samples)
# sanity check
print(len(texts) == (training_samples + validation_samples))
print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))

texts_train = texts[:training_samples]
y_train = labels[:training_samples]
texts_test = texts[training_samples:]
y_test = labels[training_samples:]

# Example
toy_samples = ["It is sunny today and I like it, ", 
               "she does not like hamburger"]
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# tokenize the document (split texts into each unique words)
vect.fit(toy_samples)
print("tokenization")
print(vect.vocabulary_, "\n")

# transform document into a matrix(the number indicates the number of words showing up in the document)
bag_of_words = vect.transform(toy_samples)
print("Transformed sparse matrix is: ")
print(bag_of_words.toarray())

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(texts_train)
X_train = vect.transform(texts_train)
print(repr(X_train))

X_train.shape

X_test = vect.transform(texts_test)

np.bincount(y_train)

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)
logreg_train = grid.fit(X_train, y_train)

print(grid.best_estimator_)

# logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg_train.predict(X_test)
print("accuracy is: ", grid.score(X_test, y_test))

confusion = confusion_matrix(y_test, pred_logreg)
print(confusion)

#Improve
#check vocab
features_names = vect.get_feature_names()
print(len(features_names))
print("\n")
# print first 20 features
print(features_names[:20])
print("\n")
# print last 20 features
print(features_names[-20:])
print("\n")
# print every 50th word
print(features_names[::400])

# We will use only the words that appear in at least 3 tweets -- In other words, we will use frequent words which are also likely to be in the test set
# min_df controls this condition(min_df=3 means pick up words which appear
# at least 3 documents)
vect = CountVectorizer(min_df=3).fit(texts_train)
X_train = vect.transform(texts_train)
X_test = vect.transform(texts_test)
print(repr(X_train))

features_names = vect.get_feature_names()
print(len(features_names))
print("\n")
# print first 20 features
print(features_names[:20])
print("\n")
# print last 20 features
print(features_names[-20:])
print("\n")
# print every 50th word
print(features_names[::400])

logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)
logreg_train = grid.fit(X_train, y_train)

# logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg_train.predict(X_test)
print("accuracy is: ", grid.score(X_test, y_test))

confusion = confusion_matrix(y_test, pred_logreg)
print(confusion)

# remove stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Number of stop words is :", len(ENGLISH_STOP_WORDS), "\n")
print("Examples: ", list(ENGLISH_STOP_WORDS)[::10])

vect = CountVectorizer(min_df=3, stop_words='english').fit(texts_train)
X_train = vect.transform(texts_train)
X_test = vect.transform(texts_test)
print(repr(X_train))

logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)
logreg_train = grid.fit(X_train, y_train)

# logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg_train.predict(X_test)
print("accuracy is: ", grid.score(X_test, y_test))
confusion = confusion_matrix(y_test, pred_logreg)
print("confusion matrix \n", confusion)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

logreg = LogisticRegression()
pipe = make_pipeline(TfidfVectorizer(min_df=3, norm=None, stop_words='english'), logreg)
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)
logreg_train = grid.fit(texts_train, y_train)

logreg.get_params().keys()

print(grid.best_estimator_)

# logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg_train.predict(texts_test)
print("accuracy is: ", grid.score(texts_test, y_test))
confusion = confusion_matrix(y_test, pred_logreg)
print("confusion matrix \n", confusion)

# Check which words are considered to be low tfidf(widely used words across many tweets) and high tfidf (used only in a few tweets)
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
X_train = vectorizer.transform(texts_train)
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()

feature_names = np.array(vectorizer.get_feature_names())

print("features with lowest tfidf")
print(feature_names[sorted_by_tfidf[:20]], '\n')

print("featues with hightest tfidf")
print(feature_names[sorted_by_tfidf[-20:]])

get_ipython().system('pip install mglearn')

import mglearn 
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps['logisticregression'].coef_, 
                                   feature_names, n_top_features=40)
plt.title("tfidf-cofficient")

pipe = make_pipeline(TfidfVectorizer(min_df=3, stop_words='english'), logreg)
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100], 
              'tfidfvectorizer__ngram_range': [(1,1), (1,2), (1,3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
logreg_train = grid.fit(texts_train, y_train)

# logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg_train.predict(texts_test)
print("accuracy is: ", grid.score(texts_test, y_test))
confusion = confusion_matrix(y_test, pred_logreg)
print("confusion matrix \n", confusion)

print(grid.best_estimator_)

scores = [s.mean_validation_score for s in grid.grid_scores_]
scores = np.array(scores).reshape(-1, 3).T

heatmap = mglearn.tools.heatmap(scores, xlabel="C", ylabel="ngram_range", 
                                xticklabels=param_grid['logisticregression__C'], 
                                yticklabels=param_grid['tfidfvectorizer__ngram_range'], 
                                cmap='viridis', fmt="%.3f")
plt.colorbar(heatmap);

feature_names = np.array(grid.best_estimator_.named_steps['tfidfvectorizer'].get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef,feature_names, n_top_features=40)
plt.title("tfidf-cofficient")



