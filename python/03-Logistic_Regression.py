from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import sklearn.metrics as sk

import pandas as pd
from collections import Counter
import numpy as np
import nltk

import matplotlib.pyplot as plt
import seaborn
get_ipython().magic('matplotlib inline')

modern = pd.read_pickle('data/5color_modern_no_name_hardmode.pkl')

Counter(modern.colors)

UG = modern.loc[modern['colors'].isin(['Blue', 'Red'])]

UG.reset_index(inplace=True)
UG.pop('index')

UG[['name', 'colors', 'cmc', 'text']].sample(6)

dummies = pd.get_dummies(UG.colors)
# dummies['Green']

vectorizer = CountVectorizer()

vec_X = vectorizer.fit_transform(UG['text'])

X_train, X_test, y_train, y_test = train_test_split(vec_X, dummies['Red'],
                                             random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

# Linear regression 
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
mses = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# Multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

mses = cross_val_score(nb, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(nb, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# Bernoulli naive bayes
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()

mses = cross_val_score(bnb, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(bnb, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

y = pd.get_dummies(modern.colors)['Green']

X = vectorizer.fit_transform(modern.text)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

# Logistic regression 

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver = 'lbfgs', multi_class='multinomial')

mses = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# Multinomial naive bayes

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
mses = cross_val_score(nb, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(nb, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

modern['bincolor'] = pd.Categorical.from_array(modern.colors).codes

vectorizer = CountVectorizer()

y = modern.bincolor

X = vectorizer.fit_transform(modern.text)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

clf = LogisticRegression(C=2, multi_class='ovr', solver='liblinear')

mses = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

clf.fit(X_train, y_train)

label = ["Black", "Blue", "Green", "Red", "White"]

cm = sk.confusion_matrix(y_test, clf.predict(X_test), labels=None)

# plot code adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, title='Logistic Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(label))
    plt.xticks(tick_marks, label, rotation=45)
    plt.yticks(tick_marks, label)    
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm)   

def show_most_informative_features(vectorizor, clf, color=0, n=20):
    feature_names = vectorizor.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[color], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

show_most_informative_features(vectorizer, clf, color=0, n=20)

for l in xrange(5):
    print label[l], 'most negative featues         top features '
    show_most_informative_features(vectorizer, clf, color=l, n=10)
    print '\n'

modern['bincolor'] = pd.Categorical.from_array(modern.colors).codes

a = modern.groupby('rarity').get_group('Uncommon')
b = modern.groupby('rarity').get_group('Rare')
c = modern.groupby('rarity').get_group('Mythic Rare')
no_commons = pd.concat([a,b,c])


no_commons['bincolor'] = pd.Categorical.from_array(no_commons.colors).codes

vectorizer = CountVectorizer()

y = no_commons.bincolor

X = vectorizer.fit_transform(no_commons.text)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

clf = LogisticRegression(C=2, multi_class='ovr', solver='liblinear')

mses = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

modern['bincolor'] = pd.Categorical.from_array(modern.colors).codes

vectorizer = CountVectorizer()

y = modern.bincolor

X = vectorizer.fit_transform(modern.text)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

from sklearn.linear_model import BayesianRidge, LinearRegression
import sklearn.metrics as sk

clf = BayesianRidge(compute_score=True)
clf.fit(X_train.toarray(), y_train)


print (sum(y_test == clf.predict(X_test).round()) * 1.0) /  len(y_test)

mse = cross_val_score(clf, X_train.toarray(), y_train,
                       cv=10, scoring='mean_squared_error') * -1
print np.mean(mse)

# Plot true weights, estimated weights and histogram of the weights

ols = LinearRegression()
ols.fit(X, y)

n_features = len(vectorizer.vocabulary_)
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)

plt.figure(figsize=(16, 8))
plt.title("Weights of the model")
plt.plot(clf.coef_, 'b-', label="Bayesian Ridge estimate")
plt.plot(w, 'g-', label="Ground truth")
plt.plot(ols.coef_, 'r--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.figure(figsize=(16, 8))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, log=True)
plt.plot(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)),
         'ro', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="lower left")

plt.figure(figsize=(16, 8))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_)
plt.ylabel("Score")
plt.xlabel("Iterations")
plt.show()



vectorizer = CountVectorizer(stop_words="english")

flavor_df = modern[pd.notnull(modern.flavor)]

y = flavor_df.bincolor

X = vectorizer.fit_transform(flavor_df.flavor)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

clf = LogisticRegression(C=1)

mses = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='mean_squared_error') * -1

acc = cross_val_score(clf, X_train, y_train,
                       cv=10, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

clf.fit(X_train, y_train)

cm = sk.confusion_matrix(y_test, clf.predict(X_test), labels=None)

plot_confusion_matrix(cm)  

show_most_informative_features(vectorizer, clf)

# Todo: add softmax

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/(sum(np.exp(x)))



