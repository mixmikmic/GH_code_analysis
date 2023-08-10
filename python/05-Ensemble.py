from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import sklearn.metrics as sk
from sklearn.grid_search import GridSearchCV

import pandas as pd
from collections import Counter
import numpy as np
import nltk

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# read in and vectorize 

modern = pd.read_pickle('data/cards_modern_no_name.pkl')

modern['bincolor'] = pd.Categorical.from_array(modern.colors).codes

y = modern.bincolor

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(modern.text)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                             random_state=42)

print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

# Extreme Trees 

from sklearn.ensemble import ExtraTreesClassifier

rf = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1)
rf.fit(X_train, y_train)

mses = cross_val_score(rf, X_train, y_train,
                       cv=3, scoring='mean_squared_error') * -1

acc = cross_val_score(rf, X_train, y_train,
                       cv=3, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# Grid search extreme trees

param_grid = { 
    'n_estimators': [80, 100, 200, 500],
    'min_samples_leaf': [1, 2, 5],
    'max_depth': [None, 8, 10],
    'bootstrap': [True, False],
    'max_features': ['log2', None]}

CV_rf = GridSearchCV(estimator=rf, 
                     param_grid=param_grid, 
                     cv= 3,
                     verbose=True,
                     scoring='mean_squared_error')

print "Best parameters:", CV_rf.best_params_

CV_rf = ExtraTreesClassifier(n_estimators=500, 
                             max_features=None, 
                             min_samples_leaf=2, 
                             bootstrap=True, 
                             n_jobs=-1)

CV_rf.fit(X_train, y_train)

mses = cross_val_score(rf, X_train, y_train,
                       cv=3, scoring='mean_squared_error') * -1

acc = cross_val_score(rf, X_train, y_train,
                       cv=3, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# feature ranking 

importances = CV_rf.feature_importances_

indices = np.argsort(importances)[::-1]

indices[:9]

print("Feature ranking:")
print 

for i in xrange(20):
    for key, value in vectorizer.vocabulary_.items():
        if value == indices[i]:
            print("%d. %s (%f)" % (i + 1, key, importances[indices[i]]))

num_feat = 100 

plt.figure()
plt.title("Feature importances")
plt.bar(range(num_feat), importances[indices[:num_feat]],
       color="r", align="center")
plt.xlim([-1, num_feat])
plt.show()

n, acc = 1, [[],[]]
while n <= 1000:
    et = ExtraTreesClassifier(n_estimators=n, 
                              min_samples_leaf=2, 
                              bootstrap=True,
                              n_jobs=-1)
    
    xacc = cross_val_score(et, X_train, y_train,
                          cv=3, scoring='accuracy') 
    acc[0] += [n]
    acc[1] += [xacc.mean()]
    
    if n < 25:
        n += 2
    if n < 100 and n >= 25:
        n += 10
    if n >= 100:
        n += 20
    
plt.figure()
plt.title("Accuracy VS Num Features")
plt.scatter(acc[0], acc[1])
plt.show();

from sklearn.ensemble import GradientBoostingClassifier

gd = GradientBoostingClassifier(n_estimators = 40,
                                learning_rate = 0.5,
                                min_samples_leaf = 1,
                                random_state=42)

mses = cross_val_score(gd, X_train.toarray(), y_train,
                       cv=3, scoring='mean_squared_error') * -1

acc = cross_val_score(gd, X_train.toarray(), y_train,
                       cv=3, scoring='accuracy') 

print "MSE: %s, Accuracy: %s" % (mses.mean().round(3), acc.mean().round(3))

# Gradient Descent Boost grid search 

from sklearn.ensemble import GradientBoostingClassifier

gd_boost_grid = {'learning_rate': [.1, .5],
            'n_estimators': [100, 200],
              'min_samples_leaf': [1, 2]}

gd_gridsearch = GridSearchCV(GradientBoostingClassifier(random_state=42),
                             param_grid=gd_boost_grid,
                             verbose=True,
                             cv= 3,
                             scoring='mean_squared_error')

gd_gridsearch.fit(X_train.toarray(), y_train)

print "GD boost best parameters:", gd_gridsearch.best_params_

# Gradient Boost confusion matrix 

gd_gridsearch = GradientBoostingClassifier(n_estimators = 80,
                                            learning_rate = 0.5,
                                            min_samples_leaf = 1,
                                            random_state=42)

gd_gridsearch.fit(X_train.toarray(), y_train)


cm2 = sk.confusion_matrix(y_test, gd_gridsearch.predict(X_test.toarray()))

label = ["Black", "Blue", "Green", "Red", "White"]

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

plot_confusion_matrix(cm2, title="GD Gridsearch Confusion Matrix");

# Extra Trees confusion matrix 

cm = sk.confusion_matrix(y_test, CV_rf.predict(X_test))

plot_confusion_matrix(cm, title="RF Confusion Matrix");  

# logistic confusion matrix 

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=2)

clf.fit(X_train, y_train)

cm3 = sk.confusion_matrix(y_test, clf.predict(X_test), labels=None)

plot_confusion_matrix(cm3, title="Logistic Confusion Matrix");  

from sklearn.metrics import roc_curve

def get_scores(model, **kwargs):
    model.fit(X_train.toarray(), y_train)
    y_prods = model.predict_proba(X_test.toarray())
    y_prod = [b[i] for i, b in zip(y_test, y_prods)]
    y_pred = model.predict(X_test.toarray()) 
    y_true = [1 if a==b else 0 for a,b in zip(y_test, y_pred)]
    recall, precision, thresh = roc_curve(y_true, y_prod)
    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show();

classifiers = [rf, CV_rf, gd, gd_gridsearch, clf]

for c in classifiers:
    print c
    get_scores(c)
    print 

y_prods = clf.predict_proba(X_test.toarray())
y_prod = [b[i] for i, b in zip(y_test, y_prods)]
y_pred = clf.predict(X_test.toarray()) 
y_true = [1 if a==b else 0 for a,b in zip(y_test, y_pred)]
recall, precision, thresh = roc_curve(y_true, y_prod)
plt.clf()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show();

y_prods[0]

