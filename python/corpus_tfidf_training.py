import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from nltk.corpus import stopwords

corpus = pd.read_csv('corpus.csv.gz', compression='gzip')
corpus = corpus[corpus['qual_a_melhor_classificao_para_esse_texto:confidence'] == 1]
stopwords = stopwords.words("portuguese")

# fix labels to binary
def classFit(x):
    if x['qual_a_melhor_classificao_para_esse_texto'] == "diario":
        return 1
    else:
        return -1
    
corpus['class'] = corpus.apply(classFit,axis=1)
target = corpus['class'].values

print(corpus['qual_a_melhor_classificao_para_esse_texto'].values[:2])
print(corpus['class'][:2])

data = TfidfVectorizer(max_features=1000, strip_accents='unicode', stop_words=stopwords).fit_transform(corpus.content)
data.shape

from sklearn.naive_bayes import MultinomialNB, GaussianNB

model = MultinomialNB(alpha=0.001)

precision = cross_val_score(model, data.toarray(), target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data.toarray(), target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data.toarray(), target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

model = svm.LinearSVC(C=2.15)

precision = cross_val_score(model, data.toarray(), target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data.toarray(), target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data.toarray(), target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

c_range = np.logspace(-3,7,7)
param_grid = [
    {'kernel': ['rbf', 'linear'], 'C': c_range},
]
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=10, verbose=3, n_jobs=10)
grid_search.fit(data, target)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)

model = svm.SVC(kernel='linear',C=2.15,gamma=0.1)

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

nb_params = { 'alpha': np.logspace(-3, 3, 7)}

grid_search = GridSearchCV(MultinomialNB(), nb_params, cv=10, verbose=3, n_jobs=10)
grid_search.fit(data, target)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)

from sklearn.naive_bayes import BernoulliNB

nb_params = { 'alpha': np.logspace(-3, 3, 7) }

grid_search = GridSearchCV(BernoulliNB(), nb_params, cv=10, verbose=3, n_jobs=10)
grid_search.fit(data, target)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)

model = BernoulliNB(alpha=0.001)

precision = cross_val_score(model, data.toarray(), target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data.toarray(), target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data.toarray(), target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.tree import DecisionTreeClassifier 

model = DecisionTreeClassifier()

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()

print(precision)
print(acc)
print(recall)

from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.tree import ExtraTreeClassifier

model = ExtraTreeClassifier()

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import oll

## manual 10-fold cross-validation
kf = KFold(n_splits=2, random_state=None, shuffle=False)

methods = ["P" ,"AP" ,"PA" ,"PA1","PA2" ,"PAK","CW" ,"AL"]

for m in methods:
    model = oll.oll(m, C=1)

    accuracy = []
    precision = []
    recall = []
    
    for train_index, test_index in kf.split(data):

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, predicted))
        precision.append(precision_score(y_test, predicted))
        recall.append(recall_score(y_test, predicted))

    print(m + ': acc(' + str(np.mean(accuracy)) 
          + '), prec(' + str(np.mean(precision))
          + '), rec(' + str(np.mean(recall)) + ')'
         )

import oll

model = oll.oll("CW", C=1)
model.fit(data, target)
predicted = model.predict(data)
scores = model.scores(data)

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    loss='exponential',
    n_estimators=500,
    subsample=0.5,
    max_depth=7,
    max_features=0.5,
    learning_rate=0.1
)

acc = cross_val_score(model, data.toarray(), target, cv=10, scoring='accuracy').mean()
fscore = cross_val_score(model, data.toarray(), target, cv=10, scoring='f1').mean()
print(acc)
print(fscore)

param_grid = {'learning_rate': [0.01, 0.1, 0.1, 0.5, 1.0],
              'max_depth':[1, 3, 5, 7, 9],
              'n_estimators': [100, 300, 500],
              'loss' : ['deviance', 'exponential'],
              'subsample':[0.2, 0.5, 0.8, 1],
              'max_features': [0.5, 1]}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=5, verbose=3, n_jobs=3, scoring='accuracy')
grid_search.fit(data.toarray(), target)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)



