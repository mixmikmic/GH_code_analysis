import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

get_ipython().magic('matplotlib inline')

train = pd.read_csv('./data/synthesized/train_mean.csv')
test = pd.read_csv('./data/synthesized/test_mean.csv')

features = train.columns[1:-1]

X = train[features]
y = train.Loan_Status

test = test[features]

def do_pca(X, y):
    pca = PCA(n_components=2, whiten=True)
    X = pca.fit_transform(X, y)
    
    colors = ['r', 'g']
    markers = ['s', 'x']
    
    for l, c, m in zip(np.unique(y), colors, markers):
        class_label = ( y == l ).values
        plt.scatter(X[class_label, 0], X[class_label, 1], c=c, label=l, marker=m)
    
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='best')
    plt.show();

X = scale(X)

do_pca(X, y)

train = pd.read_csv('./data/synthesized/train_median.csv')
test = pd.read_csv('./data/synthesized/test_median.csv')

features = train.columns[1:-1]

X = train[features]
y = train.Loan_Status

test = test[features]

X = scale(X)

do_pca(X, y)

train = pd.read_csv('./data/synthesized/train_mode.csv')
test = pd.read_csv('./data/synthesized/test_mode.csv')

features = train.columns[1:-1]

X = train[features]
y = train.Loan_Status

test = test[features]

X = X.fillna(-1)
test = test.fillna(-1)

X = scale(X)

do_pca(X, y)

train = pd.read_csv('./data/synthesized/train_mean.csv')
test = pd.read_csv('./data/synthesized/test_mean.csv')

features = train.columns[1:-1]

X = train[features]
y = train.Loan_Status

test = test[features]

np.random.seed(251)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

def get_cross_val_score(X, y, cv):
    k_vals = np.arange(1, 100, 1)
    quality_by_k = [
        cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='uniform'), X, y, cv=cv).mean()
        for k in k_vals
    ]
    return k_vals, quality_by_k

skf = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=1279)

k_vals, quality_by_k = get_cross_val_score(X_train, y_train, skf)

plt.scatter(k_vals, quality_by_k)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy');

print 'Best value for k %d and highest accuracy score %f' %(k_vals[np.argmax(quality_by_k)], max(quality_by_k))

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

k_vals, quality_by_k = get_cross_val_score(X_train, y_train, skf)

plt.scatter(k_vals, quality_by_k)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy');

print 'Best value for k %d and highest accuracy score %f' %(k_vals[np.argmax(quality_by_k)], max(quality_by_k))

def get_cross_val_score_by_p(X, y, cv):
    p_vals = np.linspace(1, 10, 100)
    quality_by_p = [
        cross_val_score(KNeighborsClassifier(n_neighbors=57, weights='uniform', metric='minkowski', p=p), X, y, cv=cv).mean()
        for p in p_vals
    ]
    return p_vals, quality_by_p

p_vals, quality_by_p = get_cross_val_score_by_p(X_train, y_train, skf)

plt.scatter(p_vals, quality_by_p)
plt.xlabel('P value')
plt.ylabel('Accuarcy');

p_vals[np.argmax(quality_by_p)]

def score_by_thresholds(X, y, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=57, n_jobs=-1, weights='uniform', p=1)
    thresholds = np.linspace(0.1, 0.9, 40)
    quality_by_threshold = []
    knn.fit(X, y)
    
    for threshold in thresholds:
        preds = knn.predict_proba(X_test)[:, 1]
        mapped_preds = map(lambda x: 1 if x > threshold else 0, preds)
        score = accuracy_score(y_test, mapped_preds)
        
        quality_by_threshold.append(score)
    
    return thresholds, quality_by_threshold

thresholds, quality_by_thresholds = score_by_thresholds(X_train, y_train, X_test, y_test)

plt.scatter(thresholds, quality_by_thresholds)
plt.xlabel('Thresholds')
plt.ylabel('Accuaracy Score')
plt.title('Relationship between Threshold and accuracy score');

print thresholds[np.argmax(quality_by_thresholds)]

knn = KNeighborsClassifier(n_neighbors=57, n_jobs=-1, weights='uniform', p=1.0)
knn.fit(X_train, y_train)

preds = knn.predict_proba(X_test)[:, 1]
mapped_preds = map(lambda x: 1 if x > 0.1 else 0, preds)
print 'Accuracy ', accuracy_score(y_test, mapped_preds)

knn.fit(X, y)
preds = knn.predict_proba(test)[:, 1]
mapped_preds = map(lambda x: 1 if x > 0.1 else 0, preds)

sub = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')
test = pd.read_csv('./data/test_Y3wMUE5.csv')

sub['Loan_ID'] = test.Loan_ID
sub['Loan_Status'] = mapped_preds

sub.Loan_Status.value_counts()



