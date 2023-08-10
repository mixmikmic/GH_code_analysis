get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.reset_defaults()
from scipy import special

# plt.plot(range(2,11),[sum(special.binom(k,1+j) for j in range(k)) for k in range(2,11)])

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from secoc import estimator; reload(estimator)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.datasets import make_classification, make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(centers=4, n_samples=1000, n_features=100, cluster_std=30)

train_index, test_index = next(StratifiedShuffleSplit(n_splits=1, test_size=.2).split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
# X_train_small = X_train
# y_train_small = y_tr

reload(estimator)
from sklearn.tree import DecisionTreeClassifier
est = estimator.SlidingECOC(
    LogisticRegressionCV(), n_estimators_window=50, window_size=25,
    oob_score=True, code_size=2000, single_seed_features=True, single_seed_samples=True,
    circular_features=True, n_estimators=None, stride=1, n_jobs=-1)

est_proba = estimator.SlidingECOCProba(
    LogisticRegressionCV(), n_estimators_window=50, window_size=25,
    oob_score=True, code_size=2000, single_seed_features=True, single_seed_samples=True,
    circular_features=True, n_estimators=None, stride=1, n_jobs=-1)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# pipe = make_pipeline(est, LogisticRegression())
pipe = make_pipeline(est, LogisticRegressionCV())
pipe2 = make_pipeline(est_proba, LogisticRegressionCV())

pipe.fit(X_train, y_train)

pipe2.fit(X_train, y_train)

print pipe.score(X_test, y_test)
print pipe2.score(X_test, y_test)

LogisticRegressionCV().fit(X_train, y_train).score(X_test, y_test)

est.fit(X_train, y_train)

encoding = est_proba.transform(X_train)

encoding

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(metric='hamming', algorithm='brute', n_jobs=-1, n_neighbors=5)
code_range = map(int, np.linspace(10,1000, 100))
tr_score, ts_score = [], []
for code_size in code_range:
    y_pred = est.predict(X_test, classifier=clf, code_size=code_size)
    ts_score.append(accuracy_score(y_test, y_pred))
    y_pred = est.predict(X_train, classifier=clf, code_size=code_size)
    tr_score.append(accuracy_score(y_train, y_pred))

X_small_train = X_train[est.estimators_samples_[0]]
y_small_train = y_train[est.estimators_samples_[0]]
lr = LogisticRegression().fit(X_small_train, y_small_train)
base_train_score_logistic = lr.score(X_train, y_train)
base_test_score_logistic = lr.score(X_test, y_test)

from sklearn.ensemble import BaggingClassifier
X_small_train = X_train[est.estimators_samples_[0]]
y_small_train = y_train[est.estimators_samples_[0]]
bc = BaggingClassifier(LogisticRegression(), n_estimators=2000, max_samples=.5, max_features=1.0,
                  bootstrap=False, bootstrap_features=False, oob_score=0, warm_start=False, n_jobs=-1)
bc.fit(X_small_train, y_small_train)
base_train_score = bc.score(X_train, y_train)
base_test_score = bc.score(X_test, y_test)

plt.figure(figsize=(15,20));
f, ax = plt.subplots(1,2, figsize=(15,5));

ax[0].plot(code_range, ts_score, label="ts_score", c='navy');
ax[1].plot(code_range, tr_score, label="tr_score", c='orange');

ax[1].axhline(base_train_score, label='bag train', c='lightblue', ls='--');
ax[0].axhline(base_test_score, label='bag test', c='navy', ls='--');
ax[1].axhline(base_train_score_logistic, label='base train', c='red', ls='--');
ax[0].axhline(base_test_score_logistic, label='base test', c='darkred', ls='--');

ax[0].set_title("Test scores");
ax[1].set_title("Train scores");
ax[0].set_xlabel("code_size");
ax[1].set_xlabel("code_size");
ax[0].set_ylabel("accuracy");
ax[0].legend();
ax[1].legend();
ax[0].set_ylim([0.25,1])
ax[1].set_ylim([0.25,1])

plt.show()

accuracy_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=850).fit(X_train_small, y_train_small)

rfc.score(X_test, np.where(y_test)[1])

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def visualise(X, y, title=''):
#     tr = TSNE(n_components=2).fit_transform(X)
    tr = PCA(n_components=2).fit_transform(X)
    for i in np.unique(y):
        plt.plot(tr[y == i,0],tr[y == i,1], 'o', label=i)
        
    plt.legend(bbox_to_anchor=(0., -.252, 1., .102), loc='lower center',
           ncol=5, mode="expand", borderaxespad=0.)
    plt.title(title)
    plt.show()

encoding.shape

visualise(X_train, y_train)

visualise(pipe.steps[0][1].transform(X_train), y_train)

visualise(pipe2.steps[0][1].transform(X_train), y_train)

visualise(encoding, y_train_small, "encoding")

visualise(X_train_small, y_train_small, "CNN features")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def visualise_lda(X, y):
    tr = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    for i in np.unique(y):
        plt.plot(tr[y == i,0],tr[y == i,1], 'o', label=i)
        
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
    plt.show()

visualise_lda(X_train_small, y_train_small)

visualise_lda(encoding, y_train_small)



