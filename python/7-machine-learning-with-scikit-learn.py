# Plotting imports
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

import numpy as np
import pandas as pd
from sklearn import datasets

d = datasets.load_breast_cancer()

print(d['DESCR'])

d.keys()

d['data']

d['target']

d['target_names']

df = pd.DataFrame(d['data'],
                  columns=d['feature_names'])

target = pd.Series(d['target'],
                   name='target')

df.info()

df.hist(bins=50, figsize=(20,15));

sns.pairplot(data=df[df.columns[:5]], size=2);

from pandas.plotting import scatter_matrix

scatter_matrix(df[df.columns[:5]], figsize=(10, 10));

mclust = sns.clustermap(df.corr(),
                        figsize=(10, 10),
                        cmap='RdBu')
mclust.ax_heatmap.set_yticklabels(mclust.ax_heatmap.get_yticklabels(),
                                  rotation=0);

feat_corr = df.corr()
feat_corr.stack().head(5)

feat_corr = df.corr()
# ignore the diagonal, obviously
np.fill_diagonal(feat_corr.values, np.nan)
feat_corr = feat_corr.stack()

feat_corr[feat_corr > 0.7].head(5)

# can you think a smarter way to perform this operation?
high_corr = feat_corr[feat_corr > 0.7]
discarded = set()
saved = set()
for feat1 in {x[0] for x in high_corr.index}:
    if feat1 in discarded:
        continue
    saved.add(feat1)
    for feat2 in high_corr.loc[feat1].index:
        discarded.add(feat2)

saved

discarded

sns.pairplot(data=df.loc[:, sorted(set(df.columns) - discarded)], size=2);

df = df.loc[:, sorted(set(df.columns) - discarded)]

from sklearn import model_selection

df_train, df_test = model_selection.train_test_split(df,
                                                     test_size=0.2)
df_train.head(5)

target_train, target_test = target.iloc[df_train.index], target.iloc[df_test.index]
target_train.head(5)

target_train[target_train == 0].shape[0] / target_train[target_train == 1].shape[0]

target_test[target_test == 0].shape[0] / target_test[target_test == 1].shape[0]

cv = model_selection.StratifiedShuffleSplit(n_splits=1,
                                            test_size=0.2,
                                            random_state=42)

cv.split(df, target)

train_idx, test_idx = next(cv.split(df, target))
train_idx

df_train, target_train = df.iloc[train_idx], target[train_idx]
df_test, target_test = df.iloc[test_idx], target[test_idx]
df_train.head(5)

target_train[target_train == 0].shape[0] / target_train[target_train == 1].shape[0]

target_test[target_test == 0].shape[0] / target_test[target_test == 1].shape[0]

from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
std_scaler.fit(df_train)

std_scaler.fit_transform(df_train)

# let's bring it back to a dataframe
df_scaled = pd.DataFrame(std_scaler.fit_transform(df_train),
                         columns=df_train.columns)

df_scaled.hist(bins=50, figsize=(20,15));

from sklearn import svm

svm.LinearSVC()

clf = svm.LinearSVC()
clf = clf.fit(df_scaled, target_train)

clf.predict(df_scaled)

predict_train = pd.Series(clf.predict(df_scaled),
                          index=target_train.index,
                          name='prediction')

combined = target_train.to_frame().join(predict_train.to_frame())
combined.head(10)

combined[combined['target'] == combined['prediction']].shape[0] / combined.shape[0]

from sklearn import metrics

plt.figure(figsize=(6, 3))

plt.subplot(121)

fpr, tpr, thresholds = metrics.roc_curve(combined['target'],
                                         combined['prediction'])
plt.plot(fpr, tpr,
         'r-')
plt.plot([0, 1],
         [0, 1],
         '--',
         color='grey')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.subplot(122)

prec, rec, thresholds = metrics.precision_recall_curve(combined['target'],
                                                        combined['prediction'])
plt.plot(rec, prec,
         'r-')

plt.xlabel('precision')
plt.ylabel('recall')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.tight_layout()

metrics.roc_auc_score(combined['target'],
                      combined['prediction'])

model_selection.cross_val_score(clf, df_scaled, target_train,
                                cv=10,
                                scoring=metrics.make_scorer(metrics.roc_auc_score))

param_grid = {'C': np.linspace(0.01, 10),
              'loss': ['hinge', 'squared_hinge']}

clf = svm.LinearSVC()
grid_search = model_selection.GridSearchCV(clf,
                                           param_grid=param_grid,
                                           cv=10,
                                           scoring=metrics.make_scorer(metrics.roc_auc_score))
grid_search.fit(df_scaled, target_train)

grid_search.cv_results_.keys()

plt.figure(figsize=(7, 3))

for mean_score, params in zip(grid_search.cv_results_["mean_test_score"],
                              grid_search.cv_results_["params"]):
    if params['loss'] == 'hinge':
        plt.plot(params['C'],
                 mean_score,
                 'bo')
    else:
        plt.plot(params['C'],
                 mean_score,
                 'ro')
plt.xlabel('C')
plt.ylabel('ROC AUC');

from sklearn import pipeline

breast_pipeline = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                                     ('classifier', svm.LinearSVC(C=2., loss='hinge'))])

breast_pipeline = breast_pipeline.fit(df_train, target_train)
breast_pipeline.predict(df_test)

predict_test = pd.Series(breast_pipeline.predict(df_test),
                         index=target_test.index,
                         name='prediction')
combined = target_test.to_frame().join(predict_test.to_frame())
combined.head(10)

plt.figure(figsize=(6, 3))

plt.subplot(121)

fpr, tpr, thresholds = metrics.roc_curve(combined['target'],
                                         combined['prediction'])
plt.plot(fpr, tpr,
         'r-')
plt.plot([0, 1],
         [0, 1],
         '--',
         color='grey')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.subplot(122)

prec, rec, thresholds = metrics.precision_recall_curve(combined['target'],
                                                        combined['prediction'])
plt.plot(rec, prec,
         'r-')

plt.xlabel('precision')
plt.ylabel('recall')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.tight_layout()

metrics.roc_auc_score(combined['target'],
                      combined['prediction'])

iris = datasets.load_iris()

# rows are observations, columns are features
iris.data

# true label for each observation
iris.target

# label names
iris.target_names

# the simplest preprocessing is to standardize the data
std_scaler = preprocessing.StandardScaler()
iris.data = std_scaler.fit_transform(iris.data)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(iris.data)

kmeans.labels_

# once the model has been fitted, we can add a new observation and can try to predict to which cluster they belong to
kmeans.predict([[5.8,  2.7,  4.0,  1.25],])

plt.figure(figsize=(15, 7))

plt.subplot(121)
for label, glyph in zip(set(iris.target), ('o', 'D', '^')):
    for cluster, color in zip(set(kmeans.labels_), ('b', 'r', 'g')):
        plt.plot(iris.data[(iris.target == label) & (kmeans.labels_ == cluster)][:, 0],
                 iris.data[(iris.target == label) & (kmeans.labels_ == cluster)][:, 1],
                 marker=glyph,
                 linestyle='',
                 color=color,
                 label='{0} - {1}'.format(iris.target_names[label],
                                          cluster))
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.subplot(122)
for label, glyph in zip(set(iris.target), ('o', 'D', '^')):
    for cluster, color in zip(set(kmeans.labels_), ('b', 'r', 'g')):
        plt.plot(iris.data[(iris.target == label) & (kmeans.labels_ == cluster)][:, 2],
                 iris.data[(iris.target == label) & (kmeans.labels_ == cluster)][:, 3],
                 marker=glyph,
                 linestyle='',
                 color=color,
                 label='{0} - {1}'.format(iris.target_names[label],
                                          cluster))
plt.xlabel('feature 2')
plt.ylabel('feature 3')
plt.legend(loc=(1, 0));

metrics.homogeneity_score(iris.target, kmeans.labels_)

from sklearn.linear_model import RidgeClassifier

ridge = RidgeClassifier(alpha=1.0)
ridge.fit(iris.data, iris.target)

predictions = ridge.predict(iris.data)

metrics.f1_score(iris.target, predictions, average=None)

plt.figure(figsize=(15, 7))

plt.subplot(121)
plt.plot(iris.data[iris.target == predictions][:, 0],
         iris.data[iris.target == predictions][:, 1],
         'k.',
         label='correct predictions')
plt.plot(iris.data[iris.target != predictions][:, 0],
         iris.data[iris.target != predictions][:, 1],
         'ro',
         label='incorrect predictions')
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.subplot(122)
plt.plot(iris.data[iris.target == predictions][:, 2],
         iris.data[iris.target == predictions][:, 3],
         'k.',
         label='correct predictions')
plt.plot(iris.data[iris.target != predictions][:, 2],
         iris.data[iris.target != predictions][:, 3],
         'ro',
         label='incorrect predictions')
plt.xlabel('feature 2')
plt.ylabel('feature 3')
plt.legend(loc=(1, 0));

