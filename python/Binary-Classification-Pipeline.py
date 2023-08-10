from __future__ import division
import numpy as np
RNG = 10
np.random.seed(RNG)

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 ## Output Type 3 (Type3) or Type 42 (TrueType)
rcParams['font.sans-serif'] = 'Arial'
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().magic('matplotlib inline')

# Create synthetic data
X, y = make_classification(n_classes=2, class_sep=2, 
                           weights=[0.1, 0.9], # the ratios of the two classes
                           n_informative=3, n_redundant=4, flip_y=0.02,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=RNG)

# Force them to be on different scales
X = np.multiply(X, np.arange(20)) + np.arange(20)

print 'Shape of X: ', X.shape, 'shape of y:', y.shape

# Examine the distribution of the 10 features over all the samples
fig, ax = plt.subplots()
for j in range(20):
    ax = sns.kdeplot(X[:,j], ax=ax, label='feature%d' % j)

ax.set_ylabel('Density')
ax.set_xlabel('Values')
plt.show()

from sklearn.preprocessing import StandardScaler
# Scale the features using z-score
scl = StandardScaler()
X = scl.fit_transform(X)
# Examine the distribution of the 10 features over all the samples again
fig, ax = plt.subplots()
for j in range(20):
    ax = sns.kdeplot(X[:,j], ax=ax, label='feature%d' % j)

ax.set_ylabel('Density')
ax.set_xlabel('Values')
plt.show()

from sklearn.decomposition import PCA

## Use PCA to reduce the dimensionality 
pca = PCA(n_components=10)
X = pca.fit_transform(X)
print 'Shape of X after PCA to the first 10 dimensions: ', X.shape

## PCA can also be used to visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
plt.show()

from sklearn.cross_validation import StratifiedKFold, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                 test_size=0.3, 
                 stratify=None, ## Change this
                 random_state=RNG)

print 'Before spliting:'
print X.shape, y.shape, y.sum()/y.shape[0]

print 'After spliting:'
print X_train.shape, y_train.shape, y_train.sum()/y_train.shape[0]
print X_test.shape, y_test.shape, y_test.sum()/y_test.shape[0]

from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

## Initiate a FS-Logit pipeline
fs = SelectKBest(f_classif, k=2)
logit = LogisticRegression()

pipeline = Pipeline([
        ('fs', fs),
        ('logit', logit)
    ])

# Train classifier
pipeline.fit(X_train, y_train)
# Get prediction on test set
y_test_preds = pipeline.predict(X_test)
y_test_probas = pipeline.predict_proba(X_test)
print y_test_preds.shape, y_test_probas.shape

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Evaluate predictions
print 'Accuracy: %.5f' % accuracy_score(y_test, y_test_preds)
print 'F1 score: %.5f' % f1_score(y_test, y_test_preds)
print 'AUROC: %.5f' % roc_auc_score(y_test, y_test_probas[:, 1])
print 'AUPRC: %.5f' % average_precision_score(y_test, y_test_probas[:, 1])

## Results from not straitified split
# Accuracy: 0.98000
# F1 score: 0.98909
# AUROC: 0.93825
# AUPRC: 0.98502

## Results from straitified split
# Accuracy: 0.99000
# F1 score: 0.99443
# AUROC: 0.94018
# AUPRC: 0.98791

# To plot ROC and PRC
from sklearn.metrics import roc_curve, precision_recall_curve

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_probas[:, 1])
precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas[:, 1])

# Plot
fig, axes = plt.subplots(1, 2)
axes[0].plot(fpr, tpr)
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC')

axes[1].plot(recall, precision)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PRC')

axes[0].set_xlim([-.05, 1.05])
axes[0].set_ylim([-.05, 1.05])
axes[1].set_xlim([-.05, 1.05])
axes[1].set_ylim([-.05, 1.05])

fig.tight_layout()
plt.show()

