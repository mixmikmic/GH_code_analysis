import matplotlib
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score as AUC

# make some data which has a clear covariate shift
n_samples_Z = 1000
n_samples_X = 100
x = 11*np.random.random(n_samples_Z)- 6.0
y = x**2 + 10*np.random.random(n_samples_Z) - 5
Z = np.c_[x, y]
    
x = 2*np.random.random(n_samples_X) - 6.0
y = x**2 + 10*np.random.random(n_samples_X) - 5
X = np.c_[x, y]

# plot the data
sns.plt.scatter(Z[:,0], Z[:,1], marker='o', s=4, c='b', label='Z')
sns.plt.scatter(X[:,0], X[:,1], marker='o', s=4, c='r', label='X')
sns.plt.legend()

X = pd.DataFrame(X)
Z = pd.DataFrame(Z)
X['is_z'] = 0 # 0 means test set
Z['is_z'] = 1 # 1 means training set
XZ = pd.concat( [X, Z], ignore_index=True, axis=0 )

labels = XZ['is_z'].values
XZ = XZ.drop('is_z', axis=1).values
X, Z = X.values, Z.values

# can use a non-linear learner, but make sure to restrict how 
# much it can learn or it will discriminate too well.
clf = RFC(max_depth=2)
# because we can see a learn divide in the above data we 
# could simply use logistic regression here.
# clf = LR()

predictions = np.zeros(labels.shape)
skf = SKF(n_splits=20, shuffle=True, random_state=1234)
for fold, (train_idx, test_idx) in enumerate(skf.split(XZ, labels)):
    print 'Training discriminator model for fold {}'.format(fold)
    X_train, X_test = XZ[train_idx], XZ[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
        
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    predictions[test_idx] = probs

print 'ROC-AUC for X and Z distributions:', AUC(labels, predictions)

# first, isolate the training part (recall we joined them above)
predictions_Z = predictions[len(X):]
weights = (1./predictions_Z) - 1. 
weights /= np.mean(weights) # we do this to re-normalize the computed log-loss
sns.plt.xlabel('Computed sample weight')
sns.plt.ylabel('# Samples')
sns.distplot(weights, kde=False)

Zsize = 0.1 + weights*15
sns.plt.scatter(Z[:,0], Z[:,1], marker='o', s=Zsize, c='b', label='Z')
sns.plt.scatter(X[:,0], X[:,1], marker='o', s=4, c='r', label='X')
sns.plt.legend()

