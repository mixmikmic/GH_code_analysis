from sklearn.decomposition import PCA as sPCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc



from matplotlib.mlab import PCA as mPCA


import seaborn as sns
import pandas as pd
import numpy as np



import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

from __future__ import division
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
get_ipython().magic('autoreload 2')

from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, decomposition, ensemble
from sklearn import lda, manifold, random_projection, preprocessing


from statsmodels.regression import linear_model

import statsmodels.api as sm
import seaborn as sea
sea.set()

import pandas as pd
import numpy as np

digits = load_digits()

get_ipython().magic('pylab inline')
get_ipython().magic('autoreload 2')

df = pd.read_csv('data.csv', index_col=0)

df = pd.read_csv('data.csv')

df[:2]

sam = np.random.choice(df.index.values, df.shape[0]*0.1 )
df_sam = df.ix[sam]

# df_sam.describe()

y = df_sam.pop('target').values

X_inte = df_sam[df_sam.columns[df_sam.dtypes == 'int64']]

X_inte[:3]

feature_names = X_inte.columns

feature_names

y_ = df.pop('target').values

X_int = df[df.columns[df.dtypes == 'int64']]

feature_names = X_inte.columns

x_train, x_test, y_train, y_test = train_test_split(X_int, y_)

model = RandomForestClassifier(n_jobs=-1)
model.fit(x_train, y_train)

model.score(x_test, y_test)

important_features = X_int[X_int.columns[model.feature_importances_.argsort()[::-1][:15]]]

important_features.head()

categorical = []

for col in important_features:
    if len(important_features[col].unique()) <= 100:
        categorical.append(col)

categorical

len(categorical)

def bin_categorical(s, n=5, na_value=-99999):
    return pd.cut(s.replace(na_value, float('nan')), 5)

important_dummies = important_features.copy()

for col in categorical:
    important_dummies[col] = bin_categorical(important_features[col])

important_dummies = pd.get_dummies(important_features, dummy_na=True)

x_train, x_test, y_train, y_test = train_test_split(important_features, y_)

imodel = RandomForestClassifier(n_jobs=-1)
imodel.fit(x_train, y_train)

imodel.score(x_test, y_test)

scaler = StandardScaler()

feature_scaling = scaler.fit_transform(X_inte)

pca = PCA(70)

principal_features = pca.fit_transform(feature_scaling, y)

def evariance_pca(pca):
    components = pca.n_components_
    xaxis = np.arange(components)
    evariance = pca.explained_variance_ratio_
    
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.bar(xaxis, evariance, 0.4, 
           color=['red', 'green', 'blue', 'cyan', 'magenta'], alpha=0.5)
    
    for i in xrange(components):
        ax.annotate('{:1.4}%'.format((str(evariance[i]*100))),
                   (xaxis[i], evariance[i]), 
                    fontsize=12)
    
#     ax.set_xticklabels(xaxis, fontsize=12)
    
    ax.set_ylim(0, max(evariance)+0.02)
    ax.set_xlim(0-0.45, 8+0.45)
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Pareto')

evariance_pca(pca)

model = sm.OLS(y, sm.add_constant(principal_features))
results = model.fit()

print results.rsquared
print results.rsquared_adj
results.summary()





















































































