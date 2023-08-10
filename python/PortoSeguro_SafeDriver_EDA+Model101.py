from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np # linear algebra
import pandas as pd #data processing, CSV file I/O
import os
import seaborn as sns # visualization

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

#Tackel warnings while running
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

#Load back-up train set while keep the original train set untouched
train = pd.read_csv("./input/train-1.csv")
train.head()

#Shape of the train set
print("Train shape : ", train.shape)

#Load the test set
test = pd.read_csv("./input/test.csv")
test.head()

#Shape of the train set
print("Test shape : ", test.shape)

#Obseveration on Memory Usage
train.info(verbose=False),test.info(verbose=False)

# statistics of the numerial features
train.describe()

# look at the distribution of our interest: target
ratio_target = train['target'].value_counts()/len(train)
print("Ratio of Target: SafeDriver vs Not-A-SafeDriver")
print(ratio_target)

plt.figure(figsize=(8, 8))
values= [train[train.target==1].shape[0],train[train.target==0].shape[0]]
colors = ['cyan', 'grey']
labels = ['Not-A-SafeDriver', 'SafeDriver']

explode = (0.1, 0)  # explode 1st slice
plt.pie(values, labels=labels, colors=colors, shadow=True, autopct='%.2f%%')
plt.axis('equal')
plt.show()

#Use blank/white to visualize all the missing values accross the variables
train_copy = train
train_copy = train_copy.replace(-1, np.NaN)
import missingno as msno
# Nullity or missing values by columns
msno.matrix(df=train_copy.iloc[:,2:42], figsize=(20, 14), color=(0.42, 0.1, 0.05))

#count the numbers of int64, float64, bool or object/string features
int_features = train.select_dtypes(include = ['int64']).columns.values
float_features = train.select_dtypes(include = ['float64']).columns.values
bool_features= train.select_dtypes(include = ['bool']).columns.values
categorical_features = train.select_dtypes(include = ['object']).columns.values
print('int_features:', int_features)
print('float_features:', float_features)
print('bool_features:', bool_features)
print('categorical_features:', categorical_features)

train_copy = train.drop(["id", "target"], axis=1)
train_int_name = train_copy.select_dtypes(include=['int64','float64']).columns.values
corr = train[train_int_name].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)


plt.title('Pearson Correlation of All Features', y=1.05, size=20)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
save_fig("covariance matrix heatmap")

train_float = train.select_dtypes(include=['float64'])
colormap = plt.cm.inferno
plt.figure(figsize=(12,10))
plt.title('Pearson correlation of continuous features', y=1.05, size=20)
sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=cmap, linecolor='white', annot=True)

from sklearn.decomposition import PCA
covar_matrix = PCA(n_components = 57)
X_train=train.drop(["id"], axis=1)
covar_matrix.fit(X_train)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)

plt.figure(figsize=(8,8))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(90,100.5)
plt.style.context('seaborn-whitegrid')

plt.plot(var)

var

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

train = pd.read_csv("./input/train-1.csv")
X = train.drop(['id','target'], axis=1).values
y = train.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 24)

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# Create an XGBoost-compatible metric from Gini

def gini_coefficient(preds,dtrain):
    y = dtrain.get_label()
    return 'gini', -gini_normalized(y,preds)

# try simple models 101 without class_weight='balanced'
clfs = {'LogisticRegression':LogisticRegression(class_weight='balanced'),
        #'SVC': SVC(), Not good idea to run svc at this time since the SVC complexity = O(m*n^3)
              'KNeighborsClassifier': KNeighborsClassifier(n_neighbors = 2),
              'GaussianNB': GaussianNB(), 'Perceptron': Perceptron(), 
              'LinearSVC': LinearSVC(), 'SGDClassifier': SGDClassifier(), 
              'DecisionTreeClassifier': DecisionTreeClassifier(),
              'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
              'XGBoostClassifier': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
       }

for name, clf in clfs.items():
    clf.fit(X_train,y_train)
    ypred = clf.predict(X_test)
    gini_ = gini_normalized(y_test,ypred)
    precision_ = precision_score(ypred, y_test)
    accuracy_ = accuracy_score(ypred,y_test)
    f1_ = f1_score(ypred,y_test)
    print('%s classifier: gini = %.4f, precision = %.4f, accuracy = %.4f, f1 score = %.4f' 
          %(name, gini_, precision_, accuracy_, f1_))
print("-----------------------------------------------")

# try simple models 101 with class_weight='balanced'
import time
start = time.time()
clfs = {'LogisticRegression':LogisticRegression(class_weight='balanced'),
        #'SVC': SVC(), Not good idea to run svc at this time since the SVC complexity = O(m*n^3)
              'KNeighborsClassifier': KNeighborsClassifier(n_neighbors = 3),
              'GaussianNB': GaussianNB(), 'Perceptron': Perceptron(class_weight='balanced'), 
              'LinearSVC': LinearSVC(class_weight='balanced'), 'SGDClassifier': SGDClassifier(class_weight='balanced'), 
              'DecisionTreeClassifier': DecisionTreeClassifier(class_weight='balanced'),
              'RandomForestClassifier': RandomForestClassifier(n_estimators=100,class_weight='balanced'),
              'XGBoostClassifier': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)}

for name, clf in clfs.items():
    clf.fit(X_train,y_train)
    ypred = clf.predict(X_test)
    gini_ = gini_normalized(y_test,ypred)
    precision_ = precision_score(ypred, y_test)
    accuracy_ = accuracy_score(ypred,y_test)
    f1_ = f1_score(ypred,y_test)
    print('%s classifier: gini = %.4f, precision = %.4f, accuracy = %.4f, f1 score = %.4f' 
          %(name, gini_, precision_, accuracy_, f1_))

end = time.time()
runningtime = end - start
print('RunningTime=%.3fs'%(runningtime))
print("-----------------------------------------------")

#from guo li-Porto_Seguro/Analysis/03_xgb_test
import time
start = time.time()

train3 = pd.read_csv("./input/train-1.csv")
train3 = train3.reset_index(drop=True)

features = list(train3.columns)
target = 'target'
features.remove(target)

X = np.array(train3[features])
y = train3[target]

from sklearn.model_selection import KFold
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=7, 
                    learning_rate=0.05, 
                    n_estimators=1000, 
                    objective='binary:logistic', 
                    nthread=-1, 
                    gamma=0, 
                    colsample_bytree=0.8, 
                    colsample_bylevel=1, 
                    scale_pos_weight=30, 
                    missing=None)

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X):
    train_X, test_X = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    xgb.fit(train_X, train_y, 
            eval_set=[(train_X,train_y),(test_X,test_y)], 
            eval_metric=gini_coefficient,
            early_stopping_rounds=10)
    
end = time.time()
runningtime = end - start
print('RunningTime=%.3fs'%(runningtime))
print("-----------------------------------------------")

# #summarize model score
# models = pd.DataFrame({
#     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#               'Random Forest', 'Naive Bayes', 'Perceptron', 
#               'Stochastic Gradient Decent', 'Linear SVC', 
#               'Decision Tree'],
#     'Score': [acc_svc, acc_knn, acc_log, 
#               acc_random_forest, acc_gaussian, acc_perceptron, 
#               acc_sgd, acc_linear_svc, acc_decision_tree]})
# # models.sort_values(by='Score', ascending=False)



