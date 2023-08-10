# Import useful libraries used in the notebook
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

# Pull in the models we'll be building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Functions are defined in this module
from acm_imbalanced_library import *

# Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Show plots inline 
get_ipython().magic('matplotlib inline')

# Auto-reload external modules
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Read in the dataset, look at the column names and NAs
credit_df = pd.read_csv('cs-training.csv')
credit_df.head()

# Clean up the imported raw CSV file. Rename columns, fill in NA values,
# and change dependents to integer.
credit_df = cleanCreditDf(credit_df)

# Check how the data looks now
credit_df.head()

# Get some summary statistics from the dataframe
credit_df.describe()

# Plot out the distributions to see what's going on, colour code by target
import seaborn as sns; sns.set(style="ticks", color_codes=True)
g = sns.pairplot(credit_df[['age', 'debt_ratio', 'lines', 'target']], 
                 hue='target', palette=sns.diverging_palette(220,20,n=2))

# Separate out the ID and target values from the dataset
X, y, id_val = removeIDTargetFromCreditDf(credit_df)

# Train a logistic regression model using the balanced class weights 

default_log_reg = LogisticRegression(class_weight=None)
balanced_log_reg = LogisticRegression(class_weight='balanced')

# Important to use a stratified K folds to make sure representative
# proportion of minority examples are used in each fold
cv = StratifiedKFold(y, n_folds=5, shuffle=True)

scores = cross_val_score(estimator=default_log_reg, X=X, y=y, cv=cv, scoring='accuracy')
def_log_reg_acc = np.mean(scores)

scores = cross_val_score(estimator=balanced_log_reg, X=X, y=y, cv=cv, scoring='accuracy')
bal_log_reg_acc = np.mean(scores)

scores = cross_val_score(estimator=default_log_reg, X=X, y=y, cv=cv, scoring='roc_auc')
def_log_reg_roc = np.mean(scores)

scores = cross_val_score(estimator=balanced_log_reg, X=X, y=y, cv=cv, scoring='roc_auc')
bal_log_reg_roc = np.mean(scores)

print 'Default logistic regression accuracy  = {:.6f}, AUC = {:.6f}'.format(def_log_reg_acc, def_log_reg_roc)
print 'Balanced logistic regression accuracy = {:.6f}, AUC = {:.6f}'.format(bal_log_reg_acc, bal_log_reg_roc)

# Train a linear SVM model using the balanced class weights 

default_lin_svm = LinearSVC(class_weight=None)
balanced_lin_svm = LinearSVC(class_weight='balanced')

scores = cross_val_score(estimator=default_lin_svm, X=X, y=y, cv=cv,  scoring='roc_auc')
def_lin_svm_roc = np.mean(scores)

scores = cross_val_score(estimator=balanced_lin_svm, X=X, y=y, cv=cv, scoring='roc_auc')
bal_lin_svm_roc = np.mean(scores)

print 'Default linear SVM ROC = {:.6f}'.format(def_lin_svm_roc)
print 'Balanced linear SVM ROC = {:.6f}'.format(bal_lin_svm_roc)

# Train a Decision tree classifier

default_tree = DecisionTreeClassifier(class_weight=None)
balanced_tree = DecisionTreeClassifier(class_weight='balanced')

scores = cross_val_score(estimator=default_tree, X=X, y=y, cv=cv, scoring='roc_auc')
def_tree_roc = np.mean(scores)

scores = cross_val_score(estimator=balanced_tree, X=X, y=y, cv=cv, scoring='roc_auc')
bal_tree_roc = np.mean(scores)

print 'Default decision tree ROC = {:.6f}'.format(def_tree_roc)
print 'Balanced decision tree ROC = {:.6f}'.format(bal_tree_roc)

# Load in the R pre-processed dataframes, check AUC on each
print 'Loading R pre-processed dataframes'
credit_df_cnn = pd.read_csv('cs-training-CNN.csv')
credit_df_oss = pd.read_csv('cs-training-OSS.csv')
credit_df_smote = pd.read_csv('cs-training-smote.csv')
credit_df_tomek = pd.read_csv('cs-training-tomek.csv')

print 'Creating logistic regression setup'
bal_log_reg = LogisticRegression(class_weight='balanced')
# cv = StratifiedKFold(y, n_folds=5, shuffle=True)

eval_data = [credit_df_cnn, 
             credit_df_oss, 
             credit_df_smote, 
             credit_df_tomek]

eval_title = ['CNN', 'OSS', 'SMOTE', 'Tomek']

eval_results = {}

print 'Running evaluation of processed dataframes'
for data_idx, data in enumerate(eval_data):
    print '-> {} of {}'.format(data_idx+1, len(eval_data))
    data = cleanRProcessedCreditDf(data)
    X, y, id_val = removeIDTargetFromCreditDf(data)
    cv = StratifiedKFold(y, n_folds=5, shuffle=True)
    scores = cross_val_score(estimator=bal_log_reg, X=X, y=y, cv=cv, scoring='roc_auc')
    eval_results[eval_title[data_idx]] = np.mean(scores)
    
print 'Baseline logistic regression AUC = {:.6f}'.format(bal_log_reg_roc)


for result, value in eval_results.items():
    print '{} AUC = {:.6f}'.format(result, value)

# Show a bar chart of the ROC results for each of the algorithms

ROC_labels = ['logreg', 'lin_svm', 'tree', 'logreg\n+CNN', 
            'logreg\n+OSS', 'logreg\n+SMOTE', 'logreg\n+Tomek']
ROC_vals = [bal_log_reg_roc, bal_lin_svm_roc, bal_tree_roc, 
            eval_results['CNN'], eval_results['OSS'], 
            eval_results['SMOTE'], eval_results['Tomek']]

width = 0.5
ind = np.arange(len(ROC_labels))
base_roc = [bal_log_reg_roc, bal_lin_svm_roc, bal_tree_roc, 
            bal_log_reg_roc, bal_log_reg_roc, 
            bal_log_reg_roc, bal_log_reg_roc]
inc_roc = [0,0,0, eval_results['CNN']-bal_log_reg_roc, 
           eval_results['OSS']-bal_log_reg_roc, 
            eval_results['SMOTE']-bal_log_reg_roc, 
           eval_results['Tomek']-bal_log_reg_roc]

p1 = plt.bar(ind, base_roc, color='b')
p2 = plt.bar(ind, inc_roc, color='g', bottom = base_roc)
plt.ylabel('AUC', fontsize=18)
plt.title('AUC by algorithm and preprocessing type', fontsize=20)
plt.xticks(ind + width/2, ROC_labels, fontsize=14)
plt.yticks(np.arange(0, 1.3, 0.1))
plt.legend((p1[0], p2[0]), ('Balanced algorithm', 'Preprocessing'), fontsize=14)
plt.show()

