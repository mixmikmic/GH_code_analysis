import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().magic('matplotlib inline')

# Loading the data saved from the last notebook
X_train = np.load('./_data/X_train.npy')
y_train = np.load('./_data/y_train.npy')
X_val = np.load('./_data/X_val.npy')
y_val = np.load('./_data/y_val.npy')
X_test = np.load('./_data/X_test.npy')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Instatiate and fit the logistic regression model
logr = LogisticRegression()
logr.fit(X_train,y_train)

print(logr.predict_proba(X_train).shape)
logr.predict_proba(X_train)

# Print shape of the X_Train
print(logr.predict_proba(X_train)[:,1].shape)
logr.predict_proba(X_train)[:,1]

# Find the ROC/AUC on one column of the logistic regression predict_proba on the training dataset
roc_auc_score(y_train,logr.predict_proba(X_train)[:,1])

# Find the AUC on one column of the logistic regression predict_proba on the validation dataset
roc_auc_score(y_val,logr.predict_proba(X_val)[:,1])

from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.preprocessing import PolynomialFeatures

# Set up feature selection by selecting most relative 50 percent of features
select_feature = SelectPercentile(f_classif, percentile=50)
# Fit the feature selection model
select_feature.fit(X_train,y_train)

# Now we'll make an array of booleans for those upper 50 percent of features
feature_filter = select_feature.get_support()

# Plot the filter results
sns.countplot(feature_filter)
plt.title('Features Filtered')
plt.savefig('./_assets/3-1-feature-filter')
plt.show()

# Here is the mean and array of f_classif p-values of the features
print('f_classif p-values mean:', select_feature.pvalues_.mean())

# Plot histogram of p-value scores
sns.distplot(select_feature.pvalues_, kde=False)
plt.axvline(.05, linestyle='dashed', label='p-val of 0.05', color='g')
plt.axvline(select_feature.pvalues_.mean(), label='mean', linestyle='dashed', color='r')
plt.title('Histogram of p-value scores')
plt.legend()
plt.savefig('./_assets/3-2-hist-pvals')
plt.show()

# Here is the mean and array of f_classif scores of the features
print('f_classif scores mean:', select_feature.scores_.mean())

# Plot histogram of f_classif scores
sns.distplot(select_feature.scores_, kde=False)
plt.axvline(select_feature.scores_.mean(), label='mean', linestyle='dashed', color='r')
plt.title('Histogram of f_classif feature scores')
plt.legend()
plt.savefig('./_assets/3-3-hist-f_classif-scores')
plt.show()

# Create array of booleans of features with scores greater than 10
feature_filter = select_feature.scores_ > 10

# Print sum of 'True' in feature_filter
print('Number of filtered features:', np.sum(feature_filter))
print('Number of un-filtered features:', len(feature_filter) - np.sum(feature_filter))

# Plot the filter results
sns.countplot(feature_filter)
plt.title('Features Filtered by Threshold')    
plt.savefig('./_assets/3-4-feature-filter-threshold')
plt.show()

# Generate a new feature matrix consisting of all polynomial combinations of
# the features with degrees less than or equal to 2
interactions = PolynomialFeatures(degree=2, interaction_only=True)
X_interactions = interactions.fit_transform(X_train[:,feature_filter])
print('No of features and their interactions:', X_interactions.shape[1])

# Fit_transform to see if there's any relationships in the model
logr.fit(X_interactions, y_train)
X_val_filtered = interactions.fit_transform(X_val[:,feature_filter])
# Print validation score on the filtered X
roc_auc_score(y_val, logr.predict_proba(X_val_filtered)[:,1])

# np array of whether or not the feature is important
feature_filter

counter = -1
important_features = []
for i in feature_filter:
    counter += 1
    if i == True:
        important_features.append(counter)
print('Number of important features:', len(important_features))
print('List of important features:', important_features)



