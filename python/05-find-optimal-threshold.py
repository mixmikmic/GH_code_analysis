import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score

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

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Instatiate and fit the logistic regression model
logr = LogisticRegression()
logr.fit(X_train,y_train)

# Build a function that makes ranges with floats
def frange(start, stop, step, roundval):
    i = start
    while i < stop:
        yield round(i, roundval)
        i += step

# Make list of thresholds to test
threshold_list = list(frange(0.01, .13, .01, 2))
threshold_list

features_list = []
rocauc_list = []

for thresh in threshold_list:
    stability_selection = RandomizedLogisticRegression(n_resampling=300,
                                                       n_jobs=1,
                                                       random_state=101,
                                                       scaling=0.15,
                                                       sample_fraction=0.50,
                                                       selection_threshold=thresh)
    interactions = PolynomialFeatures(degree=4, interaction_only=True)
    model = make_pipeline(stability_selection, interactions, logr)
    model.fit(X_train, y_train)
    feature_filter = model.steps[0][1].all_scores_ >= thresh
    
    counter = -1
    important_features = []
    for i in feature_filter:
        counter += 1
        if i == True:
            important_features.append(counter)
    print('Number of important features:', len(important_features))
    print('List of important features:', important_features)
    
    
    
    features_list.append(np.sum(model.steps[0][1].all_scores_ >= thresh))
    rocauc_list.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

# I should have saved the list of important features for each threshold as a dictionary or something
# but this code takes like 30 minutes to run and I am out of time

results_df = pd.DataFrame({'threshold': threshold_list,
                           'score': rocauc_list,
                           'no-of-features': features_list})
results_df.set_index('threshold',inplace=True)
results_df

# Pickle DataFrame
results_df.to_pickle('./_data/results_df.p')

# Plot ROC-AUC score vs number of features
plt.plot(features_list, rocauc_list)
plt.axvline(13, c='r', label='13 features\nroc-auc: -.91')
plt.xlabel('No. of Features')
plt.ylabel('ROC-AUC Score')
plt.title('ROC-AUC Scores\nvs No. of Features')
plt.legend()
plt.savefig('./_assets/5-2-rocauc-scores-vs-features.png')
plt.show()



