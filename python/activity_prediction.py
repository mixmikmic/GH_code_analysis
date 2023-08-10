import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix
from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

with open('/Users/nickbecker/Downloads/HAPT Data Set/features.txt') as handle:
    features = handle.readlines()
    features = map(lambda x: x.strip(), features)

with open('/Users/nickbecker/Downloads/HAPT Data Set/activity_labels.txt') as handle:
    activity_labels = handle.readlines()
    activity_labels = map(lambda x: x.strip(), activity_labels)

activity_df = pd.DataFrame(activity_labels)
activity_df = pd.DataFrame(activity_df[0].str.split(' ').tolist(),
                           columns = ['activity_id', 'activity_label'])
activity_df

x_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/X_train.txt',
             header = None, sep = " ", names = features)
x_train.iloc[:10, :10].head()

y_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/y_train.txt',
             header = None, sep = " ", names = ['activity_id'])
y_train.head()

x_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/X_test.txt',
             header = None, sep = " ", names = features)
x_test.iloc[:10, :10].head()

y_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/y_test.txt',
             header = None, sep = " ", names = ['activity_id'])
y_test.head()

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

C_params = np.logspace(-6, 3, 10)
svc_2 = LinearSVC(random_state = 12)

train_scores, test_scores = validation_curve(
    svc_2, x_train.values, y_train.values.flatten(),
    param_name="C", param_range=C_params,
    cv=5, scoring="accuracy", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

import seaborn as sns

y_min = 0.5
y_max = 1.0

f = plt.figure(figsize = (12, 8))
ax = plt.axes()
sns.set(font_scale = 1.25)
sns.set_style("darkgrid")
plt.title("SVM Training and Validation Accuracy")
plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + .01, .05))
plt.semilogx(C_params, train_scores_mean, label="CV Training Accuracy", color="red")
plt.fill_between(C_params, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="red")
plt.semilogx(C_params, test_scores_mean, label="CV Validation Accuracy",
             color="green")
plt.fill_between(C_params, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="green")
plt.legend(loc="best")
plt.show()

from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf.fit(x_train.values, y_train.values.flatten())

print clf.best_estimator_
print clf.best_params_
print clf.best_score_

clf.score(x_test, y_test)

y_test.activity_id.value_counts().values[0] / y_test.activity_id.value_counts().values.sum()

crosstab = pd.crosstab(y_test.values.flatten(), clf.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab

#percentages_crosstab = crosstab.iloc[:-1, :-1].apply(lambda x: x/x.sum(), axis = 1)
percentages_crosstab = crosstab.iloc[:-1, :-1]
percentages_crosstab.columns = activity_df.activity_label.values
percentages_crosstab.index = activity_df.activity_label.values
percentages_crosstab

y_train['walking_flag'] = y_train.activity_id.apply(lambda x: 1 if x <= 3 else 0)
y_test['walking_flag'] = y_test.activity_id.apply(lambda x: 1 if x <= 3 else 0)

from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf_binary = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf_binary.fit(x_train.values, y_train.walking_flag.values.flatten())

print clf_binary.best_estimator_
print clf_binary.best_params_
print clf_binary.best_score_

clf_binary.score(x_test, y_test.walking_flag)

crosstab_binary = pd.crosstab(y_test.walking_flag.values.flatten(), clf_binary.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab_binary



