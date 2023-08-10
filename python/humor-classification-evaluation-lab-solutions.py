import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

hsq = pd.read_csv('/Users/david.yan/hello_world/week-05/evaluation-classifiers_confusion_matrix_roc-lab/datasets/hsq_data.csv')

# correct spelling
hsq.rename(columns={'agressive':'aggressive'}, inplace=True)

hsq.shape

# looks like there are 4 genders but most are just male and female
print hsq.gender.unique()
print hsq.gender.value_counts()

# Set any of the -1 values in the question answers to np.nan
for col in [c for c in hsq.columns if c.startswith('Q')]:
    hsq[col] = hsq[col].map(lambda x: np.nan if x == -1 else x)

# check null values:
hsq.isnull().sum()

# drop the nulls
hsq.dropna(inplace=True)
print hsq.shape

hsq.age.unique()

# set hsq to be only valid ages:
hsq = hsq[hsq.age <= 100]

# only keep male and female
hsq = hsq[hsq.gender.isin([1,2])]

hsq.shape

# not including the "aggregate" measures (affiliative, selfenhancing, etc.) as they are combinations
# of the original questions.
predictors = [x for x in hsq.columns if 'Q' in x]
predictors = predictors + ['age', 'accuracy']
print predictors

# set up y variable
y = hsq.gender.map(lambda x: 1 if x == 1 else 0)

X = hsq[predictors]

# baseline:
print np.mean(y)

# good practice to standardize! 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xs = ss.fit_transform(X)

from sklearn.model_selection import cross_val_score

lr = LogisticRegression()

# using a 25-fold cross-val for fun
scores = cross_val_score(lr, Xs, y, cv=25)
print scores
print np.mean(scores)

# this is higher than the baseline accuracy. 54% --> 60%
# not bad.

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.5)
lr.fit(X_train, y_train)

# predictions and pred prob.
yhat = lr.predict(X_test)
yhat_pp = lr.predict_proba(X_test)

tp = np.sum((y_test == 1) & (yhat == 1))
fp = np.sum((y_test == 0) & (yhat == 1))
tn = np.sum((y_test == 0) & (yhat == 0))
fn = np.sum((y_test == 1) & (yhat == 0))
print tp, fp, tn, fn

conmat = np.array(confusion_matrix(y_test, yhat, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['is_male', 'is_female'],
                         columns=['predicted_male','predicted_female'])
confusion

pp = pd.DataFrame(yhat_pp, columns=['female','male'])
pp.head()

for thresh in np.arange(1,100)/100.:
    labeled_male = np.array([1 if x >= thresh else 0 for x in pp.male.values])
    print 'Threshold:', thresh, 'false positives:', np.sum((y_test == 0) & (labeled_male == 1))

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, yhat_pp[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=[8,8])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is male', fontsize=18)
plt.legend(loc="lower right")
plt.show()

from sklearn.linear_model import LogisticRegressionCV

lr_ridge = LogisticRegressionCV(penalty='l2', Cs=200, cv=25)
lr_ridge.fit(X_train, y_train)

print lr_ridge.C_

yhat_ridge = lr_ridge.predict(X_test)
yhat_ridge_pp = lr_ridge.predict_proba(X_test)

conmat = np.array(confusion_matrix(y_test, yhat_ridge, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['is_male', 'is_female'],
                         columns=['predicted_male','predicted_female'])
confusion

fpr_ridge, tpr_ridge, _ = roc_curve(y_test, yhat_ridge_pp[:,1])
roc_auc_ridge = auc(fpr_ridge, tpr_ridge)

plt.figure(figsize=[8,8])

plt.plot(fpr, tpr, label='Original (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot(fpr_ridge, tpr_ridge, label='Ridge (area = %0.2f)' % roc_auc_ridge, 
         linewidth=4, color='darkred')

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is male', fontsize=18)
plt.legend(loc="lower right")
plt.show()

# Looks like they perform the same on my training/testing set.

lr_lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=100, cv=10)
lr_lasso.fit(X_train, y_train)

print lr_lasso.C_

yhat_lasso = lr_lasso.predict(X_test)
yhat_lasso_pp = lr_lasso.predict_proba(X_test)

conmat = np.array(confusion_matrix(y_test, yhat_lasso, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['is_male', 'is_female'],
                         columns=['predicted_male','predicted_female'])
confusion

fpr_lasso, tpr_lasso, _ = roc_curve(y_test, yhat_lasso_pp[:,1])
roc_auc_lasso = auc(fpr_lasso, tpr_lasso)

plt.figure(figsize=[8,8])

plt.plot(fpr, tpr, label='Original (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot(fpr_ridge, tpr_ridge, label='Ridge (area = %0.2f)' % roc_auc_ridge, 
         linewidth=4, color='darkred')
plt.plot(fpr_lasso, tpr_lasso, label='Lasso (area = %0.2f)' % roc_auc_lasso, 
         linewidth=4, color='darkgoldenrod')

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is male', fontsize=18)
plt.legend(loc="lower right")
plt.show()

coefs_vars = pd.DataFrame({
        'coef':lr_lasso.coef_[0],
        'variable':X.columns,
        'abscoef':np.abs(lr_lasso.coef_[0])
    })
coefs_vars.sort_values('abscoef', ascending=False, inplace=True)
coefs_vars

# Q15 is by far the most important predictor (largest coef)
# 15. I do not like it when people use humor as a way of criticizing or putting someone down.
# A higher score on this predicts female.

