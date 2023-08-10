import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

hsq = pd.read_csv('./datasets/hsq_data.csv')

# A:
hsq.head()

hsq.info()

hsq.describe()

# Check the target column
hsq['gender'].value_counts()

# Check the age column
hsq['age'].value_counts().sort_values()

# drop all rows containing 3 and 0 for gender
temp_df = hsq[hsq['gender'].isin([1,2])]
# transform 1=male, 2=female to 0=male, 1=female
temp_df['gender'] = temp_df['gender'].map(lambda x: 1 if x == 1 else 0)
temp_df['gender'].value_counts()

# nan for age columns that are above 100
temp_df['age'] = temp_df['age'].map(lambda x: np.nan if x > 100 else x)

# nan for answer that are -1
cols = [x for x in temp_df.columns if x.startswith('Q')]
for col in cols:
    temp_df[col] = temp_df[col].map(lambda x: np.nan if x == -1 else x)

temp_df.dropna(inplace=True)

temp_df['age'].value_counts()

temp_df.info()

temp_df.describe()

# A:
# remove summary columns because we want our predictors to be as granular as possible
df = temp_df.drop(labels=['affiliative','selfenhancing','agressive','selfdefeating'], axis=1)

# get target and predictor columns
y = df['gender']
X = df.drop(labels='gender',axis=1)

# get baseline acc
baseline_acc = y.mean()
baseline_acc

# initialize standardization protocol
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# standardize predictors
Xs = scaler.fit_transform(X)

# get cross validated accuracy, total true / total sample size
lrm = LogisticRegression()
scores = cross_val_score(lrm, Xs, y, cv=25)

print(scores)
print(np.mean(scores))

# only slightly better than baseline, not very good

# A:
from sklearn.model_selection import train_test_split

# Get Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.5)

# Fit the logistic model
lrm.fit(X_train,y_train)

# get predictions & predicted probabilities
pred = lrm.predict(X_test)
pred_prob = lrm.predict_proba(X_test)

print(len(pred == 1))
print(np.sum(pred == 1))
print(np.sum(y_test == 1))
print(np.sum((y_test==1) & (pred==1)))

# A:
tp = np.sum((y_test==1) & (pred==1))
fp = np.sum((y_test==0) & (pred==1))
tn = np.sum((y_test==0) & (pred==0))
fn = np.sum((y_test==1) & (pred==0))
print(tp, fp, tn, fn)

# A:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred, labels=[1,0])
cm_df = pd.DataFrame(cm, index=['is_male','is_female'], columns=['predicted_male','predicted_female'])
cm_df

# A:
pp_df = pd.DataFrame(pred_prob, columns=['female','male'])
pp_df.head()

for thresh in np.arange(1,100)/100.:
    male_label = np.array([1 if x >= round(thresh,2) else 0 for x in pp_df.male.values])
    print('Threshold: ', round(thresh,2), ', False Positive = ', np.sum((y_test==0) & (male_label==1)))

from sklearn.metrics import roc_curve, auc

# A:
fpr, tpr, _ = roc_curve(y_test, pred_prob[:,1])
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

# A:
lr_ridge = LogisticRegressionCV(penalty='l2', Cs=200, cv=25)
lr_ridge.fit(X_train, y_train)

print lr_ridge.C_

# A:
ridge_pred = lr_ridge.predict(X_test)
ridge_pp = lr_ridge.predict_proba(X_test)

cm_df

# A:
ridge_cm = confusion_matrix(y_test, ridge_pred, labels=[1,0])
ridge_cm_df = pd.DataFrame(ridge_cm, index=['is_male','is_female'], columns=['predicted_male','predicted_female'])
ridge_cm_df

ridge_fpr, ridge_tpr, _ = roc_curve(y_test, ridge_pp[:,1])
ridge_roc_auc = auc(ridge_fpr, ridge_tpr)

# A:
plt.figure(figsize=[8,8])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot(ridge_fpr, ridge_tpr, label='Ridge ROC curve (area = %0.2f)' % ridge_roc_auc, linewidth=4, color='darkred')

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is male', fontsize=18)
plt.legend(loc="lower right")
plt.show()

# A:
lr_ridge = LogisticRegressionCV(penalty='l1', Cs=200, cv=25,solver='liblinear')
lr_ridge.fit(X_train, y_train)

lr_ridge.C_

# A:
lr_pred = lr_ridge.predict(X_test)
lr_pp = lr_ridge.predict_proba(X_test)

lr_cm = confusion_matrix(y_test, lr_pred, labels=[1,0])
lr_cm_df = pd.DataFrame(lr_cm, index=['is_male','is_female'], columns=['predicted_male','predicted_female'])
lr_cm_df

# A:
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pp[:,1])
lr_roc_auc = auc(lr_fpr, lr_tpr)

plt.figure(figsize=[8,8])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
plt.plot(ridge_fpr, ridge_tpr, label='Ridge ROC curve (area = %0.2f)' % ridge_roc_auc, linewidth=4, color='darkred')
plt.plot(lr_fpr, lr_tpr, label='Lasso ROC curve (area = %0.2f)' % lr_roc_auc, linewidth=4, color='darkorange')

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is male', fontsize=18)
plt.legend(loc="lower right")
plt.show()

lr_ridge.coef_.shape

# A:
lr_betas = pd.DataFrame({'predictors':X.columns,'coef':lr_ridge.coef_[0],'abs_coef':np.abs(lr_ridge.coef_[0])})
lr_betas.sort_values('abs_coef', ascending=False, inplace=True)
lr_betas

# Question 15 has the most effect in the model
# I do not like it when people use humor as a way of criticizing or putting someone down.
# Higher score, meaning the more the surveyer agrees with this statement, 
# indicates higher likelihood of surveyer being female with all other variables constant

